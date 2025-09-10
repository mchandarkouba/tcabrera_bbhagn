### This module handles the simulation of a follow-up suite.

### Inputs:
#     - DF_GW (GWname --> skymap, f_cover)
#     - sim config (lambda, H0, Om0, AGN distribution, flare model)

### Outputs:
# - GW events
# - Flares

### Steps:
# Select a number of GW events; assign times if needed
# Draw background flares from AGN distribution in GW volumes
# - Define easily sampleable volume (i.e. cosmo sphere)
# - Determine average number of flares in volume; draw from Poisson
# - Sample times for flares
# - Sample locations for flares from whole volume
# - Keep flares within GW volume (if f_cover available, randomly drop flares appropriately)
# Select GW events producing AGN flares
# - Draw flare time
# - Draw flare location from GW skymap

# Do a version where only massive events produce flares as well, and see if it is distinguishable from the other version

import multiprocessing as mp
import os.path as pa
import sys

import astropy.units as u
import astropy_healpix as ah
import ligo.skymap.distance as lsm_dist
import ligo.skymap.moc as lsm_moc
import numpy as np
import pandas as pd
import parmap
import yaml
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM, z_at_value
from astropy.io import fits
from astropy.time import Time
from ligo.skymap.io import read_sky_map
from ligo.skymap.postprocess.crossmatch import crossmatch

PROJDIR = pa.dirname(pa.dirname(__file__))
sys.path.append(PROJDIR)
from myagn import distributions as myagndistributions


def draw_continuous_from_discrete(
    x_grid,
    x_pdf,
    rng=np.random.default_rng(12345),
    n_draws=1,
):
    """Chooses a bin by trapezoid probability per bin, then draws from linear distribution within bin."""
    # Calculate probabilities per bin
    bin_probs = 0.5 * (x_pdf[:-1] + x_pdf[1:]) * (x_grid[1:] - x_grid[:-1])
    bin_probs /= np.sum(bin_probs)
    # Choose bins
    idxs = rng.choice(np.arange(len(bin_probs)), p=bin_probs, size=n_draws)
    # Draw uniformly from bin (tried with trapezoidal area, but a bit complicated)
    fracs = rng.uniform(0, 1, n_draws)
    x_los = x_grid[idxs]
    x_his = x_grid[idxs + 1]
    draws = x_los + fracs * (x_his - x_los)
    if n_draws == 1:
        return draws[0]
    else:
        return draws


def _crossmatch_skymap(
    gw_row,
    scs,
    cosmo,
):
    """Generates mask for flares in GW volume.
    Uses the df_gw_input and sc_flares variables defined above.
    Assumes that the f_cover for each skymap corresponds to the highest probability interval;
    in practice this is not always the case due to observational constraints.

    Parameters
    ----------
    sm_index : int
        Integer indexing into df_gw_input.

    Returns
    -------
    _type_
        Boolean mask on df_gw_input; True if flare is in GW volume.
    """
    # gw_row = df_gw_input.iloc[gw_idx]
    sm = read_sky_map(gw_row["skymap_path"], moc=True)
    cm = crossmatch(sm, scs, cosmology=True, cosmo=cosmo)
    mask = cm.searched_prob_vol <= gw_row["f_cover"]
    return mask


def _draw_gw_flare_coords(
    gw_row,
    dl_grid,
    dt_followup,
    cosmo,
):
    """Draws flare coordinates from GW skymap.
    Uses the df_gw_input and sc_flares variables defined above.

    Parameters
    ----------
    gw_idx : int
        Integer indexing into df_gw_input.

    Returns
    -------
    _type_
        SkyCoord object for flare.
    """
    rng = np.random.default_rng(gw_row["rng_seed"])
    # Draw time
    t = gw_row["t"] + rng.uniform(0, dt_followup)
    # Draw location
    # Draw pixel
    sm = read_sky_map(gw_row["skymap_path"], moc=True)
    sm["PIXAREA"] = lsm_moc.uniq2pixarea(sm["UNIQ"])
    sm["PROB"] = sm["PROBDENSITY"] * sm["PIXAREA"]
    hpx_idx = rng.choice(
        np.arange(len(sm)),
        p=sm["PROB"] / np.sum(sm["PROB"]),
    )
    hpx_row = sm[hpx_idx]
    uniq = hpx_row["UNIQ"]
    distmu = hpx_row["DISTMU"]
    distsigma = hpx_row["DISTSIGMA"]
    distnorm = hpx_row["DISTNORM"]
    del sm
    # Draw RA, dec
    level, ipix = lsm_moc.uniq2nest(uniq)
    nside = ah.level_to_nside(level)
    ra, dec = ah.healpix_to_lonlat(
        ipix,
        nside,
        dx=rng.uniform(0, 1),
        dy=rng.uniform(0, 1),
        order="nested",
    )
    ra = ra.to(u.deg).value
    dec = dec.to(u.deg).value
    # Draw redshift
    dl_pdf = lsm_dist.conditional_pdf(
        dl_grid.to(u.Mpc).value,
        [distmu] * len(dl_grid),
        [distsigma] * len(dl_grid),
        [distnorm] * len(dl_grid),
    )
    dl = draw_continuous_from_discrete(
        dl_grid.to(u.Mpc).value,
        dl_pdf,
        rng=rng,
    )
    z = z_at_value(cosmo.luminosity_distance, dl * u.Mpc)
    # Return
    return {
        "mjd": t,
        "ra": ra,
        "dec": dec,
        "Redshift": z.value,
    }


def simulate_flares(
    lamb,
    cosmo,
    df_gw_path,  # Should include skymap_path, t, f_cover
    agn_dist,
    dt_followup,
    z_grid=np.linspace(0, 2, 1000),
    rng=np.random.default_rng(12345),
    nproc=16,
):
    # Setup
    dl_grid = cosmo.luminosity_distance(z_grid)
    dcm_grid = cosmo.comoving_distance(z_grid)
    df_gw_input = pd.read_csv(df_gw_path)

    # Get GW times
    ts = []
    for _, row in df_gw_input.iterrows():
        with fits.open(row["skymap_path"]) as hdul:
            t = Time(hdul[1].header["MJD-OBS"], format="mjd").mjd
            ts.append(t)
    df_gw_input["t"] = ts

    print("Generating background flares...")
    # Background flares
    # Determine number of flares
    dn_dz = 4 * np.pi * u.sr * agn_dist.dn_dOmega_dz(z_grid)
    n_flares_bg_avg = np.trapezoid(dn_dz, z_grid)
    n_flares_bg_avg = 100
    n_flares_bg = rng.poisson(n_flares_bg_avg)
    print(f"Average number of background flares: {n_flares_bg_avg}")
    # Draw times
    t_min_bg = np.nanmin(df_gw_input["t"])
    t_max_bg = np.nanmax(df_gw_input["t"])
    t_flares_bg = rng.uniform(t_min_bg, t_max_bg + dt_followup, n_flares_bg) * u.day
    # Draw RAs, decs
    ra_flares_bg = rng.uniform(0, 360, n_flares_bg)
    dec_flares_bg = 180 / np.pi * np.arcsin(rng.uniform(-1, 1, n_flares_bg))
    # Draw redshifts
    z_flares_bg = draw_continuous_from_discrete(
        z_grid,
        dn_dz.value,
        rng=rng,
        n_draws=n_flares_bg,
    )
    dl_flares_bg = cosmo.luminosity_distance(z_flares_bg)
    sc_flares_bg = SkyCoord(
        ra_flares_bg,
        dec_flares_bg,
        unit=(u.deg, u.deg, dl_flares_bg.unit),
        distance=dl_flares_bg.value,
    )

    # Keep flares in GW volumes (using f_cover)
    print("Crossmatching flares with GW skymaps...")
    df_rows = [row for _, row in df_gw_input.iterrows()]
    cm_masks = parmap.map(
        _crossmatch_skymap,
        df_rows,
        sc_flares_bg,
        cosmo,
        pm_processes=nproc,
    )
    cm_mask = np.any(cm_masks, axis=0)
    df_flares_bg = pd.DataFrame(
        {
            "mjd": t_flares_bg,
            "ra": ra_flares_bg,
            "dec": dec_flares_bg,
            "Redshift": z_flares_bg,
        }
    )[cm_mask]
    df_flares_bg["gw"] = False

    print("Generating GW flares...")
    # GW flares
    # Determine number of flares
    n_flares_gw_avg = lamb * df_gw_input.shape[0]
    print(f"Average number of GW flares: {n_flares_gw_avg}")
    n_flares_gw = rng.poisson(n_flares_gw_avg)
    # Select GW events
    gw_em_idxs = rng.choice(
        np.arange(df_gw_input.shape[0]),
        n_flares_gw,
        replace=False,
    )

    print("Drawing GW flare coordinates...")
    # Draw coordinates
    df_temp = df_gw_input.copy()
    df_temp["rng_seed"] = rng.integers(0, 2**32 - 1, df_temp.shape[0])
    gw_rows = [df_temp.iloc[i] for i in gw_em_idxs]
    gw_flare_coords = parmap.map(
        _draw_gw_flare_coords,
        gw_rows,
        dl_grid,
        dt_followup,
        cosmo,
        pm_processes=nproc,
    )
    df_flares_gw = pd.DataFrame(gw_flare_coords)
    df_flares_gw["gw"] = True

    # Return
    df_flares = pd.concat([df_flares_bg, df_flares_gw], ignore_index=True)
    df_flares["flarename"] = np.arange(df_flares.shape[0])
    return df_flares


def main(config_path):
    config = yaml.safe_load(open(config_file))

    # Parse AGN distribution config
    for k, v in config["agn_distribution"].items():
        if v["model"] == "ConstantPhysicalDensity":
            v["args"] = v["args"] * u.Mpc**-3
        if "brightness_limits" in v["density_kwargs"]:
            if "brightness_units" not in v["density_kwargs"]:
                raise ValueError(
                    "Must specify brightness_units if brightness_limits is given."
                )
            if v["density_kwargs"]["brightness_units"] == "ABmag":
                bu = u.ABmag
            elif v["density_kwargs"]["brightness_units"] == "erg/s":
                bu = u.erg / u.s
            else:
                raise ValueError("brightness_units must be 'ABmag' or 'erg/s'.")
            v["density_kwargs"]["brightness_limits"] = [
                float(bl) for bl in v["density_kwargs"]["brightness_limits"]
            ] * bu
            v["density_kwargs"].pop("brightness_units")

    # Setup
    cosmo = FlatLambdaCDM(H0=config["H0"], Om0=config["Om0"])
    agndist_config = config["agn_distribution"]
    agn_dist = getattr(
        myagndistributions,
        agndist_config["observed"]["model"],
    )(
        *agndist_config["observed"]["args"],
        **agndist_config["observed"]["kwargs"],
    )

    # Simulate
    df_flares = simulate_flares(
        config["lambda"],
        cosmo,
        config["gw_csv"],
        agn_dist,
        config["dt_followup"],
    )

    # Save
    df_flares.to_csv(pa.join(pa.dirname(config_path), "flares.csv"), index=False)

    # Done
    print("Done.")
    return


if __name__ == "__main__":

    # Load yaml file
    config_file = sys.argv[
        1
    ]  # "/hildafs/home/tcabrera/HIPAL/bbhagn/bbhagn/config.yaml"
    main(config_file)
