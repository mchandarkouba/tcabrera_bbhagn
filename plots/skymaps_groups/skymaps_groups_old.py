import os.path as pa
import sys
import time

import healpy as hp
import ligo.skymap.io.fits as lsm_fits
import ligo.skymap.moc as lsm_moc
import ligo.skymap.plot
import matplotlib
import matplotlib.lines as mpl_lines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import ICRS, Galactic, SkyCoord

# Local imports
sys.path.append(pa.dirname(pa.dirname(pa.dirname(__file__))))
import utils.io as io
from utils.paths import PROJDIR

# 20230310: This doesn't work with ligo.skymap.plot
plt.style.use(f"{PROJDIR}/plots/matplotlibrc.mplstyle")

###############################################################################


def hourangle_axis(axes):
    """Change the labels on the axis to hour angle.
    Assumes a format of [+-]XX°, by taking all but the last character.

    Parameters
    ----------
    axes : matplotlib.axes
        The axes to change the xaxis to hourangle
    """
    # Get the original labels
    labels = axes.get_xticklabels()

    # Convert to hourangle
    ha_labels = []
    for label in labels:
        ha_l = float(label.get_text()[:-1])
        ha_l = 12 / 180 * (ha_l % 360)
        ha_labels.append("%dh" % ha_l)

    # Set labels
    axes.set_xticklabels(ha_labels)

    return ha_labels


def hist2d_hpx_to_mollweide(
    axes,
    m,
    nest=False,
    xsize=1000,
    cutoff=0.0,
    kw_rotator={},
    kw_pcolormesh={},
):
    """Add a HEALPix map as a 2D histogram to an existing Mollweide axes.
    Largely inspired by hp.newvisufunc.projview.

    Parameters
    ----------
    axes : matplotlib.axes.Axes
        Axes to add histogram to
    m : _type_
        HEALPix map
    nest : bool or str, optional
        Indicate whether the HEALPix map is NESTED, by default False (RING)
        Can also be a string to specify other schemes, such as "NUNIQ".
    xsize : int, optional
        Number of bins to use in phi dimension, by default 1000.
        The number of theta bins will be set to half this, rounded down.
    kw_rotator : dict, optional
        Keyword arguments to pass to healpy rotator, by default {}
    kw_pcolormesh : dict, optional
        Keyword arguments to pass to matplotlib pcolormesh, by default {}

    Returns
    -------
    _type_
        _description_
    """

    # Make the theta, phi meshgrid
    ysize = xsize // 2
    phi = np.linspace(-np.pi, np.pi, xsize)
    theta = np.linspace(np.pi, 0, ysize)
    phi, theta = np.meshgrid(phi, theta)

    # Rotate meshgrid if needed
    if kw_rotator:
        r = hp.rotator.Rotator(**kw_rotator)
        theta, phi = r(theta.flatten(), phi.flatten())
        theta = theta.reshape(ysize, xsize)
        phi = phi.reshape(ysize, xsize)

    # Convert to NESTED if NUNIQ
    if nest == "NUNIQ":
        for c in list(m.columns):
            if c not in ["UNIQ", "PROBDENSITY"]:
                m.remove_column(c)
        m = m.as_array()
        m = lsm_moc.rasterize(m)
        nest = True

    # Get map values at grid points
    nside = hp.pixelfunc.npix2nside(len(m))
    grid_pix = hp.pixelfunc.ang2pix(nside, theta, phi, nest=nest)
    grid_map = m[grid_pix].astype(float)

    # Mask the low values
    grid_map[grid_map < cutoff] = np.nan

    # Get longitude and latitude
    longitude = np.linspace(-np.pi, np.pi, xsize)
    latitude = np.linspace(-np.pi / 2, np.pi / 2, ysize)

    # Plot map
    ret = axes.pcolormesh(
        # phi,
        # theta,
        longitude,
        latitude,
        grid_map,
        **kw_pcolormesh,
    )

    return ret


def contour2d_hpx_to_mollweide(
    axes,
    m,
    nest=False,
    xsize=1000,
    levels=[0.68, 0.9],
    kw_rotator={},
    kw_contour={},
    kw_compute_contours={},
):
    """Add a HEALPix map as a 2D contour to an existing Mollweide axes.
    Largely inspired by hp.newvisufunc.projview.

    Parameters
    ----------
    axes : matplotlib.axes.Axes
        Axes to add histogram to
    m : _type_
        HEALPix map
    nest : bool or str, optional
        Indicate whether the HEALPix map is NESTED, by default False (RING)
        Can also be a string to specify other schemes, such as "NUNIQ".
    xsize : int, optional
        Number of bins to use in phi dimension, by default 1000.
        The number of theta bins will be set to half this, rounded down.
    kw_rotator : dict, optional
        Keyword arguments to pass to healpy rotator, by default {}
    kw_pcolormesh : dict, optional
        Keyword arguments to pass to matplotlib pcolormesh, by default {}

    Returns
    -------
    _type_
        _description_
    """

    # Make the theta, phi meshgrid
    ysize = xsize // 2
    phi = np.linspace(-np.pi, np.pi, xsize)
    theta = np.linspace(np.pi, 0, ysize)
    phi, theta = np.meshgrid(phi, theta)

    # Rotate meshgrid if needed
    if kw_rotator:
        r = hp.rotator.Rotator(**kw_rotator)
        theta, phi = r(theta.flatten(), phi.flatten())
        theta = theta.reshape(ysize, xsize)
        phi = phi.reshape(ysize, xsize)

    # Convert to NESTED if NUNIQ
    if nest == "NUNIQ":
        for c in list(m.columns):
            if c not in ["UNIQ", "LEVEL"]:
                m.remove_column(c)
        m = m.as_array()
        m = lsm_moc.rasterize(m)
        nest = True

    # Get map values at grid points
    nside = hp.pixelfunc.npix2nside(len(m))
    grid_pix = hp.pixelfunc.ang2pix(nside, theta, phi, nest=nest)
    grid_map = m[grid_pix].astype(float)

    # Get longitude and latitude (for ra/dec plot)
    longitude = np.linspace(-np.pi, np.pi, xsize)
    latitude = np.linspace(-np.pi / 2, np.pi / 2, ysize)

    # Plot map
    ret = axes.contour(
        longitude,
        latitude,
        grid_map,
        levels=levels,
        **kw_contour,
    )

    return ret


def get_credible_region_hpix(skymap, credible_fraction=0.9, moc=True):
    """Given a credible region fraction, finds the HEALPix that compose the credible region for the skymap,
    iteratively adding HEALPix by highest probabilities.

    Parameters
    ----------
    skymap : astropy.Table
        Skymap, as read with ligo.skymap.fits.read_sky_map
    credible_fraction : float, optional
        Credible fraction to get HEALPix of, by default 0.9
    moc : bool, optional
        If `True`, the skymap is treated as multi-order, by default True

    Returns
    -------
    astropy.Table
        Subset of original skymap, containing the rows that make up the credible region
    """
    # Get probs as pd.Series to preserve indexing
    # Sort by prob/probdensity
    # If multi-order, multiply by area afterwards
    if moc:
        probs = pd.DataFrame(skymap.as_array())
        probs.sort_values("PROBDENSITY", inplace=True)
        probs = probs["PROBDENSITY"] * lsm_moc.uniq2pixarea(probs["UNIQ"])
    else:
        probs = pd.Series(skymap["PROB"])
        probs.sort_values(inplace=True)

    # Paranoid normalize, and cumulative sum probs
    probs /= probs.sum()
    pd_cumsum = np.cumsum(probs)

    # Cut at credible fraction, inclusive
    mask = pd_cumsum > (1 - credible_fraction)
    is_credfrac = list(pd_cumsum[mask].index)

    return skymap[is_credfrac]


###############################################################################

# Groups
gwlists = [
    ["GW190424_180648", "GW190403_051519", "GW190514_065416", "GW190521"],
    ["GW190803_022701", "GW190909_114149*"],
]
flarelists = [
    ["J124942.30+344928.9", "J181719.94+541910.0", "J224333.95+760619.2"],
    ["J120437.98+500024.0"],
]

# Initialize figure
mosaic = np.arange(len(gwlists)).reshape((2, 1))
fig, axd = plt.subplot_mosaic(
    mosaic,
    figsize=(6, 6),
    subplot_kw={"projection": "astro hours mollweide"},
)

# Plot skymap
cmaps = [
    "Purples",
    "Greens",
    "Reds",
    "Blues",
]
credfracs = [0.9, 0.68]
for i, (axi, ax) in enumerate(axd.items()):
    # Title
    gwlist = gwlists[i]
    flarelist = flarelists[i]

    # Iterate over GWs
    legend_handles = []
    legend_labels = []
    for gi, g in enumerate(gwlist):
        # Load skymap; copy for contours
        skymap_path = io.get_gwtc_skymap_path(
            "/hildafs/projects/phy220048p/share/skymaps", g
        )
        skymap = lsm_fits.read_sky_map(skymap_path, moc=True)
        skymap_contours = pd.DataFrame(skymap.as_array()).set_index("UNIQ")
        skymap_contours["LEVEL"] = 1

        # Iterate over credible fractions; must be from largest to smallest
        for credfrac in credfracs:
            skymap_cr = get_credible_region_hpix(skymap, credible_fraction=credfrac)

            # Update contour skymap; this step is what requires the fractions to be descending
            uniqs_cr = [u for u in skymap_contours.index if u in skymap_cr["UNIQ"]]
            skymap_contours.loc[uniqs_cr, "LEVEL"] = credfrac

            # Shade area
            if credfrac == np.max(credfracs):
                hist2d_hpx_to_mollweide(
                    ax,
                    skymap_cr,
                    nest="NUNIQ",
                    cutoff=skymap_cr["PROBDENSITY"].min(),
                    kw_pcolormesh={
                        "rasterized": True,
                        "edgecolors": "face",
                        "cmap": cmaps[gi],
                        "alpha": 0.5,
                        "vmin": 0,
                    },
                    kw_rotator={
                        # "coord": "GC",
                    },
                )

        # Append levels to skymap_contours
        skymap["LEVEL"] = skymap_contours["LEVEL"]
        del skymap_contours

        # Plot contours
        contour2d_hpx_to_mollweide(
            ax,
            skymap,
            nest="NUNIQ",
            xsize=100,
            levels=credfracs[::-1],
            kw_contour={
                "cmap": cmaps[gi],
                "alpha": 0.5,
                "vmin": 0,
            },
            kw_rotator={
                # "coord": "GC",
            },
        )

        # Add artist/label to legend handles/labels
        legend_handles.append(
            mpl_lines.Line2D(
                [0],
                [0],
                color=matplotlib.cm.get_cmap(cmaps[gi])(0.75),
                lw=3,
            )
        )
        legend_labels.append(g)

    # Adjust longitude/latitude grids
    ax.grid(c="gray", alpha=0.5, zorder=-3)
    ax.set_longitude_grid(60)
    ax.set_longitude_grid_ends(90)
    ax.set_latitude_grid(30)

    # Plot flare locations
    ra_strs = [":".join([f[1:3], f[3:5], f[5:10]]) for f in flarelist]
    dec_strs = [":".join([f[10:13], f[13:15], f[15:]]) for f in flarelist]
    print("ra_strs:", ra_strs)
    print("dec_strs:", dec_strs)
    sc = SkyCoord(ra=ra_strs, dec=dec_strs, unit=(u.hourangle, u.deg))
    x = sc.ra.radian
    y = sc.dec.radian
    sc = sc.transform_to(Galactic)
    # x = sc.l.radian
    # y = sc.b.radian
    x[x > np.pi] = x[x > np.pi] - 2 * np.pi
    flare_handle = ax.plot(
        x,
        y,
        marker="X",
        markerfacecolor="gold",
        markeredgecolor="k",
        linestyle="",
        markersize=10,
        zorder=2,
        rasterized=True,
    )
    legend_handles.append(flare_handle[0])
    legend_labels.append("Flare(s)")

    # Change x-axis labels to hour angle
    hourangle_axis(ax)

    # Legend
    ax.legend(
        legend_handles,
        legend_labels,
        loc="lower left",
        frameon=True,
        shadow=False,
        framealpha=0.9,
        facecolor="w",
        edgecolor="w",
    )

# Clean up and save
plt.tight_layout()
plt.savefig(__file__.replace(".py", ".pdf"))
plt.savefig(__file__.replace(".py", ".png"))
plt.close()
