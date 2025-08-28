import os.path as pa
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

# Local imports
sys.path.append(pa.dirname(pa.dirname(pa.dirname(__file__))))
from utils.paths import PROJDIR
from myagn.distributions import QLFHopkins

# Style file
plt.style.use(f"{PROJDIR}/plots/matplotlibrc.mplstyle")

################################################################################

# Form: M / (10**7 Msun) = a * (L_44 / (10**44 Lsun))**b = a * (L_bol / 9 / (10**44 Lsun))**b
# --> log10(M_7) = log10(a) + b * (log10(L_bol_44 / 9))
params = {
    "Wandel+99 (unweighted)": {
        "a": 8.71,
        "alo": 0,
        "ahi": 0,
        "b": 0.54,
        "blo": 0,
        "bhi": 0,
    },
    "Wandel+99 (weighted)": {
        "a": 8.32,
        "alo": -0.73,
        "ahi": 0.80,
        "b": 0.77,
        "blo": -0.07,
        "bhi": 0.07,
    },
    "Kaspi+00 (mean)": {
        "a": 5.71,
        "alo": -0.37,
        "ahi": 0.46,
        "b": 0.545,
        "blo": -0.036,
        "bhi": 0.036,
    },
    "Kaspi+00 (rms)": {
        "a": 5.75,
        "alo": -0.36,
        "ahi": 0.39,
        "b": 0.402,
        "blo": -0.039,
        "bhi": 0.039,
    },
    "Peterson+04": {
        "a": 7.59,
        "alo": -1.33,
        "ahi": 1.33,
        "b": 0.79,
        "blo": -0.09,
        "bhi": 0.09,
    },
}


def plot_mass_luminosity_function(ax, label, params):
    logLbol44_grid = np.linspace(-5, 5, 100)
    logM7 = np.log10(params["a"]) + params["b"] * (logLbol44_grid - np.log10(9))
    logM7_lo = np.log10(params["a"] + params["alo"]) + (params["b"] + params["blo"]) * (
        logLbol44_grid - np.log10(9)
    )
    logM7_hi = np.log10(params["a"] + params["ahi"]) + (params["b"] + params["bhi"]) * (
        logLbol44_grid - np.log10(9)
    )
    f = interp1d(logM7, logLbol44_grid)
    Lcut = f(-1)
    ax.plot(
        logLbol44_grid,
        logM7,
        label=f"{label} (x={Lcut:.3f}, 10$^{{\\rm 44 + x}}$={10**(44+Lcut):.3e})",
        rasterized=True,
    )
    ax.fill_between(
        logLbol44_grid,
        logM7_lo,
        logM7_hi,
        alpha=0.2,
        rasterized=True,
        color=ax.lines[-1].get_color(),
    )
    return Lcut


def plot_mass_luminosity_functions(ax, params):
    for k, v in params.items():
        plot_mass_luminosity_function(
            ax,
            k,
            v,
        )
    ax.hlines(
        -1,
        -5,
        5,
        color="k",
        linestyle="--",
        label=r"$M = 10^6 M_\odot$",
        rasterized=True,
    )
    ax.set_xlabel(r"$\log_{10} (L_{\rm bol} / (10^{44} L_\odot))$")
    ax.set_ylabel(r"$\log_{10} (M / (10^7 M_\odot))$")
    ax.legend(
        loc="upper center",
        title=r"Source ($\log_{10} (L_{\rm bol} / (10^{44} L_\odot))_{\rm M=10^6 M_\odot}$, $L_{\rm bol, M=10^6 M_\odot}$)",
    )


def plot_distributions(ax, params):
    qlf = QLFHopkins()
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    logLbol44_grid = np.linspace(-5, 5, 100)
    zs = np.linspace(0, 2, 47)
    for k, v in params.items():
        # Get luminosity cutoff
        logM7 = np.log10(v["a"]) + v["b"] * (logLbol44_grid - np.log10(9))
        f = interp1d(logM7, logLbol44_grid)
        Lcut = f(-1)
        # Get distribution
        dn_d3Mpc = qlf.dn_d3Mpc(
            zs=zs,
            cosmo=cosmo,
            brightness_limits=[10 ** (44 + Lcut), np.inf] * u.erg / u.s,
        )
        y = np.log10(dn_d3Mpc.value)
        # Plot
        ax.plot(
            zs,
            y,
            label=f"{k} ($\\log(L_{{\\rm bol}}/(10^{{44}} L_\\odot)) > {Lcut:.3f}$)",
            rasterized=True,
        )
    # Get distribution
    dn_d3Mpc = qlf.dn_d3Mpc(
        zs=zs,
        cosmo=cosmo,
        brightness_limits=[20.5, -np.inf] * u.ABmag,
    )
    y = np.log10(dn_d3Mpc.value)
    # Plot
    ax.plot(
        zs,
        y,
        label=r"$m_g < 20.5$",
        rasterized=True,
    )
    ax.set_xlabel("Redshift")
    ax.set_ylabel(r"$\log_{10} (dn/d(Mpc^{-3}))$")
    ax.legend()


def main():
    fig, axs = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(12, 5),
    )
    plot_mass_luminosity_functions(axs[0], params)
    plot_distributions(axs[1], params)
    plt.tight_layout()
    plt.savefig(__file__.replace(".py", ".png"))
    plt.savefig(__file__.replace(".py", ".pdf"))
    plt.close()


#########################################################################################

if __name__ == "__main__":
    main()
