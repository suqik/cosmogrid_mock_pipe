''' Validate mock shape catalogs: measured xi+/xi- (treecorr) vs pyccl theory '''

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyccl as ccl
import treecorr
from astropy.table import Table
from loguru import logger

COSMO_INDICES = [0, 250, 500, 750, 999]
TOMOS = [3, 4, 5]
NPATCH = 60

SHAPE_FMT = (
    "/public/share/ace66so15x/suchen/FastPM/MockCatalogs/Shapes/"
    "cosmo_{:06d}_realization_{:04d}_kids1000_north_3tomos.fits"
)
NOFZ_FFMT = (
    "/public/home/suchen/Programs/cosmogrid_mock_pipe/extras/NOfZ/"
    "kids1000_nofzs/K1000_NS_V1.0.0A_ugriZYJHKs_photoz_SG_mask_LF_"
    "svn_309c_2Dbins_v2_SOMcols_Fid_blindC_TOMO{}_Nz.asc"
)
COSMO_LIST = (
    "/public/home/suchen/Programs/Simtool/Pipeline/"
    "cfgs/fiducial/cosmo_list.txt"
)
OUT_XIP = Path(__file__).resolve().parent / "validation_shear_xip.png"
OUT_XIM = Path(__file__).resolve().parent / "validation_shear_xim.png"

# palette: validated light-mode categorical slots (dataviz reference palette)
BLUE = "#2a78d6"      # xi+
ORANGE = "#eb6834"    # xi-
INK = "#0b0b0b"
SEC_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

FIXED = dict(hubble=0.6727, Omegab=0.0491, ns=0.9667, mnu=0.06)


def get_cosmo(icosmo):
    Om, S8 = np.loadtxt(COSMO_LIST, comments="#")[icosmo]
    return ccl.Cosmology(
        h=FIXED["hubble"],
        Omega_b=FIXED["Omegab"],
        Omega_c=Om - FIXED["Omegab"],
        sigma8=S8 / np.sqrt(Om / 0.3),
        n_s=FIXED["ns"],
        m_nu=FIXED["mnu"],
        w0=-1.0,
        wa=0.0,
        # linear power only: maps were generated from linear power spectra
    )


def get_dndz(tomo):
    z, nz = np.loadtxt(NOFZ_FFMT.format(tomo)).T
    zc = z + 0.025  # bin centers (dz=0.05)
    keep = z < 2.0  # catalog truncated at z=2 (same cut as the mock)
    nz_keep = nz[keep]
    return zc[keep], nz_keep / nz_keep.sum()


def theory_xipm(cosmo, tomo, theta_deg):
    z, nz = get_dndz(tomo)
    tracer = ccl.WeakLensingTracer(cosmo, dndz=(z, nz), has_shear=True)
    ells = np.unique(np.geomspace(2, 8000, 200).astype(int))
    cl = ccl.angular_cl(cosmo, tracer, tracer, ells)
    xip = ccl.correlation(cosmo, ell=ells, C_ell=cl, theta=theta_deg, type="GG+")
    xim = ccl.correlation(cosmo, ell=ells, C_ell=cl, theta=theta_deg, type="GG-")
    return xip, xim


def measure_xipm(fname, tomo):
    cat = Table.read(fname)
    sel = cat[cat["tomo"] == tomo]
    tc = treecorr.Catalog(
        ra=sel["ra"], dec=sel["dec"], g1=sel["g1"], g2=sel["g2"], w=sel["w"],
        ra_units="deg", dec_units="deg",
        npatch=NPATCH,  # built-in spherical kmeans patches for jackknife
    )
    gg = treecorr.GGCorrelation(
        nbins=10, min_sep=5, max_sep=300, sep_units="arcmin",
        var_method="jackknife",
    )
    gg.process(tc)
    theta_deg = gg.meanr / 60.0
    return (
        theta_deg, gg.xip, gg.xim,
        np.sqrt(gg.varxip), np.sqrt(gg.varxim),
    )


def make_panels(y_scale, ylabel):
    fig, axes = plt.subplots(
        5, 3, figsize=(14, 18), sharex=True, sharey=True,
        facecolor=SURFACE,
    )
    for r, icosmo in enumerate(COSMO_INDICES):
        cosmo = get_cosmo(icosmo)
        fname = SHAPE_FMT.format(icosmo, 0)
        logger.info(f"cosmo {icosmo}: measuring")
        for c, tomo in enumerate(TOMOS):
            ax = axes[r][c]
            theta, xip, xim, ep, em = measure_xipm(fname, tomo)
            xip_t, xim_t = theory_xipm(cosmo, tomo, theta)

            if y_scale == "log":
                ax.plot(theta, xip_t, "-", color=BLUE, lw=2)
                ax.errorbar(
                    theta, xip, yerr=ep, fmt="o", ms=5, mfc="white",
                    mec=BLUE, ecolor=BLUE, elinewidth=1, mew=1.5,
                )
                ax.set_yscale("log")
            else:
                # xi- figure: xi- with sign, symlog to show the zero crossing
                ax.plot(theta, xim_t, "-", color=ORANGE, lw=2)
                ax.errorbar(
                    theta, xim, yerr=em, fmt="o", ms=5, mfc="white",
                    mec=ORANGE, ecolor=ORANGE, elinewidth=1, mew=1.5,
                )
                ax.axhline(0, color=BASELINE, lw=1)
                ax.set_yscale("symlog", linthresh=1e-8, linscale=0.5)

            ax.set_xscale("log")
            ax.set_facecolor(SURFACE)
            ax.grid(True, which="both", color=GRID, lw=0.6)
            for spine in ax.spines.values():
                spine.set_color(BASELINE)
            ax.tick_params(colors=MUTED, labelsize=9)
            ax.set_title(
                f"cosmo {icosmo:04d}  |  tomo{tomo}",
                fontsize=10, color=SEC_INK, loc="left", pad=6,
            )

            ratio = (xip / xip_t) if y_scale == "log" else (xim / xim_t)
            logger.info(
                f"  cosmo{icosmo} tomo{tomo}: ratio "
                f"{ratio.min():.2f}-{ratio.max():.2f}"
            )

    for c, tomo in enumerate(TOMOS):
        axes[-1][c].set_xlabel(r"$\theta$ [deg]", color=SEC_INK)
    for r in range(5):
        axes[r][0].set_ylabel(ylabel, color=SEC_INK)
    return fig


if __name__ == "__main__":

    fig = make_panels("log", r"$\xi_+$")
    handles = [
        plt.Line2D([], [], color=BLUE, lw=2, label=r"$\xi_+$ theory"),
        plt.Line2D([], [], marker="o", ms=6, mfc="white", mec=BLUE,
                   lw=0, label=r"$\xi_+$ measured (jackknife)"),
    ]
    fig.legend(
        handles=handles, loc="upper center", ncol=2,
        frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.995),
    )
    fig.suptitle(
        r"$\xi_+$: treecorr (jackknife) vs pyccl linear theory",
        color=INK, fontsize=13, y=1.01,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(OUT_XIP, dpi=150, facecolor=SURFACE)
    logger.info(f"saved {OUT_XIP}")

    fig = make_panels("symlog", r"$\xi_-$")
    handles = [
        plt.Line2D([], [], color=ORANGE, lw=2, label=r"$\xi_-$ theory"),
        plt.Line2D([], [], marker="o", ms=6, mfc="white", mec=ORANGE,
                   lw=0, label=r"$\xi_-$ measured (jackknife)"),
    ]
    fig.legend(
        handles=handles, loc="upper center", ncol=2,
        frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.995),
    )
    fig.suptitle(
        r"$\xi_-$: treecorr (jackknife) vs pyccl linear theory",
        color=INK, fontsize=13, y=1.01,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(OUT_XIM, dpi=150, facecolor=SURFACE)
    logger.info(f"saved {OUT_XIM}")
