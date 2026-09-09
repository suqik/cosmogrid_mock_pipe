#!/usr/bin/env python
'''
Visualize one FastPM mock galaxy catalog: sky distribution (RA/Dec) and n(z).

Picks one cosmology (default cosmo_000000) and one HOD catalog, and makes a
two-panel figure:
  left  - angular positions on the sky, colored by survey
  right - stacked redshift distribution n(z), colored by the same surveys

The catalogs combine four surveys: BOSS LOWZ NGC, LOWZE2 NGC, LOWZE3 NGC
and 2dFLenS South. LOWZE2/LOWZE3 are folded into a single "LOWZ-ext" group
(the two extension samples are small and spatially mixed with LOWZ); set
SPLIT_LOWZE = True below to keep them separate (re-validate colors if so).

Usage:
  python plot_mock_gals.py                     # cosmo 0, realization 0, HOD 0
  python plot_mock_gals.py --cosmo 42 --hod 7  # another cosmology / HOD draw
'''

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------

CATALOG_DIR = "/public/share/ace66so15x/suchen/FastPM/MockCatalogs/Gals"
CATALOG_FMT = (
    "{catalog_dir}/cosmo_{cosmo:06d}_realization_{rlz:04d}_HOD_{hod:d}_"
    "boss_lowz_north_2dflens_south.fits"
)
COSMO_PAR_FNAME = (
    "/public/home/suchen/Programs/Simtool/Pipeline/"
    "cfgs/fiducial/cosmo_list.txt"
)

# ----------------------------------------------------------------------
# Survey groups and colors (light-mode categorical palette, validated:
# all-pairs CVD DeltaE 9.2, normal-vision 24.0; aqua needs labels/legend
# relief -> direct labels + legend are shipped)
# ----------------------------------------------------------------------

SPLIT_LOWZE = False  # if True, LOWZE2 and LOWZE3 become their own series

SURVEY_NAMES = {0: "BOSS LOWZ North", 1: "LOWZE2 North",
                2: "LOWZE3 North", 3: "2dFLenS South"}
SURVEY_COLORS = {0: "#2a78d6", 1: "#eb6834",
                 2: "#eb6834", 3: "#1baf7a"}

# Surface & ink tokens (light mode)
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS_LINE = "#c3c2b7"

ZBINS = np.arange(0.20, 0.4001, 0.004)  # n(z) bins


def load_cosmo_params(cosmo_idx):
    '''Return (OmegaM, S8) for the cosmo index; fixed pars from the header.'''
    with open(COSMO_PAR_FNAME) as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    omega_m, s8 = lines[cosmo_idx].split()
    return float(omega_m), float(s8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cosmo", type=int, default=0, help="cosmology index")
    parser.add_argument("--rlz", type=int, default=0, help="realization index")
    parser.add_argument("--hod", type=int, default=0, help="HOD draw index")
    parser.add_argument("--outdir", type=str, default=None,
                        help="output dir (default: this script's dir)")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    outdir = Path(args.outdir) if args.outdir else Path(__file__).resolve().parent
    fname = Path(CATALOG_FMT.format(
        catalog_dir=CATALOG_DIR, cosmo=args.cosmo, rlz=args.rlz, hod=args.hod))

    # ---- data --------------------------------------------------------
    data = fits.open(fname)[1].data
    ra, dec, zrsd, survey = data["ra"], data["dec"], data["zrsd"], data["survey"]
    ntot = len(ra)

    omega_m, s8 = load_cosmo_params(args.cosmo)

    # survey -> series groups
    if SPLIT_LOWZE:
        groups = [(name, [k]) for k, name in SURVEY_NAMES.items()]
    else:
        groups = [
            (SURVEY_NAMES[0], [0]),
            ("BOSS LOWZ-ext North (LOWZE2+3)", [1, 2]),
            (SURVEY_NAMES[3], [3]),
        ]

    # ---- figure ------------------------------------------------------
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "text.color": INK_PRIMARY,
        "axes.edgecolor": AXIS_LINE,
        "axes.labelcolor": INK_SECONDARY,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "grid.color": GRID,
        "grid.linewidth": 1.0,
    })

    fig, (ax_sky, ax_nz) = plt.subplots(
        1, 2, figsize=(13.2, 5.4),
        gridspec_kw={"width_ratios": [1.30, 1.0], "wspace": 0.24})
    fig.subplots_adjust(top=0.80, bottom=0.11, left=0.07, right=0.98)

    # ---- left panel: sky map -----------------------------------------
    handles = []
    for name, surv_ids in groups:
        m = np.isin(survey, surv_ids)
        color = SURVEY_COLORS[surv_ids[0]]
        # draw later groups on top of earlier ones
        ax_sky.scatter(ra[m], dec[m], s=1.6, c=color, alpha=0.35,
                       linewidths=0, rasterized=True, zorder=len(surv_ids))
        handles.append(Patch(facecolor=color, edgecolor="none",
                             label=f"{name}  ({m.sum():,})"))

    # astronomical convention: RA increases to the left
    ax_sky.set_xlim(360, 0)
    ax_sky.set_ylim(-40, 72)
    ax_sky.set_xticks(np.arange(0, 361, 60))
    ax_sky.set_yticks(np.arange(-40, 81, 20))
    ax_sky.set_xlabel("RA [deg]")
    ax_sky.set_ylabel("Dec [deg]")
    ax_sky.grid(True, which="major")
    ax_sky.set_title("Sky distribution", color=INK_PRIMARY, fontweight="bold",
                     fontsize=12, loc="left", pad=10)
    ax_sky.tick_params(top=False, right=False, length=0)

    # direct labels (ink text on the dense colored regions)
    label_kw = dict(fontsize=11, color=INK_PRIMARY, ha="center", va="center",
                    fontweight="bold")
    ax_sky.text(210, 55, SURVEY_NAMES[0], **label_kw)
    ax_sky.text(237, 25, "LOWZ-ext\n(LOWZE2+3)", **label_kw)
    ax_sky.text(110, -33, SURVEY_NAMES[3], **label_kw)

    ax_sky.legend(handles=handles, loc="upper left", frameon=False,
                  fontsize=10, labelcolor=INK_SECONDARY)

    # ---- right panel: stacked n(z) ------------------------------------
    hist_kw = dict(bins=ZBINS, histtype="stepfilled",
                   edgecolor=SURFACE, linewidth=1.0)
    bottom = np.zeros(len(ZBINS) - 1)
    hists = {}
    for name, surv_ids in groups:  # bottom-up stacking order
        m = np.isin(survey, surv_ids)
        hists[name] = np.histogram(zrsd[m], bins=ZBINS)[0]
        ax_nz.hist(zrsd[m], weights=None, bottom=bottom,
                   color=SURVEY_COLORS[surv_ids[0]], **hist_kw)
        bottom += hists[name]

    ax_nz.set_xlim(0.2, 0.4)
    ax_nz.set_xlabel("Observed redshift $z$ (RSD)")
    ax_nz.set_ylabel(f"Galaxies per $\\Delta z$ = {ZBINS[1] - ZBINS[0]:.3f}")
    ax_nz.grid(True, axis="y", which="major")
    ax_nz.set_title("Redshift distribution $n(z)$", color=INK_PRIMARY,
                    fontweight="bold", fontsize=12, loc="left", pad=10)
    ax_nz.tick_params(top=False, right=False, length=0)
    ax_nz.yaxis.set_major_formatter(
        matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))

    # interior direct label on the dominant LOWZ segment
    lowz_h = hists[SURVEY_NAMES[0]]
    z_peak = ZBINS[:-1][np.argmax(lowz_h)]
    other_top = bottom - lowz_h
    ax_nz.text(z_peak + 0.012, 0.55 * (other_top + bottom)[np.argmax(lowz_h)],
               SURVEY_NAMES[0], fontsize=11, color=INK_PRIMARY,
               ha="center", va="center", fontweight="bold")

    # ---- titles & save ------------------------------------------------
    fig.suptitle(
        "FastPM mock galaxies — "
        f"cosmo_{args.cosmo:06d}   "
        f"($\\Omega_m$ = {omega_m:.4f},  $S_8$ = {s8:.4f}   |   "
        f"h = 0.6727,  $\\Omega_b$ = 0.0491,  $n_s$ = 0.9667)",
        fontsize=13, fontweight="bold", color=INK_PRIMARY, y=0.965)
    fig.text(0.5, 0.875,
             f"realization {args.rlz:04d} · HOD {args.hod} · "
             f"N = {ntot:,} · lightcone z = 0.2–0.4",
             fontsize=10, color=INK_SECONDARY, ha="center")

    out = outdir / f"mock_gals_cosmo_{args.cosmo:06d}_HOD_{args.hod}_sky_nz.png"
    fig.savefig(out, dpi=args.dpi)
    print(f"saved {out}")
    for name, surv_ids in groups:
        m = np.isin(survey, surv_ids)
        print(f"  {name:34s} N = {m.sum():,}")
    print(f"  total N = {ntot:,}  |  ra [{ra.min():.1f}, {ra.max():.1f}]  "
          f"dec [{dec.min():.1f}, {dec.max():.1f}]  z_rsd "
          f"[{zrsd.min():.3f}, {zrsd.max():.3f}]")


if __name__ == "__main__":
    main()
