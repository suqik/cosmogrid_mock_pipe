''' Compare ggl across the 4 (flip_g1, flip_g2) shear-convention combos, HOD 0,
    per-HOD jackknife errors. '''

import numpy as np
from astropy.table import Table
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

base = "/public/home/suchen/Programs/cosmogrid_mock_pipe/measurements/results/ggl"
colors = {"tomo3": "#2a78d6", "tomo4": "#eb6834", "tomo5": "#1baf7a"}
combos = [("f00", "no flip"), ("f10", "flip g1"), ("f01", "flip g2"), ("f11", "flip g1+g2")]
modes = [("boss_ngc_kids1000_3tomos", "pure shear (g_pure)"),
         ("boss_ngc_kids1000_3tomos_wSN", "with shape noise (wSN)")]

fig, axes = plt.subplots(2, 4, figsize=(17.5, 8.2), sharex=True)
fig.patch.set_facecolor("#fcfcfb")

summary = {}
for imode, (mode, mode_title) in enumerate(modes):
    for ic, (suffix, combo_title) in enumerate(combos):
        ax = axes[imode, ic]
        ax.set_facecolor("#fcfcfb")
        t = Table.read(f"{base}/{mode}_{suffix}/cosmo000000_HOD0_ggl.fits")
        for tomo in (3, 4, 5):
            sel = t[t["tomo"] == tomo]
            ax.errorbar(sel["rp"], sel["ds"], yerr=sel["ds_err_jk"],
                        color=colors[f"tomo{tomo}"], marker="o", ms=4, lw=1.8,
                        elinewidth=1.1, capsize=2, zorder=3)
            summary[(mode, suffix, tomo)] = (
                np.mean(sel["ds"]), np.max(np.abs(sel["ds"])),
                np.mean(sel["ds_err_jk"]))
        ax.axhline(0, color="#c3c2b7", lw=1.1, ls="--", zorder=2)
        ax.set_xscale("log")
        ax.grid(True, which="both", color="#e1e0d9", lw=0.55, zorder=0)
        for spine in ax.spines.values():
            spine.set_color("#c3c2b7")
        ax.tick_params(colors="#898781", labelsize=9)
        if imode == 0:
            ax.set_title(f"{combo_title}\n[{suffix}]", color="#0b0b0b", fontsize=11)
        if ic == 0:
            ax.set_ylabel(f"{mode_title}\n"
                          r"$\Delta\Sigma$ [M$_\odot$/pc$^2$]", color="#0b0b0b", fontsize=10)
        if imode == 1:
            ax.set_xlabel(r"$r_p$ [Mpc/$h$]", color="#0b0b0b", fontsize=11)

handles = [plt.Line2D([], [], color=colors[f"tomo{t}"], lw=2, label=f"tomo{t}")
           for t in (3, 4, 5)]
fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
           fontsize=10, labelcolor="#0b0b0b")
fig.suptitle("ggl - cosmo_000000, HOD 0: rlz2 KiDS-like sources x rlz0 BOSS-like lenses\n"
             "shear-convention combos; rand-subtracted; error bars = per-HOD jackknife",
             color="#0b0b0b", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = f"{base}/ggl_cosmo000000_flip_combos.png"
fig.savefig(out, dpi=150, facecolor="#fcfcfb")
print("saved:", out)

print("\nmean(ds) / max|ds| / mean(jk err) per (mode, combo, tomo):")
for mode, _ in modes:
    for suffix, _c in combos:
        for tomo in (3, 4, 5):
            m, mx, e = summary[(mode, suffix, tomo)]
            print(f"  {mode.split('_')[-1] or 'pure':4s} {suffix} tomo{tomo}: "
                  f"{m:+7.4f} / {mx:.4f} / {e:.4f}")
