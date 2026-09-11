"""Measure FastPM mock void lensing, following run_vl.py and run_ggl_fastpm.py.

Run with: python measurements/run_vl_fastpm.py
The radial scale is the median Rv of the selected, footprint-matched voids,
as implemented by GGLCalculator (not a separate radial scale for each void).
"""

from pathlib import Path

import numpy as np

from container import SurveyData
from calculator import GGLCalculator, GGLConfig


NJOBS = 8
NGAL = 3.5e-4  # Fixed tracer number density in (h/Mpc)^3 for all samples.

wdir = Path(__file__).resolve().parents[1]
data_dirbase = Path("/Users/suqikuai777/Dataspace/FastPM/MockCatalogs")
lens_dir = data_dirbase / "Voids"
lens_fmt = "cosmo_{:06d}_realization_0000_HOD_{:d}_boss_lowz_north_2dflens_south.fits"
rand_file = data_dirbase / "Rand/DS20/boss_cmasslowztot_ngc_z0.2_0.4_official.fits"
srcs_dir = data_dirbase / "Shape/kids1000_north_3tomos"
srcs_fmt = "cosmo_{:06d}_realization_0002_kids1000_north_3tomos.fits"
out_dir = wdir / "results/vl_fpm/boss_ngc_kids1000_3tomos"
out_fmt = "cosmo{:06d}_HOD{:d}_vl.fits"
cosmo_param_info_file = data_dirbase.parent / "cosmo_list.txt"


def load_cosmo_params(fname):
    """Read the fixed hubble parameter and row-indexed FastPM cosmologies."""
    with open(fname) as f:
        fixed = dict(token.split("=", 1) for token in f.readline().lstrip("#").split())
    hubble = float(fixed["hubble"])
    rows = np.loadtxt(fname, ndmin=2)
    return {
        icosmo: {"Om0": float(row[0]), "H0": 100.0 * hubble, "w0": -1.0}
        for icosmo, row in enumerate(rows)
    }


def main():
    cosmo_labels = [0]
    hod_labels = range(1)
    cosmo_params = load_cosmo_params(cosmo_param_info_file)

    ggl_config = GGLConfig(
        rp_min=0.1,
        rp_max=3.0,
        rp_bins=13,
        rp_unit="Rv",
        bin_type="log",
        flip_g1=False,
        flip_g2=True,  # Same FastPM shear convention as run_ggl_fastpm.py.
        wRSD=False,  # Void catalogs already contain the void-center redshift.
        wSN=False,
        wPhZ=False,
    )
    ggl_instance = GGLCalculator(config=ggl_config)
    mock_rand_boss = SurveyData.load_cosmogrid_rand(rand_file)
    out_dir.mkdir(parents=True, exist_ok=True)

    for icosmo in cosmo_labels:
        cosmo_dict = cosmo_params[icosmo]
        mock_shape_kids = SurveyData.load_cosmogrid_shape(srcs_dir / srcs_fmt.format(icosmo))
        if len(mock_shape_kids) == 0:
            raise ValueError(f"Empty shape catalog for cosmology {icosmo}")
        srcs_table = ggl_instance.mk_srcs_cat(mock_shape_kids)
        mock_rand_boss_matched = mock_rand_boss.match_to_reference(
            mock_shape_kids, nside=256, in_place=False
        )
        if len(mock_rand_boss_matched) == 0:
            raise ValueError(f"No random points overlap the shapes for cosmology {icosmo}")

        for ihod in hod_labels:
            mock_void = SurveyData.load_cosmogrid_void(lens_dir / lens_fmt.format(icosmo, ihod))
            mock_void_boss = mock_void[mock_void.survey != 3]
            rescaled_Rv = mock_void_boss.Rv * np.cbrt(NGAL)
            mock_void_boss = mock_void_boss[rescaled_Rv > 1.0]
            mock_void_boss_matched = mock_void_boss.match_to_reference(
                mock_shape_kids, nside=256, in_place=False
            )
            if len(mock_void_boss_matched) == 0:
                raise ValueError(f"No selected voids overlap the shapes: cosmo={icosmo}, HOD={ihod}")

            lens_table = ggl_instance.mk_lens_cat(mock_void_boss_matched)
            ggl_instance.get_Rv_mean_mpch(mock_void_boss_matched)
            # Rv (and hence physical bin edges) changes with each void sample.
            # Start from fresh randoms and recompute pairs for every HOD.
            rand_table = ggl_instance.mk_lens_cat(mock_rand_boss_matched)
            lens_table = ggl_instance.compute_pairs(
                cosmo_dict, lens_table, srcs_table, n_jobs=NJOBS
            )
            rand_table = ggl_instance.compute_pairs(
                cosmo_dict, rand_table, srcs_table, n_jobs=NJOBS
            )
            if len(lens_table) == 0 or len(rand_table) == 0:
                raise ValueError(f"No usable lens/random-source pairs: cosmo={icosmo}, HOD={ihod}")
            esd = ggl_instance.stack_signals(lens_table, rand_table)
            esd.meta["NGAL"] = NGAL
            esd.meta["RV_MED"] = float(ggl_instance.Rv_mean_mpch)
            esd.write(out_dir / out_fmt.format(icosmo, ihod), overwrite=True)


if __name__ == "__main__":
    main()
