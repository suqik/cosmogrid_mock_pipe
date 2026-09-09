''' Test FastPM mock void catalog on a single cosmology and HOD '''

import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from astropy.table import Table
from loguru import logger

from handler import PipeConfig
from runner import FastPMRunner

if __name__ == "__main__":

    icosmo = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    ihod = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    fastpm_config = PipeConfig(
        ### fixed siminfo
        Lbox = 1000.0,
        Npart = 1024,
        redshift = 0.3,
        # ### HOD model params
        model = 2, # label of model name.
        model_params_names = ('logMcut', 'sigma_logM', 'logM1', 'k', 'alpha', 'fic'),
        nhod_per_cosmo = 10, # Number of varied HOD parameter values per cosmology
        Num_ptcl_requirement = 12,
        verbose = True,
        num_seeds = 1,
        init_seed = 33000, ## initial seed for generating galaxy catalog
        ngal_ref = 3.5e-4,
        z_space = False, ## RSD in box. Note if need RSD in survey-like, do not open this.

        ### HOD param sampling
        param_prior_low  = np.array([12.0, 0.1, 13, 0.00, 0.0]),
        param_prior_high = np.array([13.6, 0.6, 15.0, 10.0, 1.5]),

        ### lightcone redshift range
        zmin_lightcone = 0.2,
        zmax_lightcone = 0.4,
        ctr_lightcone = [0,0,0],
        rsd_lightcone = True,

        ### nofz
        nofz_method = "downsample", # can be `rank`, `downsample`, or `const`

        dive_exec_path = "/public/home/suchen/applications/DIVE/DIVE"
    )

    cosmo_par_fname = (
        "/public/home/suchen/Programs/Simtool/Pipeline/"
        "cfgs/fiducial/cosmo_list.txt"
    )

    wdir = "/public/home/suchen/Programs/cosmogrid_mock_pipe/extras"

    ### Geometry & masks
    mask_boss_fdir = f"{wdir}/masks/boss_geom/"

    mask_fnames_dict = {
        'boss_lowz_ngc': mask_boss_fdir + "mask_DR12v5_LOWZ_North.ply", # Note LOWZE2 and LOWZE3 need LOWZ for trimming
        'boss_lowze2_ngc': mask_boss_fdir + "mask_DR12v5_LOWZE2_North.ply",
        'boss_lowze3_ngc': mask_boss_fdir + "mask_DR12v5_LOWZE3_North.ply",
        'boss_veto': [
            mask_boss_fdir + "badfield_mask_postprocess_pixs8.ply",
            mask_boss_fdir + "badfield_mask_unphot_seeing_extinction_pixs8_dr12.ply",
            mask_boss_fdir + "allsky_bright_star_mask_pix.ply",
            mask_boss_fdir + "bright_object_mask_rykoff_pix.ply",
            mask_boss_fdir + "collision_priority_mask_dr12.ply",
            mask_boss_fdir + "centerpost_mask_dr12.ply"
        ],
        '2dflens_south': f"{wdir}/masks/2dflens_geom/2dFLens_mask_weight_South.fits"
    }

    ### N of Z
    nz_fbase = f"{wdir}/NOfZ/"
    nofz_fnames_dict = {
        'boss_lowz_ngc': nz_fbase + "boss_nofzs/nbar_DR12v5_LOWZ_North_om0p31_Pfkp10000.dat",
        'boss_lowze2_ngc': nz_fbase + "boss_nofzs/nbar_DR12v5_LOWZE2_North_om0p31_Pfkp10000.dat",
        'boss_lowze3_ngc': nz_fbase + "boss_nofzs/nbar_DR12v5_LOWZE3_North_om0p31_Pfkp10000.dat",
        '2dflens_south': nz_fbase + "2dflens_nofzs/nbar_2dFLens_south_data.dat"
    }

    ### survey labels
    survey_labels_dict = {
        'boss_lowz_ngc': 0,
        'boss_lowze2_ngc': 1,
        'boss_lowze3_ngc': 2,
        '2dflens_south': 3
    }

    dive_input_fmt = "/public/home/suchen/Programs/cosmogrid_mock_pipe/tmp/dive_tmps/input_rank{}.tmp"
    dive_output_fmt = "/public/home/suchen/Programs/cosmogrid_mock_pipe/tmp/dive_tmps/output_rank{}.tmp"

    galcone_fmt = (
        "/public/share/ace66so15x/suchen/FastPM/MockCatalogs/Gals/"
        "cosmo_{:06d}_realization_{:04d}_HOD_{:d}_"
        "boss_lowz_north_2dflens_south.fits"
    )
    voidcone_fmt = (
        "/public/share/ace66so15x/suchen/FastPM/MockCatalogs/Voids/"
        "cosmo_{:06d}_realization_{:04d}_HOD_{:d}_"
        "boss_lowz_north_2dflens_south.fits"
    )

    Path(voidcone_fmt).parent.mkdir(parents=True, exist_ok=True)

    fastpm_runner = FastPMRunner.build_void_runner(
        config=fastpm_config,
        cosmo_par_fname=cosmo_par_fname,
        fore_mask_fnames_dict=mask_fnames_dict,
        fore_nofz_fnames_dict=nofz_fnames_dict,
        fore_survey_labels_dict=survey_labels_dict,
        void_ofmt=voidcone_fmt,
    )

    galcone = Table.read(galcone_fmt.format(icosmo, 0, ihod))
    logger.info(f"input galaxy catalog: {len(galcone)} galaxies")

    result = fastpm_runner.gen_mock_void(
        icosmo, irlz=0, ihod=ihod, galcone_survey=galcone,
        dive_input=dive_input_fmt.format(0),
        dive_output=dive_output_fmt.format(0),
        save=True,
    )

    logger.info(f"total voids: {len(result)}")
    for survey_name, survey_label in survey_labels_dict.items():
        sel = result[result["survey"] == survey_label]
        if len(sel):
            logger.info(
                f"  {survey_name}: {len(sel)} voids, "
                f"Rv range [{sel['Rv'].min():.1f}, {sel['Rv'].max():.1f}] Mpc/h, "
                f"z range [{sel['z'].min():.3f}, {sel['z'].max():.3f}]"
            )
        else:
            logger.info(f"  {survey_name}: 0 voids")

    out_fname = voidcone_fmt.format(icosmo, 0, ihod)
    logger.info(f"saved to {out_fname} ({os.path.getsize(out_fname)/1e6:.1f} MB)")
