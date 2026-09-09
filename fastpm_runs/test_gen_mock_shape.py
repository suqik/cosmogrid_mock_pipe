''' Test FastPM mock shape catalog on a single cosmology '''

import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from loguru import logger
from astropy.table import Table

from handler import PipeConfig
from runner import FastPMRunner

if __name__ == "__main__":

    icosmo = int(sys.argv[1]) if len(sys.argv) > 1 else 1

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
        nofz_method = "downsample", # can be `rank`, `downsample`, or `const`,

        dive_exec_path = "/home/suchen/applications/DIVE/DIVE",

        sigma_e = 0.3,
        seed_SN_ini = 0,
        sigma_phz = 0.01,
        seed_Phz_ini = 26120,
    )

    cosmo_par_fname = (
        "/public/home/suchen/Programs/Simtool/Pipeline/"
        "cfgs/fiducial/cosmo_list.txt"
    )
    shear_sim_fmt = (
        "/public/share/ace66so15x/suchen/FastPM/Shear/products/"
        "cosmo_{:06d}/realization_{:04d}.npz"
    )

    wdir = "/public/home/suchen/Programs/cosmogrid_mock_pipe/extras"
    mask_dirbase = f"{wdir}/masks"
    nofz_dirbase = f"{wdir}/NOfZ"

    back_mask_fnames_dict = {
        'KiDS1000-North': f"{mask_dirbase}/kids1000_geom/mask_KiDS_North_1024.fits"
    }

    back_survey_labels_dict = {
        'KiDS1000-North': 0
    }

    back_nofz_ffmt = nofz_dirbase + "/kids1000_nofzs/K1000_NS_V1.0.0A_ugriZYJHKs_photoz_SG_mask_LF_svn_309c_2Dbins_v2_SOMcols_Fid_blindC_TOMO{}_Nz.asc"

    back_ngals_dict = {'tomo3': 1.85,
                       'tomo4': 1.26,
                       'tomo5': 1.31}
    tomo_labels_dict = {'tomo3': 3,
                        'tomo4': 4,
                        'tomo5': 5}
    back_nofz_fnames_dict = {'tomo3': back_nofz_ffmt.format(3),
                             'tomo4': back_nofz_ffmt.format(4),
                             'tomo5': back_nofz_ffmt.format(5)}

    shapecone_fmt = (
        "/public/share/ace66so15x/suchen/FastPM/MockCatalogs/Shapes/"
        "cosmo_{:06d}_realization_{:04d}_kids1000_north_3tomos.fits"
    )

    Path(shapecone_fmt).parent.mkdir(parents=True, exist_ok=True)

    fastpm_runner = FastPMRunner.build_shape_runner(
        config=fastpm_config,
        cosmo_par_fname=cosmo_par_fname,
        shear_sim_fmt=shear_sim_fmt,
        back_mask_fnames_dict=back_mask_fnames_dict,
        back_nofz_fnames_dict=back_nofz_fnames_dict,
        back_survey_labels_dict=back_survey_labels_dict,
        back_ngals_dict=back_ngals_dict,
        tomo_labels_dict=tomo_labels_dict,
        shear_ofmt=shapecone_fmt,
    )

    result = fastpm_runner.gen_mock_shear(
        icosmo=icosmo,
        irlz=0,
        save=True,
    )

    logger.info(f"total galaxies: {len(result)}")
    for tomo_label in (3, 4, 5):
        sel = result[result["tomo"] == tomo_label]
        dz = sel["z"] - sel["z_true"]
        logger.info(
            f"  tomo{tomo_label}: {len(sel)} galaxies, "
            f"z_true mean {sel['z_true'].mean():.3f}, "
            f"photo-z scatter (z-z_true) std {dz.std():.4f} "
            f"(sigma_phz={fastpm_config.sigma_phz})"
        )
    g1n = result["g1"]; g1p = result["g1_pure"]
    noise = g1n - g1p
    logger.info(
        f"shape noise: std(g1 - g1_pure) = {noise.std():.4f} "
        f"(sigma_e={fastpm_config.sigma_e}), "
        f"pure shear std = {g1p.std():.4f}"
    )

    out_fname = shapecone_fmt.format(icosmo, 0)
    logger.info(f"saved to {out_fname} ({os.path.getsize(out_fname)/1e6:.1f} MB)")
