''' Script to generate mock shape catalog from Abacus shear maps '''

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from loguru import logger

from handler import PipeConfig
from runner import AbacusRunner

if __name__ == "__main__":
    cosmogridV1_config = PipeConfig(
        ### fixed siminfo
        Lbox = 900.0,
        Npart = 832,
        redshift = 0.5125,
        # ### HOD model params
        model = 2, # label of model name.
        model_params_names = ('logMcut', 'sigma_logM', 'logM1', 'k', 'alpha', 'fic'),
        nhod_per_cosmo = 10,
        Num_ptcl_requirement = 12,
        verbose = True,
        num_seeds = 1,
        init_seed = 33000,
        ngal_ref = 4e-4,
        z_space = False,

        ### HOD param sampling
        param_prior_low  = np.array([13, 0.1, 13, 0.00, 0.0]),
        param_prior_high = np.array([13.6, 0.6, 15.0, 10.0, 1.5]),

        ### lightcone redshift range
        zmin_lightcone = 0.4,
        zmax_lightcone = 0.6,
        ctr_lightcone = [0,0,0],
        rsd_lightcone = True,

        ### nofz
        nofz_method = "downsample", # can be `rank`, `downsample`, or `const`,

        dive_exec_path = "/home/suchen/applications/DIVE/DIVE",

        ### shape noise / photo-z
        sigma_e = 0.3,
        seed_SN = 0,
        sigma_phz = 0.01,
        seed_Phz = 26120,
    )

    shear_map_fmt = "/data2/suchen/Abacus/shear_maps/gamma{:d}_rt_z{:.2f}.fits"

    wdir = "/home/suchen/Program/CosmoGrid"
    mask_dirbase = f"{wdir}/catalogs/masks"
    nofz_dirbase = f"{wdir}/catalogs/NOfZ"

    ### background survey: KiDS1000-North, 5 tomographic bins
    back_mask_fnames_dict = {
        'KiDS1000-North': f"{mask_dirbase}/kids1000_geom/mask_KiDS_North_1024.fits",
    }
    back_survey_labels_dict = {
        'KiDS1000-North': 0,
    }

    back_ngals_dict = {'tomo1': 0.62,
                       'tomo2': 1.18,
                       'tomo3': 1.85,
                       'tomo4': 1.26,
                       'tomo5': 1.31}
    tomo_labels_dict = {'tomo1': 1,
                        'tomo2': 2,
                        'tomo3': 3,
                        'tomo4': 4,
                        'tomo5': 5}
    nz_kids1000_fbase = f"{nofz_dirbase}/kids1000_nofzs"
    back_nofz_ffmt = nz_kids1000_fbase + "/K1000_NS_V1.0.0A_ugriZYJHKs_photoz_SG_mask_LF_svn_309c_2Dbins_v2_SOMcols_Fid_blindC_TOMO{}_Nz.asc"
    back_nofz_fnames_dict = {'tomo1': back_nofz_ffmt.format(1),
                             'tomo2': back_nofz_ffmt.format(2),
                             'tomo3': back_nofz_ffmt.format(3),
                             'tomo4': back_nofz_ffmt.format(4),
                             'tomo5': back_nofz_ffmt.format(5)}

    ### Abacus source shells: complete grid z = 0.05 - 2.00 (step 0.05)
    redshift_src_list = [
        0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
        0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
        0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20,
        1.25, 1.30, 1.35, 1.40, 1.45, 1.50, 1.55, 1.60,
        1.65, 1.70, 1.75, 1.80, 1.85, 1.90, 1.95, 2.00,
    ]

    ### source clustering: mass maps & shell correspondence (test version)
    mass_map_fmt = "/data2/suchen/Abacus/mass_maps/shell_{:d}.fits"
    z_to_mass_label = {z: (12 if z < 0.5 else 13) for z in redshift_src_list}
    position_method = "density"  # "random" or "density"
    bias = 1.0

    shape_ofmt = "/data2/suchen/Abacus/shape_cats/abacus_run_0_kids_north_5tomos_density.fits"

    abacus_runner = AbacusRunner.build_shape_runner(
        config=cosmogridV1_config,
        shear_map_fmt=shear_map_fmt,
        back_mask_fnames_dict=back_mask_fnames_dict,
        back_nofz_fnames_dict=back_nofz_fnames_dict,
        back_survey_labels_dict=back_survey_labels_dict,
        back_ngals_dict=back_ngals_dict,
        tomo_labels_dict=tomo_labels_dict,
        redshift_src_list=redshift_src_list,
        shear_ofmt=shape_ofmt,
        mass_map_fmt=mass_map_fmt,
        z_to_mass_label=z_to_mass_label,
        position_method=position_method,
        bias=bias,
    )

    logger.info("Generate Abacus shape catalog ...")
    shapecone_survey = abacus_runner.gen_mock_shear(save=True)
    logger.info(f"Done: {len(shapecone_survey)} galaxies written to {shape_ofmt}")
