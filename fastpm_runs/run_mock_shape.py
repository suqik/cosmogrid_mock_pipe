''' Script to generate FastPM mock shape catalogs '''

import os
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from loguru import logger

from handler import PipeConfig
from runner import FastPMRunner

def divide_MPI_chunks(data, size):
    k, m = divmod(len(data), size)
    chunks = [data[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(size)]
    return chunks

def get_cosmo_labels_processed(fname:str):
    '''
    Read cosmo labels from hod param json file.
    '''

    cosmo_hod_info = load_hod_samples(fname)

    cosmo_labels = []
    for icosmo_str in cosmo_hod_info.keys():
        cosmo_labels.append(int(icosmo_str[6:]))

    return cosmo_labels

def load_hod_samples(fname:str):
    '''
    Load (cosmo_label, hod_params) pairs.
    '''

    if not os.path.isdir(os.path.dirname(fname)):
        raise FileNotFoundError(f"Dictionary {os.path.dirname(fname)} not found !")

    with open(fname, "r") as f:
        cosmo_hod_pairs = json.load(f)

    return cosmo_hod_pairs

if __name__ == "__main__":
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
        ngal_ref = 4e-4,
        z_space = False, ## RSD in box. Note if need RSD in survey-like, do not open this.

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

        sigma_e = 0.3,
        seed_SN = 0,
        sigma_phz = 0.01,
        seed_Phz = 26120,
    )

    cosmo_par_fname = (
        "/Users/suqikuai777/Dataspace/FastPM/Cosmology/cosmo_list.txt"
    )
    shear_sim_fmt = (
        "/Users/suqikuai777/Workspace/fast_shear_map/outputs/"
        "dz_tomography_acceptance_v2/products/cosmo_{:06d}/"
        "realization_{:04d}.npz"
    )

    wdir = "/home/suchen/Program/CosmoGrid"
    mask_dirbase = f"{wdir}/catalogs/masks"
    nofz_dirbase = f"{wdir}/catalogs/NOfZ"

    back_mask_fnames_dict = {
        # 'KiDS1000-North': f"{mask_dirbase}/mask_KiDS_North_1024.fits",
        'boss_cmass_ngc': f"{mask_dirbase}/boss_geom/mask_DR12v5_CMASS_North.ply"
    }

    back_survey_labels_dict = {
        # 'KiDS1000-North': 0,
        'boss_cmass_ngc': 2
    }

    tomo_name_list = ['tomo{}'.format(i) for i in range(1,6)]
    ngal_list = [0.62, 1.18, 1.85, 1.26, 1.31]
    tomo_label_list = np.arange(5) + 1
    back_nofz_ffmt = nofz_dirbase + "/srcs/K1000_NS_V1.0.0A_ugriZYJHKs_photoz_SG_mask_LF_svn_309c_2Dbins_v2_SOMcols_Fid_blindC_TOMO{}_Nz.asc"
    back_nofz_fnames = [back_nofz_ffmt.format(i) for i in range(1, 6)]

    # back_ngals_dict = dict(zip(tomo_name_list, ngal_list))
    # tomo_labels_dict = dict(zip(tomo_name_list, tomo_label_list))
    # back_nofz_fnames_dict = dict(zip(tomo_name_list, back_nofz_fnames))

    back_ngals_dict = {'tomo4': 1.26,
                    'tomo5': 1.31}
    tomo_labels_dict = {'tomo4': 4,
                        'tomo5': 5}
    back_nofz_fnames_dict = {'tomo4': back_nofz_ffmt.format(4),
                            'tomo5': back_nofz_ffmt.format(5)}

    cosmo_hod_file = (
        "/Users/suqikuai777/Dataspace/FastPM/MockCatalogs/"
        "cfgs/hod/cosmo_hod_pairs.json"
    )
    shapecone_fmt = (
        "/Users/suqikuai777/Dataspace/FastPM/MockCatalogs/Shapes/"
        "cosmo_{:06d}_realization_{:04d}_boss_north_2tomos.fits"
    )

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:

        logger.info("Read cosmo labels")

        cosmo_labels_global = get_cosmo_labels_processed(cosmo_hod_file)

        chunks = divide_MPI_chunks(cosmo_labels_global, size)
        Path(shapecone_fmt).parent.mkdir(parents=True, exist_ok=True)

    else:
        chunks = None

    if rank == 0:

        logger.info("Scattering labels")

    cosmo_labels_local = comm.scatter(chunks, root=0)

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

    NRLZS_PER_COSMO = fastpm_runner.config.nrlzs_per_cosmo

    ### Loop from cosmo_labels
    for icosmo in cosmo_labels_local:

        logger.info(f"Rank {rank}: start processing cosmo_{icosmo:06d}")

        for irlz in range(NRLZS_PER_COSMO):

            _ = fastpm_runner.gen_mock_shear(
                icosmo=icosmo,
                irlz=irlz,
                save=True,
            )
