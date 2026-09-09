''' Script to generate FastPM mock void catalogs '''

import os
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from astropy.table import Table
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

    cosmo_hod_file = (
        "/public/home/suchen/Programs/cosmogrid_mock_pipe/"
        "fastpm_runs/cosmo_hod_pairs.json"
    )
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

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:

        logger.info("Read cosmo labels")

        cosmo_labels_global = get_cosmo_labels_processed(cosmo_hod_file)

        chunks = divide_MPI_chunks(cosmo_labels_global, size)
        Path(voidcone_fmt).parent.mkdir(parents=True, exist_ok=True)

    else:
        chunks = None

    if rank == 0:

        logger.info("Scattering labels")

    cosmo_labels_local = comm.scatter(chunks, root=0)

    fastpm_runner = FastPMRunner.build_void_runner(
        config=fastpm_config,
        cosmo_par_fname=cosmo_par_fname,
        fore_mask_fnames_dict=mask_fnames_dict,
        fore_nofz_fnames_dict=nofz_fnames_dict,
        fore_survey_labels_dict=survey_labels_dict,
        void_ofmt=voidcone_fmt,
    )

    NHOD_PER_COSMO = fastpm_runner.config.nhod_per_cosmo
    NRLZS_PER_COSMO = fastpm_runner.config.nrlzs_per_cosmo

    ### Loop from cosmo_labels
    for icosmo in cosmo_labels_local:

        logger.info(f"Rank {rank}: start processing cosmo_{icosmo:06d}")

        for irlz in range(NRLZS_PER_COSMO):

            for ihod in range(NHOD_PER_COSMO):

                try:
                    galcone = Table.read(galcone_fmt.format(icosmo, irlz, ihod))
                    _ = fastpm_runner.gen_mock_void(
                        icosmo, irlz, ihod, galcone,
                        dive_input=dive_input_fmt.format(rank),
                        dive_output=dive_output_fmt.format(rank),
                        save=True,
                    )
                except Exception as err:
                    logger.error(f"Rank {rank}: cosmo_{icosmo:06d} irlz={irlz} ihod={ihod} failed: {err}")
