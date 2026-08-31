''' Script to sample FastPM HOD parameters '''

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

def get_cosmo_labels_initial(fname:str):
    '''
    Read zero-based cosmology labels from the FastPM parameter table.
    '''
    if not os.path.exists(fname):
        raise FileNotFoundError(f"File {fname} not found!")

    rows = np.loadtxt(fname, comments="#", ndmin=2)
    return list(range(len(rows)))

def get_hod_params_container(params_list):
    params_container = {}
    if params_list is not None:
        for i in range(len(params_list)):
            params_container[f'HOD{i}'] = params_list[i].tolist()

    return params_container

def merge_hod_sample_parts(cosmo_hod_pairs_parts):
    cosmo_hod_pairs = {}
    for cosmo_hod_pairs_part in cosmo_hod_pairs_parts:
        cosmo_hod_pairs.update(cosmo_hod_pairs_part)

    return cosmo_hod_pairs

def save_hod_samples(fname:str, cosmo_hod_pairs:dict):
    '''
    Save (cosmo_label, hod_params) pairs.
    '''

    if not os.path.isdir(os.path.dirname(fname)):
        raise FileNotFoundError(f"Dictionary {os.path.dirname(fname)} not found !")

    with open(fname, "w+") as f:
        json.dump(cosmo_hod_pairs, f)

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
        param_prior_low  = np.array([13, 0.1, 13, 0.00, 0.0]),
        param_prior_high = np.array([13.6, 0.6, 15.0, 10.0, 1.5]),

        ### lightcone redshift range
        zmin_lightcone = 0.4,
        zmax_lightcone = 0.6,
        ctr_lightcone = [0,0,0],
        rsd_lightcone = True,

        ### nofz
        nofz_method = "const", # can be `rank`, `downsample`, or `const`
    )

    cosmo_par_fname = (
        "/Users/suqikuai777/Dataspace/FastPM/Cosmology/cosmo_list.txt"
    )
    halo_fmt = (
        "/Users/suqikuai777/Dataspace/FastPM/Cosmology/"
        "L1000_N1024_1000cosmo/cosmo{:d}/"
        "a_{:5.4f}/rstar/out_0_wPID.list"
    )
    hod_samples_output = Path(
        "/Users/suqikuai777/Dataspace/FastPM/MockCatalogs/"
        "cfgs/hod/cosmo_hod_pairs.json"
    )

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:

        logger.info("Read cosmo labels")

        cosmo_labels_global = get_cosmo_labels_initial(cosmo_par_fname)

        chunks = divide_MPI_chunks(cosmo_labels_global, size)

        hod_samples_output.parent.mkdir(parents=True, exist_ok=True)

    else:
        chunks = None

    if rank == 0:

        logger.info("Scattering labels")

    cosmo_labels_local = comm.scatter(chunks, root=0)

    fastpm_runner = FastPMRunner.build_hod_runner(
        config=fastpm_config,
        halo_fmt=halo_fmt,
        cosmo_par_fname=cosmo_par_fname,
    )

    cosmo_hod_pairs_local = {}

    ### Loop from cosmo_labels
    for icosmo in cosmo_labels_local:

        logger.info(f"Rank {rank}: start processing cosmo_{icosmo:06d}")

        hod_params_alive = fastpm_runner.sample_hod_params(icosmo, 0)

        cosmo_hod_pairs_local[f'cosmo_{icosmo:06d}'] = get_hod_params_container(hod_params_alive)

    logger.info(f"Rank {rank} finished.")

    cosmo_hod_pairs_global = comm.gather(cosmo_hod_pairs_local, root=0)

    if rank == 0:
        merged_hod_samples = merge_hod_sample_parts(cosmo_hod_pairs_global)
        save_hod_samples(hod_samples_output, merged_hod_samples)
