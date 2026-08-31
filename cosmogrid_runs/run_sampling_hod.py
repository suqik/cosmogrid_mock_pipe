''' Script to generate mock galaxy catalog '''

import os
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from loguru import logger

from handler import PipeConfig
from runner import CosmoGridRunner

def divide_MPI_chunks(data, size):
    k, m = divmod(len(data), size)
    chunks = [data[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(size)]
    return chunks

def get_cosmo_labels_initial(fname:str):
    '''
    Read cosmo labels from original cosmo file.
    '''
    if not os.path.exists(fname):
        raise FileNotFoundError(f"File {fname} not found!")
    
    with open(fname, "r") as f:
        cosmo_labels = []
        for line in f.readlines():
            cosmo_labels.append(int(line.strip("\n").split("_")[1]))

    return cosmo_labels

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

    cosmogridV1_config = PipeConfig(
        ### fixed siminfo
        Lbox = 900.0,
        Npart = 832,
        redshift = 0.5125,
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

    sim_fmt = "/data3/suchen/CosmoGridV1/grid/cosmo_{:06d}/run_{:d}/"
    halo_fmt = "pkd_halos/CosmoML.{:05d}.fofstats.0"
    lb_z_file = "/data3/suchen/CosmoGridV1/grid_info/label_z_table.txt"

    wdir = "/home/suchen/Program/CosmoGrid"

    ### Geometry & masks
    mask_boss_fdir = f"{wdir}/catalogs/masks/boss_geom/"

    mask_fnames_dict = {
        # 'boss_lowz_ngc': mask_boss_fdir + "mask_DR12v5_LOWZ_North.ply", # Note LOWZE2 and LOWZE3 need LOWZ for trimming
        # 'boss_lowze2_ngc': mask_boss_fdir + "mask_DR12v5_LOWZE2_North.ply",
        # 'boss_lowze3_ngc': mask_boss_fdir + "mask_DR12v5_LOWZE3_North.ply",
        'boss_cmass_ngc': mask_boss_fdir + "mask_DR12v5_CMASS_North.ply",
        'boss_veto': [
            mask_boss_fdir + "badfield_mask_postprocess_pixs8.ply",
            mask_boss_fdir + "badfield_mask_unphot_seeing_extinction_pixs8_dr12.ply",
            mask_boss_fdir + "allsky_bright_star_mask_pix.ply",
            mask_boss_fdir + "bright_object_mask_rykoff_pix.ply", 
            mask_boss_fdir + "collision_priority_mask_dr12.ply", 
            mask_boss_fdir + "centerpost_mask_dr12.ply"
        ],
        '2dflens_south': f"{wdir}/catalogs/masks/2dflens_geom/2dFLens_mask_weight_South.fits"
    }

    ### N of Z
    nz_fbase = f"{wdir}/catalogs/NOfZ/lens/"
    nofz_fnames_dict = {
        # 'boss_lowz_ngc': nz_fbase + "nbar_DR12v5_LOWZ_North_om0p31_Pfkp10000.dat",
        # 'boss_lowze2_ngc': nz_fbase + "nbar_DR12v5_LOWZE2_North_om0p31_Pfkp10000.dat",
        # 'boss_lowze3_ngc': nz_fbase + "nbar_DR12v5_LOWZE3_North_om0p31_Pfkp10000.dat",
        'boss_cmass_ngc': nz_fbase + "nbar_DR12v5_CMASS_North_om0p31_Pfkp10000.dat",
        '2dflens_south': nz_fbase + "nbar_2dFLens_south_data.dat"
    }

    ### survey labels
    survey_labels_dict = {
        # 'boss_lowz_ngc': 0,
        # 'boss_lowze2_ngc': 1,
        # 'boss_lowze3_ngc': 2,
        'boss_cmass_ngc': 4,
        '2dflens_south': 3
    }

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        
        logger.info("Read cosmo labels")

        cosmo_labels_global = get_cosmo_labels_initial("/data3/suchen/CosmoGridV1/grid/dirnames.txt")

        chunks = divide_MPI_chunks(cosmo_labels_global, size)

        hod_samples_output = f"{wdir}/cfgs/hod/cosmo_hod_pairs.json"

    else:
        chunks = None

    if rank == 0:

        logger.info("Scattering labels")

    cosmo_labels_local = comm.scatter(chunks, root=0)

    cosmogrid_runner = CosmoGridRunner.build_hod_runner(
        config=cosmogridV1_config,
        sim_fmt=sim_fmt,
        halo_fmt=halo_fmt,
        lb_z_file=lb_z_file,
    )
    
    cosmo_hod_pairs_local = {}

    ### Loop from cosmo_labels
    for icosmo in cosmo_labels_local:

        logger.info(f"Rank {rank}: start processing cosmo_{icosmo:06d}")

        hod_params_alive = cosmogrid_runner.sample_hod_params(icosmo, 0)

        cosmo_hod_pairs_local[f'cosmo_{icosmo:06d}'] = get_hod_params_container(hod_params_alive)

    logger.info(f"Rank {rank} finished.")

    cosmo_hod_pairs_global = comm.gather(cosmo_hod_pairs_local, root=0)

    if rank == 0:
        merged_hod_samples = merge_hod_sample_parts(cosmo_hod_pairs_global)
        save_hod_samples(hod_samples_output, merged_hod_samples)
