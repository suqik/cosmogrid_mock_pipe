''' Script to generate mock galaxy catalog '''

import os
import json
import numpy as np
from loguru import logger

from handler import PipeConfig
from runner import CosmoGridRunner

def divide_MPI_chunks(data, size):
    k, m = divmod(len(data), size)
    chunks = [data[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(size)]
    return chunks

def get_cosmo_labels_processed(fname:str):
    '''
    Read cosmo labels from hod param json file.
    '''

    if not os.path.exists(fname):
        raise FileNotFoundError(f"File {fname} not found!")

    with open(fname, "r") as f:
        cosmo_hod_info = json.load(f)
    
    cosmo_labels = []
    for icosmo_str in cosmo_hod_info.keys():
        if len(cosmo_hod_info[icosmo_str]) == 11:
            cosmo_labels.append(int(icosmo_str[5:]))

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

    cosmo_hod_file = f"{wdir}/cfgs/hod/hod_5params_dict_free_ngal_cmass_v2.json"
    galcone_fmt = "/data2/suchen/CosmoGrid/Free_NGAL_wrsd/HOD_cmass/grid/Gals/cosmo_{:06d}_run_{:d}_HOD_{:d}_run_0_boss_north_2dflens_south.fits"

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        
        logger.info("Read cosmo labels")

        cosmo_labels_global = get_cosmo_labels_processed(cosmo_hod_file)

        chunks = divide_MPI_chunks(cosmo_labels_global, size)

    else:
        chunks = None

    if rank == 0:

        logger.info("Scattering labels")

    cosmo_labels_local = comm.scatter(chunks, root=0)

    cosmo_hod_pairs = load_hod_samples(cosmo_hod_file)

    cosmogrid_runner = CosmoGridRunner(config=cosmogridV1_config, 
                                    sim_fmt=sim_fmt,
                                    halo_fmt=halo_fmt,
                                    lb_z_file=lb_z_file,
                                    mask_fnames_dict=mask_fnames_dict,
                                    nofz_fnames_dict=nofz_fnames_dict,
                                    survey_labels_dict=survey_labels_dict,
                                    gal_fmt=galcone_fmt)
    
    NHOD_PER_COSMO = cosmogrid_runner.config.nhod_per_cosmo
    NRLZS_PER_COSMO = cosmogrid_runner.config.nrlzs_per_cosmo
    
    ### Loop from cosmo_labels
    for icosmo in cosmo_labels_local:

        logger.info(f"Rank {rank}: start processing cosmo_{icosmo:06d}")

        curr_hod_params_dict = cosmo_hod_pairs[f'cosmo{icosmo:06d}']

        for irlz in range(NRLZS_PER_COSMO):

            for ihod in range(NHOD_PER_COSMO):

                curr_hod_param = curr_hod_params_dict[f'HOD{ihod}']
                _ = cosmogrid_runner.gen_mock_gal(icosmo, irlz=irlz, ihod=ihod, ihod_param=curr_hod_param, save=True)