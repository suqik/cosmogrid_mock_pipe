'''
Build kde of p(z) or p(z, Rv) from given discrete samples.
'''

import os
import numpy as np
from loguru import logger

from utils.io_func import *
from utils.mkfore_utils import get_abundance

hod_param_fname = "cfgs/hod/hod_5params_dict.json"
vfmt = "/data2/suchen/CosmoGrid/fiducial_suits/Void/cosmo_{:06d}_run_0_HOD_{}_run_0_boss_north.npy"
ofmt = "results/VSF/fiducial/cosmo_{:06d}_run_0_HOD_{}_run_0_boss_north.npz"
   
if __name__ == "__main__":
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    '''  ---------------  HOD run  ---------------- '''
    if rank == 0:

        if not os.path.isdir(os.path.dirname(ofmt)):
            os.makedirs(os.path.dirname(ofmt))
        
        logger.info("Read cosmo labels")

        cosmo_labels_tot = get_cosmo_name_list_process(hod_param_fname)
        # cosmo_labels_tot = [1,2,3,4]

        k, m = divmod(len(cosmo_labels_tot), size)
        chunks = [cosmo_labels_tot[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(size)]
    else:
        chunks = None

    if rank == 0:
        logger.info("Scattering labels")

    cosmo_labels = comm.scatter(chunks, root=0)

    # with open("/data2/suchen/CosmoGrid/fix_HOD_suits/HOD/ngals.json", "r") as f:
    #     ngal_dict = json.load(f)

    bin_edges = np.linspace(0, 3, 101)

    for icosmo in cosmo_labels:
        for ihod in range(10):
            fnamebase = vfmt.format(icosmo, ihod)

            logger.info(f"Load void catalog from {fnamebase}")

            vcat = np.load(fnamebase)

            curr_ngal = 3.5
            scaled_Rv = vcat['Rv']*np.cbrt(curr_ngal*1e-4)
            slt = scaled_Rv < 3.0
            scaled_Rv = scaled_Rv[slt]

            fRv, fedges = get_abundance(scaled_Rv, bin_edges)
            fRv_normal = fRv/np.sum(fRv)

            np.savez(ofmt.format(icosmo, ihod), fRv=fRv_normal, fedges=fedges)

    '''  ---------------  fix HOD run  ---------------- '''
    # if rank == 0:
        
    #     logger.info("Read cosmo labels")

    #     cosmo_labels_tot = get_cosmo_name_list_original("/data3/suchen/CosmoGridV1/grid/dirnames.txt")
    #     # cosmo_labels_tot = [1,2,3,4]

    #     k, m = divmod(len(cosmo_labels_tot), size)
    #     chunks = [cosmo_labels_tot[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(size)]
    # else:
    #     chunks = None

    # if rank == 0:
    #     logger.info("Scattering labels")

    # cosmo_labels = comm.scatter(chunks, root=0)

    # with open("/data2/suchen/CosmoGrid/fix_HOD_suits/HOD/ngals.json", "r") as f:
    #     ngal_dict = json.load(f)

    # bin_edges = np.linspace(0, 3, 101)

    # for icosmo in cosmo_labels:
    #     fnamebase = "/data2/suchen/CosmoGrid/fix_HOD_suits/Void/cosmo_{:06d}_run_0_HOD_0_run_0_boss_north.npy".format(icosmo)

    #     logger.info(f"Load void catalog from {fnamebase}")

    #     vcat = np.load(fnamebase)

    #     curr_ngal = ngal_dict[f'cosmo{icosmo:06d}']
    #     scaled_Rv = vcat['Rv']*np.cbrt(curr_ngal*1e-4)
    #     slt = scaled_Rv < 3.0
    #     scaled_Rv = scaled_Rv[slt]

    #     fRv, fedges = get_abundance(scaled_Rv, bin_edges)
    #     fRv_normal = fRv/np.sum(fRv)

    #     np.savez(ofmt.format(icosmo, 0), fRv=fRv_normal, fedges=fedges)