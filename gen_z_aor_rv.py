'''
Build kde of p(z) or p(z, Rv) from given discrete samples.
'''

import numpy as np
import pickle
from loguru import logger

from utils.io_func import *
from utils.mkfore_utils import bounded_kde_transform

hod_param_fname = "cfgs/hod/hod_5params_dict.json"
vfmt = "/data2/suchen/CosmoGrid/Void/cosmo_{:06d}_run_0_HOD_{}_run_0_boss_north.npy"
ofmt = "/data2/suchen/CosmoGrid/NofZ/Void/cosmo_{:06d}_run_0_HOD_{}_run_0_boss_north.pkl"

if __name__ == "__main__":
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        
        logger.info("Read cosmology and HOD parameters")

        hod_params_dict = get_hod_params(hod_param_fname)
        cosmo_labels_tot = []

        for icosmo_str in hod_params_dict.keys():
            if len(hod_params_dict[icosmo_str]) > 0:
                cosmo_labels_tot.append(int(icosmo_str[5:]))

        k, m = divmod(len(cosmo_labels_tot), size)
        clb_chunks_tbs = [cosmo_labels_tot[i*k + min(i,m) : (i + 1)*k + min(i+1, m)] for i in range(size)]
        chunks = []
        for ichunk in clb_chunks_tbs:
            hod_dict_tbs = {}
            for icosmo in ichunk:
                hod_dict_tbs["cosmo{:06d}".format(icosmo)] = hod_params_dict["cosmo{:06d}".format(icosmo)]
            chunks.append(hod_dict_tbs)
    
    else:
        chunks = None

    hod_param_dict_local = comm.scatter(chunks, root=0)

    for icosmo_str in hod_param_dict_local.keys():
        cosmo_label = int(icosmo_str[5:])
        for ihod in range(len(hod_param_dict_local[icosmo_str])):
            fnamebase = vfmt.format(cosmo_label, ihod)

            logger.info(f"Load void catalog from {fnamebase}")

            vcat = np.load(fnamebase)
            rv_bounds = [0, 40]
            rvcut = (vcat['Rv'] > rv_bounds[0]) & (vcat['Rv'] < rv_bounds[1])
            vcat = vcat[rvcut]

            zmin = np.minimum(vcat['z'].min(), 0.2)
            zmax = np.maximum(vcat['z'].max(), 0.4)
            Rvmin = np.minimum(vcat['Rv'].min(), rv_bounds[0])
            Rvmax = np.maximum(vcat['Rv'].max(), rv_bounds[1])

            logger.info("Generate redshifts that follows void redshifts distribution")

            z_rv_bounds = [(zmin, zmax), (Rvmin, Rvmax)]
            bounded_kde = bounded_kde_transform(np.c_[vcat['z'], vcat['Rv']], z_rv_bounds)

            logger.info("Save kde to file.")

            with open(ofmt.format(cosmo_label, ihod), 'wb') as f:
                pickle.dump(bounded_kde, f)
            