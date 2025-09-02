'''
Make background galaxies.
Random sampling galaxies from background shear map, and add shape noise
'''

import numpy as np
import h5py
# from tqdm import trange

from io_func import *
from mkback_utils import *

from loguru import logger

''' simulation info '''
sim_fmt = "/data3/suchen/CosmoGridV1/raytrace/cosmo_{:06d}/nside0512/raytracing_z{:.2f}_nufft.hdf5"
redshift_src_list = np.concatenate([np.arange(0.1,1.0,0.05), np.arange(1.0, 2.0, 0.1), np.array([2.0,2.2,2.4,2.8,3.2,3.6])])

''' galaxy params '''
ngal_list = [0.62, 1.18, 1.85, 1.26, 1.31] # arcmin^-2, from Table. 1 in 2404.15402
sigma_e = 0.3
seed = 0
mask_file = "/home/suchen/Program/CosmoGrid/catalogs/masks/mask_KiDS_North_1024.fits"
nofz_file_fmt = "catalogs/NOfZ/srcs/K1000_NS_V1.0.0A_ugriZYJHKs_photoz_SG_mask_LF_svn_309c_2Dbins_v2_SOMcols_Fid_blindC_TOMO{}_Nz.asc"
nz_tomo_bins = 5

''' output file info'''
out_dir = "/data2/suchen/CosmoGrid/Shape/"
out_fmt = "cosmo_{:06d}_run_0_kids_north_tomo{:d}_wo_noise.txt"

''' main routine '''
### read shear map
def read_shear_maps(cosmo_label:int, redshift_src_list:list) -> dict:
    shear_map_dict = {}
    for ishell in range(len(redshift_src_list)):
        redshift_src = redshift_src_list[ishell]

        shear_map_dict[f"shell{ishell}"] = {}
        shear_map_dict[f"shell{ishell}"]['redshift'] = redshift_src

        sim_fname = sim_fmt.format(cosmo_label, redshift_src)
        with h5py.File(sim_fname, 'r') as f:
            A = np.array(f["Distortion_matrix"]["Raytraced"])

        gamma1 = -(A[0][0] - A[1][1]) / 2
        gamma2 = -(A[0][1] + A[1][0]) / 2

        shear_map_dict[f"shell{ishell}"]['gamma1'] = gamma1
        shear_map_dict[f"shell{ishell}"]['gamma2'] = gamma2

    return shear_map_dict

if __name__ == "__main__":
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    ### read shear maps
    import datetime
    start = datetime.datetime.now()

    ### read cosmo labels
    if rank == 0:
        
        logger.info("Read cosmo labels")

        # with open("/data3/suchen/CosmoGridV1/grid/dirnames.txt", "r") as f:
        with open("/data2/suchen/CosmoGrid/Shape/missing_cosmo_labels.txt", "r") as f:
            dirnames = f.readlines()
            cosmo_labels_tot = [int(i.strip("\n").split("_")[1]) for i in dirnames]

        k, m = divmod(len(cosmo_labels_tot), size)
        chunks = [cosmo_labels_tot[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(size)]
    else:
        chunks = None

    if rank == 0:
        logger.info("Scattering labels")

    cosmo_labels = comm.scatter(chunks, root=0)

    ### read mask
    logger.info("Load mask")

    mask = loadFitsMaps(mask_file)
    mask = mask[0]

    ### FIXME: For test, KiDS-North mask was downloaded from kids-sbi repository
    ### https://github.com/mwiet/kids_sbi . However, I cannot read the binary type
    ### file, which may consider some effects depending on position. So I just read
    ### the mask as a float64 array and convert it to boolean
    mask = np.where(mask > 0, 1, 0)

    # ###############   For Test   #################
    # logger.info("Generate 8 rotated masks")

    # nside = hp.npix2nside(len(mask))
    # rot_degrees_list = [
    #     [0,50,0],
    #     [90,0,-50],
    #     [180,-50,0],
    #     [270,0,50],
    #     [0,-50,0],
    #     [90,0,50],
    #     [180,50,0],
    #     [270,0,-50],
    # ]

    # new_mask_list = []
    # for rot_degrees in rot_degrees_list:
    #     new_mask = np.zeros_like(mask)
    #     new_mask_pix = rotate_pix(np.argwhere(mask!=0).flatten(), nside=nside, rot_degrees=rot_degrees)
    #     new_mask[new_mask_pix] = 1
    #     new_mask_list.append(new_mask)
    # ###############################################

    ### read nofz
    logger.info("Load nofz")

    nofz_dict = {}
    for i in range(1, nz_tomo_bins+1):
        filename = nofz_file_fmt.format(i)
        tmp = np.loadtxt(filename)
        nofz_dict[f'tomo{i}'] = make_nofz(tmp[:,0], tmp[:,1])

    # cosmo_label = 3
    for cosmo_label in cosmo_labels:
        logger.info("Process cosmo {:06d}\n".format(cosmo_label))
        logger.info("Load shear maps")

        shear_map_dict = read_shear_maps(cosmo_label, redshift_src_list)

        ### generate background galaxies for each tomographic bin
        logger.info("Generate background galaxies")

        for itomo in range(nz_tomo_bins):
            
            logger.info("Tomographic bin {}".format(itomo+1))

            bg_galcat = gen_gal_positions(ngal_list[itomo], mask, nofz_dict[f'tomo{itomo+1}'])
            # bg_galcat = get_gal_shear(bg_galcat, shear_map_dict, sigma_e=sigma_e, seed=seed+1)
            bg_galcat = get_gal_shear(bg_galcat, shear_map_dict, sigma_e=None, seed=seed+1)

            np.savetxt(out_dir + out_fmt.format(cosmo_label, itomo+1), bg_galcat, fmt='%.3f %.3f %.5f %.5f %.8f %.8f %.3f')

        # ############################      For Test      ###########################
        # out_fmt = "cosmo_{:06d}_run_0_kids_north_tomo{:d}_wo_noise_part{}.txt"
        # for ipart in range(8):
            
        #     logger.info(f"Generate Part {ipart}")
            
        #     curr_mask = new_mask_list[ipart]

        #     for itomo in range(2,3):
                
        #         logger.info("Tomographic bin {}".format(itomo+1))

        #         bg_galcat = gen_gal_positions(ngal_list[itomo], curr_mask, nofz_dict[f'tomo{itomo+1}'])
        #         # bg_galcat = get_gal_shear(bg_galcat, shear_map_dict, sigma_e=sigma_e, seed=seed+1)
        #         bg_galcat = get_gal_shear(bg_galcat, shear_map_dict, sigma_e=None, seed=seed+1)

        #         np.savetxt(out_dir + out_fmt.format(cosmo_label, itomo+1, ipart+1), bg_galcat, fmt='%.3f %.3f %.5f %.5f %.8f %.8f %.3f')
        # ############################################################################

    logger.info("Done.")
    end = datetime.datetime.now()
    logger.info("Elapsed time: {}".format(end - start))