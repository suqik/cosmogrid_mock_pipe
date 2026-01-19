'''
Make background galaxies.
Random sampling galaxies from background shear map, and add shape noise
'''

import os, sys
import numpy as np
import h5py

from utils.io_func import *
from utils.mkback_utils import *

from loguru import logger

''' simulation info '''
sim_fmt = "/data3/suchen/CosmoGridV1/raytrace/cosmo_{:06d}/nside0512/raytracing_z{:.2f}_nufft.hdf5"
redshift_src_list = np.concatenate([np.arange(0.1,1.0,0.05), np.arange(1.0, 2.0, 0.1), np.array([2.0,2.2,2.4,2.8,3.2,3.6])])

''' galaxy params '''
SURVEY_NAME = "KiDS"

sigma_e = 0.3 # Set `None` if do not add shape noise
seed = 0 ### initial seed of shape noise

if SURVEY_NAME == "KiDS":
    #### specify total tomographic bins
    nz_tomo_bins = 5
    #### specify ngal in each tomograpic bins, in arcmin^-2
    ngal_list = [0.62, 1.18, 1.85, 1.26, 1.31] # arcmin^-2, from Table. 1 in 2404.15402
    #### specify mask file.
    mask_file = "/home/suchen/Program/CosmoGrid/catalogs/masks/mask_KiDS_North_1024.fits"
    #### specify nofz file.
    nofz_file_fmt = "catalogs/NOfZ/srcs/K1000_NS_V1.0.0A_ugriZYJHKs_photoz_SG_mask_LF_svn_309c_2Dbins_v2_SOMcols_Fid_blindC_TOMO{}_Nz.asc"

elif SURVEY_NAME == "FullSky":
    nz_tomo_bins = 5
    ngal_list = [0.62, 1.18, 1.85, 1.26, 1.31] # arcmin^-2, from Table. 1 in 2404.15402
    nofz_file_fmt = None
    nofz = 1.0

elif SURVEY_NAME == "Custom":
    nz_tomo_bins = 5
    ngal_list = [0.62, 1.18, 1.85, 1.26, 1.31] # arcmin^-2, from Table. 1 in 2404.15402
    nofz_file_fmt = "catalogs/NOfZ/srcs/K1000_NS_V1.0.0A_ugriZYJHKs_photoz_SG_mask_LF_svn_309c_2Dbins_v2_SOMcols_Fid_blindC_TOMO{}_Nz.asc"

else:
    raise NotImplementedError

''' output file info'''
out_dir = "/data2/suchen/CosmoGrid/Shape/KiDS_ngal_suits/"
out_fmt = "cosmo_{:06d}_run_0_kids_north_tomo{:d}.npy"

''' If rotate the survey footprint'''
ROT = False
if ROT:
    logger.info("Generate 8 rotated masks")

    rot_degrees_list = [
        [0,50,0],
        [90,0,-50],
        [180,-50,0],
        [270,0,50],
        [0,-50,0],
        [90,0,50],
        [180,50,0],
        [270,0,-50],
    ]

logger.info(f"survey name: {SURVEY_NAME}")
logger.info(f"Nz tomos: {nz_tomo_bins}")
logger.info(f"ngals: {ngal_list}")
if sigma_e is not None:
    logger.info( "Add shape noise")
    logger.info(f"sigma_e = {sigma_e:.2f}")
if nofz_file_fmt is None:
    logger.info( "Use single plane zsrc(s)")
    logger.info(f"zsrc(s): {nofz}")
if ROT:
    logger.info(f"{len(rot_degrees_list)} Rotation angles combinations")
    logger.info(f"{rot_degrees_list}")

''' Useful functions'''
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

def make_mask_given_range(ra_min, ra_max, dec_min, dec_max, nside=1024):
    """
    Use healpy with lonlat=True to generate a rectangular RA/Dec mask.
    """

    npix = hp.nside2npix(nside)
    mask = np.zeros(npix)

    # 像素中心的 lon/lat（即 ra/dec）
    lon, lat = hp.pix2ang(nside, np.arange(npix), lonlat=True, nest=False)

    # RA wrap（跨越 0° 区间，例如 350→10）
    if ra_min <= ra_max:
        ra_sel = (lon >= ra_min) & (lon <= ra_max)
    else:
        ra_sel = (lon >= ra_min) | (lon <= ra_max)

    dec_sel = (lat >= dec_min) & (lat <= dec_max)

    mask[ra_sel & dec_sel] = 1

    return mask

if __name__ == "__main__":

    TEST_MODE = False
    if len(sys.argv) > 1 and sys.argv[-1] == "test":
        TEST_MODE = True

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    import datetime
    start = datetime.datetime.now()

    ### read cosmo labels
    if rank == 0:
        
        logger.info("Read cosmo labels")

        # hod_param_fname = "cfgs/hod/hod_5params_dict.json"
        # cosmo_labels_tot = get_cosmo_name_list_process(hod_param_fname)

        cosmo_labels_tot = get_cosmo_name_list_original("/data3/suchen/CosmoGridV1/grid/dirnames.txt")

        ####    For test    ####
        if TEST_MODE:
            cosmo_labels_tot = [1]
        ########################

        k, m = divmod(len(cosmo_labels_tot), size)
        chunks = [cosmo_labels_tot[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(size)]
    else:
        chunks = None

    if rank == 0:
        logger.info("Scattering labels")

    cosmo_labels = comm.scatter(chunks, root=0)

    ### read mask
    logger.info("Load mask")

    if SURVEY_NAME == "KiDS":
        mask = loadFitsMaps(mask_file)
        mask = mask[0]

        # ### FIXME: For test, KiDS-North mask was downloaded from kids-sbi repository
        # ### https://github.com/mwiet/kids_sbi . However, I cannot read the binary type
        # ### file, which may consider some effects depending on position. So I just read
        # ### the mask as a float64 array and convert it to boolean
        mask = np.where(mask > 0, 1, 0)

    ### fullsky mask
    if SURVEY_NAME == "FullSky":
        nside = 1024
        mask = np.ones_like(12*nside*nside)

    ### mask with given range
    if SURVEY_NAME == "Custom":
        mask = make_mask_given_range(105, 270, -5, 75, nside=1024)

    ### if apply rotation
    if ROT:
        logger.info("Generate 8 rotated masks")
    
        nside = hp.npix2nside(len(mask))
        new_mask_list = []
        for rot_degrees in rot_degrees_list:
            new_mask = np.zeros_like(mask)
            new_mask_pix = rotate_pix(np.argwhere(mask!=0).flatten(), nside=nside, rot_degrees=rot_degrees)
            new_mask[new_mask_pix] = 1
            new_mask_list.append(new_mask)

    ### read nofz
    if nofz_file_fmt is not None:
        logger.info("Load nofz")

        nofz_dict = {}
        for i in range(1, nz_tomo_bins+1):
            filename = nofz_file_fmt.format(i)
            tmp = np.loadtxt(filename)
            nofz_dict[f'tomo{i}'] = make_nofz(tmp[:,0], tmp[:,1])

    ### main loop
    for idx, cosmo_label in enumerate(cosmo_labels):
        logger.info("Process {}th cosmo {:06d}\n".format(idx, cosmo_label))
        logger.info("Load shear maps")

        shear_map_dict = read_shear_maps(cosmo_label, redshift_src_list)

        ### generate background galaxies for each tomographic bin
        logger.info("Generate background galaxies")

        for itomo in range(3,4):
            logger.info("Tomographic bin {}".format(itomo+1))
            if not ROT:
                ofilename = os.path.join(out_dir, out_fmt.format(cosmo_label, itomo+1))

                if nofz_file_fmt is not None:
                    bg_galcat = gen_gal_positions(ngal_list[itomo], mask, nofz_dict[f'tomo{itomo+1}'], logger)
                else:
                    bg_galcat = gen_gal_positions(ngal_list[itomo], mask, nofz, logger)
                bg_galcat = get_gal_shear(bg_galcat, shear_map_dict, sigma_e=sigma_e, seed=seed+idx)

                np.save(ofilename, bg_galcat)
            
            ### if apply rotations
            else:
                for ipart in range(8):
                    
                    logger.info(f"Rotation {ipart}, rotation angle {rot_degrees_list[ipart]}")
                    
                    curr_mask = new_mask_list[ipart]

                    bg_galcat = gen_gal_positions(ngal_list[itomo], curr_mask, nofz_dict[f'tomo{itomo+1}'])
                    bg_galcat = get_gal_shear(bg_galcat, shear_map_dict, sigma_e=sigma_e, seed=seed+idx)

                    np.save(out_dir + out_fmt.format(cosmo_label, itomo+1, ipart), bg_galcat)

    logger.info("Done.")
    end = datetime.datetime.now()
    logger.info("Elapsed time: {}".format(end - start))
