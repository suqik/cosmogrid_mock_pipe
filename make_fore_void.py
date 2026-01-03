'''
Script to find voids in lightcone.
'''

import pyccl as ccl
import numpy as np
import pymangle
import healpy as hp
from loguru import logger

from utils.io_func import *
from utils.mkfore_utils import *
from utils.mkback_utils import rotate_pix

wdir = "/home/suchen/Program/CosmoGrid"

''' cosmology, HOD and galaxy catalog info '''
sim_fmt = "/data3/suchen/CosmoGridV1/grid/cosmo_{:06d}/run_0/"
hod_param_fname = f"{wdir}/cfgs/hod/hod_5params_dict_high_ngal_wcosmo2.json"
zbin_lb = 2
gfmt = f"/data2/suchen/CosmoGrid/high_ngal_suits/HOD_bin{zbin_lb}/" + "cosmo_{:06d}_run_0_HOD_{}_run_0_boss_north_2dflens_south.npy"
''' output info '''
vfmt = f"/data2/suchen/CosmoGrid/high_ngal_suits/Void_bin{zbin_lb}/" + "cosmo_{:06d}_run_0_HOD_{}_run_0_boss_north_2dflens_south.npy"

if zbin_lb == 1:
    zmin = 0.2
    zmax = 0.4
elif zbin_lb == 2:
    zmin = 0.4
    zmax = 0.6

''' mask file info'''

mask_boss_fdir = f"{wdir}/catalogs/masks/boss_geom/"
### geometry files
geom_boss_fname_list = [
    mask_boss_fdir + "mask_DR12v5_CMASSLOWZ_North.ply",
    mask_boss_fdir + "mask_DR12v5_LOWZE2_North.ply",
    mask_boss_fdir + "mask_DR12v5_LOWZE3_North.ply",
    mask_boss_fdir + "mask_DR12v5_LOWZ_North.ply", # For trimming LOWZE2 and LOWZE3 regions
    mask_boss_fdir + "mask_DR12v5_CMASS_North.ply"
]
### mask files corresponding to observational effects
mask_boss_fname_list = [
    mask_boss_fdir + "badfield_mask_postprocess_pixs8.ply",
    mask_boss_fdir + "badfield_mask_unphot_seeing_extinction_pixs8_dr12.ply",
    mask_boss_fdir + "allsky_bright_star_mask_pix.ply",
    mask_boss_fdir + "bright_object_mask_rykoff_pix.ply", 
    mask_boss_fdir + "collision_priority_mask_dr12.ply", 
    mask_boss_fdir + "centerpost_mask_dr12.ply"
]

mask_weight_2df_fname = f"{wdir}/catalogs/masks/2dflens_geom/2dFLens_mask_weight.fits"

src_mask_file = f"{wdir}/catalogs/masks/mask_KiDS_North_1024.fits"

### Can only activate one of these three modes
HALO_ONLY = False # only use halo, which preserve the ngal but not G-H connection
FIX_HOD = False # use the same G-H connection but cannot preserve the ngal
VARY_HOD = True # preserve the ngal, as well as vary G-H connection

if VARY_HOD:
    nhod_per_cosmo = 10

### if apply rotations
ROT = False
### setup rotation angles
if ROT:
    ### load shear mask to reduce storage
    src_mask = loadFitsMaps(src_mask_file)
    src_mask = src_mask[0]

    src_mask = np.where(src_mask > 0, 1, 0)
    nside = hp.npix2nside(len(src_mask))
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

    new_src_mask_list = []
    for rot_degrees in rot_degrees_list:
        new_src_mask = np.zeros_like(src_mask)
        new_src_mask_pix = rotate_pix(np.argwhere(src_mask!=0).flatten(), nside=nside, rot_degrees=rot_degrees)
        new_src_mask[new_src_mask_pix] = 1
        new_src_mask_list.append(new_src_mask)


''' =========================================      useful functions       ================================ '''


def gal2void(tracer_pos_cart, cosmo_ccl, survey:int, rank=0):
    void_pos_cart, void_radii = find_void(tracer_pos_cart, 
                                    dive_input='tmp/tmp_tracer_{}.dat'.format(rank), 
                                    dive_output = 'tmp/tmp_void_{}.dat'.format(rank))
    void_ra, void_dec, void_z, phys_cut = Cart2Sph(cosmo_ccl, pos=void_pos_cart)
    void_radii = void_radii[phys_cut]
    
    void_lcone = np.empty(len(void_ra), dtype=fvoid_type)
    void_lcone['ra'] = void_ra
    void_lcone['dec'] = void_dec
    void_lcone['z'] = void_z
    void_lcone['Rv'] = void_radii
    void_lcone['w'] = 1.0
    void_lcone['survey'] = survey

    return void_lcone

def find_voids_fullsky(galcone:np.ndarray, 
                       cosmo_ccl:ccl.Cosmology, 
                       zmin:float, zmax:float,
                       rank:int=None):
    if rank is None:
        rank = 0
    ### Transform galaxy catalog to Cartesian coordinates
    logger.info("Transform to Cart coord")

    gal_pos_cart = Sph2Cart(cosmo_ccl, ra=galcone['ra'], dec=galcone['dec'], z=galcone['z'])
    ### find voids, and transform to Spherical coordinates
    logger.info("Find voids")

    void_lcone = gal2void(gal_pos_cart, cosmo_ccl, survey=0, rank=rank)

    ### apply redshift cut
    zcut = (void_lcone['z'] >= zmin) & (void_lcone['z'] <= zmax)
    void_lcone = void_lcone[zcut]

    return void_lcone

def find_voids_with_boundary_effect(galcone:np.ndarray, 
                                    geoms:dict, masks:dict, 
                                    cosmo_ccl:ccl.Cosmology, 
                                    zmin:float, zmax:float,
                                    rank:int, 
                                    rot_degrees=None):

    ### Transform galaxy catalog to Cartesian coordinates
    logger.info("Transform to Cart coord")

    gal_pos_cart = Sph2Cart(cosmo_ccl, ra=galcone['ra'], dec=galcone['dec'], z=galcone['z'])
    ### find voids, and transform to Spherical coordinates
    logger.info("Find voids")
    tmp_void_list = []
    survey_name = ["LOWZ", "LOWZE2", "LOWZE3", "2dFLens", "CMASS"]

    for survey_lb in [0,1,2,4,3]:
        logger.info(f"Survey {survey_name[survey_lb]}")
        select = galcone['survey'] == survey_lb
        if np.sum(select) > 0:
            tmp_void_lcone = gal2void(gal_pos_cart[select], cosmo_ccl, survey_lb, rank)
            tmp_void_list.append(tmp_void_lcone)

    void_lcone = np.concatenate(tmp_void_list)

    ### apply redshift cut
    zcut = (void_lcone['z'] >= zmin) & (void_lcone['z'] <= zmax)
    void_lcone = void_lcone[zcut]

    if rot_degrees is not None:
        void_lcone_xyz = Sph2Cart(cosmo_ccl, ra=void_lcone['ra'], dec=void_lcone['dec'], z=void_lcone['z'])
        void_lcone_rot_xyz = rotate_lightcone(void_lcone_xyz, rot_degrees, inv=True)
        void_rot_ra, void_rot_dec, void_rot_z, phys_cut = Cart2Sph(cosmo_ccl, pos=void_lcone_rot_xyz)
        void_lcone = void_lcone[phys_cut]
        void_lcone['ra'] = void_rot_ra
        void_lcone['dec'] = void_rot_dec
        void_lcone['z'] = void_rot_z

    ### apply survey geometry cut
    logger.info("Apply survey geometry cut")
    tmp_void_list = []
    ## CMASSLOWZ
    logger.info("LOWZ")
    select = void_lcone['survey'] == 0
    if np.sum(select) > 0:
        boss_lowzcmass_void, _ = apply_boss_geometry(void_lcone[select], geoms['lowzcmass'], masks, galcone_ids=None)
        tmp_void_list.append(boss_lowzcmass_void)
    ## LOWZE2
    logger.info("LOWZE2")
    select = void_lcone['survey'] == 1
    if np.sum(select) > 0:
        boss_lowze2_void, _ = apply_boss_geometry(void_lcone[select], geoms['lowze2'], masks, galcone_ids=None)
        boss_lowze2_void, _ = apply_boss_lowze2e3_trim(boss_lowze2_void, geoms['lowz'], galcone_ids=None)
        tmp_void_list.append(boss_lowze2_void)
    ## LOWZE3
    logger.info("LOWZE3")
    select = void_lcone['survey'] == 2
    if np.sum(select) > 0:
        boss_lowze3_void, _ = apply_boss_geometry(void_lcone[select], geoms['lowze3'], masks, galcone_ids=None)
        boss_lowze3_void, _ = apply_boss_lowze2e3_trim(boss_lowze3_void, geoms['lowz'], galcone_ids=None)
        tmp_void_list.append(boss_lowze3_void)
    ## CMASS
    logger.info("CMASS")
    select = void_lcone['survey'] == 4
    if np.sum(select) > 0:
        boss_lowzcmass_void, _ = apply_boss_geometry(void_lcone[select], geoms['cmass'], masks, galcone_ids=None)
        tmp_void_list.append(boss_lowzcmass_void)
    ## 2dFLens
    logger.info("2dFLens")
    select = void_lcone['survey'] == 3
    if np.sum(select) > 0:
        t2dflens_void, _ = apply_2dflens_geometry(void_lcone[select], geoms['2dflens'], galcone_ids=None)
        tmp_void_list.append(t2dflens_void)

    void_lcone = np.concatenate(tmp_void_list)

    if rot_degrees is not None:
        void_lcone_xyz = Sph2Cart(cosmo_ccl, ra=void_lcone['ra'], dec=void_lcone['dec'], z=void_lcone['z'])
        void_lcone_rot_xyz = rotate_lightcone(void_lcone_xyz, rot_degrees, inv=False)
        void_rot_ra, void_rot_dec, void_rot_z, phys_cut = Cart2Sph(cosmo_ccl, pos=void_lcone_rot_xyz)
        void_lcone = void_lcone[phys_cut]
        void_lcone['ra'] = void_rot_ra
        void_lcone['dec'] = void_rot_dec
        void_lcone['z'] = void_rot_z

    return void_lcone

def select_voids_in_mask(void_lcone:np.ndarray, mask:np.ndarray):
    nside = hp.npix2nside(len(mask))
    void_pix = hp.ang2pix(nside, void_lcone['ra'], void_lcone['dec'], lonlat=True)
    select = np.isin(void_pix, np.where(mask != 0)[0])

    void_lcone = void_lcone[select]

    return void_lcone


''' =========================================      main routine       ================================ '''



if __name__ == "__main__":
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        
        logger.info("Read cosmo labels")

        # cosmo_labels_tot = get_cosmo_name_list_original("/data3/suchen/CosmoGridV1/grid/dirnames.txt")
        cosmo_labels_tot = get_cosmo_name_list_process(hod_param_fname)
        ####  For test  ####
        # cosmo_labels_tot = [1]
        ####################
        k, m = divmod(len(cosmo_labels_tot), size)
        chunks = [cosmo_labels_tot[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(size)]
    else:
        chunks = None

    if rank == 0:

        logger.info("Scattering labels")

    cosmo_labels = comm.scatter(chunks, root=0)

    logger.info("Load mask files")

    geoms = {}
    geoms['lowzcmass'] = pymangle.Mangle(geom_boss_fname_list[0])
    geoms['lowze2'] = pymangle.Mangle(geom_boss_fname_list[1])
    geoms['lowze3'] = pymangle.Mangle(geom_boss_fname_list[2])
    geoms['lowz'] = pymangle.Mangle(geom_boss_fname_list[3])
    geoms['cmass'] = pymangle.Mangle(geom_boss_fname_list[4])
    geoms['2dflens'] = loadFitsMaps(mask_weight_2df_fname)

    masks = []
    for mask_fname in mask_boss_fname_list:
        masks.append(pymangle.Mangle(mask_fname))

    logger.info("Load shear catalog mask")

    for cosmo_label in cosmo_labels:
        if rank == 0:
            logger.info(f"Start processing cosmo_label {cosmo_label}")

        cosmo_ccl = get_cosmo_from_file(sim_fmt.format(cosmo_label) + "params.yml", otype='ccl')

        if ROT:
            for imask, new_src_mask in enumerate(new_src_mask_list):

                logger.info(f"Rotation {imask}, rotation angle {rot_degrees_list[imask]}")

                gfile = "/data2/suchen/CosmoGrid/fix_HOD_rots/cosmo_{:06d}_run_0_HOD_0_run_0_boss_north_2dflens_south_rot{}.npy".format(cosmo_label, imask)
                galcone = np.load(gfile)
                ## boss galaxy
                boss_cut = galcone['survey'] != 3
                galcone = galcone[boss_cut]

                boss_lowzcmasstot_void = find_voids_with_boundary_effect(galcone, geoms, masks, cosmo_ccl, zmin, zmax, rank, rot_degrees=rot_degrees_list[imask])

                boss_lowzcmasstot_void = select_voids_in_mask(boss_lowzcmasstot_void, new_src_mask)

                logger.info("Save void catalog")

                vfile = "/data2/suchen/CosmoGrid/fix_HOD_Void_rots/cosmo_{:06d}_run_0_HOD_0_run_0_boss_north_rot{}.npy".format(cosmo_label, imask)
                np.save(vfile, boss_lowzcmasstot_void)
        elif VARY_HOD:
            for ihod in range(nhod_per_cosmo):
                gfile = gfmt.format(cosmo_label, ihod)
                galcone = np.load(gfile)
                # ### boss galaxy
                # boss_cut = galcone['survey'] != 3
                # galcone = galcone[boss_cut]

                boss_lowzcmasstot_2dflens_void = find_voids_with_boundary_effect(galcone, geoms, masks, cosmo_ccl, zmin, zmax, rank)

                # boss_lowzcmasstot_void = select_voids_in_mask(boss_lowzcmasstot_void, src_mask)

                logger.info("Save void catalog")

                vfile = vfmt.format(cosmo_label, ihod)
                np.save(vfile, boss_lowzcmasstot_2dflens_void)