'''
Script to find voids in lightcone.
'''

import pyccl as ccl
import numpy as np
import pymangle
from loguru import logger

from utils.io_func import *
from utils.mkfore_utils import *

''' cosmology, HOD and galaxy catalog info '''
sim_fmt = "/data3/suchen/CosmoGridV1/grid/cosmo_{:06d}/run_0/"
hod_param_fname = "cfgs/hod/hod_5params_dict.json"
gfmt = "/data2/suchen/CosmoGrid/HOD/cosmo_{:06d}_run_0_HOD_{}_run_0_boss_north_2dflens_south.npy"

''' Output info '''
vfmt = "/data2/suchen/CosmoGrid/Void/cosmo_{:06d}_run_0_HOD_{}_run_0_boss_north.npy"

### FIXME: for test
zmin = 0.2
zmax = 0.4

''' mask file info'''

mask_boss_fdir = "catalogs/masks/boss_geom/"
### geometry files
geom_boss_fname_list = [
    mask_boss_fdir + "mask_DR12v5_CMASSLOWZ_North.ply",
    mask_boss_fdir + "mask_DR12v5_LOWZE2_North.ply",
    mask_boss_fdir + "mask_DR12v5_LOWZE3_North.ply",
    mask_boss_fdir + "mask_DR12v5_LOWZ_North.ply" # For trimming LOWZE2 and LOWZE3 regions
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

mask_weight_2df_fname = "catalogs/masks/2dflens_geom/2dFLens_mask_weight.fits"

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

def find_voids_with_boundary_effect(galcone:np.ndarray, 
                                    geoms:dict, masks:dict, 
                                    cosmo_ccl:ccl.Cosmology, 
                                    zmin:float, zmax:float):

    ### Transform galaxy catalog to Cartesian coordinates
    logger.info("Transform to Cart coord")

    gal_pos_cart = Sph2Cart(cosmo_ccl, ra=galcone['ra'], dec=galcone['dec'], z=galcone['z'])
    ### find voids, and transform to Spherical coordinates
    logger.info("Find voids")

    ### BOSS CMASSLOWZ
    logger.info("LOWZCMASS")
    select = galcone['survey'] == 0
    void_lcone = gal2void(gal_pos_cart[select], cosmo_ccl, 0, rank)
    ### BOSS LOWZE2
    logger.info("LOWZE2")
    select = galcone['survey'] == 1
    void_lcone = np.append(void_lcone, gal2void(gal_pos_cart[select], cosmo_ccl, 1, rank))
    ### BOSS LOWZE3
    logger.info("LOWZE3")
    select = galcone['survey'] == 2
    void_lcone = np.append(void_lcone, gal2void(gal_pos_cart[select], cosmo_ccl, 2, rank))

    ### apply redshift cut
    zcut = (void_lcone['z'] >= zmin) & (void_lcone['z'] <= zmax)
    void_lcone = void_lcone[zcut]

    ### apply survey geometry cut
    logger.info("Apply survey geometry cut")
    ## CMASSLOWZ
    logger.info("LOWZCMASS")
    select = void_lcone['survey'] == 0
    boss_lowzcmass_void, _ = apply_boss_geometry(void_lcone[select], geoms['lowzcmass'], masks, galcone_ids=None)
    ## LOWZE2
    logger.info("LOWZE2")
    select = void_lcone['survey'] == 1
    boss_lowze2_void, _ = apply_boss_geometry(void_lcone[select], geoms['lowze2'], masks, galcone_ids=None)
    boss_lowze2_void, _ = apply_boss_lowze2e3_trim(boss_lowze2_void, geoms['lowz'], galcone_ids=None)
    ## LOWZE3
    logger.info("LOWZE3")
    select = void_lcone['survey'] == 2
    boss_lowze3_void, _ = apply_boss_geometry(void_lcone[select], geoms['lowze3'], masks, galcone_ids=None)
    boss_lowze3_void, _ = apply_boss_lowze2e3_trim(boss_lowze3_void, geoms['lowz'], galcone_ids=None)

    void_lcone = np.append(boss_lowzcmass_void, boss_lowze2_void)
    void_lcone = np.append(void_lcone, boss_lowze3_void)

    return void_lcone

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

        # ############# read legacy cosmology #############
        # with open("/data2/suchen/CosmoGrid/diff_cosmo.txt", "r") as f:
        #     cosmo_labels_tot = [int(line.split('_')[1]) for line in f.readlines()]
        # #################################################

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

    logger.info("Load mask files")

    geoms = {}
    geoms['lowzcmass'] = pymangle.Mangle(geom_boss_fname_list[0])
    geoms['lowze2'] = pymangle.Mangle(geom_boss_fname_list[1])
    geoms['lowze3'] = pymangle.Mangle(geom_boss_fname_list[2])
    geoms['lowz'] = pymangle.Mangle(geom_boss_fname_list[3])

    masks = []
    for mask_fname in mask_boss_fname_list:
        masks.append(pymangle.Mangle(mask_fname))

    logger.info("Main process")

    for icosmo_str in hod_param_dict_local.keys():
        cosmo_label = int(icosmo_str[5:])
        for ihod in range(len(hod_param_dict_local[icosmo_str])):
            
            logger.info("Load Cosmology")
            
            cpar_file = sim_fmt.format(cosmo_label) + "params.yml"
            cosmo_ccl = get_cosmo_from_file(cpar_file, otype='ccl')

            logger.info("Load galaxy catalog")

            gfile = gfmt.format(cosmo_label, ihod)

            vdir = "/data2/suchen/CosmoGrid/Void/"
            ofmt = "cosmo_{:06d}_run_0_HOD_0_run_0_{}.npy"

            logger.info("Load galaxy catalog")

            galcone = np.load(gfile)
            ## boss galaxy
            boss_cut = galcone['survey'] != 3
            galcone = galcone[boss_cut]

            boss_lowzcmasstot_void = find_voids_with_boundary_effect(galcone, geoms, masks, cosmo_ccl, zmin, zmax)

            logger.info("Save void catalog")

            vfile = vfmt.format(cosmo_label, ihod)
            np.save(vfile, boss_lowzcmasstot_void)
