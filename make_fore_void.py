'''
Script to find voids in lightcone.
'''

import pyccl as ccl
import numpy as np
import pymangle
from loguru import logger

from utils.io_func import *
from utils.mkfore_utils import *

''' Setup '''
sim_fmt = "/data3/suchen/CosmoGridV1/grid/cosmo_{:06d}/run_0/"
halo_fmt = "pkd_halos/CosmoML.{:05d}.fofstats.0"
redshift_label = 120 # corresponding to z~0.3

lb_z_file = "/data3/suchen/CosmoGridV1/label_z_table.txt"
lb_z_tb = np.loadtxt(lb_z_file)

Lbox = 900.0
Nside = 832 # Npart = Nside**3
redshift = lb_z_tb[redshift_label,1]
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

def find_voids_with_boundary_effect(galcone:np.ndarray, 
                                    geoms:dict, masks:dict, 
                                    cosmo_ccl:ccl.Cosmology, 
                                    zmin:float, zmax:float):

    ### Transform galaxy catalog to Cartesian coordinates
    logger.info("Transform to Cart coord")

    gal_pos_cart = Sph2Cart(cosmo_ccl, ra=galcone['ra'], dec=galcone['dec'], z=galcone['z'])
    ### find voids, and transform to Spherical coordinates
    logger.info("Find voids")

    void_pos_cart, void_radii = find_void(gal_pos_cart)
    void_ra, void_dec, void_z, phys_cut = Cart2Sph(cosmo_ccl, pos=void_pos_cart)
    void_radii = void_radii[phys_cut]
    
    void_lcone = np.empty(len(void_ra), dtype=fvoid_type)
    void_lcone['ra'] = void_ra
    void_lcone['dec'] = void_dec
    void_lcone['z'] = void_z
    void_lcone['Rv'] = void_radii
    void_lcone['w'] = np.ones(len(void_ra))

    ### apply redshift cut
    zcut = (void_lcone['z'] >= zmin) & (void_lcone['z'] <= zmax)
    void_lcone = void_lcone[zcut]

    ### apply survey geometry cut
    logger.info("Apply survey geometry cut")
    ## CMASSLOWZ
    logger.info("LOWZCMASS")
    boss_lowzcmass_void, _ = apply_boss_geometry(void_lcone, geoms['lowzcmass'], masks, galcone_ids=None)
    ## LOWZE2
    logger.info("LOWZE2")
    boss_lowze2_void, _ = apply_boss_geometry(void_lcone, geoms['lowze2'], masks, galcone_ids=None)
    boss_lowze2_void, _ = apply_boss_lowze2e3_trim(void_lcone, geoms['lowz'], galcone_ids=None)
    ## LOWZE3
    logger.info("LOWZE3")
    boss_lowze3_void, _ = apply_boss_geometry(void_lcone, geoms['lowze3'], masks, galcone_ids=None)
    boss_lowze3_void, _ = apply_boss_lowze2e3_trim(void_lcone, geoms['lowz'], galcone_ids=None)

    return boss_lowzcmass_void, boss_lowze2_void, boss_lowze3_void

if __name__ == "__main__":
    cosmo_label = 1

    cpar_file = sim_fmt.format(cosmo_label) + "params.yml"
    cosmo_ccl = get_cosmo_from_file(cpar_file, otype='ccl')

    gdir = "/data2/suchen/CosmoGrid/HOD/"
    # TODO: change to npy
    ifmt = "cosmo_{:06d}_run_0_HOD_0_run_0_boss_north_2dflens_south.txt"
    gfile = gdir + ifmt.format(cosmo_label)

    vdir = "/data2/suchen/CosmoGrid/Void/"
    ofmt = "cosmo_{:06d}_run_0_HOD_0_run_0_{}.npy"

    logger.info("Load galaxy catalog")
    # TODO: change to npy
    galcone = np.loadtxt(gfile, dtype=fgal_type)
    ## boss galaxy
    boss_cut = galcone['dec'] > -10
    galcone = galcone[boss_cut]

    logger.info("Load mask files")

    geoms = {}
    geoms['lowzcmass'] = pymangle.Mangle(geom_boss_fname_list[0])
    geoms['lowze2'] = pymangle.Mangle(geom_boss_fname_list[1])
    geoms['lowze3'] = pymangle.Mangle(geom_boss_fname_list[2])
    geoms['lowz'] = pymangle.Mangle(geom_boss_fname_list[3])

    masks = []
    for mask_fname in mask_boss_fname_list:
        masks.append(pymangle.Mangle(mask_fname))

    boss_lowzcmass_void, boss_lowze2_void, boss_lowze3_void = find_voids_with_boundary_effect(galcone, geoms, masks, cosmo_ccl, zmin, zmax)

    logger.info("Save void catalog")

    np.save(vdir + ofmt.format(cosmo_label, "lowzcmass"), boss_lowzcmass_void)
    np.save(vdir + ofmt.format(cosmo_label, "lowze2"), boss_lowze2_void)
    np.save(vdir + ofmt.format(cosmo_label, "lowze3"), boss_lowze3_void)