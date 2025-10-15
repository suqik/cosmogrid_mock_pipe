'''
Script to generate boss-like galaxies/voids
'''

import sys
sys.path.append("/home/suchen/Program/CosmoGrid/")

import numpy as np
import pymangle
import sys
from loguru import logger

from utils.io_func import *
from utils.mkfore_utils import *

''' simulation info '''
sim_fmt = "/data3/suchen/CosmoGridV1/grid/cosmo_{:06d}/run_0/"
halo_fmt = "pkd_halos/CosmoML.{:05d}.fofstats.0"
redshift_label = 120 # corresponding to z~0.3

lb_z_file = "/data3/suchen/CosmoGridV1/label_z_table.txt"
lb_z_tb = np.loadtxt(lb_z_file)

Lbox = 900.0
Nside = 832 # Npart = Nside**3
redshift = lb_z_tb[redshift_label,1]
# redshift_before = lb_z_tb[redshift_label-1,1]
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

''' n(z) file info '''

### n(z) files
nz_fbase = "catalogs/NOfZ/lens/"
nz_boss_fname_list = [
    nz_fbase + "nbar_DR12v5_CMASSLOWZ_North_om0p31_Pfkp10000.dat",
    nz_fbase + "nbar_DR12v5_LOWZE2_North_om0p31_Pfkp10000.dat",
    nz_fbase + "nbar_DR12v5_LOWZE3_North_om0p31_Pfkp10000.dat"
]

nz_2dflens_fname = nz_fbase + "nbar_2dFLens_south_data.dat"

def load_halocat(cosmo_label, ofmt='hod'):
    cpar_file = sim_fmt.format(cosmo_label) + "params.yml"

    logger.info(f"Load cosmology from file {cpar_file}")
    
    cosmo_ccl = get_cosmo_from_file(cpar_file, otype='ccl')
    OmegaM = cosmo_ccl.to_dict()["Omega_c"] + cosmo_ccl.to_dict()["Omega_b"]
    pmass = rhoc0*OmegaM*(Lbox/Nside)**3 # Msun/h

    ### load pkd halo catalog
    halo_file = sim_fmt.format(cosmo_label) + halo_fmt.format(redshift_label)
    
    logger.info(f"Load PKD halo from file {halo_file}")

    if ofmt == 'pos':
        pkd_halo_infos = get_pkd_halo_attrs(halo_file, attrs=["pos"], Lbox=Lbox, redshift=redshift)

        return pkd_halo_infos['pos']

    if ofmt == 'hod':
        pkd_halo_infos = get_pkd_halo_attrs(halo_file, attrs=["pos", "vel", "mass"], Lbox=Lbox, redshift=redshift)
        hod_halocat = pkd_to_hod_type(pkd_halo_infos, cosmo_ccl, pmass)

        return hod_halocat

def snap2lcone(gal_pos:np.ndarray, cosmo_ccl:ccl.Cosmology, zmin:float, zmax:float) -> tuple[np.ndarray, np.ndarray]:
    assert isinstance((gal_pos), np.ndarray)
    assert isinstance((cosmo_ccl), ccl.Cosmology)

    logger.info("Calculate cosmology-dependent quantities.")
    
    hubble = cosmo_ccl.to_dict()["h"]
    chi_min = ccl.comoving_radial_distance(cosmo_ccl, 1./(1 + zmin))*hubble # Mpc/h
    chi_max = ccl.comoving_radial_distance(cosmo_ccl, 1./(1 + zmax))*hubble # Mpc/h

    logger.info("Transforming")

    ### transform box data to lightcone 
    ### there are 4 cols in galcone: first 3 are (x,y,z), and the last one is the id of the galaxy
    galcone = make_lightcone_tiles(gal_pos, boxsize=Lbox, chi_min=chi_min, chi_max=chi_max)
    
    galcone_ra, galcone_dec = hp.vec2ang(galcone[:,:-1], lonlat=True)
    galcone_chi = np.linalg.norm(galcone[:,:-1], axis=1)
    
    del galcone

    galcone_output = np.empty((len(galcone_ra),), dtype=fgal_type)
    galcone_output["ra"] = galcone_ra
    galcone_output["dec"] = galcone_dec
    galcone_output["z"] = 1./ccl.scale_factor_of_chi(cosmo_ccl, galcone_chi/hubble) - 1

    galcone_id = np.arange(len(galcone_output))

    return galcone_output, galcone_id

def load_boss_geom_files(geom_boss_fname_list, mask_boss_fname_list):
    masks = {}
    masks['boss_gemo'] = []
    masks['boss_masks'] = []
    logger.info("Load mask files.")
    ### load boss survey geometry
    for geom_file in geom_boss_fname_list:
        masks['boss_gemo'].append(pymangle.Mangle(geom_file))
    for mask_file in mask_boss_fname_list:
        masks['boss_masks'].append(pymangle.Mangle(mask_file))

    return masks

def load_boss_nz_files(nz_boss_fname_list):
    logger.info("Load n(z) files.")
    ### load nofz information
    nofz_info = {}
    ### boss lowzcmass
    boss_part_names = ['boss_lowzcmass', 'boss_lowze2', 'boss_lowze3']
    for ipart, nz_boss_fname in enumerate(nz_boss_fname_list):
        nofz = np.loadtxt(nz_boss_fname, usecols=(1,2,3,5)) # zmin, zmax, nz, shell_vol
        argstart = np.argwhere(nofz[:,0] == zmin)[0,0]
        argend = np.argwhere(nofz[:,1] == zmax)[0,0]

        nofz_info = make_nofz_info(nofz_info, boss_part_names[ipart], nofz[argstart:argend+2,0], nofz[argstart:argend+1,3], nofz[argstart:argend+1,2])

    return nofz_info

def apply_boss_lowz_cut(galcone_output, geom_boss, masks_boss, galcone_id=None):
    galcone_boss, galcone_id_boss = apply_boss_geometry(galcone_output, geom_boss, masks_boss, galcone_ids=galcone_id)
    return galcone_boss, galcone_id_boss

def apply_boss_lowze2e3_cut(galcone_output, geom_boss, masks_boss, masks_boss_lowz, galcone_id=None):
    galcone_boss, galcone_id_boss = apply_boss_geometry(galcone_output, geom_boss, masks_boss, galcone_ids=galcone_id)
    galcone_boss, galcone_id_boss = apply_boss_lowze2e3_trim(galcone_boss, masks_boss_lowz, galcone_ids=galcone_id_boss)
    return galcone_boss, galcone_id_boss

def apply_boss_lowz_nz_dsample(galcone_boss, nofz_boss, galcone_id_boss=None):
    galcone_boss, galcone_id_boss = apply_nz_downsample(galcone_boss, nofz_boss, galcone_ids=galcone_id_boss)
    return galcone_boss, galcone_id_boss

# def find_void(halo_pos, boxsize=None):
#     exec_path = "/home/suchen/applications/DIVE/DIVE"
#     dive_input = "aux/tmp_halo.dat"
#     np.savetxt(dive_input, halo_pos, fmt='%.3f')
#     dive_output = "aux/tmp_void.dat"
#     cmd = exec_path + " -i " + dive_input + " -o " + dive_output
#     if boxsize is not None:
#         cmd += " -u " + str(boxsize)
#     print(cmd)
#     os.system(cmd)
#     print(f"rm {dive_input}")
#     os.system(f"rm {dive_input}")

#     void_info = np.loadtxt(dive_output)
#     void_pos = void_info[:,:-1]
#     void_radius = void_info[:,-1]

#     print(f"rm {dive_output}")
#     os.system(f"rm {dive_output}")

#     return void_pos, void_radius

if __name__ == "__main__":
    ''' >>>==========   pre-defined routines   ============<<< '''

    run_name_list = ['gen_boss_gal', 'find_void_wob', 'find_void_wb', 'find_boss_void']
    if len(sys.argv) > 1:
        run_name = sys.argv[1]
    else:
        print("Usage: python apply_survey_geom.py <run_name>")
        print(f"Current support run_name: {run_name_list}")
        exit()

    if run_name not in run_name_list:
        print(f"Cannot find run_name: {run_name}")
        print(f"Current support run_name: {run_name_list}")
        exit()

    ################################   generate boss-like galaxies   ##############################
    if run_name == "gen_boss_gal":
        cosmo_label = 1
        ### initialize cosmology
        logger.info("Initialize cosmology.")
        cosmo_ccl = get_cosmo_from_file(sim_fmt.format(cosmo_label) + "params.yml", otype="ccl")
        ### load pkdgrav halo file
        logger.info("Load pkdgrav halo file.")
        halo_pos = load_halocat(cosmo_label, ofmt='pos')
        ### transform to lightcone format
        logger.info("Transform to lightcone format.")
        halo_lcone, halo_id = snap2lcone(halo_pos, cosmo_ccl, zmin, zmax)
        
        ### apply boss geometry
        ### apply boss_lowz, boss_lowe2, and boss_lowe3 separately, and save.
        ### since void finding algorithm is sensitive to galaxy number density
        logger.info("Apply boss geometry.")
        boss_geoms = load_boss_geom_files(geom_boss_fname_list, mask_boss_fname_list)
        boss_lowzcmass_halo, _ = apply_boss_lowz_cut(halo_lcone, boss_geoms['boss_gemo'][0], boss_geoms['boss_masks'], galcone_id=None)
        boss_lowze2_halo, _ = apply_boss_lowze2e3_cut(halo_lcone, boss_geoms['boss_gemo'][1], boss_geoms['boss_masks'], boss_geoms['boss_gemo'][-1], galcone_id=None)
        boss_lowze3_halo, _ = apply_boss_lowze2e3_cut(halo_lcone, boss_geoms['boss_gemo'][2], boss_geoms['boss_masks'], boss_geoms['boss_gemo'][-1], galcone_id=None)
        
        # ### apply n(z) downsample
        # logger.info("Apply n(z) downsample.")
        # boss_nofzs = load_boss_nz_files(nz_boss_fname_list)
        # boss_lowzcmass_halo, _ = apply_boss_lowz_nz_dsample(boss_lowzcmass_halo, boss_nofzs['boss_lowzcmass'], galcone_id_boss=None)
        # boss_lowze2_halo, _ = apply_boss_lowz_nz_dsample(boss_lowze2_halo, boss_nofzs['boss_lowze2'], galcone_id_boss=None)
        # boss_lowze3_halo, _ = apply_boss_lowz_nz_dsample(boss_lowze3_halo, boss_nofzs['boss_lowze3'], galcone_id_boss=None)

        ### save
        logger.info("Save to file.")
        np.save("aux/catalogs/boss_lowzcmass.npy", boss_lowzcmass_halo)
        np.save("aux/catalogs/boss_lowze2.npy", boss_lowze2_halo)
        np.save("aux/catalogs/boss_lowze3.npy", boss_lowze3_halo)
    ################################################################################################

    #############################   find void without boundary effect   ############################
    if run_name == "find_void_wob":
        cosmo_label = 1
        ### initialize cosmology
        logger.info("Initialize cosmology.")
        cosmo_ccl = get_cosmo_from_file(sim_fmt.format(cosmo_label) + "params.yml", otype="ccl")
        ### load pkdgrav halo file
        logger.info("Load pkdgrav halo file.")
        halo_pos = load_halocat(cosmo_label, ofmt='pos')
        ### find void in box data
        logger.info("Find void in box data.")
        void_pos, void_radius = find_void(halo_pos, boxsize=None)
        void_rcut = (void_radius > 15) & (void_radius < 25)
        void_pos = void_pos[void_rcut]
        void_radius = void_radius[void_rcut]
        ### transform to lightcone format
        logger.info("Transform to lightcone format.")
        void_lcone, void_id = snap2lcone(void_pos, cosmo_ccl, zmin, zmax)
        
        ### apply boss geometry
        ### apply boss_lowz, boss_lowe2, and boss_lowe3 separately, and save.
        ### since void finding algorithm is sensitive to galaxy number density
        logger.info("Apply boss geometry.")
        boss_geoms = load_boss_geom_files(geom_boss_fname_list, mask_boss_fname_list)
        # boss_nofzs = load_boss_nz_files(nz_boss_fname_list)
        boss_lowzcmass_void, boss_lowzcmass_void_id = apply_boss_lowz_cut(void_lcone, boss_geoms['boss_gemo'][0], boss_geoms['boss_masks'], galcone_id=void_id)
        boss_lowze2_void, boss_lowze2_void_id = apply_boss_lowze2e3_cut(void_lcone, boss_geoms['boss_gemo'][1], boss_geoms['boss_masks'], boss_geoms['boss_gemo'][-1], galcone_id=void_id)
        boss_lowze3_void, boss_lowze3_void_id = apply_boss_lowze2e3_cut(void_lcone, boss_geoms['boss_gemo'][2], boss_geoms['boss_masks'], boss_geoms['boss_gemo'][-1], galcone_id=void_id)
    
        # ### apply n(z) downsample
        # logger.info("Apply n(z) downsample.")
        # boss_lowzcmass_void, boss_lowzcmass_void_id = apply_boss_lowz_nz_dsample(boss_lowzcmass_void, boss_nofzs['boss_lowzcmass'], galcone_id_boss=boss_lowzcmass_void_id)
        # boss_lowze2_void, boss_lowze2_void_id = apply_boss_lowz_nz_dsample(boss_lowze2_void, boss_nofzs['boss_lowze2'], galcone_id_boss=boss_lowze2_void_id)
        # boss_lowze3_void, boss_lowze3_void_id = apply_boss_lowz_nz_dsample(boss_lowze3_void, boss_nofzs['boss_lowze3'], galcone_id_boss=boss_lowze3_void_id)

        ### save
        logger.info("Save to file.")
        np.save("aux/catalogs/boss_lowzcmass_void_wob.npy", boss_lowzcmass_void)
        np.save("aux/catalogs/boss_lowze2_void_wob.npy", boss_lowze2_void)
        np.save("aux/catalogs/boss_lowze3_void_wob.npy", boss_lowze3_void)
    #################################################################################################

    ###############################   find void with boundary effect   ##############################
    ### TODO: replace sph2cart/cart2sph to functions in `mkfore_utils.py`
    if run_name == "find_void_wb":

        def sph2cart(sph_pos, cosmo_ccl):
            chi_radial = ccl.comoving_radial_distance(cosmo_ccl, 1./(1+sph_pos['z'])) # Mpc
            chi_radial *= cosmo_ccl.to_dict()["h"] # Mpc/h
            pos = hp.ang2vec(sph_pos['ra'], sph_pos['dec'], lonlat=True) # Actually norm of position
            pos = (pos.T * chi_radial).T

            return pos

        def cart2sph(cart_pos, cosmo_ccl):
            chi_radial = np.linalg.norm(cart_pos, axis=1) # Mpc/h
            chi_radial /= cosmo_ccl.to_dict()["h"] # Mpc
            redshifts = 1./ccl.scale_factor_of_chi(cosmo_ccl, chi_radial) - 1.
            ra, dec = hp.vec2ang(cart_pos, lonlat=True)
            pos = np.empty(len(ra), dtype=fgal_type)
            pos['ra'] = ra
            pos['dec'] = dec
            pos['z'] = redshifts
            pos['w'] = np.ones(len(ra))

            return pos
        
        def find_void_from_lcone(halo_lcone, cosmo_ccl):
            zmin = halo_lcone['z'].min()
            zmax = halo_lcone['z'].max()
            halo_lcone_cart = sph2cart(halo_lcone, cosmo_ccl)
            void_pos, void_radius = find_void(halo_lcone_cart, boxsize=None)
            void_rcut = (void_radius > 15) & (void_radius < 25)
            void_pos = void_pos[void_rcut]
            void_radius = void_radius[void_rcut]

            void_lcone = cart2sph(void_pos, cosmo_ccl)
            void_zcut = (void_lcone['z'] > zmin) & (void_lcone['z'] < zmax)
            void_lcone = void_lcone[void_zcut]
            void_radius = void_radius[void_zcut]

            return void_lcone, void_radius
        
        cosmo_label = 1
        ### initialize cosmology
        logger.info("Initialize cosmology.")
        cosmo_ccl = get_cosmo_from_file(sim_fmt.format(cosmo_label) + "params.yml", otype="ccl")

        ### apply boss geometry
        ### apply boss_lowz, boss_lowe2, and boss_lowe3 separately, and save.
        ### since void finding algorithm is sensitive to galaxy number density
        logger.info("Apply boss geometry.")
        boss_geoms = load_boss_geom_files(geom_boss_fname_list, mask_boss_fname_list)
        
        import os

        if os.path.exists("aux/catalogs/boss_lowzcmass.npy") and os.path.exists("aux/catalog/boss_lowze2.npy") and os.path.exists("aux/catalog/boss_lowze3.npy"):
            boss_lowzcmass_halo = np.load("aux/catalogs/boss_lowzcmass.npy")
            boss_lowze2_halo = np.load("aux/catalogs/boss_lowze2.npy")
            boss_lowze3_halo = np.load("aux/catalogs/boss_lowze3.npy")
        else:
            ### load pkdgrav halo file
            logger.info("Load pkdgrav halo file.")
            halo_pos = load_halocat(cosmo_label, ofmt='pos')
            ### transform to lightcone format
            logger.info("Transform to lightcone format.")
            halo_lcone, halo_id = snap2lcone(halo_pos, cosmo_ccl, zmin, zmax)

            # boss_nofzs = load_boss_nz_files(nz_boss_fname_list)
            boss_lowzcmass_halo, _ = apply_boss_lowz_cut(halo_lcone, boss_geoms['boss_gemo'][0], boss_geoms['boss_masks'], galcone_id=None)
            boss_lowze2_halo, _ = apply_boss_lowze2e3_cut(halo_lcone, boss_geoms['boss_gemo'][1], boss_geoms['boss_masks'], boss_geoms['boss_gemo'][-1], galcone_id=None)
            boss_lowze3_halo, _ = apply_boss_lowze2e3_cut(halo_lcone, boss_geoms['boss_gemo'][2], boss_geoms['boss_masks'], boss_geoms['boss_gemo'][-1], galcone_id=None)
            
            # ### apply n(z) downsample
            # logger.info("Apply n(z) downsample.")
            # boss_lowzcmass_halo, _ = apply_boss_lowz_nz_dsample(boss_lowzcmass_halo, boss_nofzs['boss_lowzcmass'], galcone_id_boss=None)
            # boss_lowze2_halo, _ = apply_boss_lowz_nz_dsample(boss_lowze2_halo, boss_nofzs['boss_lowze2'], galcone_id_boss=None)
            # boss_lowze3_halo, _ = apply_boss_lowz_nz_dsample(boss_lowze3_halo, boss_nofzs['boss_lowze3'], galcone_id_boss=None)

            ### save
            logger.info("Save to file.")
            np.save("aux/catalogs/boss_lowzcmass.npy", boss_lowzcmass_halo)
            np.save("aux/catalogs/boss_lowze2.npy", boss_lowze2_halo)
            np.save("aux/catalogs/boss_lowze3.npy", boss_lowze3_halo)

        ### find void in lightcone data
        logger.info("Find void in lightcone data.")
        boss_lowzcmass_void, boss_lowzcmass_void_radius = find_void_from_lcone(boss_lowzcmass_halo, cosmo_ccl)
        boss_lowze2_void, boss_lowze2_void_radius = find_void_from_lcone(boss_lowze2_halo, cosmo_ccl)
        boss_lowze3_void, boss_lowze3_void_radius = find_void_from_lcone(boss_lowze3_halo, cosmo_ccl)
        
        boss_lowzcmass_void, _ = apply_boss_lowz_cut(boss_lowzcmass_void, boss_geoms['boss_gemo'][0], boss_geoms['boss_masks'], galcone_id=None)
        boss_lowze2_void, _ = apply_boss_lowze2e3_cut(boss_lowze2_void, boss_geoms['boss_gemo'][1], boss_geoms['boss_masks'], boss_geoms['boss_gemo'][-1], galcone_id=None)
        boss_lowze3_void, _ = apply_boss_lowze2e3_cut(boss_lowze3_void, boss_geoms['boss_gemo'][2], boss_geoms['boss_masks'], boss_geoms['boss_gemo'][-1], galcone_id=None)

        ### save
        logger.info("Save to file.")
        np.save("aux/catalogs/boss_lowzcmass_void_wb.npy", boss_lowzcmass_void)
        np.save("aux/catalogs/boss_lowze2_void_wb.npy", boss_lowze2_void)
        np.save("aux/catalogs/boss_lowze3_void_wb.npy", boss_lowze3_void)
    #################################################################################################

    ###############################  find void in BOSS LOWZ-E3 data   ###############################
    if run_name == "find_boss_void":
        
        from astropy.table import Table
        def load_boss_data(fname, zmin, zmax):
            boss_data_tb = Table.read(fname)
            boss_data = np.empty(len(boss_data_tb), dtype=fgal_type)
            boss_data['ra'] = boss_data_tb['RA']
            boss_data['dec'] = boss_data_tb['DEC']
            boss_data['z'] = boss_data_tb['Z']
            boss_data['w'] = boss_data_tb['WEIGHT_SYSTOT']

            zcut = (boss_data['z'] > zmin) & (boss_data['z'] < zmax)
            boss_data = boss_data[zcut]

            return boss_data

        def find_void_from_lcone(halo_lcone, cosmo_ccl, survey:int=0):
            zmin = halo_lcone['z'].min()
            zmax = halo_lcone['z'].max()
            halo_lcone_cart = Sph2Cart(cosmo_ccl, ra=halo_lcone['ra'], dec=halo_lcone['dec'], z=halo_lcone['z'])
            void_pos, void_radius = find_void(halo_lcone_cart, boxsize=None)
            # void_rcut = (void_radius > 15) & (void_radius < 25)
            # void_pos = void_pos[void_rcut]
            # void_radius = void_radius[void_rcut]

            v_ra, v_dec, v_z, phys_cut = Cart2Sph(cosmo_ccl, pos=void_pos)
            void_lcone = np.empty(len(v_ra), dtype=fvoid_type)
            void_lcone['ra'] = v_ra
            void_lcone['dec'] = v_dec
            void_lcone['z'] = v_z
            void_lcone['Rv'] = void_radius[phys_cut]
            void_lcone['survey'] = survey

            void_zcut = (void_lcone['z'] > zmin) & (void_lcone['z'] < zmax)
            void_lcone = void_lcone[void_zcut]

            return void_lcone
        
        ### initialize cosmology
        logger.info("Assume Planck 2015 cosmology.")
        cosmo_ccl = ccl.Cosmology(Omega_c=0.26, Omega_b=0.049, h=0.6774, sigma8=0.816, n_s=0.9667)

        ### load boss data
        zmin = 0.2
        zmax = 0.4
        boss_lowzcmass_halo = load_boss_data("/data2/suchen/BOSS_dr12/galaxy_DR12v5_CMASSLOWZ_North.fits", zmin, zmax)
        boss_lowze2_halo = load_boss_data("/data2/suchen/BOSS_dr12/SDSS_DR12_orig/galaxy_DR12v5_LOWZE2_North_trimmed.fits", zmin, zmax)
        boss_lowze3_halo = load_boss_data("/data2/suchen/BOSS_dr12/SDSS_DR12_orig/galaxy_DR12v5_LOWZE3_North_trimmed.fits", zmin, zmax)

        ### apply boss geometry
        ### apply boss_lowz, boss_lowe2, and boss_lowe3 separately, and save.
        ### since void finding algorithm is sensitive to galaxy number density
        logger.info("Apply boss geometry.")
        boss_geoms = load_boss_geom_files(geom_boss_fname_list, mask_boss_fname_list)

        ### find void in lightcone data
        logger.info("Find void in lightcone data.")
        boss_lowzcmass_void = find_void_from_lcone(boss_lowzcmass_halo, cosmo_ccl, survey=0)
        boss_lowze2_void = find_void_from_lcone(boss_lowze2_halo, cosmo_ccl, survey=1)
        boss_lowze3_void = find_void_from_lcone(boss_lowze3_halo, cosmo_ccl, survey=2)
        
        boss_lowzcmass_void, _ = apply_boss_lowz_cut(boss_lowzcmass_void, boss_geoms['boss_gemo'][0], boss_geoms['boss_masks'], galcone_id=None)
        boss_lowze2_void, _ = apply_boss_lowze2e3_cut(boss_lowze2_void, boss_geoms['boss_gemo'][1], boss_geoms['boss_masks'], boss_geoms['boss_gemo'][-1], galcone_id=None)
        boss_lowze3_void, _ = apply_boss_lowze2e3_cut(boss_lowze3_void, boss_geoms['boss_gemo'][2], boss_geoms['boss_masks'], boss_geoms['boss_gemo'][-1], galcone_id=None)

        boss_lowzcmasstot_void = np.append(boss_lowzcmass_void, np.append(boss_lowze2_void, boss_lowze3_void))

        del boss_lowzcmass_void, boss_lowze2_void, boss_lowze3_void

        ### save
        logger.info("Save to file.")
        np.save("aux/catalogs/Data/bossdata_lowzcmasstot_void.npy", boss_lowzcmasstot_void)
    #################################################################################################