''' 
From halo to galaxy. 
apply HOD and make lightcone.
'''

import sys
import json
import numpy as np
from scipy.stats import qmc, truncnorm
import pymangle
import datetime
from loguru import logger

from utils.io_func import *
from utils.mkfore_utils import *

wdir = "/home/suchen/Program/CosmoGrid"

''' 1. Simulation info '''

sim_fmt = "/data3/suchen/CosmoGridV1/grid/cosmo_{:06d}/run_{:d}/"
halo_fmt = "pkd_halos/CosmoML.{:05d}.fofstats.0"
redshift_label = 120 # corresponding to z~0.3
# redshift_label = 110 # corresponding to z~0.51

rlz_label = 0 # realization label. Note only partial cosmologies have multiple rlzs. (~400)

lb_z_file = "/data3/suchen/CosmoGridV1/label_z_table.txt"
lb_z_tb = np.loadtxt(lb_z_file)

Lbox = 900.0
Nside = 832 # Npart = Nside**3
redshift = lb_z_tb[redshift_label,1]

### Use different redshift range depending on redshift_label
if redshift_label == 120:
    zmin = 0.2
    zmax = 0.4
    zbin_name = "lowz"
    boss_part_names = ['boss_lowzcmass', 'boss_lowze2', 'boss_lowze3']
    # boss_part_names = []

elif redshift_label == 110:
    zmin = 0.4
    zmax = 0.6
    zbin_name = "cmass"
    boss_part_names = ['boss_cmass']
    # boss_part_names = []

MK_2DFLENS = True # if make 2dflens mock

if len(boss_part_names) > 0:
    survey_specify = "boss_north"
    if MK_2DFLENS:
        survey_specify += "_2dflens_south"
else:
    if MK_2DFLENS:
        survey_specify = "2dflens_south"
    else:
        survey_specify = ""

print(survey_specify)

''' 2. Mask file info'''

mask_boss_fdir = f"{wdir}/catalogs/masks/boss_geom/"

geom_boss_fname_list = {
    'boss_lowzcmass': mask_boss_fdir + "mask_DR12v5_CMASSLOWZ_North.ply",
    'boss_lowze2': mask_boss_fdir + "mask_DR12v5_LOWZE2_North.ply",
    'boss_lowze3': mask_boss_fdir + "mask_DR12v5_LOWZE3_North.ply",
    'boss_lowz': mask_boss_fdir + "mask_DR12v5_LOWZ_North.ply",
    'boss_cmass': mask_boss_fdir + "mask_DR12v5_CMASS_North.ply",
}

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

''' 3. n(z) file info '''

### n(z) files
nz_fbase = f"{wdir}/catalogs/NOfZ/lens/"
nz_boss_fname_list = {
    'boss_lowzcmass': nz_fbase + "nbar_DR12v5_LOWZ_North_om0p31_Pfkp10000.dat",
    'boss_lowze2': nz_fbase + "nbar_DR12v5_LOWZE2_North_om0p31_Pfkp10000.dat",
    'boss_lowze3': nz_fbase + "nbar_DR12v5_LOWZE3_North_om0p31_Pfkp10000.dat",
    'boss_cmass': nz_fbase + "nbar_DR12v5_CMASS_North_om0p31_Pfkp10000.dat",
}

nz_2dflens_fname = nz_fbase + "nbar_2dFLens_south_data.dat"

''' 4. Output files '''

out_dir = f"/data2/suchen/CosmoGrid/high_ngal_suits_wrsd/HOD_{zbin_name}/"
out_fmt = "cosmo_{:06d}_run_{:d}_HOD_{:d}_run_{:d}_{:s}.npy"
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

hod_param_out = f"{wdir}/cfgs/hod/hod_5params_dict_high_ngal_wcosmo2.json"

''' 5. HOD setup '''

model = 2
num_params = 6 # Number of parameters of HOD model
nhod_per_cosmo = 10 # Number of varied HOD parameter values per cosmology
model_params_names = 'logMcut', 'sigma_logM', 'logM1', 'k', 'alpha', 'fic' # for model == 2
# model_params_names = 'logMcut', 'sigma_logM', 'logM1', 'logM0', 'alpha', 'fic' # SIMBIG HOD params, for model==3
Num_ptcl_requirement = 12
verbose = True
num_seeds = 1
init_seed = 33000
z_space = False
ngal_ref = 4e-4

''' 6. Running modes specifications '''

### 6.1 If apply RSD effect
ADD_RSD = True

### 6.2 If using rotations to augment data
ROT = False # if True, will rotate the galaxy catalogs
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

### 6.3 Running modes, can only activate one of these three
HALO_ONLY = False # only use halo, which preserve the ngal but not G-H connection
FIX_HOD = False # use the same G-H connection but cannot preserve the ngal
VARY_HOD = True # preserve the ngal, as well as vary G-H connection

if VARY_HOD:
    LOAD_HOD_PAR = True # if load exist hod params
    #### prior for model == 2
    param_prior_low  = np.array([12.5, 1e-5, 12.5, 0.00, 0.0])
    param_prior_high = np.array([13.5, 3.00, 15.0, 10.0, 2.0])

    #### prior from SIMBIG, for model == 3
    # param_prior_low = np.array([12., 0.1, 13., 13., 0.0])
    # param_prior_high = np.array([14., 0.6, 15., 15., 1.5])

if FIX_HOD:
    raise NotImplementedError("FIX HOD run has been deprecated. Please wait for future updates.")
    # # fid_hod_model_param = np.array([12.59102404,  2.10923402, 14.06049531,  0.07197861,  0.25447211, 1.0])
    # fid_hod_model_param = np.array([12.72, 0.67, 12.86, 0.32, 0.21, 1.0])
    # # fid_hod_model_param = np.array([13.2, 0.62, 14.32, 13.24, 0.93, 1.0])

    # fixed_model_params_dict = {}
    # for i in range(num_params):
    #     fixed_model_params_dict[model_params_names[i]] = fid_hod_model_param[i]

''' 7. Show config info '''

logger.info(f"used simulation redshift: {redshift:.4f}")
logger.info(f"Simulating redshift range: {zmin:.4f} - {zmax:.4f}")
logger.info(f"RSD: {ADD_RSD}")

if HALO_ONLY:
    logger.info("HALO only mode")
    logger.info(f"Ngal ref: {ngal_ref*1e4:.2f} e-4")
# elif FIX_HOD:
#     logger.info("FIX_HOD mode")
#     logger.info(f"FIX_HOD: {fixed_model_params_dict}")
elif VARY_HOD:
    logger.info("VARY_HOD mode")
    logger.info(f"Ngal ref: {ngal_ref*1e4:.2f} e-4")
    if LOAD_HOD_PAR:
        logger.info("Load HOD pars from:")
        logger.info(f"{hod_param_out}")
    else:
        logger.info(f"HOD prior low: {param_prior_low}")
        logger.info(f"HOD prior high: {param_prior_high}")
        logger.info( "Will save cosmo and HOD pars to:")
        logger.info(f"{hod_param_out}")

if ROT:
    logger.info("Use rotation mode")
    logger.info(f"Rotation degrees: {rot_degrees_list}")


''' ================================================================================================================== '''


''' main routines '''
def load_halocat(cpar_fname, halo_fname, ofmt='hod', clean=True):
    # cpar_file = sim_fmt.format(cosmo_label, rlz_label) + "params.yml"

    logger.info(f"Load cosmology from file {cpar_fname}")
    
    cosmo_ccl = get_cosmo_from_file(cpar_fname, otype='ccl')
    OmegaM = cosmo_ccl.to_dict()["Omega_c"] + cosmo_ccl.to_dict()["Omega_b"]
    pmass = rhoc0*OmegaM*(Lbox/Nside)**3 # Msun/h
    
    logger.info(f"Load PKD halo from file {halo_fname}")
    
    pkd_halo_infos = get_pkd_halo_attrs(halo_fname, attrs=["pos","vel","mass","rHalf"], Lbox=Lbox, redshift=redshift)

    if clean:
        phys_cut = (
            pkd_halo_infos['rHalf'] > 0.1
        )

        for iattr in pkd_halo_infos.keys():
            pkd_halo_infos[iattr] = pkd_halo_infos[iattr][phys_cut]

    if ofmt == 'pkd':
        return pkd_halo_infos

    if ofmt == 'hod':
        ## Initialize HOD model class
        hod_halo_cat = pkd_to_hod_type(pkd_halo_infos, cosmo=cosmo_ccl, pmass=pmass, boxsize=Lbox, redshift=redshift)

        return hod_halo_cat, OmegaM

def find_fic(halo_mass, hod_param_vals, model_lb=2):
    if isinstance(hod_param_vals, list) or isinstance(hod_param_vals, np.ndarray):
        model_params_dict = dict(zip(model_params_names, hod_param_vals))
    elif isinstance(hod_param_vals, dict):
        model_params_dict = hod_param_vals.copy()
    else:
        raise ValueError("hod_param_vals should be list or dict")

    Mmin = halo_mass.min()
    Mmax = halo_mass.max()

    nMbin = 30
    dlgM = np.log10(Mmax/Mmin)/(nMbin-1)
    Mbin_edges = np.logspace(np.log10(Mmin)-dlgM, np.log10(Mmax)+dlgM, nMbin)

    massbin, NM = get_HMF(halo_mass, Mbin_edges, boxsize=Lbox, isedge=True, bin_scale='log', ifcum=False)

    tmp_dict = model_params_dict.copy()
    tmp_dict['fic'] = 1.0
    ctr = MWCens_IC(redshift=redshift)
    ctr.param_dict = tmp_dict
    Nctr = ctr.mean_occupation(prim_haloprop=massbin)

    if model_lb == 2:
        sat = MWSats(redshift=redshift, cenocc_model=ctr, modulate_with_cenocc=True)
    elif model_lb == 3:
        sat = MWSats2(redshift=redshift, cenocc_model=ctr, modulate_with_cenocc=True)

    sat.param_dict = tmp_dict
    Nsat = sat.mean_occupation(prim_haloprop=massbin)

    f_ic = ngal_ref/(np.sum(Nctr*NM) + np.sum(Nsat*NM))
    
    return f_ic, Nsat

def apply_hod(halo_file, hod_halo_cat, OmegaM, model_params_dict, indx):
    hod_model = ModelClass(
        [halo_file], [hod_halo_cat], 
        model=model, 
        num_params=num_params,
        param_names=model_params_names,
        redshift=redshift, 
        box_size=Lbox,
        Omega_m=OmegaM, 
        init_seed=init_seed,
        num_seeds=num_seeds,
        z_space=z_space,
        Num_ptcl_requirement=Num_ptcl_requirement,
        verbose=verbose
        )

    dict_of_gsamples = hod_model.populate_mock(model_params_dict, ref_num_dens=ngal_ref, indx=indx, ifcheck=False)

    return dict_of_gsamples

def find_hod_params_alive(halo_mass, num_pool=30000, seedini=9782, seed_offset=0):
    ## Sample HOD parameters
    count = 0
    idx = 0
    seed = seedini + seed_offset
    lhc_sampler = qmc.LatinHypercube(d=len(param_prior_low), seed=seed)
    hod_params_pool = lhc_sampler.random(n=num_pool)
    hod_params_pool = qmc.scale(hod_params_pool, param_prior_low, param_prior_high)

    # ##################    apply Gaussian prior of parameter alpha    ##################
    # mu = 1.0
    # sigma = 0.5
    # lower_bound = 0.0
    # upper_bound = 2.0
    # hod_params_pool[:,4] = truncnorm(
    #     (lower_bound - mu)/sigma, (upper_bound - mu)/sigma, loc=mu, scale=sigma
    #     ).rvs(size=num_pool)
    # ####################################################################################

    ## Main loop to find HOD parameters that matches reference galaxy number density
    hod_params_alive = []
    FAILED_FLAG = False
    while(count < nhod_per_cosmo):
        try:
            curr_hod_params = hod_params_pool[idx,:]
        except:
            logger.warning("cosmo_{:06d} cannot find HOD parameters that matches reference galaxy number density.".format(cosmo_label))
            logger.warning("Found {} HOD parameters that matches reference galaxy number density.".format(count))
            FAILED_FLAG = True
            break

        ## update fic
        if model == 2 or model == 3:
            f_ic, Nsat = find_fic(halo_mass, curr_hod_params, model_lb=model)
            ## FIXME: lower bound of f_ic may need careful consideration.
            # if f_ic > 0.5 and f_ic <= 1.0 and Nsat.max() < 100: # avoid too many satellite galaxies in one halo
            if f_ic > 0 and f_ic <= 1.0 and Nsat.max() < 100: # avoid too many satellite galaxies in one halo
                count += 1  
                idx += 1
                hod_params_alive.append(np.append(curr_hod_params, f_ic))
            else:
                idx += 1
                continue

    if not FAILED_FLAG:
        return hod_params_alive
    else:
        return None

def make_survey(gal_pos:np.ndarray, masks:dict, cosmo_ccl:ccl.Cosmology, nofz_info:dict, gal_vel=None, check_repeat:bool=False, rot_degrees=None):
    assert isinstance((gal_pos), np.ndarray)

    if gal_pos.ndim != 2:
        raise ValueError("gal_pos should be 2D array")
    if gal_pos.shape[1] != 3:
        raise ValueError("gal_pos should have 3 columns")
    
    assert isinstance((masks), dict)
    assert isinstance((cosmo_ccl), ccl.Cosmology)
    assert isinstance((nofz_info), dict)

    logger.info("Calculate cosmology-dependent quantities.")
    
    hubble = cosmo_ccl.to_dict()["h"]
    chi_min = ccl.comoving_radial_distance(cosmo_ccl, 1./(1 + zmin))*hubble # Mpc/h
    chi_max = ccl.comoving_radial_distance(cosmo_ccl, 1./(1 + zmax))*hubble # Mpc/h

    ### transform box data to lightcone 
    ### Type of galcone: first 3 are galaxy Positions in Cartesian coordinates, 
    ### and the last one are the IDs of galaxies.
    ### If considering RSD, will cut thicker slice
    if ADD_RSD and gal_vel is not None:
        Delta_chi = 300.0 # Mpc/h
        galcone, galcone_vel = make_lightcone_tiles(gal_pos, 
                                       boxsize=Lbox, 
                                       chi_min=chi_min-Delta_chi, chi_max=chi_max+Delta_chi, 
                                       other_prop=gal_vel
                                       )
        
    else:
        galcone = make_lightcone_tiles(gal_pos, boxsize=Lbox, 
                                       chi_min=chi_min, chi_max=chi_max
                                       )

    ### if apply rotation
    if rot_degrees is not None:
        ## In shear catalog, we instead rotate the mask, therefore here we use the same 
        ## rotation angle but with an inverse of the rotator.
        galcone_vector = rotate_lightcone(galcone[:,:-1], rot_degrees, inv=True)
    else:
        galcone_vector = galcone[:,:-1]
        
    galcone_ra, galcone_dec, galcone_z, phys_cut = Cart2Sph(cosmo_ccl, pos=galcone_vector)
    galcone_id = galcone[phys_cut][:,-1]

    ### apply RSD effect
    if ADD_RSD and gal_vel is not None:
        galcone_vel = galcone_vel[phys_cut]
        gal_vel_los = (galcone_vel * galcone_vector).sum(axis=1) / np.linalg.norm(galcone_vector, axis=1)
        galcone_zrsd = galcone_z + gal_vel_los * (1 + galcone_z) / sol
    
    del galcone

    galcone_output = np.empty((len(galcone_ra),), dtype=fgal_type)
    galcone_output["ra"] = galcone_ra
    galcone_output["dec"] = galcone_dec
    galcone_output["z"] = galcone_z

    if ADD_RSD and gal_vel is not None:
        galcone_output["zrsd"] = galcone_zrsd
        ### and don't forget to apply a redshift cut
        zrsd_cut = ((galcone_zrsd > zmin) & (galcone_zrsd < zmax))
        galcone_output = galcone_output[zrsd_cut]
        galcone_id = galcone_id[zrsd_cut]

        del galcone_zrsd
    ### if don't consider RSD, set zsrd to be identical as zreal
    else:
        galcone_output['zrsd'] = galcone_output['z']

    del galcone_ra, galcone_dec, galcone_z

    ### survey geometry cut
    logger.info("Apply BOSS survey geometry cut and redshift downsampling")

    galcone_boss_tot = []
    galcone_id_boss_tot = []

    if len(boss_part_names) > 0:
        masks_boss = masks['boss_masks']

    if 'boss_lowzcmass' in boss_part_names:

        logger.info("Making boss_lowzcmass-like mock")

        geom_boss = masks['boss_geom']['boss_lowzcmass']
        galcone_boss, galcone_id_boss = apply_boss_geometry(galcone_output, geom_boss, masks_boss, galcone_ids=galcone_id)
        ### apply BOSS redshift downsample
        nofz_boss = nofz_info['boss_lowzcmass']
        galcone_boss, galcone_id_boss = apply_nz_downsample(galcone_boss, nofz_boss, galcone_ids=galcone_id_boss, norm=False, add_rsd=ADD_RSD)

        logger.debug(f"ngal of lowz: {len(galcone_boss)}")

        galcone_boss['survey'] = 0

        galcone_boss_tot.append(galcone_boss)
        galcone_id_boss_tot.append(galcone_id_boss)
    
    if 'boss_lowze2' in boss_part_names:

        logger.info("Making boss_lowze2-like mock")

        geom_boss = masks['boss_geom']['boss_lowze2']
        galcone_boss, galcone_id_boss = apply_boss_geometry(galcone_output, geom_boss, masks_boss, galcone_ids=galcone_id)
        ### apply BOSS redshift downsample
        nofz_boss = nofz_info['boss_lowze2']
        galcone_boss, galcone_id_boss = apply_nz_downsample(galcone_boss, nofz_boss, galcone_ids=galcone_id_boss, norm=False, add_rsd=ADD_RSD)

        logger.info("Trimming boss_lowze2 region")

        galcone_boss, galcone_id_boss = apply_boss_lowze2e3_trim(galcone_boss, masks['boss_geom']['boss_lowz'], galcone_ids=galcone_id_boss)

        logger.debug(f"ngal of lowze2: {len(galcone_boss)}")

        galcone_boss['survey'] = 1

        galcone_boss_tot.append(galcone_boss)
        galcone_id_boss_tot.append(galcone_id_boss)

    if 'boss_lowze3' in boss_part_names:

        logger.info("Making boss_lowze3-like mock")

        geom_boss = masks['boss_geom']['boss_lowze3']
        galcone_boss, galcone_id_boss = apply_boss_geometry(galcone_output, geom_boss, masks_boss, galcone_ids=galcone_id)
        ### apply BOSS redshift downsample
        nofz_boss = nofz_info['boss_lowze3']
        galcone_boss, galcone_id_boss = apply_nz_downsample(galcone_boss, nofz_boss, galcone_ids=galcone_id_boss, norm=False, add_rsd=ADD_RSD)

        logger.info("Trimming boss_lowze3 region")

        galcone_boss, galcone_id_boss = apply_boss_lowze2e3_trim(galcone_boss, masks['boss_geom']['boss_lowz'], galcone_ids=galcone_id_boss)

        logger.debug(f"ngal of lowze3: {len(galcone_boss)}")

        galcone_boss['survey'] = 2

        galcone_boss_tot.append(galcone_boss)
        galcone_id_boss_tot.append(galcone_id_boss)

    if 'boss_cmass' in boss_part_names:
        
        logger.info(f"Making CMASS-like mock")

        geom_boss = masks['boss_geom']['boss_cmass']
        galcone_boss, galcone_id_boss = apply_boss_geometry(galcone_output, geom_boss, masks_boss, galcone_ids=galcone_id)
        ### apply BOSS redshift downsample
        nofz_boss = nofz_info['boss_cmass']
        ###############################################################################################
        galcone_boss, galcone_id_boss = apply_nz_downsample(galcone_boss, nofz_boss, galcone_ids=galcone_id_boss, norm=False, add_rsd=ADD_RSD)

        logger.debug(f"ngal of cmass: {len(galcone_boss)}")
        
        galcone_boss['survey'] = 4

        galcone_boss_tot.append(galcone_boss)
        galcone_id_boss_tot.append(galcone_id_boss)

    if len(galcone_boss_tot) > 0:
        galcone_boss = np.concatenate(galcone_boss_tot)
        galcone_id_boss = np.concatenate(galcone_id_boss_tot)
    else:
        galcone_boss = []
        galcone_id_boss = []

    if MK_2DFLENS:

        logger.info("Making 2dFLenS-like mock")

        ### apply 2dFLens survey geometry cut
        masks_2dflens = masks['2dflens']
        galcone_2dflens, galcone_id_2dflens = apply_2dflens_geometry(galcone_output, masks_2dflens, galcone_ids=galcone_id)
        galcone_2dflens, galcone_id_2dflens = apply_nz_downsample(galcone_2dflens, nofz_info["2dflens"], galcone_ids=galcone_id_2dflens, norm=False, add_rsd=ADD_RSD)

        galcone_2dflens['survey'] = 3

        logger.debug(f"ngal of 2dflens: {len(galcone_2dflens)}")

        if len(galcone_boss) > 0:
            galcone_id = np.append(galcone_id_boss, galcone_id_2dflens)
            galcone_output = np.append(galcone_boss, galcone_2dflens)
        else:
            galcone_id = galcone_id_2dflens
            galcone_output = galcone_2dflens
    
    else:
        galcone_output = galcone_boss
        galcone_id = galcone_id_boss
    
    ### check replication
    if check_repeat:
        repeat_ratio = np.unique(galcone_id).shape[0]/len(galcone_id)
        print(f"Repeat ratio: {1-repeat_ratio:.3f}")

    del galcone_id

    ### finally do not forget to rotate back
    if rot_degrees is not None:
        galcone_output_vec = Sph2Cart(cosmo_ccl, ra=galcone_output["ra"], dec=galcone_output["dec"], z=galcone_output["z"])
        galcone_output_vec = rotate_lightcone(galcone_output_vec, rot_degrees, inv=False) # since rotate back there we use foreward rotator
        galcone_ra, galcone_dec, galcone_z, phys_cut = Cart2Sph(cosmo_ccl, pos=galcone_output_vec)

        galcone_output = galcone_output[phys_cut]
        galcone_output["ra"] = galcone_ra
        galcone_output["dec"] = galcone_dec
        galcone_output["z"] = galcone_z
        
    return galcone_output


''' ===========================================   Running pipeline   ================================================= '''
def run_halo_only(halo_pos_mass_dict):
    ### choose the most massive halos according to reference ngal
    Ngal = int(ngal_ref*Lbox*Lbox*Lbox)
    logger.debug(f"Nhalo: {len(halo_pos_mass_dict['mass'])}   Ngal targer: {Ngal}")
    halo_pos_mass = np.empty((len(halo_pos_mass_dict["mass"])), dtype=np.dtype([("pos", 'f4', 3), ("mass", 'f4')]))
    halo_pos_mass["pos"] = halo_pos_mass_dict["pos"]
    halo_pos_mass["mass"] = halo_pos_mass_dict["mass"]
    halo_pos_mass.sort(order="mass")
    halo_pos_mass = halo_pos_mass[::-1]

    gal_pos = halo_pos_mass["pos"][:Ngal]
    galcone_output = make_survey(gal_pos, masks, cosmo_ccl, nofz_info, check_repeat=False)

    return galcone_output

def run_vary_hod(halo_file, hod_halocat, OmegaM, hod_params_alive):
    galcone_output_dict = {}
    ### apply HOD and geometry cut
    for ihod, each_hod_params in enumerate(hod_params_alive):
        galcone_output_dict[f'hod{ihod}'] = {}
        #### populate galaxies
        model_params_dict = dict(zip(model_params_names, each_hod_params))
        dict_of_gsamples = apply_hod(halo_file, hod_halocat, OmegaM, model_params_dict, indx=cosmo_label*ihod)
        #### apply geometry cut 
        for iseed, key in enumerate(dict_of_gsamples.keys()):
            galcone_output_dict[f'hod{ihod}'][f'seed{iseed}'] = {}
            x_c, y_c, z_c = dict_of_gsamples[key]["x"], dict_of_gsamples[key]["y"], dict_of_gsamples[key]["z"]
            x_c = (x_c + Lbox) % Lbox
            y_c = (y_c + Lbox) % Lbox
            z_c = (z_c + Lbox) % Lbox
            gal_pos = np.c_[x_c, y_c, z_c]
            if ADD_RSD:
                vx_c, vy_c, vz_c = dict_of_gsamples[key]["vx"], dict_of_gsamples[key]["vy"], dict_of_gsamples[key]["vz"]
                gal_vel = np.c_[vx_c, vy_c, vz_c]
                galcone_output = make_survey(gal_pos, masks, cosmo_ccl, nofz_info, gal_vel, check_repeat=False)
            else:
                galcone_output = make_survey(gal_pos, masks, cosmo_ccl, nofz_info, check_repeat=False)
            ## save to cache
            galcone_output_dict[f'hod{ihod}'][f'seed{iseed}'] = galcone_output

    return galcone_output_dict

def initial_hod_param_dict(cosmo_labels, hod_param_out=None):
    if not LOAD_HOD_PAR:
        ### initial hod parameter dictionary
        hod_param_dict_tot = {}
        for cosmo_label in cosmo_labels:
            hod_param_dict_tot['cosmo{:06d}'.format(cosmo_label)] = {}
    else:
        hod_param_dict_tot = get_hod_params(hod_param_out, otype='dict')

    return hod_param_dict_tot

def prepare_hod_params(hod_halocat, seed_offset=0):
    if not LOAD_HOD_PAR:
        halo_mass = hod_halocat.halo_table["halo_mvir"].value
        hod_params_alive = find_hod_params_alive(halo_mass, num_pool=30000, seedini=9782, seed_offset=seed_offset)
    else:
        if len(hod_param_dict_tot['cosmo{:06d}'.format(cosmo_label)]) == 11:
            hod_params_alive = [hod_param_dict_tot['cosmo{:06d}'.format(cosmo_label)]['HOD{}'.format(jhod)] \
                                for jhod in range(nhod_per_cosmo)]
        else:
            hod_params_alive = None

    return hod_params_alive

def save_hod_param_dict(all_hod_params, hod_param_out):
    # Save HOD params
    logger.info(f"Save HOD parameters")
    final_hod_params = {}
    for d in all_hod_params:
        final_hod_params.update(d)

    with open(hod_param_out, "w+") as f:
        json.dump(final_hod_params, f, indent=4)

    return None

''' ================================================================================================================== '''

# >>> ======   main routine   ====== <<<
if __name__ == "__main__":

    TEST_MODE = False

    if len(sys.argv) > 1 and sys.argv[-1] == "test":
        TEST_MODE = True

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        
        logger.info("Read cosmo labels")

        cosmo_labels_tot = get_cosmo_name_list_original("/data3/suchen/CosmoGridV1/grid/dirnames.txt")
        ######## For Test #######
        if TEST_MODE:
            cosmo_labels_tot = [1]
        #########################
        k, m = divmod(len(cosmo_labels_tot), size)
        chunks = [cosmo_labels_tot[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(size)]
    else:
        chunks = None

    if rank == 0:

        logger.info("Scattering labels")

    cosmo_labels = comm.scatter(chunks, root=0)

    start = datetime.datetime.now()

    # >>> =================================       preparation       ================================= <<<

    masks = {}
    masks['boss_geom'] = {}
    masks['boss_masks'] = []

    logger.info("Load mask files.")

    for ipart_name in boss_part_names:
        masks['boss_geom'][ipart_name] = pymangle.Mangle(geom_boss_fname_list[ipart_name])

    if 'boss_lowze2' in boss_part_names or 'boss_lowze3' in boss_part_names:
        masks['boss_geom']['boss_lowz'] = pymangle.Mangle(geom_boss_fname_list['boss_lowz'])

    if len(boss_part_names) > 0:
        for mask_file in mask_boss_fname_list:
            masks['boss_masks'].append(pymangle.Mangle(mask_file))

    ### load 2dflens survey geometry
    if MK_2DFLENS:
        masks['2dflens'] = loadFitsMaps(mask_weight_2df_fname)

    logger.info("Load n(z) files.")

    ### load nofz information
    nofz_info = {}

    for ipart_name in boss_part_names:
        nofz = np.loadtxt(nz_boss_fname_list[ipart_name], usecols=(1,2,3,5)) # zmin, zmax, nz, shell_vol
        argstart = np.argwhere(nofz[:,0] == zmin)[0,0]
        argend = np.argwhere(nofz[:,1] == zmax)[0,0]

        nofz_info = make_nofz_info(nofz_info, ipart_name, nofz[argstart:argend+2,0], nofz[argstart:argend+1,3], nofz[argstart:argend+1,2])

    ### FIXME: it seems that BOSS CMASS data has a different number density than that given in nofz
    if 'boss_cmass' in boss_part_names:
        nofz_info['boss_cmass']['nz_ref'] *= 0.93 

    ### 2dflens
    if MK_2DFLENS:
        nofz_info['2dflens'] = {}
        nofz = np.loadtxt(nz_2dflens_fname, usecols=(1,2,3,4)) # zmin, zmax, nz, shell_vol
        argstart = np.argwhere(nofz[:,0] == zmin)[0,0]
        argend = np.argwhere(nofz[:,1] == zmax)[0,0]
        nofz_info = make_nofz_info(nofz_info, '2dflens', nofz[argstart:argend+2,0], nofz[argstart:argend+1,3], nofz[argstart:argend+1,2])

    # ====================================================================================================

    # >>> =================================       main process       ================================= <<<


    logger.info("Main process.")


    if VARY_HOD:
        hod_param_dict_tot = initial_hod_param_dict(cosmo_labels, hod_param_out)

    for cosmo_label in cosmo_labels:

        logger.info(f"Start processing cosmo_label {cosmo_label}")

        ### specify cosmo par file name & halo file name
        cpar_fname = os.path.join(sim_fmt.format(cosmo_label, rlz_label), "params.yml")
        halo_fname = os.path.join(sim_fmt.format(cosmo_label, rlz_label), halo_fmt.format(redshift_label))

        ### initialize cosmology
        cosmo_ccl = get_cosmo_from_file(cpar_fname, otype="ccl")

        ### initialize parameter dictionary
        if VARY_HOD and not LOAD_HOD_PAR:
            hod_param_dict_tot['cosmo{:06d}'.format(cosmo_label)]['cpar'] = {
                "Om": cosmo_ccl['Omega_c'] + cosmo_ccl['Omega_b'],
                "Ob": cosmo_ccl['Omega_b'],
                "H0" : cosmo_ccl['h'],
                "ns": cosmo_ccl['n_s'],
                "s8": cosmo_ccl['sigma8'],
                "w0": cosmo_ccl['w0']
            }

        if HALO_ONLY:
            #### load halo info
            halo_pos_mass_dict = get_pkd_halo_attrs(halo_fname, ["pos", "mass"], Lbox, redshift)
            #### main process
            galcone_output = run_halo_only(halo_pos_mass_dict)
            #### save galcone
            gal_fname = os.path.join(out_dir, out_fmt.format(cosmo_label, rlz_label, 0, 0, survey_specify))
            np.save(gal_fname, galcone_output)

        else:
            #### load halo catalog
            halo_fname = sim_fmt.format(cosmo_label, rlz_label) + halo_fmt.format(redshift_label)
            hod_halocat, OmegaM = load_halocat(cpar_fname, halo_fname, ofmt='hod')

            ### vary HOD run
            if VARY_HOD:
                #### prepare HOD params
                hod_params_alive = prepare_hod_params(hod_halocat, seed_offset=cosmo_label)
                #### loop over hod params
                if hod_params_alive is not None:
                    if not LOAD_HOD_PAR:
                        #### save HOD parameter to param dict
                        for ihod, each_hod_params in enumerate(hod_params_alive):
                            hod_param_dict_tot['cosmo{:06d}'.format(cosmo_label)]['HOD{}'.format(ihod)] = list(each_hod_params)
                    #### vary HOD main process, including apply hod and geometry cut
                    galcone_output_dict = run_vary_hod(halo_fname, hod_halocat, OmegaM, hod_params_alive)
                    #### save to file
                    for ihod in range(nhod_per_cosmo):
                        for iseed in range(len(galcone_output_dict[f'hod{ihod}'])):
                            gal_fname = os.path.join(out_dir, out_fmt.format(cosmo_label, rlz_label, ihod, iseed, survey_specify))
                            np.save(gal_fname, galcone_output_dict[f'hod{ihod}'][f'seed{iseed}'])   
                else:
                    continue

    if VARY_HOD and not LOAD_HOD_PAR:
        all_hod_params = comm.gather(hod_param_dict_tot, root=0)

        if rank == 0:
            save_hod_param_dict(all_hod_params, hod_param_out)

    end = datetime.datetime.now()
    logger.info(f"Time elapsed: {end-start}")

    # ====================================================================================================