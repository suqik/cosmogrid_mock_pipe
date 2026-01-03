''' 
From halo to galaxy. 
apply HOD and make lightcone.
'''

import json
import numpy as np
from scipy.stats import qmc, truncnorm
import pymangle
import datetime
from loguru import logger

from utils.io_func import *
from utils.mkfore_utils import *

wdir = "/home/suchen/Program/CosmoGrid"

''' simulation info '''
sim_fmt = "/data3/suchen/CosmoGridV1/grid/cosmo_{:06d}/run_0/"
halo_fmt = "pkd_halos/CosmoML.{:05d}.fofstats.0"
redshift_label = 120 # corresponding to z~0.3
# redshift_label = 110 # corresponding to z~0.51

lb_z_file = "/data3/suchen/CosmoGridV1/label_z_table.txt"
lb_z_tb = np.loadtxt(lb_z_file)

Lbox = 900.0
Nside = 832 # Npart = Nside**3
redshift = lb_z_tb[redshift_label,1]

### Use different redshift range depending on redshift_label
if redshift_label == 120:
    zmin = 0.2
    zmax = 0.4
    zbin_lb = 1
elif redshift_label == 110:
    zmin = 0.4
    zmax = 0.6
    zbin_lb = 2

''' mask file info'''

boss_part_names = ['boss_lowzcmass', 'boss_lowze2', 'boss_lowze3'] # define which BOSS regions to make  'boss_lowzcmass', 'boss_lowze2', 'boss_lowze3', 'boss_cmass'
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

''' n(z) file info '''

### n(z) files
nz_fbase = f"{wdir}/catalogs/NOfZ/lens/"
nz_boss_fname_list = {
    'boss_lowzcmass': nz_fbase + "nbar_DR12v5_LOWZ_North_om0p31_Pfkp10000.dat",
    'boss_lowze2': nz_fbase + "nbar_DR12v5_LOWZE2_North_om0p31_Pfkp10000.dat",
    'boss_lowze3': nz_fbase + "nbar_DR12v5_LOWZE3_North_om0p31_Pfkp10000.dat",
    'boss_cmass': nz_fbase + "nbar_DR12v5_CMASS_North_om0p31_Pfkp10000.dat",
}

nz_2dflens_fname = nz_fbase + "nbar_2dFLens_south_data.dat"

''' HOD params '''
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

''' output files '''
out_dir = f"/data2/suchen/CosmoGrid/high_ngal_suits/HOD_bin{zbin_lb}/"
out_fmt = "cosmo_{:06d}_run_0_HOD_{:d}_run_{:d}_{:s}.npy"
hod_param_out = f"{wdir}/cfgs/hod/hod_5params_dict_high_ngal_wcosmo2.json"

''' Modes specifications '''
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

### Can only activate one of these three modes
HALO_ONLY = False # only use halo, which preserve the ngal but not G-H connection
FIX_HOD = False # use the same G-H connection but cannot preserve the ngal
VARY_HOD = True # preserve the ngal, as well as vary G-H connection

if VARY_HOD:
    LOAD_HOD_PAR = True # if load exist hod params

if FIX_HOD:
    # fid_hod_model_param = np.array([12.59102404,  2.10923402, 14.06049531,  0.07197861,  0.25447211, 1.0])
    fid_hod_model_param = np.array([12.72, 0.67, 12.86, 0.32, 0.21, 1.0])
    # fid_hod_model_param = np.array([13.2, 0.62, 14.32, 13.24, 0.93, 1.0])

    fixed_model_params_dict = {}
    for i in range(num_params):
        fixed_model_params_dict[model_params_names[i]] = fid_hod_model_param[i]

### prior for model == 2
param_prior_low  = np.array([12.5, 1e-5, 12.5, 0.00, 0.0])
param_prior_high = np.array([13.5, 3.00, 15.0, 10.0, 2.0])

### prior from SIMBIG, for model == 3
# param_prior_low = np.array([12., 0.1, 13., 13., 0.0])
# param_prior_high = np.array([14., 0.6, 15., 15., 1.5])

logger.info(f"used simulation redshift: {redshift:.4f}")
logger.info(f"Simulating redshift range: {zmin:.4f} - {zmax:.4f}")

if HALO_ONLY:
    logger.info("HALO only mode")
    logger.info(f"Ngal ref: {ngal_ref*1e4:.2f} e-4")
elif FIX_HOD:
    logger.info("FIX_HOD mode")
    logger.info(f"FIX_HOD: {fixed_model_params_dict}")
elif VARY_HOD:
    logger.info("VARY_HOD mode")
    logger.info(f"Ngal ref: {ngal_ref*1e4:.2f} e-4")
    logger.info(f"HOD prior low: {param_prior_low}")
    logger.info(f"HOD prior high: {param_prior_high}")

if ROT:
    logger.info("Use rotation mode")
    logger.info(f"Rotation degrees: {rot_degrees_list}")


''' ================================================================================================================== '''


''' main routines '''
def load_halocat(cosmo_label, ofmt='hod'):
    cpar_file = sim_fmt.format(cosmo_label) + "params.yml"

    logger.info(f"Load cosmology from file {cpar_file}")
    
    cosmo_ccl = get_cosmo_from_file(cpar_file, otype='ccl')
    OmegaM = cosmo_ccl.to_dict()["Omega_c"] + cosmo_ccl.to_dict()["Omega_b"]
    pmass = rhoc0*OmegaM*(Lbox/Nside)**3 # Msun/h

    ### load pkd halo catalog
    halo_file = sim_fmt.format(cosmo_label) + halo_fmt.format(redshift_label)
    
    logger.info(f"Load PKD halo from file {halo_file}")
    
    pkd_halo_infos = get_pkd_halo_attrs(halo_file, attrs=["pos","vel","mass"], Lbox=Lbox, redshift=redshift)

    if ofmt == 'pkd':
        return pkd_halo_infos

    if ofmt == 'hod':
        ## Initialize HOD model class
        hod_halo_cat = pkd_to_hod_type(pkd_halo_infos, cosmo=cosmo_ccl, pmass=pmass, boxsize=Lbox, redshift=redshift)

        return halo_file, hod_halo_cat, OmegaM

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

def apply_hod(cosmo_label, halo_file, hod_halo_cat, OmegaM, model_params_dict, idx_hod):
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

    dict_of_gsamples = hod_model.populate_mock(model_params_dict, ref_num_dens=ngal_ref, indx=cosmo_label*idx_hod, ifcheck=False)

    return dict_of_gsamples

def find_hod_params_alive(cosmo_label, halo_mass, num_pool=30000, seedini=9782):
    ## Sample HOD parameters
    count = 0
    idx = 0
    seed = seedini + cosmo_label
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

def make_survey2(gal_pos:np.ndarray, masks:dict, cosmo_ccl:ccl.Cosmology, nofz_info:dict, check_repeat:bool=False, rot_degrees=None):
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
    galcone = make_lightcone_tiles(gal_pos, boxsize=Lbox, chi_min=chi_min, chi_max=chi_max)

    ### if apply rotation
    if rot_degrees is not None:
        ## In shear catalog, we instead rotate the mask, therefore here we use the same 
        ## rotation angle but with an inverse of the rotator.
        galcone_vector = rotate_lightcone(galcone[:,:-1], rot_degrees, inv=True)
    else:
        galcone_vector = galcone[:,:-1]
        
    galcone_ra, galcone_dec, galcone_z, phys_cut = Cart2Sph(cosmo_ccl, pos=galcone_vector)
    galcone_id = galcone[phys_cut][:,-1]
    
    del galcone

    galcone_output = np.empty((len(galcone_ra),), dtype=fgal_type)
    galcone_output["ra"] = galcone_ra
    galcone_output["dec"] = galcone_dec
    galcone_output["z"] = galcone_z

    del galcone_ra, galcone_dec, galcone_z

    ### survey geometry cut
    logger.info("Apply BOSS survey geometry cut and redshift downsampling")

    galcone_boss_tot = []
    galcone_id_boss_tot = []
    masks_boss = masks['boss_masks']

    if 'boss_lowzcmass' in boss_part_names:

        logger.info("Making boss_lowzcmass-like mock")

        geom_boss = masks['boss_geom']['boss_lowzcmass']
        galcone_boss, galcone_id_boss = apply_boss_geometry(galcone_output, geom_boss, masks_boss, galcone_ids=galcone_id)
        ### apply BOSS redshift downsample
        nofz_boss = nofz_info['boss_lowzcmass']
        galcone_boss, galcone_id_boss = apply_nz_downsample(galcone_boss, nofz_boss, galcone_ids=galcone_id_boss, norm=False)

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
        galcone_boss, galcone_id_boss = apply_nz_downsample(galcone_boss, nofz_boss, galcone_ids=galcone_id_boss, norm=False)

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
        galcone_boss, galcone_id_boss = apply_nz_downsample(galcone_boss, nofz_boss, galcone_ids=galcone_id_boss, norm=False)

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
        galcone_boss, galcone_id_boss = apply_nz_downsample(galcone_boss, nofz_boss, galcone_ids=galcone_id_boss, norm=False)

        logger.debug(f"ngal of cmass: {len(galcone_boss)}")
        
        galcone_boss['survey'] = 4

        galcone_boss_tot.append(galcone_boss)
        galcone_id_boss_tot.append(galcone_id_boss)

    galcone_boss = np.concatenate(galcone_boss_tot)
    galcone_id_boss = np.concatenate(galcone_id_boss_tot)

    logger.info("Apply 2dFLens survey geometry cut")

    ### apply 2dFLens survey geometry cut
    masks_2dflens = masks['2dflens']
    galcone_2dflens, galcone_id_2dflens = apply_2dflens_geometry(galcone_output, masks_2dflens, galcone_ids=galcone_id)
    galcone_2dflens, galcone_id_2dflens = apply_nz_downsample(galcone_2dflens, nofz_info["2dflens"], galcone_ids=galcone_id_2dflens, norm=False)

    galcone_2dflens['survey'] = 3

    galcone_id = np.append(galcone_id_boss, galcone_id_2dflens)
    galcone_output = np.append(galcone_boss, galcone_2dflens)
    
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


''' ================================================================================================================== '''


# >>> ======   main routine   ====== <<<
if __name__ == "__main__":
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        
        logger.info("Read cosmo labels")

        cosmo_labels_tot = get_cosmo_name_list_original("/data3/suchen/CosmoGridV1/grid/dirnames.txt")
        ######## For Test #######
        # cosmo_labels_tot = [1]
        #########################
        k, m = divmod(len(cosmo_labels_tot), size)
        chunks = [cosmo_labels_tot[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(size)]
    else:
        chunks = None

    if rank == 0:

        logger.info("Scattering labels")

    cosmo_labels = comm.scatter(chunks, root=0)

    start = datetime.datetime.now()

    masks = {}
    masks['boss_geom'] = {}
    masks['boss_masks'] = []

    logger.info("Load mask files.")

    for ipart_name in boss_part_names:
        masks['boss_geom'][ipart_name] = pymangle.Mangle(geom_boss_fname_list[ipart_name])

    if 'boss_lowze2' in boss_part_names or 'boss_lowze3' in boss_part_names:
        masks['boss_geom']['boss_lowz'] = pymangle.Mangle(geom_boss_fname_list['boss_lowz'])

    for mask_file in mask_boss_fname_list:
        masks['boss_masks'].append(pymangle.Mangle(mask_file))

    ### load 2dflens survey geometry
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
    nofz_info['2dflens'] = {}
    nofz = np.loadtxt(nz_2dflens_fname, usecols=(1,2,3,4)) # zmin, zmax, nz, shell_vol
    argstart = np.argwhere(nofz[:,0] == zmin)[0,0]
    argend = np.argwhere(nofz[:,1] == zmax)[0,0]
    nofz_info = make_nofz_info(nofz_info, '2dflens', nofz[argstart:argend+2,0], nofz[argstart:argend+1,3], nofz[argstart:argend+1,2])


    logger.info("Main process.")


    if VARY_HOD:
        hod_param_dict_tot = {}

    for cosmo_label in cosmo_labels:

        logger.info(f"Start processing cosmo_label {cosmo_label}")

        ### initialize cosmology
        cosmo_ccl = get_cosmo_from_file(sim_fmt.format(cosmo_label) + "params.yml", otype="ccl")

        ### initialize parameter dictionary
        if VARY_HOD:
            if not LOAD_HOD_PAR:
                ### initial hod parameter dictionary
                hod_param_dict_tot['cosmo{:06d}'.format(cosmo_label)] = {}
                hod_param_dict_tot['cosmo{:06d}'.format(cosmo_label)]['cpar'] = cosmo_ccl.to_dict()
            else:
                hod_param_dict_tot = get_hod_params(hod_param_out, otype='dict')
        ###################################

        if HALO_ONLY:
            halo_pos_mass_dict = get_pkd_halo_attrs(os.path.join(sim_fmt.format(cosmo_label), halo_fmt.format(redshift_label)), 
                                                    ["pos", "mass"], Lbox, redshift)

            ### choose the most massive halos according to reference ngal
            Ngal = int(ngal_ref*Lbox*Lbox*Lbox)
            logger.debug(f"Nhalo: {len(halo_pos_mass_dict['mass'])}   Ngal targer: {Ngal}")
            halo_pos_mass = np.empty((len(halo_pos_mass_dict["mass"])), dtype=np.dtype([("pos", 'f4', 3), ("mass", 'f4')]))
            halo_pos_mass["pos"] = halo_pos_mass_dict["pos"]
            halo_pos_mass["mass"] = halo_pos_mass_dict["mass"]
            halo_pos_mass.sort(order="mass")
            halo_pos_mass = halo_pos_mass[::-1]

            gal_pos = halo_pos_mass["pos"][:Ngal]
            galcone_output = make_survey2(gal_pos, masks, cosmo_ccl, nofz_info, check_repeat=False)
            ## save as binary file
            out_dir = "/data2/suchen/CosmoGrid/HALO_only_suits/HOD_bin2/"
            np.save(out_dir + out_fmt.format(cosmo_label, 0, 0, "boss_north_2dflens_south"), galcone_output)
        else:
            ### load halo catalog
            halo_file, hod_halocat, OmegaM = load_halocat(cosmo_label, ofmt='hod')
            ### fix HOD run
            if FIX_HOD:
                ### fix HOD run
                dict_of_gsamples = apply_hod(cosmo_label, halo_file, hod_halocat, OmegaM, fixed_model_params_dict, 0)

                for iseed, key in enumerate(dict_of_gsamples.keys()):
                    x_c, y_c, z_c = dict_of_gsamples[key]["x"], dict_of_gsamples[key]["y"], dict_of_gsamples[key]["z"]
                    x_c = (x_c + Lbox) % Lbox
                    y_c = (y_c + Lbox) % Lbox
                    z_c = (z_c + Lbox) % Lbox
                    gal_pos = np.c_[x_c, y_c, z_c]

                    curr_ngals = len(gal_pos) / Lbox / Lbox / Lbox
                    logger.debug(f"Ngal: {curr_ngals*1e4:.2f}")

                    if ROT:
                        for irot, rot_degrees in enumerate(rot_degrees_list):

                            logger.info(f"Rotation {irot}: rot_angles (zyx) = {rot_degrees}")

                            galcone_output = make_survey2(gal_pos, masks, cosmo_ccl, nofz_info, check_repeat=False, rot_degrees=rot_degrees)
                            out_dir = "/data2/suchen/CosmoGrid/fix_HOD_rots/"
                            np.save(out_dir + out_fmt.format(cosmo_label, 0, iseed, f"boss_north_2dflens_south_rot{irot}"), galcone_output)
                    else:
                        galcone_output = make_survey2(gal_pos, masks, cosmo_ccl, nofz_info, check_repeat=False)
                        ## save as binary file
                        out_dir = "/data2/suchen/CosmoGrid/fix_HOD_suits/HOD_bin2/"
                        np.save(out_dir + out_fmt.format(cosmo_label, 0, iseed, "boss_north_2dflens_south"), galcone_output)
            ### vary HOD run
            else:
                if not LOAD_HOD_PAR:
                    halo_mass = hod_halocat.halo_table["halo_mvir"].value
                    hod_params_alive = find_hod_params_alive(cosmo_label, halo_mass, num_pool=30000, seedini=9782)
                else:
                    if len(hod_param_dict_tot['cosmo{:06d}'.format(cosmo_label)]) == 11:
                        hod_params_alive = [hod_param_dict_tot['cosmo{:06d}'.format(cosmo_label)]['HOD{}'.format(jhod)] for jhod in range(nhod_per_cosmo)]
                    else:
                        hod_params_alive = None
                if hod_params_alive is not None:
                    gsamples_list = []
                    ### apply HOD
                    for ihod, each_hod_params in enumerate(hod_params_alive):
                        ## save hod parameter to param dict
                        if not LOAD_HOD_PAR:
                            hod_param_dict_tot['cosmo{:06d}'.format(cosmo_label)]['HOD{}'.format(ihod)] = list(each_hod_params)
                        ### populate galaxies
                        model_params_dict = dict(zip(model_params_names, each_hod_params))
                        dict_of_gsamples = apply_hod(cosmo_label, halo_file, hod_halocat, OmegaM, model_params_dict, ihod)
                        for iseed, key in enumerate(dict_of_gsamples.keys()):
                            print(len(dict_of_gsamples[key]["x"])/Lbox/Lbox/Lbox)
                        
                        gsamples_list.append(dict_of_gsamples)

                    logger.info(f'{len(gsamples_list)} HODs are found.')

                    ### apply survey geometry
                    logger.info('Begin Loops')

                    for ihod, dict_of_gsamples in enumerate(gsamples_list):
                        for iseed, key in enumerate(dict_of_gsamples.keys()):
                            x_c, y_c, z_c = dict_of_gsamples[key]["x"], dict_of_gsamples[key]["y"], dict_of_gsamples[key]["z"]
                            x_c = (x_c + Lbox) % Lbox
                            y_c = (y_c + Lbox) % Lbox
                            z_c = (z_c + Lbox) % Lbox
                            gal_pos = np.c_[x_c, y_c, z_c]
                            galcone_output = make_survey2(gal_pos, masks, cosmo_ccl, nofz_info, check_repeat=False)
                            ## save as binary file
                            np.save(out_dir + out_fmt.format(cosmo_label, ihod, iseed, "boss_north_2dflens_south"), galcone_output)

                    logger.info('Loops end')

                else:
                    continue

    if VARY_HOD:
        all_hod_params = comm.gather(hod_param_dict_tot, root=0)

        # Save HOD params
        if not LOAD_HOD_PAR:
            if rank == 0:
                logger.info(f"Save HOD parameters")
                final_hod_params = {}
                for d in all_hod_params:
                    final_hod_params.update(d)

                with open(hod_param_out, "w+") as f:
                    json.dump(final_hod_params, f, indent=4)

    end = datetime.datetime.now()
    logger.info(f"Time elapsed: {end-start}")