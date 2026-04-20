''' 
Useful functions in making galaxy catalog
'''

import json
import numpy as np
from scipy.stats import qmc, truncnorm
import pymangle
import warnings
from loguru import logger

from utils.io_func import *
from utils.mkfore_utils import *


def load_halocat(
        cpar_fname, halo_fname, 
        Lbox, Nside, redshift,
        ofmt='hod', clean=True):
    '''
    Load halo catalog, and transfer to HOD type
    '''

    print(f"Load cosmology from file {cpar_fname}")
    
    cosmo_ccl = get_cosmo_from_file(cpar_fname, otype='ccl')
    OmegaM = cosmo_ccl.to_dict()["Omega_c"] + cosmo_ccl.to_dict()["Omega_b"]
    pmass = rhoc0*OmegaM*(Lbox/Nside)**3 # Msun/h
    
    print(f"Load PKD halo from file {halo_fname}")
    
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

        return hod_halo_cat
    

def get_ngal(
        halo_mass, Lbox, redshift,
        model_lb, model_params_names, hod_param_vals, 
        ):
    '''
    Calculate theoretical predictions of ngal given HMF.
    '''
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

    if model_lb == 0:
        ctr = MWCens(redshift=redshift)
    elif model_lb == 2 or model_lb == 3:
        ctr = MWCens_IC(redshift=redshift)
    elif model_lb == 4:
        ctr = ABMWCens_IC(redshift=redshift)

    ctr.param_dict = tmp_dict
    Nctr = ctr.mean_occupation(prim_haloprop=massbin)

    if model_lb == 0 or model_lb == 2:
        sat = MWSats(redshift=redshift, cenocc_model=ctr, modulate_with_cenocc=True)
    elif model_lb == 3:
        sat = MWSats2(redshift=redshift, cenocc_model=ctr, modulate_with_cenocc=True)
    elif model_lb == 4:
        sat = ABMWSats(redshift=redshift, cenocc_model=ctr, modulate_with_cenocc=True)

    sat.param_dict = tmp_dict
    Nsat = sat.mean_occupation(prim_haloprop=massbin)

    ngal_mock = (np.sum(Nctr*NM) + np.sum(Nsat*NM))

    Nsat_frac = Nsat.sum()/(Nctr+Nsat).sum()

    return ngal_mock, Nsat_frac


def check_mask(survey_names, masks):
    miss_keys = []

    HAVE_BOSS = False
    HAVE_2DFLENS = False
    for isurvey_name in survey_names:
        if "boss" in isurvey_name:
            HAVE_BOSS = True
        if "2dflens" in isurvey_name:
            HAVE_2DFLENS = True
        
    if HAVE_BOSS:
        if not "boss_geom" in masks.keys():
            miss_keys.append("boss_geom")
        if not "boss_masks" in masks.keys():
            miss_keys.append("boss_masks")
    
    if HAVE_2DFLENS:
        if not "2dflens_geom" in masks.keys():
            miss_keys.append("2dflens_geom")
        
    if len(miss_keys) > 0:
        raise ValueError(f"Need {miss_keys}")

    for isurvey_name in survey_names:
        if "boss" in isurvey_name:
            if not isurvey_name in masks["boss_geom"].keys():
                miss_keys.append(isurvey_name)
        if "2dflens" in isurvey_name:
            if not isurvey_name in masks["2dflens_geom"].keys():
                miss_keys.append(isurvey_name)
        
    if len(miss_keys) > 0:
        raise ValueError(f"Need {miss_keys}")


def initial_hod_param_dict(load_hod_par=True, hod_param_file=None):
    if not load_hod_par:
        ### initial hod parameter dictionary
        cosmo_hod_dict_tot = {}
    else:
        cosmo_hod_dict_tot = get_hod_params(hod_param_file, otype='dict')

    return cosmo_hod_dict_tot


def save_hod_param_dict(all_hod_params, hod_param_file):
    # Save HOD params
    print(f"Save HOD parameters")
    final_hod_params = {}
    for d in all_hod_params:
        final_hod_params.update(d)

    with open(hod_param_file, "w+") as f:
        json.dump(final_hod_params, f, indent=4)

    return None

''' ================================================================================================================== '''


''' Main routine '''


def prepare_masks(
        survey_part_names:list,
        geom_fname_dict:dict,
        mask_boss_fname_list:list=None
        ):
    masks = {}

    HAVE_BOSS = False
    for isurvey_name in survey_part_names:
        if "boss" in isurvey_name:
            HAVE_BOSS = True
            assert mask_boss_fname_list is not None, "Must provide boss mask file list"
            masks['boss_geom'] = {}
            masks['boss_masks'] = []
        if "2dflens" in isurvey_name:
            masks['2dflens_geom'] = {}
        
    print("Load mask files.")

    if HAVE_BOSS:
        ### obervational masks
        for mask_file in mask_boss_fname_list:
            masks['boss_masks'].append(pymangle.Mangle(mask_file))
    ### survey geometry
    for ipart_name in survey_part_names:
        if "boss" in ipart_name:
            masks['boss_geom'][ipart_name] = pymangle.Mangle(geom_fname_dict[ipart_name])
        if "2dflens" in ipart_name:
            masks['2dflens_geom'][ipart_name] = loadFitsMaps(geom_fname_dict[ipart_name])

    if 'boss_lowze2' in survey_part_names or 'boss_lowze3' in survey_part_names:
        if 'boss_low' not in masks['boss_geom'].keys():
            masks['boss_geom']['boss_lowz'] = pymangle.Mangle(geom_fname_dict['boss_lowz'])

    return masks

def prepare_nofz(
        survey_part_names:list,
        zmin, zmax,
        nz_fname_dict:dict
):

    nofz_info = {}

    print("Load n(z) files.")

    for ipart_name in survey_part_names:
        if "boss" in ipart_name:
            nofz = np.loadtxt(nz_fname_dict[ipart_name], usecols=(1,2,3,5)) # zmin, zmax, nz, shell_vol
        if "2dflens" in ipart_name:
            nofz = np.loadtxt(nz_fname_dict[ipart_name], usecols=(1,2,3,4)) # zmin, zmax, nz, shell_vol
        # argstart = np.argwhere(nofz[:,0] == zmin)[0,0]
        # argend = np.argwhere(nofz[:,1] == zmax)[0,0]
        # nofz_info = make_nofz_info(nofz_info, ipart_name, nofz[argstart:argend+2,0], nofz[argstart:argend+1,3], nofz[argstart:argend+1,2])
        nofz_info = make_nofz_info(nofz_info, ipart_name, np.append(nofz[:,0], nofz[-1,1]), nofz[:,3], nofz[:,2])

    ### FIXME: it seems that BOSS CMASS data has a different number density than that given in nofz
    if 'boss_cmass' in nofz_info.keys():
        nofz_info['boss_cmass']['nz_ref'] *= 0.93 

    return nofz_info

def run_find_hod_params(
        hod_cat, nhod_per_cosmo,
        model_lb, model_params_names, param_prior_low, param_prior_high, ngal_ref,
        num_pool=30000, seedini=9782, seed_offset=0
        ) -> list:

    halo_mass = hod_cat.halo_table["halo_mvir"].value
    Lbox = hod_cat.Lbox[0]
    redshift = hod_cat.redshift

    ## Sample HOD parameters
    count = 0
    idx = 0
    seed = seedini + seed_offset

    ### priors do not include f_ic
    lhc_sampler = qmc.LatinHypercube(d=len(param_prior_low), seed=seed)
    hod_params_pool = lhc_sampler.random(n=num_pool)
    hod_params_pool = qmc.scale(hod_params_pool, param_prior_low, param_prior_high)

    ### prior of SIMBIG 
    if model_lb == 3:
        mu = 1.0
        sigma = 0.5
        lower_bound = param_prior_low[4]
        upper_bound = param_prior_high[4]
        hod_params_pool[:,4] = truncnorm(
            (lower_bound - mu)/sigma, (upper_bound - mu)/sigma, loc=mu, scale=sigma
            ).rvs(size=num_pool)
        
    ## Main loop to find HOD parameters that matches reference galaxy number density
    hod_params_alive = []
    while(count < nhod_per_cosmo):
        try:
            curr_hod_params = hod_params_pool[idx,:]
        except:
            warnings.warn("Found {} HOD parameters that matches reference galaxy number density.".format(count))
            break

        ngal_mock, Nsat_frac = get_ngal(
            halo_mass=halo_mass, Lbox=Lbox, redshift=redshift,
            model_lb=model_lb, model_params_names=model_params_names, hod_param_vals=curr_hod_params, 
        )

        if model_lb == 0:
            # if np.abs(ngal_mock - ngal_ref) < 0.1 and Nsat.max() < 1000: # avoid too many satellite galaxies in one halo
            if f_ic > 0 and f_ic <= 1.0 and Nsat_frac < 0.3: # avoid too many satellite galaxies in one halo
                count += 1
                idx += 1
                hod_params_alive.append(curr_hod_params)
            else:
                idx += 1
                continue

        ## update fic
        if model_lb == 2 or model_lb == 3 or model_lb == 4:
            f_ic = ngal_ref/ngal_mock

            ## FIXME: lower bound of f_ic may need careful consideration.
            # if f_ic > 0.5 and f_ic <= 1.0 and Nsat.max() < 100: # avoid too many satellite galaxies in one halo
            # if f_ic > 0 and f_ic <= 1.0 and Nsat.max() < 1000: # avoid too many satellite galaxies in one halo
            if f_ic > 0 and f_ic <= 1.0 and Nsat_frac < 0.3: # avoid too many satellite galaxies in one halo
                count += 1  
                idx += 1
                ### here we append f_ic to construct total HOD parameters
                # hod_params_alive.append(np.append(curr_hod_params, f_ic))
                hod_params_alive.append(list(curr_hod_params)+[f_ic])
            else:
                idx += 1
                continue
    
    return hod_params_alive

def run_apply_hod(
        hod_halo_cat,
        model_lb, model_params_names, model_params_dict,
        OmegaM, init_seed, num_seeds, z_space, Num_ptcl_requirement, ngal_ref, indx
        ):
    
    '''
    Halo catalog to galaxy catalog
    '''
    
    redshift = hod_halo_cat.redshift
    Lbox     = hod_halo_cat.Lbox

    num_params = len(model_params_names)

    hod_model = ModelClass(
        ["dummy"], [hod_halo_cat], 
        model=model_lb, 
        num_params=num_params,
        param_names=model_params_names,
        redshift=redshift, 
        box_size=Lbox,
        Omega_m=OmegaM, 
        init_seed=init_seed,
        num_seeds=num_seeds,
        z_space=z_space,
        Num_ptcl_requirement=Num_ptcl_requirement,
        verbose=True
        )

    dict_of_gsamples = hod_model.populate_mock(model_params_dict, ref_num_dens=ngal_ref, indx=indx, ifcheck=False)

    return dict_of_gsamples

def run_box_to_lightcone(
        gal_pos:np.ndarray, 
        cosmo_ccl:ccl.Cosmology, Lbox:float,
        zmin:float, zmax:float,
        add_rsd:bool=True,
        gal_adj_props:dict=None
        ):
    
    ngal_box = len(gal_pos)
    eff_num_den = ngal_box / Lbox / Lbox / Lbox
    print("="*40)
    print(f"Ngal in box: {eff_num_den*1e4:.3f} e-4")
    print("="*40)
    
    '''
    Transform box data to lightcone data
    '''
    if add_rsd:
        assert ("gal_vel" in gal_adj_props.keys()), "Need gal_vel to compute RSD effect!"
    if gal_pos.ndim != 2:
        raise ValueError("gal_pos should be 2D array")
    if gal_pos.shape[1] != 3:
        raise ValueError("gal_pos should have 3 columns")
    
    assert isinstance((cosmo_ccl), ccl.Cosmology)

    print("\nCalculate cosmology-dependent quantities.")
    
    hubble = cosmo_ccl.to_dict()["h"]
    chi_min = ccl.comoving_radial_distance(cosmo_ccl, 1./(1 + zmin))*hubble # Mpc/h
    chi_max = ccl.comoving_radial_distance(cosmo_ccl, 1./(1 + zmax))*hubble # Mpc/h

    ### register gal adj props
    adj_props = []
    adj_props_idx = {}
    curr_idx = 0
    for prop_name, prop_data in gal_adj_props.items():
        adj_props_idx[prop_name] = curr_idx
        adj_props.append(prop_data)
        curr_idx += 1

    ### transform box data to lightcone 
    ### Type of galcone: first 3 are galaxy Positions in Cartesian coordinates, 
    ### and the last one are the IDs of galaxies.
    ### If considering RSD, will cut thicker slice

    print("\nConstruct lightcone by replicating box data")

    if add_rsd:
        Delta_chi = 300.0 # Mpc/h
        galcone, galcone_props = make_lightcone_tiles(gal_pos, 
                                       boxsize=Lbox, 
                                       chi_min=chi_min-Delta_chi, chi_max=chi_max+Delta_chi, 
                                       other_props=adj_props
                                       )
        
    else:
        galcone = make_lightcone_tiles(gal_pos, boxsize=Lbox, 
                                       chi_min=chi_min, chi_max=chi_max, 
                                       other_props=adj_props
                                       )

    galcone_vector = galcone[:,:-1]
        
    galcone_ra, galcone_dec, galcone_z, phys_cut = Cart2Sph(cosmo_ccl, pos=galcone_vector)

    for i_adj_prop in range(len(adj_props)):
        galcone_props[i_adj_prop] = galcone_props[i_adj_prop][phys_cut]
    ### apply RSD effect
    if add_rsd:

        print("\nAdd RSD effect")

        galcone_vel = galcone_props[adj_props_idx["gal_vel"]]
        gal_vel_los = (galcone_vel * galcone_vector).sum(axis=1) / np.linalg.norm(galcone_vector, axis=1)
        galcone_zrsd = galcone_z + gal_vel_los * (1 + galcone_z) / sol
    
    del galcone

    galcone_output = np.empty((len(galcone_ra),), dtype=fgal_type)
    galcone_output["ra"] = galcone_ra
    galcone_output["dec"] = galcone_dec
    galcone_output["z"] = galcone_z

    for prop_name in gal_adj_props.keys():
        if prop_name in galcone_output.dtype.names:
            galcone_output[prop_name] = galcone_props[adj_props_idx[prop_name]]

    if add_rsd:
        galcone_output["zrsd"] = galcone_zrsd
        ### and don't forget to apply a redshift cut
        del galcone_zrsd
    ### if don't consider RSD, set zsrd to be identical as zreal
    else:
        galcone_output['zrsd'] = galcone_output['z']

    zrsd_cut = ((galcone_output['zrsd'] > zmin) & (galcone_output['zrsd'] < zmax))
    galcone_output = galcone_output[zrsd_cut]

    del galcone_ra, galcone_dec, galcone_z

    ### DEBUG ###
    eff_vol = 4*np.pi/3*(chi_max**3-chi_min**3)
    eff_num_den = len(galcone_output) / eff_vol
    print("="*40)
    print(f"Ngal in lightcone: {eff_num_den*1e4:.3f} e-4")
    print("="*40)

        
    return galcone_output

def run_apply_geometry(
        galcone,
        survey_part_names,
        masks:dict,
        nofz_info=None,
        nofz_method='downsample',
        rot_degrees=None,
        add_rsd:bool=True
    ):
    ## In shear catalog, we instead rotate the mask, therefore here we use the same 
    ## rotation angle but with an inverse of the rotator.
    if rot_degrees is not None:
        galcone = rotate_lightcone(galcone, rot_degrees=rot_degrees, inv=True, icoord='radec')
    
    print(f"\nSurveys: {survey_part_names}")

    check_mask(survey_part_names, masks)

    galcone_tot = []

    if 'boss_lowz' in survey_part_names:

        print("\nMaking boss_lowz-like mock")

        geom_boss = masks['boss_geom']['boss_lowz']
        masks_boss = masks['boss_masks']
        galcone_boss, _ = apply_boss_geometry(galcone, geom_boss, masks_boss, galcone_ids=None)
        
        if nofz_info is not None:
            nofz_boss = nofz_info['boss_lowz']
            galcone_boss, _ = apply_nz(galcone_boss, nofz_boss, nofz_method=nofz_method, norm=False, add_rsd=add_rsd)
        
        galcone_boss['survey'] = 0

        galcone_tot.append(galcone_boss)

    if 'boss_lowze2' in survey_part_names:

        print("\nMaking boss_lowze2-like mock")

        geom_boss = masks['boss_geom']['boss_lowze2']
        masks_boss = masks['boss_masks']
        galcone_boss, _ = apply_boss_geometry(galcone, geom_boss, masks_boss, galcone_ids=None)
        if nofz_info is not None:
            nofz_boss = nofz_info['boss_lowze2']
            galcone_boss, _ = apply_nz(galcone_boss, nofz_boss, nofz_method=nofz_method, norm=False, add_rsd=add_rsd)
        
        print("\nTrimming boss_lowze2 region")

        galcone_boss, _ = apply_boss_lowze2e3_trim(galcone_boss, masks['boss_geom']['boss_lowz'], galcone_ids=None)

        galcone_boss['survey'] = 1

        galcone_tot.append(galcone_boss)

    if 'boss_lowze3' in survey_part_names:

        print("\nMaking boss_lowze3-like mock")

        geom_boss = masks['boss_geom']['boss_lowze3']
        masks_boss = masks['boss_masks']
        galcone_boss, _ = apply_boss_geometry(galcone, geom_boss, masks_boss, galcone_ids=None)
        if nofz_info is not None:
            nofz_boss = nofz_info['boss_lowze3']
            galcone_boss, _ = apply_nz(galcone_boss, nofz_boss, nofz_method=nofz_method, norm=False, add_rsd=add_rsd)
        
        print("\nTrimming boss_lowze3 region")

        galcone_boss, _ = apply_boss_lowze2e3_trim(galcone_boss, masks['boss_geom']['boss_lowz'], galcone_ids=None)

        galcone_boss['survey'] = 2

        galcone_tot.append(galcone_boss)

    if 'boss_cmass' in survey_part_names:

        print("\nMaking boss_cmass-like mock")

        geom_boss = masks['boss_geom']['boss_cmass']
        masks_boss = masks['boss_masks']
        galcone_boss, _ = apply_boss_geometry(galcone, geom_boss, masks_boss, galcone_ids=None)
        print("="*40)
        print(f"After applying geometry cut: {len(galcone_boss)}")
        print("="*40)
        if nofz_info is not None:
            nofz_boss = nofz_info['boss_cmass']
            galcone_boss, _ = apply_nz(galcone_boss, nofz_boss, nofz_method=nofz_method, norm=False, add_rsd=add_rsd)

        galcone_boss['survey'] = 4

        galcone_tot.append(galcone_boss)

    if '2dflens_south' in survey_part_names:

        print("\nMaking 2dflens-like mock")

        masks_2dflens = masks['2dflens_geom']['2dflens_south']
        galcone_2dflens, _ = apply_2dflens_geometry(galcone, masks_2dflens, galcone_ids=None)
        if nofz_info is not None:
            nofz_2dflens = nofz_info['2dflens']
            galcone_2dflens, _ = apply_nz(galcone_2dflens, nofz_2dflens, nofz_method=nofz_method, norm=False, add_rsd=add_rsd)
        
        galcone_2dflens['survey'] = 3

        galcone_tot.append(galcone_2dflens)

    galcone_tot = np.concatenate(galcone_tot)

    ### finally do not forget to rotate back
    if rot_degrees is not None:
        galcone_tot = rotate_lightcone(galcone_tot, rot_degrees, inv=False, icoord='radec')

    return galcone_tot