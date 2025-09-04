''' 
From halo to galaxy. 
apply HOD and make lightcone.
'''

import numpy as np
from scipy.stats import qmc
import pymangle
import datetime
import sys
from loguru import logger

from io_func import *
from mkfore_utils import *

logger.remove()

### debug
if len(sys.argv) > 1:
    logger.add(sink=sys.stdout, level="DEBUG")
### release
else:
    logger.add(sink=sys.stdout, level="INFO")

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

''' HOD params '''
model = 2
num_params = 6 # Number of parameters of HOD model
nhod_per_cosmo = 10 # Number of varied HOD parameter values per cosmology
model_params_names = 'logMcut', 'sigma_logM', 'logM1', 'k', 'alpha', 'fic'
Num_ptcl_requirement = 12
verbose = True
num_seeds = 1
init_seed = 33000
z_space = False
ngal_ref = 3.5e-4 # BOSS LRG mean number density

# model_params_dict = {
#     'logMcut': 12.89,
#     'sigma_logM': 0.27,
#     'logM1': 13.2,
#     'k': 0.65,
#     'alpha': 1.20,
#     'fic': 1.0
# }

test_model_param = np.array([12.59102404,  2.10923402, 14.06049531,  0.07197861,  0.25447211, 1.0])
# test_model_param = np.array([12.51460811,  0.05693276, 13.49879432,  7.52035039,  1.36215534, 1.0])
model_params_dict = {}
for i in range(num_params):
    model_params_dict[model_params_names[i]] = test_model_param[i]

param_prior_low = np.array([12.5, 1e-5, 12.5, 0.0, 0.0])
param_prior_high = np.array([15, 3.0, 15.5, 10.0, 2.0])

''' output files '''
# out_dir = "/home/suchen/Program/CosmoGrid/catalogs/HOD/"
out_dir = "/data2/suchen/CosmoGrid/HOD/"
out_fmt = "cosmo_{:06d}_run_0_HOD_{:d}_run_{:d}_{:s}.npy"

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

def find_fic(halo_mass, hod_param_vals):
    if model == 2:
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

        sat = MWSats(redshift=redshift)
        sat.param_dict = tmp_dict
        Nsat = sat.mean_occupation(prim_haloprop=massbin)

        f_ic = (ngal_ref - np.sum(Nsat*NM))/np.sum(Nctr*NM)
        
        return f_ic
        
def apply_hod(cosmo_label, halo_file, hod_halo_cat, OmegaM, model_params_dict):
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

    dict_of_gsamples = hod_model.populate_mock(model_params_dict, ref_num_dens=ngal_ref, indx=cosmo_label*nhod_per_cosmo, ifcheck=False)

    return dict_of_gsamples

def find_hod_params_alive(cosmo_label, halo_mass, hod_param_file, num_pool=10000, seedini=9782):
    # ## make hod-type halo catalog
    # halo_file, hod_halo_cat, OmegaM = make_hod_halocat(cosmo_label)
    # halo_mass = hod_halo_cat.halo_mvir

    ## Sample HOD parameters
    count = 0
    idx = 0
    seed = seedini + cosmo_label
    lhc_sampler = qmc.LatinHypercube(d=len(param_prior_low), seed=seed)
    hod_params_pool = lhc_sampler.random(n=num_pool)
    hod_params_pool = qmc.scale(hod_params_pool, param_prior_low, param_prior_high)

    hod_params_alive = []

    ## Main loop to find HOD parameters that matches reference galaxy number density
    FAILED_FLAG = False
    while(count < nhod_per_cosmo):
        drop_flag = False

        try:
            curr_hod_params = hod_params_pool[idx,:]
        except:
            logger.warning("cosmo_{:06d} cannot find enough HOD parameters that matches reference galaxy number density.".format(cosmo_label))
            FAILED_FLAG = True
            break

        ## update fic
        if model == 2:
            f_ic = find_fic(halo_mass, curr_hod_params)
            if f_ic>0 and f_ic <= 1.0:
                count += 1
            else:
                drop_flag = True
        
        idx += 1
            
        if drop_flag:
            continue
        else:
            hod_params_alive.append(np.append(curr_hod_params, f_ic))

    hod_params_alive = np.asarray(hod_params_alive)

    logger.info(f"Save HOD parameters to file: {hod_param_file}")
    np.savetxt(hod_param_file, hod_params_alive)

    if not FAILED_FLAG:
        return hod_params_alive
    else:
        return None

def make_survey(gal_pos:np.ndarray, masks:dict, cosmo_ccl:ccl.Cosmology, nofz_info:dict, check_repeat:bool=False):
    assert isinstance((gal_pos), np.ndarray)
    assert isinstance((masks), dict)
    assert isinstance((cosmo_ccl), ccl.Cosmology)
    assert isinstance((nofz_info), dict)

    logger.info("Calculate cosmology-dependent quantities.")
    
    hubble = cosmo_ccl.to_dict()["h"]
    chi_min = ccl.comoving_radial_distance(cosmo_ccl, 1./(1 + zmin))*hubble # Mpc/h
    chi_max = ccl.comoving_radial_distance(cosmo_ccl, 1./(1 + zmax))*hubble # Mpc/h

    ### transform box data to lightcone 
    ### there are 4 cols in galcone: first 3 are (x,y,z), and the last one is the id of the galaxy
    galcone = make_lightcone_tiles(gal_pos, boxsize=Lbox, chi_min=chi_min, chi_max=chi_max)
    
    galcone_ra, galcone_dec = hp.vec2ang(galcone[:,:-1], lonlat=True)
    galcone_chi = np.linalg.norm(galcone[:,:-1], axis=1)
    galcone_id = galcone[:,-1]
    
    del galcone

    galcone_output = np.empty((len(galcone_ra),), dtype=fgal_type)
    galcone_output["ra"] = galcone_ra
    galcone_output["dec"] = galcone_dec
    galcone_output["z"] = 1./ccl.scale_factor_of_chi(cosmo_ccl, galcone_chi/hubble) - 1

    ### survey geometry cut
    logger.info("Apply BOSS survey geometry cut and redshift downsampling")

    ### apply BOSS LOWZCMASS survey geometry cut
    ### Separately process CMASSLOWZ, LOWZE2, LOWZE3
    ### Note LOWZE2 and LOWZE3 needs to be trimmed
    galcone_boss_tot = []
    galcone_id_boss_tot = []
    masks_boss = masks['boss_masks']
    for igeom_poly in range(3):

        logger.info(f"Making {boss_part_names[igeom_poly]}-like mock")

        geom_boss = masks['boss_gemo'][igeom_poly]
        galcone_boss, galcone_id_boss = apply_boss_geometry(galcone_output, geom_boss, masks_boss, galcone_ids=galcone_id)
        ### apply BOSS redshift downsample
        nofz_boss = nofz_info[boss_part_names[igeom_poly]]
        galcone_boss, galcone_id_boss = apply_nz_downsample(galcone_boss, nofz_boss, galcone_ids=galcone_id_boss)

        if igeom_poly != 0:
            logger.info(f"Trimming {boss_part_names[igeom_poly]} region")
            galcone_boss, galcone_id_boss = apply_boss_lowze2e3_trim(galcone_boss, masks['boss_gemo'][-1], galcone_ids=galcone_id_boss)
        galcone_boss_tot.append(galcone_boss)
        logger.debug(f"{len(galcone_boss)}")
        galcone_id_boss_tot.append(galcone_id_boss)
    
    galcone_boss = np.concatenate(galcone_boss_tot)
    galcone_id_boss = np.concatenate(galcone_id_boss_tot)

    logger.info("Apply 2dFLens survey geometry cut")

    ### apply 2dFLens survey geometry cut
    masks_2dflens = masks['2dflens']
    galcone_2dflens, galcone_id_2dflens = apply_2dflens_geometry(galcone_output, masks_2dflens, galcone_ids=galcone_id)
    galcone_2dflens, galcone_id_2dflens = apply_nz_downsample(galcone_2dflens, nofz_info["2dflens"], galcone_ids=galcone_id_2dflens)

    galcone_id = np.append(galcone_id_boss, galcone_id_2dflens)
    galcone_output = np.append(galcone_boss, galcone_2dflens)
    logger.debug(f"{len(galcone_output)}")
    
    ### check replication
    if check_repeat:
        repeat_ratio = np.unique(galcone_id).shape[0]/len(galcone_id)
        print(f"Repeat ratio: {1-repeat_ratio:.3f}")

    del galcone_id

    return galcone_output

if __name__ == "__main__":
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        
        logger.info("Read cosmo labels")

        with open("/data3/suchen/CosmoGridV1/grid/dirnames.txt", "r") as f:
            dirnames = f.readlines()
            cosmo_labels_tot = [int(i.strip("\n").split("_")[1]) for i in dirnames]

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
    masks['boss_gemo'] = []
    masks['boss_masks'] = []

    logger.info("Load mask files.")
    ### load boss survey geometry
    for geom_file in geom_boss_fname_list:
        masks['boss_gemo'].append(pymangle.Mangle(geom_file))
    for mask_file in mask_boss_fname_list:
        masks['boss_masks'].append(pymangle.Mangle(mask_file))

    ### load 2dflens survey geometry
    masks['2dflens'] = loadFitsMaps(mask_weight_2df_fname)

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

    ### 2dflens
    nofz_info['2dflens'] = {}
    nofz = np.loadtxt(nz_2dflens_fname, usecols=(1,2,3,4)) # zmin, zmax, nz, shell_vol
    argstart = np.argwhere(nofz[:,0] == zmin)[0,0]
    argend = np.argwhere(nofz[:,1] == zmax)[0,0]
    nofz_info = make_nofz_info(nofz_info, '2dflens', nofz[argstart:argend+2,0], nofz[argstart:argend+1,3], nofz[argstart:argend+1,2])

    logger.info("Main process.")
    failed_cosmo = []
    for cosmo_label in cosmo_labels:
        logger.info(f"Start processing cosmo_label {cosmo_label}")

        # ### initialize cosmology
        cosmo_ccl = get_cosmo_from_file(sim_fmt.format(cosmo_label) + "params.yml", otype="ccl")
        ### set hod parameter output file
        hod_param_file = "cfgs/hod/cosmo_{:06d}_hod_params_alive.txt".format(cosmo_label)

        ### load halo catalog
        halo_file, hod_halocat, OmegaM = load_halocat(cosmo_label, ofmt='hod')
        halo_mass = hod_halocat.halo_table["halo_mvir"].value
        hod_params_alive = find_hod_params_alive(cosmo_label, halo_mass, hod_param_file, num_pool=10000, seedini=9782)

        if hod_params_alive is not None:
            gsamples_list = []
            ### populate galaxies
            for ihod, each_hod_params in enumerate(hod_params_alive):
                model_params_dict = dict(zip(model_params_names, each_hod_params))
                dict_of_gsamples = apply_hod(cosmo_label, halo_file, hod_halocat, OmegaM, model_params_dict)
                
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

                    galcone_output = make_survey(gal_pos, masks, cosmo_ccl, nofz_info, check_repeat=False)

                    # np.savetxt(out_dir + out_fmt.format(cosmo_label, ihod, iseed, "boss_north_2dflens_south"), galcone_output, fmt="%.4f %.4f %.4f %4f")
                    np.save(out_dir + out_fmt.format(cosmo_label, ihod, iseed, "boss_north_2dflens_south"), galcone_output)

            logger.info('Loops end')

        else:
            logger.warning("Number of HODs is not equal to nhod_per_cosmo. Skip this cosmology.")
            failed_cosmo.append(cosmo_label)
            continue

        ##################################               For test              #################################
        # dict_of_gsamples = apply_hod(cosmo_label, halo_file, hod_halocat, OmegaM, model_params_dict)
        # make_survey([dict_of_gsamples], masks, cosmo_ccl, nofz_info, check_repeat=True, cosmo_label=cosmo_label, outname="boss_north_2dflens_south")
        # make_survey([dict_of_gsamples], masks, cosmo_ccl, nofz_info, check_repeat=True, cosmo_label=cosmo_label, outname="full_sky")
        ########################################################################################################

    ### save the failed cosmo labels
    # np.savetxt("Failed_cosmo_labels.txt", np.asarray(failed_cosmo), fmt="%d")

    end = datetime.datetime.now()
    logger.info(f"Time elapsed: {end-start}")
