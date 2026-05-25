''' 
From halo to galaxy. 
apply HOD and make lightcone.
'''

import sys
from astropy.table import Table
import numpy as np
import datetime
from loguru import logger

from utils.io_func import *
from utils.mkfore_utils import *
from mkfore_gal_routines import *

wdir = "/home/suchen/Program/CosmoGrid"

''' ================ 1. Simulation info ================ '''

sim_fmt = "/data3/suchen/CosmoGridV1/grid/cosmo_{:06d}/run_{:d}/"
halo_fmt = "pkd_halos/CosmoML.{:05d}.fofstats.0"
# redshift_label = 120 # corresponding to z~0.3
redshift_label = 110 # corresponding to z~0.51

rlz_label = 0 # realization label. Note only partial cosmologies have multiple rlzs. (~400)

lb_z_file = "/data3/suchen/CosmoGridV1/label_z_table.txt"
lb_z_tb = np.loadtxt(lb_z_file)

Lbox = 900.0
Nside = 832 # Npart = Nside**3
redshift = lb_z_tb[redshift_label,1]

survey_part_names = []
survey_specify = ""
### Use different redshift range depending on redshift_label
if redshift_label == 120:
    zmin = 0.2
    zmax = 0.4
    zbin_name = "lowz"
    survey_part_names += ['boss_lowz', 'boss_lowze2', 'boss_lowze3']
    survey_specify += "boss_north"

    survey_part_names += ['2dflens_south']
    survey_specify += "_2dflens_south"

elif redshift_label == 110:
    # zmin = 0.4
    zmin = 0.45
    zmax = 0.6
    zbin_name = "cmass"

    # survey_part_names += ['boss_cmass']
    # survey_specify += "boss_north"

    # survey_part_names += ['2dflens_south']
    # survey_specify += "_2dflens_south"

    survey_part_names += ['boss_cmass_sgc']
    survey_specify += "boss_south"

print(survey_specify)

''' ================ 2. Mask file info ================ '''

geom_fname_dict = {}

### boss NGC geometry
mask_boss_fdir = f"{wdir}/catalogs/masks/boss_geom/"

geom_fname_dict['boss_lowz'] = mask_boss_fdir + "mask_DR12v5_LOWZ_North.ply"
geom_fname_dict['boss_lowze2'] = mask_boss_fdir + "mask_DR12v5_LOWZE2_North.ply"
geom_fname_dict['boss_lowze3'] = mask_boss_fdir + "mask_DR12v5_LOWZE3_North.ply"
geom_fname_dict['boss_cmass'] = mask_boss_fdir + "mask_DR12v5_CMASS_North.ply"
geom_fname_dict['boss_cmass_sgc'] = mask_boss_fdir + "mask_DR12v5_CMASS_South.ply"

### mask files corresponding to observational effects
mask_boss_fname_list = [
    mask_boss_fdir + "badfield_mask_postprocess_pixs8.ply",
    mask_boss_fdir + "badfield_mask_unphot_seeing_extinction_pixs8_dr12.ply",
    mask_boss_fdir + "allsky_bright_star_mask_pix.ply",
    mask_boss_fdir + "bright_object_mask_rykoff_pix.ply", 
    mask_boss_fdir + "collision_priority_mask_dr12.ply", 
    mask_boss_fdir + "centerpost_mask_dr12.ply"
]

### 2dflens south geometry
geom_fname_dict['2dflens_south'] = f"{wdir}/catalogs/masks/2dflens_geom/2dFLens_mask_weight_South.fits"

''' ================ 3. n(z) file info ================ '''

### n(z) files
nofz_method = "const" # Can be `rank`, `downsample`, or `const`

nz_fname_dict = {}
nz_fbase = f"{wdir}/catalogs/NOfZ/lens/"

nz_fname_dict['boss_lowz'] = nz_fbase + "nbar_DR12v5_LOWZ_North_om0p31_Pfkp10000.dat"
nz_fname_dict['boss_lowze2'] = nz_fbase + "nbar_DR12v5_LOWZE2_North_om0p31_Pfkp10000.dat"
nz_fname_dict['boss_lowze3'] = nz_fbase + "nbar_DR12v5_LOWZE3_North_om0p31_Pfkp10000.dat"
nz_fname_dict['boss_cmass'] = nz_fbase + "nbar_DR12v5_CMASS_North_om0p31_Pfkp10000.dat"
nz_fname_dict['boss_cmass_sgc'] = nz_fbase + "nbar_DR12v5_CMASS_South_om0p31_Pfkp10000.dat"
nz_fname_dict['2dflens_south'] = nz_fbase + "nbar_2dFLens_south_data.dat"

''' ================ 4. HOD setup ================ '''

model = 2
### Note `fic` should be last of the `model_params_names` since 
### it is a derived parameter and not considered in the prior
if model == 2:
    model_params_names = 'logMcut', 'sigma_logM', 'logM1', 'k', 'alpha', 'fic' # for model == 2
if model == 3:
    model_params_names = 'logMcut', 'sigma_logM', 'logM1', 'logM0', 'alpha', 'fic' # SIMBIG HOD params, for model==3
if model == 4:
    model_params_names = 'logMcut', 'sigma_logM', 'logM1', 'k', 'alpha', 'AB_cen','AB_sat', 'fic' # for model == 4

num_params = len(model_params_names) # Number of parameters of HOD model
nhod_per_cosmo = 10 # Number of varied HOD parameter values per cosmology
Num_ptcl_requirement = 12
verbose = True
num_seeds = 1
init_seed = 33000 ## initial seed for generating galaxy catalog
z_space = False
ngal_ref = 2.05e-4

LOAD_HOD_PAR = False # if load exist hod params
POPULATE = True # if populate galaxy

if model == 2:
    # hod_param_out = f"{wdir}/cfgs/hod/hod_5params_dict_high_ngal_wcosmo2_ws8.json"    
    # hod_param_out = f"{wdir}/cfgs/hod/hod_5params_dict_Nsat1000_{zbin_name}.json"
    hod_param_out = f"{wdir}/cfgs/hod/hod_5params_dict_free_ngal_{zbin_name}.json"

if model == 4:
    raise NotImplementedError
    # hod_param_out = f"{wdir}/cfgs/hod/hod_7params_dict_high_ngal_wcosmo2_ws8_wAB.json"

''' ================ 5. Running modes specifications ================ '''

### 5.1 If apply RSD effect
### Note: Do not recommend to turn off `ADD_RSD`
### since when turn on RSD, the real redshift will
### als be saved int the catalog.
ADD_RSD = True

### 5.2 If using rotations to augment data
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

### 5.3 Running modes, can only activate one of these three
HALO_ONLY = False # only use halo, which preserve the ngal but not G-H connection
VARY_HOD = True # preserve the ngal, as well as vary G-H connection

if VARY_HOD:
    #### prior for model == 2
    if model == 0 or model == 2:
        param_prior_low  = np.array([12.5, 1e-5, 12.5, 0.00, 0.0])
        param_prior_high = np.array([13.5, 3.00, 15.0, 10.0, 2.0])

    #### prior from SIMBIG, for model == 3
    if model == 3:
        param_prior_low = np.array([12., 0.1, 13., 13., 0.0])
        param_prior_high = np.array([14., 0.6, 15., 15., 1.5])

    # #### prior for model == 4
    if model == 4:
        param_prior_low  = np.array([12.5, 1e-5, 12.5, 0.00, 0.0, -1.0, -1.0])
        param_prior_high = np.array([13.5, 3.00, 15.0, 10.0, 2.0,  1.0,  1.0])


''' ================ 6. Catalog output files ================ '''

dirbase = "Free_NGAL"
if ADD_RSD:
    dirbase += "_wrsd"

if model == 4:
    dirbase += "_wAB"

SAVE_BOX = False
box_out_dir = f"/data2/suchen/CosmoGrid/{dirbase}/HOD_{zbin_name}_box/grid/"
box_out_fmt = "cosmo_{:06d}_run_{:d}_HOD_{:d}_run_{:d}.fits"
if SAVE_BOX and not os.path.exists(box_out_dir):
    os.makedirs(box_out_dir)

SAVE_CONE = True
out_dir = f"/data2/suchen/CosmoGrid/{dirbase}/HOD_{zbin_name}_SGC/grid/"
out_fmt = "cosmo_{:06d}_run_{:d}_HOD_{:d}_run_{:d}_{:s}.fits"
if SAVE_CONE and not os.path.exists(out_dir):
    os.makedirs(out_dir)

''' ================ 7. Show config info ================ '''

logger.info(f"used simulation redshift: {redshift:.4f}")
logger.info(f"Simulating redshift range: {zmin:.4f} - {zmax:.4f}")
logger.info(f"RSD: {ADD_RSD}")

if HALO_ONLY:
    logger.info("HALO only mode")
    logger.info(f"Ngal ref: {ngal_ref*1e4:.2f} e-4")

if VARY_HOD:
    logger.info("VARY_HOD mode")
    logger.info(f"Ngal ref: {ngal_ref*1e4:.2f} e-4")
    if LOAD_HOD_PAR:
        logger.info(f"Load HOD pars from: {hod_param_out}")
    else:
        logger.info(f"HOD prior low: {param_prior_low}")
        logger.info(f"HOD prior high: {param_prior_high}")
        logger.info(f"Will save cosmo and HOD pars to: {hod_param_out}")
    if not POPULATE:
        logger.info("Do not populate galaxy catalogs. Only find HOD parameters")

if ROT:
    logger.info("Use rotation mode")
    logger.info(f"Rotation degrees: {rot_degrees_list}")


''' ================================================================================================================== '''


def main(cosmo_label, cosmo_ccl, hod_halocat, cosmo_hod_dict_tot):
    ### run finding HOD params for this cosmo
    if not LOAD_HOD_PAR:
        ### save cosmo par in cosmo_hod_dict
        cosmo_hod_dict_tot['cosmo{:06d}'.format(cosmo_label)] = {}
        cosmo_hod_dict_tot['cosmo{:06d}'.format(cosmo_label)]['cpar'] = {
            "Om": cosmo_ccl['Omega_c'] + cosmo_ccl['Omega_b'],
            "Ob": cosmo_ccl['Omega_b'],
            "H0" : cosmo_ccl['h']*100,
            "ns": cosmo_ccl['n_s'],
            "s8": cosmo_ccl['sigma8'],
            "w0": cosmo_ccl['w0']
        }

        ### find HOD params

        logger.info("Start finding HOD parameters")

        hod_params_alive = run_find_hod_params(
            hod_cat=hod_halocat, 
            nhod_per_cosmo=nhod_per_cosmo,
            model_lb=model,
            model_params_names=model_params_names,
            param_prior_low=param_prior_low,
            param_prior_high=param_prior_high,
            ngal_ref=ngal_ref,
            seed_offset=cosmo_label
        )

        ### save HOD par in cosmo_hod_dict
        if len(hod_params_alive) == nhod_per_cosmo:
            for i in range(nhod_per_cosmo):
                cosmo_hod_dict_tot['cosmo{:06d}'.format(cosmo_label)]['HOD{:d}'.format(i)] = hod_params_alive[i]

    ### pick HOD params corresponding to this cosmo
    else:
        if len(cosmo_hod_dict_tot['cosmo{:06d}'.format(cosmo_label)]) == int(nhod_per_cosmo + 1):
            hod_params_alive = [cosmo_hod_dict_tot['cosmo{:06d}'.format(cosmo_label)]['HOD{}'.format(jhod)] \
                                            for jhod in range(nhod_per_cosmo)]
        else:
            hod_params_alive = []

    
    if len(hod_params_alive) == 0:
        return hod_params_alive
    
    if POPULATE:

        logger.info("Populating galaxy catalogs")

        ### populate galaxies
        ### first populate in box
        for ihod, each_hod_params in enumerate(hod_params_alive):
            #####################  DEBUG  ###################
            logger.info("HOD {}".format(ihod))
            ngal_mock, _ = get_ngal(
                halo_mass=hod_halocat.halo_table["halo_mvir"].value, Lbox=Lbox, redshift=redshift,
                model_lb=model, model_params_names=model_params_names, hod_param_vals=each_hod_params, 
            )
            print("="*40)
            print(f"analytical ngal_mock: {ngal_mock*1e4:.3f} e-4")
            print("="*40)
            #################################################

            model_params_dict = dict(zip(model_params_names, each_hod_params))

            dict_of_gsamples = run_apply_hod(
                hod_halo_cat=hod_halocat,
                model_lb=model,
                model_params_names=model_params_names,
                model_params_dict=model_params_dict,
                OmegaM=cosmo_ccl['Omega_c'] + cosmo_ccl['Omega_b'],
                init_seed=init_seed,
                num_seeds=num_seeds,
                z_space=z_space,
                Num_ptcl_requirement=Num_ptcl_requirement,
                ngal_ref=ngal_ref,
                indx=cosmo_label*ihod
            )

            for iseed, key in enumerate(dict_of_gsamples.keys()):
                
                ### get galaxy positions
                x_c, y_c, z_c = dict_of_gsamples[key]["x"], dict_of_gsamples[key]["y"], dict_of_gsamples[key]["z"]
                x_c = (x_c + Lbox) % Lbox
                y_c = (y_c + Lbox) % Lbox
                z_c = (z_c + Lbox) % Lbox
                gal_pos = np.c_[x_c, y_c, z_c]
                
                ### get galaxy velocity
                vx_c, vy_c, vz_c = dict_of_gsamples[key]["vx"], dict_of_gsamples[key]["vy"], dict_of_gsamples[key]["vz"]
                gal_vel = np.c_[vx_c, vy_c, vz_c]
                gal_type = dict_of_gsamples[key]["gal_type"]
                gal_host_halo_mvir = dict_of_gsamples[key]["halo_mvir"]

                ### save to file if necessary
                if SAVE_BOX:
                    galbox_fname = os.path.join(box_out_dir, box_out_fmt.format(cosmo_label, rlz_label, ihod, iseed))
                    # np.save(galbox_fname, np.c_[gal_pos, gal_vel, gal_type, gal_host_halo_mvir])
                    tmp_tb = Table()
                    tmp_tb['x'] = gal_pos[:,0]
                    tmp_tb['y'] = gal_pos[:,1]
                    tmp_tb['z'] = gal_pos[:,2]
                    tmp_tb['gal_type'] = gal_type
                    tmp_tb['halo_mvir'] = gal_host_halo_mvir
                    tmp_tb.write(galbox_fname, overwrite=True)
                    del tmp_tb
                
                ### box to lightcone
                galcone_output = run_box_to_lightcone(
                    gal_pos = gal_pos,
                    cosmo_ccl = cosmo_ccl,
                    Lbox = Lbox,
                    zmin = zmin,
                    zmax = zmax,
                    add_rsd = ADD_RSD,
                    gal_adj_props={
                        'gal_vel': gal_vel,
                        'gal_type': gal_type,
                        'gal_host_halo_mvir': gal_host_halo_mvir
                    }
                )

                ### survey geometry and mask
                if not ROT:
                    galcone_output = run_apply_geometry(
                        galcone_output,
                        survey_part_names=survey_part_names,
                        masks=masks,
                        nofz_info=nofz_info,
                        nofz_method=nofz_method,
                        add_rsd=ADD_RSD
                    )
                else:
                    raise NotImplementedError
                
                ### save to file
                if SAVE_CONE:
                    gal_fname = os.path.join(out_dir, out_fmt.format(cosmo_label, rlz_label, ihod, iseed, survey_specify))
                    # np.save(gal_fname, galcone_output)
                    tmp_tb = Table(galcone_output)
                    tmp_tb.write(gal_fname, overwrite=True)
                    del tmp_tb

    return cosmo_hod_dict_tot

''' ================================================================================================================== '''


if __name__ == "__main__":

    TEST_MODE = False

    if len(sys.argv) > 1 and sys.argv[-1] == "test":
        TEST_MODE = True

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # >>> =================================       preparation       ================================= <<<

    ### prepare masks
    masks = prepare_masks(
        survey_part_names,
        geom_fname_dict=geom_fname_dict,
        mask_boss_fname_list=mask_boss_fname_list
    )

    ### prepare nofz info
    if nofz_method != "const":
        nofz_info = prepare_nofz(
            survey_part_names,
            zmin=zmin, zmax=zmax,
            nz_fname_dict=nz_fname_dict
        )
    else:
        nofz_info = None


    # ====================================================================================================
    # ========================               SINGLE COSMOLOGY LOOP              ==========================
    # ====================================================================================================

    ### specify output directory
    hod_param_out = f"{wdir}/cfgs/hod/hod_5params_dict_const_ngal_{zbin_name}_fiducial.json"
    box_out_dir = f"/data2/suchen/CosmoGrid/{dirbase}/HOD_{zbin_name}_box/fiducial/"
    box_out_fmt = "cosmo_{:06d}_run_{:d}_HOD_{:d}_run_{:d}.npy"

    out_dir = f"/data2/suchen/CosmoGrid/{dirbase}/HOD_{zbin_name}/fiducial/"
    out_fmt = "cosmo_{:06d}_run_{:d}_HOD_{:d}_run_{:d}_{:s}.npy"

    if not os.path.isdir(out_dir):
        logger.info(f"Create dictionary: {out_dir}")
        os.makedirs(out_dir)

    ### Initialize cosmo_hod dict
    cosmo_hod_dict_tot = initial_hod_param_dict(load_hod_par=LOAD_HOD_PAR, hod_param_file=hod_param_out)

    ### For fiducial cosmology, we assure cosmo_label = 0
    cosmo_label = 0

    ### specify cosmo par file name & halo file name
    cpar_fname = "/data3/suchen/CosmoGridV1/fid/run_0000/params.yml"
    halo_fname = "/data3/suchen/CosmoGridV1/fid/run_0000/pkd_halos/CosmoML.{:05d}.fofstats.0".format(redshift_label)

    ### initialize cosmology
    cosmo_ccl = get_cosmo_from_file(cpar_fname, otype="ccl")

    ### load halo catalog for HOD 
    hod_halocat = load_halocat(cpar_fname, halo_fname, Lbox=Lbox, Nside=Nside, redshift=redshift, ofmt='hod')

    cosmo_hod_dict_tot = main(
        cosmo_label, 
        cosmo_ccl, 
        hod_halocat, 
        cosmo_hod_dict_tot
        )
    
    if not LOAD_HOD_PAR:
        save_hod_param_dict([cosmo_hod_dict_tot], hod_param_out)

    # ====================================================================================================
    # ========================               MULTI COSMOLOGY LOOP              ===========================
    # ====================================================================================================

    # if rank == 0:
        
    #     logger.info("Read cosmo labels")

    #     cosmo_labels_tot = get_cosmo_name_list_original("/data3/suchen/CosmoGridV1/grid/dirnames.txt")
    #     ######## For Test #######
    #     if TEST_MODE:
    #         cosmo_labels_tot = [172798]
    #     #########################
    #     k, m = divmod(len(cosmo_labels_tot), size)
    #     chunks = [cosmo_labels_tot[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(size)]

    #     if not os.path.isdir(out_dir):
    #         logger.info(f"Create dictionary: {out_dir}")
    #         os.makedirs(out_dir)
    # else:
    #     chunks = None

    # if rank == 0:

    #     logger.info("Scattering labels")

    # cosmo_labels = comm.scatter(chunks, root=0)

    # start = datetime.datetime.now()


    # logger.info("Main process.")


    # cosmo_hod_dict_tot = initial_hod_param_dict(load_hod_par=LOAD_HOD_PAR, hod_param_file=hod_param_out)

    # ### Loop from cosmo_labels
    # for cosmo_label in cosmo_labels:

    #     logger.info(f"Start processing cosmo_label {cosmo_label}")

    #     ### specify cosmo par file name & halo file name
    #     cpar_fname = os.path.join(sim_fmt.format(cosmo_label, rlz_label), "params.yml")
    #     halo_fname = os.path.join(sim_fmt.format(cosmo_label, rlz_label), halo_fmt.format(redshift_label))

    #     ### initialize cosmology
    #     cosmo_ccl = get_cosmo_from_file(cpar_fname, otype="ccl")

    #     ### load halo catalog for HOD 
    #     hod_halocat = load_halocat(cpar_fname, halo_fname, Lbox=Lbox, Nside=Nside, redshift=redshift, ofmt='hod')

    #     cosmo_hod_dict_tot = main(
    #         cosmo_label, 
    #         cosmo_ccl, 
    #         hod_halocat, 
    #         cosmo_hod_dict_tot
    #         )

    # ### if this is the first run, save HOD params
    # if not LOAD_HOD_PAR:
    #     all_cosmo_hod_dict = comm.gather(cosmo_hod_dict_tot, root=0)

    #     if rank == 0:
    #         save_hod_param_dict(all_cosmo_hod_dict, hod_param_out)

    # end = datetime.datetime.now()
    # logger.info(f"Time elapsed: {end-start}")

    # ====================================================================================================

