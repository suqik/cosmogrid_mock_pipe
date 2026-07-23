'''
Make foregournd galaxy catalog
'''

from dataclasses import dataclass, field
from scipy.stats import qmc, truncnorm
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R
import healpy as hp
import pymangle

from utils.io_func import *
from utils.mkfore_utils import *
from utils.mkback_utils import *

wdir = "/home/suchen/Program/CosmoGrid"

@dataclass
class PipeConfig:
    ### fixed siminfo
    Lbox:float
    Npart:int
    redshift:float
    nrlzs_per_cosmo: int = 1 # rlzs of initial conditions
    # HOD model parameters
    model: int = 2
    model_params_names: tuple[str, ...] = (
        "logMcut",
        "sigma_logM",
        "logM1",
        "k",
        "alpha",
        "fic",
    )
    nhod_per_cosmo: int = 10 # nums of hod populations per cosmology
    Num_ptcl_requirement: int = 12
    verbose: bool = True
    num_seeds: int = 1 # rlzs of each hod population
    init_seed: int = 33000
    ngal_ref: float = 3.5e-4
    z_space: bool = False

    # HOD parameter sampling
    param_prior_low: NDArray[np.float64] = field(
        default_factory=lambda: np.array(
            [13.0, 0.1, 13.0, 0.0, 0.0],
            dtype=np.float64,
        )
    )
    param_prior_high: NDArray[np.float64] = field(
        default_factory=lambda: np.array(
            [13.6, 0.6, 15.0, 10.0, 1.5],
            dtype=np.float64,
        )
    )

    # Lightcone redshift range
    zmin_lightcone: float = 0.0
    zmax_lightcone: float = 1.0
    ctr_lightcone: list[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0]
    )
    rsd_lightcone: bool = True

    # n(z)
    nofz_method: str = "downsample"

    sigma_e: float = 0.3
    seed_SN: int = 0
    sigma_phz: float = 0.01
    seed_Phz: int = 26120

    dive_exec_path:str = "/home/suchen/applications/DIVE/DIVE"

class CatalogLoader:
    def __init__(self, config:PipeConfig):
        self.config = config

    def load_pkd_halocat(self, halo_fname, 
                         cosmo=None,
                         ofmt='hod', clean=True):
        '''
        Load halo catalog, and transfer to HOD type
        '''

        Lbox = self.config.Lbox
        redshift = self.config.redshift
        Npart = self.config.Npart
        
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
            assert cosmo is not None
            OmegaM = cosmo.omega_x(a=1.0, species='matter')
            pmass = rhoc0*OmegaM*(Lbox/Npart)**3 # Msun/h
            ## Initialize HOD model class
            halocat = pkd_to_hod_type(pkd_halo_infos, cosmo=cosmo, pmass=pmass, boxsize=Lbox, redshift=redshift)
            
            return halocat

class HODPopulator:
    '''
    Class for populate galaxies on halo
    '''

    def __init__(self, config:PipeConfig):
        self.config = config

    @property
    def num_params(self):
        return len(self.config.model_params_names)
    
    def find_hod_params(
            self, hod_halocat,
            num_pool=30000, seedini=9782, seed_offset=0
            ):
        halo_mass = hod_halocat.halo_table["halo_mvir"].value
        Lbox = self.config.Lbox
        redshift = self.config.redshift
        ## Sample HOD parameters
        idx = 0
        seed = seedini + seed_offset

        hod_params_pool = self._open_params_pool(num_pool, seed)
        
        ## Main loop to find HOD parameters that matches reference galaxy number density

        while(idx < num_pool):
            curr_hod_params = hod_params_pool[idx,:]
            # try:
            #     curr_hod_params = hod_params_pool[idx,:]
            # except:
            #     warnings.warn("Found {} HOD parameters that matches reference galaxy number density.".format(count))
            #     break

            if self.config.model == 0:
                ngal_mock, Nsat_frac = get_ngal(
                    halo_mass=halo_mass, Lbox=Lbox, redshift=redshift,
                    model_lb=self.config.model, model_params_names=self.config.model_params_names, hod_param_vals=curr_hod_params, 
                )
                
                if np.abs(ngal_mock - self.config.ngal_ref) < 0.1 and Nsat_frac < 0.3: # avoid too many satellite galaxies in one halo
                    idx += 1
                    hod_params_alive = curr_hod_params
                else:
                    idx += 1
                    continue

            ## update fic
            if self.config.model == 2 or self.config.model == 3 or self.config.model == 4:
                # ngal_mock, Nsat_frac = get_ngal(
                #     halo_mass=halo_mass, Lbox=Lbox, redshift=redshift,
                #     model_lb=self.config.model, model_params_names=self.config.model_params_names, hod_param_vals=curr_hod_params, 
                # )

                # f_ic = self.config.ngal_ref/ngal_mock

                # ### FIXME: lower bound of f_ic may need careful consideration.
                # if f_ic > 0 and f_ic <= 1.0: # and Nsat_frac < 0.3: # avoid too many satellite galaxies in one halo
                #     count += 1
                #     idx += 1
                #     ### here we append f_ic to construct total HOD parameters
                #     hod_params_alive.append(list(curr_hod_params)+[f_ic])
                # else:
                #     idx += 1
                #     continue

                hod_params_alive = list(curr_hod_params)+[1.0]
                idx += 1

        return hod_params_alive

    def _open_params_pool(self, num_pool, seed=None):
        model = self.config.model
        prior_low = self.config.param_prior_low
        prior_high = self.config.param_prior_high
        ### priors do not include f_ic
        lhc_sampler = qmc.LatinHypercube(d=len(prior_low), seed=seed)
        hod_params_pool = lhc_sampler.random(n=num_pool)
        hod_params_pool = qmc.scale(hod_params_pool, prior_low, prior_high)

        ### prior of SIMBIG 
        if model == 3:
            mu = 1.0
            sigma = 0.5
            lower_bound = prior_low[4]
            upper_bound = prior_high[4]
            hod_params_pool[:,4] = truncnorm(
                (lower_bound - mu)/sigma, (upper_bound - mu)/sigma, loc=mu, scale=sigma
                ).rvs(size=num_pool)
            
        return hod_params_pool
    
    def populate_galaxies(self, hod_halocat, model_params_dict, indx, OmegaM=None):
        if self.config.z_space:
            assert OmegaM is not None

        HOD_Model = ModelClass(
            ["dummy"], [hod_halocat],
            pipe_config=self.config,
            OmegaM=OmegaM
        )
        dict_of_gsamples = HOD_Model.populate_mock(model_params_dict, ref_num_dens=self.config.ngal_ref, indx=indx, ifcheck=False)

        return dict_of_gsamples
    
    def get_galaxy_features(self, galaxy_arr, features:list=["pos"]):
        Lbox = self.config.Lbox
        x_c, y_c, z_c = galaxy_arr["x"], galaxy_arr["y"], galaxy_arr["z"]
        x_c = (x_c + Lbox) % Lbox
        y_c = (y_c + Lbox) % Lbox
        z_c = (z_c + Lbox) % Lbox
        gal_pos = np.c_[x_c, y_c, z_c]

        outputs = [gal_pos]

        if "vel" in features:
            vx_c, vy_c, vz_c = galaxy_arr["vx"], galaxy_arr["vy"], galaxy_arr["vz"]
            gal_vel = np.c_[vx_c, vy_c, vz_c]
            outputs += [gal_vel]
        if "gal_type" in features:
            gal_type = galaxy_arr["gal_type"]
            outputs += [gal_type]
        if "gal_host_halo_mvir" in features:
            gal_host_halo_mvir = galaxy_arr["halo_mvir"]
            outputs += [gal_host_halo_mvir]

        return tuple(outputs)
    
class SurveyGenerator:
    def __init__(self, config:PipeConfig, 
                 masks:dict,
                 nofzs:dict):
        
        self.config = config
        self.masks = masks
        self.nofzs = nofzs

    def _calc_radial_dist(self, cosmo:ccl.Cosmology, zs:tuple | float):
        assert isinstance((cosmo), ccl.Cosmology)

        print("\nCalculate cosmology-dependent quantities.", flush=True)
        
        hubble = cosmo.to_dict()["h"]
        zs = np.array(zs)
        chis = ccl.comoving_radial_distance(cosmo, 1./(1 + zs))*hubble # Mpc/h

        return chis
    
    def box_to_lightcone(self, cosmo, gal_pos, gal_adj_props={}):
        cosmo_ccl = cosmo
        Lbox = self.config.Lbox
        zmin = self.config.zmin_lightcone
        zmax = self.config.zmax_lightcone
        ctr = self.config.ctr_lightcone
        add_rsd = self.config.rsd_lightcone

        chi_min, chi_max = self._calc_radial_dist(cosmo_ccl, zs=(zmin, zmax))

        # ### DEBUG ###
        # ngal_box = len(gal_pos)
        # eff_num_den = ngal_box / Lbox / Lbox / Lbox
        # print("="*40)
        # print(f"Ngal in box: {eff_num_den*1e4:.3f} e-4")
        # print("="*40)
        
        '''
        Transform box data to lightcone data
        '''
        if add_rsd:
            assert ("gal_vel" in gal_adj_props.keys()), "Need gal_vel to compute RSD effect!"
        if gal_pos.ndim != 2:
            raise ValueError("gal_pos should be 2D array")
        if gal_pos.shape[1] != 3:
            raise ValueError("gal_pos should have 3 columns")

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
        else:
            Delta_chi = 0.0

        chi_min_used = np.maximum(chi_min - Delta_chi, 0.0)
        chi_max_used = chi_max + Delta_chi

        galcone, galcone_props = make_lightcone_tiles(gal_pos, 
                                    boxsize=Lbox, 
                                    chi_min=chi_min_used, 
                                    chi_max=chi_max_used, 
                                    ctr=ctr,
                                    other_props=adj_props
                                    )

        galcone_vector = galcone[:,:-1]
        galcone_gid = galcone[:,-1]
            
        galcone_ra, galcone_dec, galcone_z, phys_cut = Cart2Sph(cosmo_ccl, pos=galcone_vector)
        galcone_gid = galcone_gid[phys_cut]

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
        galcone_output["GID"] = galcone_gid

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

        del galcone_ra, galcone_dec, galcone_z, galcone_gid

        ### DEBUG ###
        # eff_vol = 4*np.pi/3*(chi_max**3-chi_min**3)
        # eff_num_den = len(galcone_output) / eff_vol
        # print("="*40)
        # print(f"Ngal in lightcone: {eff_num_den*1e4:.3f} e-4")
        # print("="*40)

        return galcone_output

    def apply_rotation(self, galcone, rot_degrees, inv=False, icoord='radec'):

        if icoord != 'vec' and icoord != 'radec':
            raise ValueError("icoord should be either 'vec' or 'radec'!")

        r = R.from_euler('zyx', rot_degrees, degrees=True)
        if inv:
            r = r.inv()
        
        if icoord == 'vec':
            galcone_rot = r.apply(galcone)
        if icoord == 'radec':
            galcone_rot = galcone.copy()
            galcone_rot['ra'] , galcone_rot['dec'] = hp.rotator.rotateDirection(
                rotmat=r.as_matrix(),
                theta=galcone['ra'],
                phi=galcone['dec'],
                lonlat=True
            )
            galcone_rot['ra'] = np.where(
                galcone_rot['ra'] < 0,
                galcone_rot['ra'] + 360,
                galcone_rot['ra']
            )

        return galcone_rot
    
    def _check_boss_survey(self, survey_name):
        assert survey_name in self.masks['boss_geom'].keys(), f"Masks of survey {survey_name} does not exist!"
        assert survey_name in self.nofzs.keys(), f"Nofz of survey {survey_name} does not exist!"

    def _check_2dflens_survey(self, survey_name):
        assert survey_name in self.masks['2dflens_geom'].keys(), f"Masks of survey {survey_name} does not exist!"
        assert survey_name in self.nofzs.keys(), f"Nofz of survey {survey_name} does not exist!"
    
    def gen_boss_like(self, galcone, survey_name, survey_label, make_nz=True):
        self._check_boss_survey(survey_name)
        galcone_curr = apply_boss_geometry(galcone, self.masks['boss_geom'][survey_name], self.masks['boss_masks'])
        if make_nz:
            galcone_curr = apply_nz(galcone_curr, self.nofzs[survey_name], 
                                    nofz_method=self.config.nofz_method, norm=False, add_rsd=self.config.rsd_lightcone)
        galcone_curr['survey'] = survey_label

        return galcone_curr
    
    def gen_boss_like_trim(self, galcone, survey_name, survey_label, make_nz=True):
        self._check_boss_survey(survey_name)
        galcone_curr = self.gen_boss_like(galcone, survey_name, survey_label, make_nz)
        ngc_sgc = survey_name.split("_")[-1]
        galcone_curr = apply_boss_lowze2e3_trim(galcone_curr, self.masks['boss_geom']['boss_lowz_{}'.format(ngc_sgc)])

        return galcone_curr
    
    def gen_2dflens_like(self, galcone, survey_name, survey_label, make_nz=True):
        self._check_2dflens_survey(survey_name)
        galcone_curr = apply_2dflens_geometry(galcone, self.masks['2dflens_geom'][survey_name])
        if make_nz:
            galcone_curr = apply_nz(galcone_curr, self.nofzs[survey_name], 
                                    nofz_method=self.config.nofz_method, norm=False, add_rsd=self.config.rsd_lightcone)
        
        galcone_curr['survey'] = survey_label

        return galcone_curr
    
class VoidFinder:
    def __init__(self, config:PipeConfig,):
        self.config = config

    def galcone_to_voidcone(self, galcone, cosmo_ccl, survey:int, dive_input:str, dive_output:str):

        galpos_cart = Sph2Cart(cosmo_ccl, ra=galcone['ra'], dec=galcone['dec'], z=galcone['zrsd']) 
        
        void_pos_cart, void_radii = find_void(galpos_cart, 
                                        dive_input=dive_input, 
                                        dive_output=dive_output)
        
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

    
class ShearAssigner:
    def __init__(self, config:PipeConfig, masks:dict, nofzs:dict):
        self.config = config
        self.masks = masks
        self.nofzs = nofzs

    def _check_mask(self, survey_name):
        if not survey_name in list(self.masks.keys()):
            raise ValueError(f"{survey_name} - like mask not found!")

    def _guess_mask_type(self, mask):
        if isinstance(mask, np.ndarray) and hp.isnpixok(len(mask)):
            mask_type = 'healpix'
        elif isinstance(mask, pymangle.Mangle):
            mask_type = 'mangle'
        else:
            raise ValueError("Cannot recognize the mask type!")
        
        return mask_type
        
    def _downsample_array(self, array_list, Ntarget):
        Nsource = len(array_list[0])
        assert Nsource >= Ntarget
        select = np.random.choice(np.arange(Nsource), Ntarget, replace=False)
        array_list_output = []
        for iarray in array_list:
            array_list_output.append(iarray[select])
        return tuple(array_list_output)
    
    def gen_gal_positions(self, ngal:float, survey_name:str, tomo_label:int, survey_label:int):
        mask = self.masks[survey_name]
        mask_type = self._guess_mask_type(mask)
        sigma_phz = self.config.sigma_phz
        seed_Phz = self.config.seed_Phz
        nofz = self.nofzs[f'tomo{tomo_label}']

        match mask_type:
            case "healpix":
                cat_ra, cat_dec = gen_angle_positions_from_healpix(ngal, mask)
            case "mangle":
                cat_ra, cat_dec = gen_angle_positions_from_mangle(ngal, mask)

        Ngal_curr = len(cat_ra)
        cat_z, cat_zph = gen_redshifts_from_nofz(Ngal_curr, nofz, photo_z_err=sigma_phz, seed=seed_Phz)
        Ngal_curr = len(cat_z)
        cat_ra, cat_dec = self._downsample_array([cat_ra, cat_dec], Ngal_curr)

        bg_galcat = np.zeros((Ngal_curr,), dtype=bgal_type)
        bg_galcat['ra'] = cat_ra
        bg_galcat['dec'] = cat_dec
        bg_galcat['z'] = cat_zph
        bg_galcat['z_true'] = cat_z
        bg_galcat['sigz'] = sigma_phz
        bg_galcat['tomo'] = tomo_label
        bg_galcat['survey'] = survey_label

        return bg_galcat
    
    def assign_shear(self, bg_galcat:np.ndarray, shear_map_dict:list):
        sigma_e = self.config.sigma_e
        seed_SN = self.config.seed_SN
        g1_pure, g2_pure, g1_noise, g2_noise = assign_shear_vals(bg_galcat['ra'], bg_galcat['dec'], bg_galcat['z_true'], shear_map_dict, sigma_e, seed_SN)

        bg_galcat['g1'] = g1_noise
        bg_galcat['g2'] = g2_noise
        bg_galcat['g1_pure'] = g1_pure
        bg_galcat['g2_pure'] = g2_pure

        return bg_galcat
    
    def rotate_pix(self, pix, nside, rot_degrees):
        r = R.from_euler('zyx', rot_degrees, degrees=True)
        norm_vec_x, norm_vec_y, norm_vec_z = hp.pix2vec(nside=nside, ipix=pix)
        norm_vec = np.c_[norm_vec_x, norm_vec_y, norm_vec_z]
        new_vec = r.apply(norm_vec)
        pix_new = hp.vec2pix(nside=nside, x=new_vec[:,0], y=new_vec[:,1], z=new_vec[:,2])
        
        return pix_new