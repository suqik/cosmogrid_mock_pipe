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
    seed_pos: int = 0

    dive_exec_path:str = "/home/suchen/applications/DIVE/DIVE"

class CatalogLoader:
    def __init__(self, config:PipeConfig):
        self.config = config

    def load_rstar_halocat(
            self, halo_fname,
            cosmo=None,
            ofmt='hod', clean=True, host_only=True):
        '''
        Load a Rockstar halo catalog and optionally transfer it to HOD type.

        The ``clean`` argument is retained for API compatibility with
        ``load_pkd_halocat``; Rockstar halos are not filtered by ``rHalf``.
        '''
        Lbox = self.config.Lbox
        redshift = self.config.redshift
        Npart = self.config.Npart

        print(f"Load Rockstar halo from file {halo_fname}")

        rstar_halo_infos = get_rstar_halo_attrs(
            halo_fname,
            attrs=[
                "pos",
                "vel",
                "mass",
                "rvir",
                "rHalf",
                "concentration",
                "ID",
                "PID",
            ],
            host_only=host_only,
        )

        if ofmt == 'rstar':
            return rstar_halo_infos

        if ofmt == 'hod':
            assert cosmo is not None
            OmegaM = cosmo.omega_x(a=1.0, species='matter')
            pmass = rhoc0*OmegaM*(Lbox/Npart)**3  # Msun/h
            halocat = rstar_to_hod_type(
                rstar_halo_infos,
                pmass=pmass,
                boxsize=Lbox,
                redshift=redshift,
            )

            return halocat

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
        target_count = int(self.config.nhod_per_cosmo)
        if target_count <= 0:
            raise ValueError("nhod_per_cosmo must be positive")
        if num_pool < target_count:
            raise ValueError(
                "num_pool must be at least nhod_per_cosmo "
                f"({target_count})"
            )

        ## Sample HOD parameters
        seed = seedini + seed_offset

        hod_params_pool = np.asarray(self._open_params_pool(num_pool, seed))
        if len(hod_params_pool) < target_count:
            raise ValueError(
                "sampled HOD pool contains fewer rows than nhod_per_cosmo"
            )
        hod_params_alive = []
        
        ## Main loop to find HOD parameters that matches reference galaxy number density

        for curr_hod_params in hod_params_pool:
            if self.config.model == 0:
                ngal_mock, Nsat_frac = get_ngal(
                    halo_mass=halo_mass, Lbox=Lbox, redshift=redshift,
                    model_lb=self.config.model, model_params_names=self.config.model_params_names, hod_param_vals=curr_hod_params, 
                )
                
                if np.abs(ngal_mock - self.config.ngal_ref) < 0.1 and Nsat_frac < 0.3: # avoid too many satellite galaxies in one halo
                    hod_params_alive.append(curr_hod_params.tolist())
                else:
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

                hod_params_alive.append(list(curr_hod_params)+[1.0])
            elif self.config.model != 0:
                raise NotImplementedError(
                    "HOD parameter sampling is not implemented for "
                    f"model {self.config.model}"
                )

            if len(hod_params_alive) == target_count:
                break

        if len(hod_params_alive) != target_count:
            raise RuntimeError(
                "HOD parameter pool exhausted after finding "
                f"{len(hod_params_alive)} of {target_count} requested rows"
            )

        return np.asarray(hod_params_alive, dtype=float)

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
                ).rvs(size=num_pool, random_state=seed)
            
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
        if (
                "host_halo_mvir" in features
                or "gal_host_halo_mvir" in features):
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
    
    def box_to_lightcone(self, cosmo, gal_pos, gal_adj_props=None):
        if gal_adj_props is None:
            gal_adj_props = {}
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
        for prop_name, prop_data in gal_adj_props.items():
            if len(prop_data) != len(gal_pos):
                raise ValueError(
                    f"adjacent property {prop_name} has {len(prop_data)} "
                    f"rows, expected {len(gal_pos)}"
                )

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

        galcone_output = np.zeros((len(galcone_ra),), dtype=fgal_type)
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
                                        exec_path=self.config.dive_exec_path,
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
    def __init__(self, config:PipeConfig, masks:dict, nofzs:dict,
                 mass_maps:dict=None, z_to_mass_label:dict=None):
        self.config = config
        self.masks = masks
        self.nofzs = nofzs
        # mass maps for density-based position sampling:
        # mass_maps: {label: full-sky overdensity map (HEALPix RING)}
        # z_to_mass_label: {source-shell redshift: mass-map label}
        self.mass_maps = mass_maps
        self.z_to_mass_label = z_to_mass_label

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
    
    def gen_gal_positions(self, ngal:float, survey_name:str, tomo_label:int, survey_label:int,
                          method:str="random", bias:float=1.0, seed:int=None):
        '''
        Generate background galaxy angular positions.

        Parameters
        ----------
        method: str
            "random": uniformly sample positions inside the survey mask
            (original behavior). "density": Poisson-sample positions from
            the mass maps, i.e. source clustering.
        bias: float
            Linear galaxy bias used by method="density"; the expected
            number of galaxies per cell scales as (1 + bias * delta).
        seed: int
            RNG seed shared by the Poisson cell counts and the uniform
            in-cell placement (and the in-shell redshift draws).
        '''
        if method not in ("random", "density"):
            raise ValueError(f"unsupported position sampling method: {method}")

        mask = self.masks[survey_name]
        nofz = self.nofzs[f'tomo{tomo_label}']

        if method == "random":
            return self._gen_gal_position_random(ngal, mask, nofz,
                                                 tomo_label, survey_label, seed)
        return self._gen_gal_position_from_map(ngal, mask, nofz,
                                               tomo_label, survey_label,
                                               bias, seed)

    def _gen_gal_position_random(self, ngal:float, mask, nofz:dict,
                                 tomo_label:int, survey_label:int, seed:int=None):
        if seed is not None:
            np.random.seed(seed)
        sigma_phz = self.config.sigma_phz
        seed_Phz = self.config.seed_Phz

        mask_type = self._guess_mask_type(mask)
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

    def _gen_gal_position_from_map(self, ngal:float, mask, nofz:dict,
                                   tomo_label:int, survey_label:int,
                                   bias:float, seed:int=None):
        '''
        Source-clustered position sampling (GLASS-style): the expected
        number of galaxies per cell is ``ngal * A_cell * f_shell *
        (1 + bias * delta)``, where ``delta`` is the overdensity of the
        mass map assigned to the source shell. Cell counts are Poisson
        draws; positions are uniform within each cell. The same RNG seed
        drives the Poisson counts, the in-cell placement and the in-shell
        redshift draws (photo-z error keeps using seed_Phz).
        '''
        if self.mass_maps is None or self.z_to_mass_label is None:
            raise ValueError(
                "mass_maps and z_to_mass_label are required "
                "for method='density'"
            )
        if self._guess_mask_type(mask) != 'healpix':
            raise ValueError(
                "method='density' currently supports healpix masks only"
            )
        if not self.mass_maps:
            raise ValueError("mass_maps is empty")

        cell_nside = hp.npix2nside(len(next(iter(self.mass_maps.values()))))
        mask_nside = hp.npix2nside(len(mask))
        if cell_nside < mask_nside:
            raise ValueError(
                "mass-map resolution must be finer than the survey mask"
            )
        # upsample the survey mask onto the mass-map grid
        # (majority rule for boundary cells)
        mask_up = hp.ud_grade(mask.astype(np.float64), cell_nside,
                              order_in="RING", order_out="RING")
        cells = np.argwhere(mask_up >= 0.5).flatten()
        if len(cells) == 0:
            raise ValueError("no survey cells after mask upsampling")
        cell_area_arcmin2 = (
            hp.nside2pixarea(cell_nside) * (180.0/np.pi)**2 * 3600.0
        )

        rng = np.random.default_rng(seed)

        # source shells and their edges (same convention as assign_shear_vals)
        shell_zs = np.array(sorted(self.z_to_mass_label.keys()), dtype=float)
        if len(shell_zs) < 2:
            raise ValueError(
                "z_to_mass_label must contain at least two source shells"
            )
        dz_last = shell_zs[-1] - shell_zs[-2]
        edges = np.concatenate([
            [0.0],
            0.5 * (shell_zs[1:] + shell_zs[:-1]),
            [shell_zs[-1] + dz_last],
        ])

        # rebin the tomo n(z) histogram into the source-shell edges
        zedges = np.asarray(nofz['zedges'])
        nz_vals = np.asarray(nofz['nz'])
        f_shell = np.zeros(len(edges) - 1)
        for i, nzi in enumerate(nz_vals):
            if nzi <= 0:
                continue
            a, b = zedges[i], zedges[i + 1]
            for j, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
                overlap = min(b, hi) - max(a, lo)
                if overlap > 0:
                    f_shell[j] += nzi * overlap / (b - a)
        fsum = f_shell.sum()
        if fsum <= 0:
            raise ValueError(
                f"tomo {tomo_label} n(z) has no overlap with source shells"
            )
        f_shell /= fsum

        ra_parts, dec_parts, ztrue_parts = [], [], []
        nclipped = 0
        for ishell, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            f_s = f_shell[ishell]
            if f_s <= 0:
                continue
            label = self.z_to_mass_label[float(shell_zs[ishell])]
            if label not in self.mass_maps:
                raise ValueError(f"mass map for label {label} not loaded")
            delta = self.mass_maps[label][cells]
            lam = ngal * cell_area_arcmin2 * f_s * (1.0 + bias * delta)
            nclipped += int(np.sum(lam < 0))
            lam = np.clip(lam, 0.0, None)
            counts = rng.poisson(lam)
            sel = counts > 0
            if not np.any(sel):
                continue
            ra_c, dec_c = self._uniform_in_cells(cells[sel], counts[sel],
                                                 cell_nside, rng)
            z_true = rng.uniform(lo, hi, len(ra_c))
            ra_parts.append(ra_c)
            dec_parts.append(dec_c)
            ztrue_parts.append(z_true)

        if nclipped:
            print(
                f"Tomo {tomo_label}: {nclipped} cell-shell expectations "
                "clipped at 0", flush=True,
            )
        if not ra_parts:
            raise ValueError(f"no galaxies sampled for tomo {tomo_label}")

        cat_ra = np.concatenate(ra_parts)
        cat_dec = np.concatenate(dec_parts)
        cat_z = np.concatenate(ztrue_parts)

        # photo-z error, same recipe as gen_redshifts_from_nofz
        sigma_phz = self.config.sigma_phz
        if sigma_phz is not None:
            rng_phz = np.random.default_rng(self.config.seed_Phz)
            cat_zph = cat_z + rng_phz.normal(0.0, sigma_phz, len(cat_z))
            phys_cut = cat_zph > 0
            cat_ra = cat_ra[phys_cut]
            cat_dec = cat_dec[phys_cut]
            cat_z = cat_z[phys_cut]
            cat_zph = cat_zph[phys_cut]
        else:
            cat_zph = cat_z

        bg_galcat = np.zeros((len(cat_ra),), dtype=bgal_type)
        bg_galcat['ra'] = cat_ra
        bg_galcat['dec'] = cat_dec
        bg_galcat['z'] = cat_zph
        bg_galcat['z_true'] = cat_z
        bg_galcat['sigz'] = sigma_phz
        bg_galcat['tomo'] = tomo_label
        bg_galcat['survey'] = survey_label

        return bg_galcat

    @staticmethod
    def _uniform_in_cells(cells, counts, nside, rng, max_rounds=20):
        '''
        Place ``counts[i]`` points uniformly inside HEALPix cell
        ``cells[i]`` via rejection sampling: points are drawn uniformly
        over the cell corner bounding box (RA x sin(dec)) and kept only
        if they fall back into the requested cell.
        '''
        counts = np.asarray(counts).astype(np.int64)
        cells = np.asarray(cells)
        if len(cells) == 0 or int(counts.sum()) == 0:
            return np.empty(0), np.empty(0)

        # healpy 1.18 boundaries(): scalar -> (3, 4) [xyz, corner];
        # array -> (ncell, 3, 4) [cell, xyz, corner]
        corners = hp.boundaries(nside, cells)
        if corners.ndim == 2:
            corners = np.transpose(corners, (1, 0))[None, ...]  # (1, 4, 3)
        else:
            corners = np.transpose(corners, (0, 2, 1))          # (n, 4, 3)
        ra_c, dec_c = hp.vec2ang(corners.reshape(-1, 3), lonlat=True)
        ra_c = np.asarray(ra_c).reshape(len(cells), 4)
        dec_c = np.asarray(dec_c).reshape(len(cells), 4)

        ra_min = ra_c.min(axis=1)
        ra_max = ra_c.max(axis=1)
        wrap = (ra_max - ra_min) > 180.0
        if np.any(wrap):
            ra_shifted = np.where(ra_c < 180.0, ra_c + 360.0, ra_c)
            ra_min = np.where(wrap, ra_shifted.min(axis=1), ra_min)
            ra_max = np.where(wrap, ra_shifted.max(axis=1), ra_max)
        dec_min = dec_c.min(axis=1)
        dec_max = dec_c.max(axis=1)
        # sample dec uniformly via the polar angle theta = 90 - dec,
        # which is monotonic for cells straddling the equator
        # (same trick as gen_angle_positions_from_healpix)
        theta_min = 90.0 - dec_max
        theta_max = 90.0 - dec_min
        cos_lo = np.cos(np.deg2rad(theta_max))
        cos_hi = np.cos(np.deg2rad(theta_min))

        remaining = counts.copy()
        ra_out, dec_out = [], []
        for _ in range(max_rounds):
            active = remaining > 0
            if not np.any(active):
                break
            idx = np.nonzero(active)[0]
            # oversample the bounding boxes (box acceptance ~0.5)
            n_extra = np.ceil(remaining[idx] * 3.0 + 5.0).astype(np.int64)
            idx_rep = np.repeat(idx, n_extra)
            n_draw = int(n_extra.sum())
            u = rng.uniform(0.0, 1.0, n_draw)
            v = rng.uniform(0.0, 1.0, n_draw)
            ra_draw = (
                ra_min[idx_rep] + u * (ra_max[idx_rep] - ra_min[idx_rep])
            ) % 360.0
            cos_draw = cos_lo[idx_rep] + v * (cos_hi[idx_rep] - cos_lo[idx_rep])
            theta_draw = np.rad2deg(np.arccos(np.clip(cos_draw, -1.0, 1.0)))
            dec_draw = 90.0 - theta_draw

            back = hp.ang2pix(nside, ra_draw, dec_draw, lonlat=True)
            ok = back == cells[idx_rep]
            ra_ok = ra_draw[ok]
            dec_ok = dec_draw[ok]
            idx_ok = idx_rep[ok]

            order = np.argsort(idx_ok, kind="stable")
            idx_s = idx_ok[order]
            ra_s = ra_ok[order]
            dec_s = dec_ok[order]
            starts = np.searchsorted(idx_s, idx)
            ends = np.searchsorted(idx_s, idx, side="right")
            take = np.minimum(ends - starts, remaining[idx])
            take_total = int(take.sum())
            if take_total:
                offsets = np.repeat(starts, take)
                add = np.arange(take_total) - np.repeat(
                    np.cumsum(take) - take, take)
                pos_idx = offsets + add
                ra_out.append(ra_s[pos_idx])
                dec_out.append(dec_s[pos_idx])
                remaining[idx] -= take

        if np.any(remaining > 0):
            # extremely unlikely; fall back to the cell centers
            left = remaining > 0
            print(
                f"rejection sampling exhausted after {max_rounds} rounds; "
                f"placing {int(remaining[left].sum())} points at cell centers",
                flush=True,
            )
            ra0, dec0 = hp.pix2ang(nside, cells[left], lonlat=True)
            ra_out.append(np.repeat(ra0, remaining[left]))
            dec_out.append(np.repeat(dec0, remaining[left]))

        return np.concatenate(ra_out), np.concatenate(dec_out)
    
    def assign_shear(self, bg_galcat:np.ndarray, shear_map_dict:list):
        sigma_e = self.config.sigma_e
        seed_SN = self.config.seed_SN
        g1_pure, g2_pure, g1_noise, g2_noise = assign_shear_vals(bg_galcat['ra'], bg_galcat['dec'], bg_galcat['z_true'], shear_map_dict, sigma_e, seed_SN)

        bg_galcat['g1'] = g1_noise
        bg_galcat['g2'] = g2_noise
        bg_galcat['g1_pure'] = g1_pure
        bg_galcat['g2_pure'] = g2_pure

        return bg_galcat
    
    def assign_weights(self, bg_galcat:np.ndarray, weight_type='unity'):
        match weight_type:
            case 'unity':
                bg_galcat['w'] = 1.0

        return bg_galcat
    
    def rotate_pix(self, pix, nside, rot_degrees):
        r = R.from_euler('zyx', rot_degrees, degrees=True)
        norm_vec_x, norm_vec_y, norm_vec_z = hp.pix2vec(nside=nside, ipix=pix)
        norm_vec = np.c_[norm_vec_x, norm_vec_y, norm_vec_z]
        new_vec = r.apply(norm_vec)
        pix_new = hp.vec2pix(nside=nside, x=new_vec[:,0], y=new_vec[:,1], z=new_vec[:,2])
        
        return pix_new
