import json
import os
import numpy as np
from astropy.io import fits
from astropy.table import Table

from handler import *

def get_hod_seed_offset(icosmo, irlz, ihod, NRLZS_PER_COSMO=1, NHOD_PER_COSMO=10):
    offset = icosmo * NRLZS_PER_COSMO * NHOD_PER_COSMO + irlz * NHOD_PER_COSMO + ihod
    return offset

def get_pool_seed_offset(icosmo):
    return icosmo

def get_shape_seed(icosmo, irlz, NRLZS, NTOMO):
    ''' Reconstruct a per-catalog real seed from an initial seed. '''
    offset = icosmo * NRLZS * NTOMO + irlz * NTOMO
    return offset

class BaseRunner:
    # def __init__(
    #         self, config=None, sim_fmt=None, halo_fmt=None,
    #         shear_sim_fmt=None, lb_z_file=None,
    #         fore_mask_fnames_dict=None, fore_nofz_fnames_dict=None,
    #         fore_survey_labels_dict=None, back_mask_fnames_dict=None,
    #         back_nofz_fnames_dict=None, back_survey_labels_dict=None,
    #         back_ngals_dict=None, tomo_labels_dict=None,
    #         redshift_src_list=None, gal_ofmt=None, void_ofmt=None,
    #         shear_ofmt=None, runner_type=None):
    def __init__(
            self, 
            config=None,  # pipeline configure file
            ### used by generating foregrounds
            fore_mask_fnames_dict=None, fore_nofz_fnames_dict=None,
            fore_survey_labels_dict=None, 
            ### used by generating backgrounds
            back_mask_fnames_dict=None, back_nofz_fnames_dict=None, 
            back_survey_labels_dict=None, back_ngals_dict=None, 
            tomo_labels_dict=None,
            runner_type=None # which type of runner, could be "hod", "gal", "void", "shape"
        ):

        self.config = config
        self.fore_mask_fnames_dict = fore_mask_fnames_dict
        self.fore_nofz_fnames_dict = fore_nofz_fnames_dict
        self.fore_survey_labels_dict = fore_survey_labels_dict
        self.back_mask_fnames_dict = back_mask_fnames_dict
        self.back_nofz_fnames_dict = back_nofz_fnames_dict
        self.back_survey_labels_dict = back_survey_labels_dict
        self.back_ngals_dict = back_ngals_dict
        self.tomo_labels_dict = tomo_labels_dict
        self.runner_type = runner_type
        self.hod_populator = None
        self.survey_generator = None
        self.void_finder = None
        self.shear_assigner = None
        self._initialize_runner(runner_type)

    @classmethod
    def build_hod_runner(
            cls, config=None):
        return cls(
            config=config, runner_type="hod",
        )

    @classmethod
    def build_gal_runner(
            cls, config=None,
            fore_mask_fnames_dict=None, fore_nofz_fnames_dict=None,
            fore_survey_labels_dict=None):
        return cls(
            config=config,
            fore_mask_fnames_dict=fore_mask_fnames_dict,
            fore_nofz_fnames_dict=fore_nofz_fnames_dict,
            fore_survey_labels_dict=fore_survey_labels_dict,
            runner_type="gal",
        )

    @classmethod
    def build_void_runner(
            cls, config=None,
            fore_mask_fnames_dict=None, fore_nofz_fnames_dict=None,
            fore_survey_labels_dict=None):
        return cls(
            config=config,
            fore_mask_fnames_dict=fore_mask_fnames_dict,
            fore_nofz_fnames_dict=fore_nofz_fnames_dict,
            fore_survey_labels_dict=fore_survey_labels_dict,
            runner_type="void",
        )

    @classmethod
    def build_shape_runner(
            cls, config=None,
            back_mask_fnames_dict=None, back_nofz_fnames_dict=None,
            back_survey_labels_dict=None, back_ngals_dict=None,
            tomo_labels_dict=None):
        return cls(
            config=config,
            back_mask_fnames_dict=back_mask_fnames_dict,
            back_nofz_fnames_dict=back_nofz_fnames_dict,
            back_survey_labels_dict=back_survey_labels_dict,
            back_ngals_dict=back_ngals_dict,
            tomo_labels_dict=tomo_labels_dict,
            runner_type="shape",
        )

    def _initialize_hod_components(self):
        if self.hod_populator is not None:
            return
        self.hod_populator = HODPopulator(config=self.config)

    def _initialize_survey_generator(self):
        if self.survey_generator is not None:
            return
        fore_masks = self._prepare_fore_masks(self.fore_mask_fnames_dict)
        fore_nofzs = self._prepare_fore_nofzs(self.fore_nofz_fnames_dict)
        self.survey_generator = SurveyGenerator(
            config=self.config, masks=fore_masks, nofzs=fore_nofzs
        )

    def _initialize_void_finder(self):
        if self.void_finder is not None:
            return
        self.void_finder = VoidFinder(config=self.config)

    def _initialize_shear_assigner(self):
        if self.shear_assigner is not None:
            return
        back_masks = self._prepare_back_masks(self.back_mask_fnames_dict)
        back_nofzs = self._prepare_back_nofzs(self.back_nofz_fnames_dict)
        self.shear_assigner = ShearAssigner(
            config=self.config, masks=back_masks, nofzs=back_nofzs
        )

    def _initialize_runner(self, runner_type):
        if runner_type == "hod":
            self._initialize_hod_components()
            return
        if runner_type == "gal":
            self._initialize_hod_components()
            self._initialize_survey_generator()
            return
        if runner_type == "void":
            self._initialize_survey_generator()
            self._initialize_void_finder()
            return
        if runner_type == "shape":
            self._initialize_shear_assigner()
            return
    
    def _make_hod_param_dict(self, hod_param:np.ndarray):
        model_param_names = self.config.model_params_names
        hod_param_dict = dict(zip(model_param_names, hod_param))
        return hod_param_dict

    def _gsample_dict_to_array(self, dict_of_gsamples:dict):
        gcat_key = list(dict_of_gsamples.keys())[0]
        gsample_arr = dict_of_gsamples[gcat_key]
        return gsample_arr

    def _prepare_fore_masks(self, fore_mask_fnames_dict:dict):
        masks = {}
        survey_part_names = list(fore_mask_fnames_dict.keys())
        survey_part_names.remove("boss_veto")

        HAVE_BOSS = False
        for isurvey_name in survey_part_names:
            if "boss" in isurvey_name:
                HAVE_BOSS = True

                assert len(fore_mask_fnames_dict['boss_veto']) != 0, "Must provide boss mask file list"
                if 'lowze2_ngc' in isurvey_name or 'lowze3_ngc' in isurvey_name:
                    assert 'boss_lowz_ngc' in survey_part_names, "Must provide LOWZ NGC geometry for trimming E2 and E3!"
                if 'lowze2_sgc' in isurvey_name or 'lowze3_sgc' in isurvey_name:
                    assert 'boss_lowz_sgc' in survey_part_names, "Must provide LOWZ SGC geometry for trimming E2 and E3!"
                
                masks['boss_geom'] = {}
                masks['boss_masks'] = []
            if "2dflens" in isurvey_name:
                masks['2dflens_geom'] = {}
            
        print("Load foreground mask files.", flush=True)

        if HAVE_BOSS:
            ### obervational masks
            for mask_file in fore_mask_fnames_dict['boss_veto']:
                masks['boss_masks'].append(pymangle.Mangle(mask_file))
        ### survey geometry
        for ipart_name in survey_part_names:
            if "boss" in ipart_name:
                masks['boss_geom'][ipart_name] = pymangle.Mangle(fore_mask_fnames_dict[ipart_name])
            if "2dflens" in ipart_name:
                masks['2dflens_geom'][ipart_name] = loadFitsMaps(fore_mask_fnames_dict[ipart_name])

        return masks
    
    def _prepare_fore_nofzs(self, fore_nofz_fnames_dict:dict):
        nofz_info = {}
        survey_part_names = fore_nofz_fnames_dict.keys()

        print("Load foreground n(z) files.", flush=True)

        for ipart_name in survey_part_names:
            if "boss" in ipart_name:
                nofz = np.loadtxt(fore_nofz_fnames_dict[ipart_name], usecols=(1,2,3,5)) # zmin, zmax, nz, shell_vol
            if "2dflens" in ipart_name:
                nofz = np.loadtxt(fore_nofz_fnames_dict[ipart_name], usecols=(1,2,3,4)) # zmin, zmax, nz, shell_vol
            nofz_info = make_nofz_info(nofz_info, ipart_name, np.append(nofz[:,0], nofz[-1,1]), nofz[:,3], nofz[:,2])

        ### FIXME: it seems that BOSS CMASS data has a different number density than that given in nofz
        if 'boss_cmass_ngc' in nofz_info.keys():
            nofz_info['boss_cmass_ngc']['nz_ref'] *= 0.93 
        if 'boss_cmass_sgc' in nofz_info.keys():
            nofz_info['boss_cmass_sgc']['nz_ref'] *= 0.93

        return nofz_info
    
    # def _prepare_back_masks(self, back_mask_fnames_dict:dict):
    #     masks = {}

    #     print("Load background mask files.", flush=True)

    #     for isurvey_name, mask_fname in back_mask_fnames_dict.items():
    #         match isurvey_name:
    #             case "KiDS1000-North":
    #                 mask = loadFitsMaps(mask_fname)
    #                 mask = mask[0]

    #                 # ### FIXME: For test, KiDS-North mask was downloaded from kids-sbi repository
    #                 # ### https://github.com/mwiet/kids_sbi . However, I cannot read the binary type
    #                 # ### file, which may consider some effects depending on position. So I just read
    #                 # ### the mask as a float64 array and convert it to boolean
    #                 mask = np.where(mask > 0, 1, 0)
    #             case "KiDS1000-South":
    #                 mask = loadFitsMaps(mask_fname)
    #                 mask = mask[0]

    #                 # ### FIXME: For test, KiDS-North mask was downloaded from kids-sbi repository
    #                 # ### https://github.com/mwiet/kids_sbi . However, I cannot read the binary type
    #                 # ### file, which may consider some effects depending on position. So I just read
    #                 # ### the mask as a float64 array and convert it to boolean
    #                 mask = np.where(mask > 0, 1, 0)
    #             case "FullSky":
    #                 nside = 1024
    #                 mask = np.ones(12*nside*nside)

    #             case "boss_cmass_ngc":
    #                 mask = pymangle.Mangle(mask_fname)

    #         masks[isurvey_name] = mask
        
    #     return masks
    
    def _prepare_back_masks(self, mask_fnames):
        masks = {}
        for survey_name, mask_fname in mask_fnames.items():
            if survey_name in {"KiDS1000-North", "KiDS1000-South"}:
                # ### FIXME: For test, KiDS-North mask was downloaded from kids-sbi repository
                # ### https://github.com/mwiet/kids_sbi . However, I cannot read the binary type
                # ### file, which may consider some effects depending on position. So I just read
                # ### the mask as a float64 array and convert it to boolean
                with fits.open(mask_fname) as hdus:
                    mask = np.array(hdus[1].data["VALUE"]).flatten()
                mask = np.where(mask > 0, 1, 0)
            elif survey_name in {
                    "boss_cmass_ngc", "boss_lowz_ngc",
                    "boss_lowze2_ngc", "boss_lowze3_ngc"}:
                mask = pymangle.Mangle(mask_fname)
            elif survey_name == "FullSky":
                nside = 1024
                mask = np.ones(12*nside*nside)
            else:
                raise ValueError(
                    f"Unsupported background survey mask: {survey_name}"
                )
            masks[survey_name] = mask
        return masks
    
    def _get_back_ngal_tot(self):
        ngal_tot = 0.0
        for ingal in self.back_ngals_dict.values():
            ngal_tot += ingal

        return ngal_tot
    
    def _get_back_area_tot(self, back_masks):
        Area = 0.0
        RADIAN2DEG = 180.0 / np.pi
        for imask in back_masks:
            if isinstance(imask, np.ndarray) and hp.isnpixok(len(imask)):
                nside = hp.npix2nside(len(imask))
                Area += (imask > 0).sum() * hp.nside2pixarea(nside) * RADIAN2DEG**2 # deg^2
            elif isinstance(imask, pymangle.Mangle):
                Area += (imask.areas * imask.weights).sum() # deg^2
            else:
                NotImplementedError
        
        return Area
    
    def _prepare_back_nofzs(self, back_nofz_fnames_dict:dict):
        nofzs = {}

        print("Load background nofz files.", flush=True)

        for tomo_name, nofz_fname in back_nofz_fnames_dict.items():
            tmp = np.loadtxt(nofz_fname)
            nofzs[tomo_name] = make_nofz(tmp[:,0], tmp[:,1])

        return nofzs

    def _pick_gen_mock_func(self, survey_name):
        match survey_name:
            case "boss_lowz_ngc":
                gen_func = self.survey_generator.gen_boss_like
            case "boss_lowze2_ngc":
                gen_func = self.survey_generator.gen_boss_like_trim
            case "boss_lowze3_ngc":
                gen_func = self.survey_generator.gen_boss_like_trim
            case "boss_cmass_ngc":
                gen_func = self.survey_generator.gen_boss_like
            case "boss_lowz_sgc":
                gen_func = self.survey_generator.gen_boss_like
            case "boss_lowze2_sgc":
                gen_func = self.survey_generator.gen_boss_like_trim
            case "boss_lowze3_sgc":
                gen_func = self.survey_generator.gen_boss_like_trim
            case "boss_cmass_sgc":
                gen_func = self.survey_generator.gen_boss_like
            case "2dflens_south":
                gen_func = self.survey_generator.gen_2dflens_like

        return gen_func
    
    def _read_fits(self, fname):
        cat = Table.read(fname)
        return cat
    
    def _save_fits(self, fname, array_struc):
        array_tb = Table(array_struc)
        array_tb.write(fname, overwrite=True)
        return
    
    # >>> =============     APIs needs to be specifid     ============= <<<
    
    
    def get_cosmo_fname(self, icosmo, **kwargs):
        pass
    
    def get_halo_fname(self, icosmo, **kwargs):
        pass

    def get_cosmo_instance(self, icosmo, **kwargs):
        pass

    def load_hod_halocat(self, icosmo, **kwargs):
        pass

    def load_shear_maps(self, icosmo, **kwargs):
        pass

    # >>> ============================================================== <<<
    
    # >>> =================     Single run routines     ================ <<<

    def sample_hod_params(self, hod_halocat, seed_offset):

        hod_params_alive = self.hod_populator.find_hod_params(
            hod_halocat,
            seed_offset=seed_offset,
        )
        
        return hod_params_alive

    def gen_mock_gal(self, cosmo:ccl.Cosmology, hod_halocat, indx, ihod_param:np.ndarray, return_box=False):
        ''' Generate mock catalog pipeline '''
        # >>> =========   1. Input a hod_halocat   =========== <<<
        
        # >>> =========   2. Populate galaxies via HOD   =========== <<<
        OmegaM = cosmo.omega_x(a=1.0, species='matter')
        ihod_param_dict = self._make_hod_param_dict(ihod_param)
        # indx = self._get_hod_seed_offset(icosmo, irlz, ihod)
        dict_of_gsamples = self.hod_populator.populate_galaxies(hod_halocat, ihod_param_dict, indx=indx, OmegaM=OmegaM)
        gsample_arr = self._gsample_dict_to_array(dict_of_gsamples)
        gal_pos, gal_vel, gal_type, host_halo_mvir = (
            self.hod_populator.get_galaxy_features(
                gsample_arr,
                features=[
                    "pos",
                    "vel",
                    "gal_type",
                    "host_halo_mvir",
                ],
            )
        )

        # >>> =========   3. Box to lightcone and apply survey geometry & n(z)   =========== <<<
        galcone_fullsky = self.survey_generator.box_to_lightcone(
            cosmo,
            gal_pos=gal_pos,
            gal_adj_props={
                'gal_vel': gal_vel,
                'gal_type': gal_type,
                'host_halo_mvir': host_halo_mvir,
            },
        )
        galcone_survey = []
        for isurvey_name, isurvey_label in self.fore_survey_labels_dict.items():

            print(f"\nMaking {isurvey_name}-like galaxy mock")

            gen_func = self._pick_gen_mock_func(isurvey_name)
            galcone_curr = gen_func(galcone_fullsky, isurvey_name, isurvey_label)
            galcone_survey.append(galcone_curr)

        galcone_survey = np.concatenate(galcone_survey)

        if return_box:
            return np.c_[gal_pos, gal_vel, gal_type, host_halo_mvir], galcone_survey

        return galcone_survey

    def gen_mock_void(self, cosmo:ccl.Cosmology, galcone_survey, dive_input, dive_output):
        ''' Generate mock void pipeline '''

        # >>> =========   1. Input a galaxy lightcone   =========== <<<
        
        # >>> =========   2. Find void and apply geometry   =========== <<<
        voidcone_survey = []
        for isurvey_name, isurvey_label in self.fore_survey_labels_dict.items():
            select = (galcone_survey['survey'] == isurvey_label)

            if select.sum() == 0:
                continue

            galcone_curr = galcone_survey[select]

            print(f"\nMaking {isurvey_name}-like void mock")
            
            voidcone_curr = self.void_finder.galcone_to_voidcone(galcone_curr, cosmo, survey=isurvey_label,
                                                      dive_input=dive_input, dive_output=dive_output)

            zcut = (voidcone_curr['z'] >= self.config.zmin_lightcone) & (voidcone_curr['z'] <= self.config.zmax_lightcone)

            voidcone_curr = voidcone_curr[zcut]
            gen_func = self._pick_gen_mock_func(isurvey_name)
            voidcone_curr = gen_func(voidcone_curr, isurvey_name, isurvey_label, make_nz=False)
            
            voidcone_survey.append(voidcone_curr)

        voidcone_survey = np.concatenate(voidcone_survey)

        return voidcone_survey
    
    def gen_mock_void_boostrap(self, cosmo:ccl.Cosmology,
                      galcone_survey, dive_input, dive_output,
                      bs_nsamples=10, bs_seed=9857
                      ):
        ''' Generate bootstrap resampled void catalog subsamples '''

        rng = np.random.default_rng(seed=bs_seed)
        nsize_origin = len(galcone_survey)
        voidcone_survey_bs_subsamples = []
        for ibs in range(bs_nsamples):
            sub_galcone_survey = rng.choice(galcone_survey, len(galcone_survey), replace=True)
            sampled_indice = rng.choice(np.arange(nsize_origin), nsize_origin, replace=True)
            sampled_indice = np.unique(sampled_indice)
            sub_galcone_survey = galcone_survey[sampled_indice]
            print(f"Bootstrap resampling: {ibs}, curr sample size: {len(sub_galcone_survey)}", flush=True)
            sub_voidcone_survey = self.gen_mock_void(cosmo,
                                                     sub_galcone_survey, dive_input, dive_output,
                                                     )
            # if save:
            #     sub_voidcone_survey = Table(sub_voidcone_survey)
            #     sub_voidcone_survey.write(self.void_ofmt.format(icosmo, irlz, ihod, ibs))
            #     del sub_voidcone_survey
            # else:
                
            voidcone_survey_bs_subsamples.append(sub_voidcone_survey)

        return voidcone_survey_bs_subsamples
    
    def gen_mock_shape(self, shear_maps_curr, seed_Phz_off, seed_SN_off):
        ''' Generate mock shape catalog pipeline '''
        # >>> =========   1. Input shear maps   =========== <<<
        
        shear_zmax = 0.0
        for ishell in range(len(shear_maps_curr)):
            shear_zcurr = shear_maps_curr[f'shell{ishell}']['redshift']
            if shear_zcurr > shear_zmax:
                shear_zmax = shear_zcurr

        print("Maximum redshift of shear maps: {:.2f}".format(shear_zmax), flush=True)

        # >>> =========   2. Generate background positions & assign shear   =========== <<<
        shapecone_survey = []
        ### Loop of surveys
        for isurvey_name, isurvey_label in self.back_survey_labels_dict.items():

            print(f"\nMaking {isurvey_name}-like shape mock", flush=True)

            ### Loop of tomographic bins
            for itomo, (itomo_name, itomo_label) in enumerate(self.tomo_labels_dict.items()):

                seed_Phz = seed_Phz_off + itomo
                seed_SN  = seed_SN_off  + itomo
                ### Generate positions
                shapecone_curr = self.shear_assigner.gen_gal_positions(ngal=self.back_ngals_dict[itomo_name], tomo_label=itomo_label,
                                                                       survey_name=isurvey_name, survey_label=isurvey_label, seed_Phz=seed_Phz)
                ### Apply redshift cut
                zcut = (shapecone_curr['z_true'] <= shear_zmax)
                shapecone_curr = shapecone_curr[zcut]
                ### Assign shear
                shapecone_curr = self.shear_assigner.assign_shear(shapecone_curr, shear_maps_curr, seed_SN=seed_SN)
                ### FIXME: Support assigning weights by weight map in the future
                shapecone_curr = self.shear_assigner.assign_weights(shapecone_curr, weight_type='unity')

                print(f"Tomo {itomo_label}: {len(shapecone_curr)}", flush=True)

                shapecone_survey.append(shapecone_curr)

        shapecone_survey = np.concatenate(shapecone_survey)

        # >>> =========   3. (Optional) Save to file   =========== <<<
        # if save:
        #     shapecone_survey_tb = Table(shapecone_survey)
        #     shapecone_survey_tb.write(self.shear_ofmt.format(icosmo))
        #     del shapecone_survey_tb
        
        return shapecone_survey

    # >>> ============================================================== <<<    

    # >>> ====================     Pipeline runs     =================== <<<

    def pipe_run_hod_sampling(self, icosmo, irlz):
        cosmo = self.get_cosmo_instance(icosmo, irlz, otype='ccl')
        hod_halocat = self.load_hod_halocat(icosmo, irlz, cosmo=cosmo, ofmt='hod')
        pool_seed_offset = get_pool_seed_offset(icosmo)

        hod_params_alive = self.sample_hod_params(hod_halocat, seed_offset=pool_seed_offset)

        return hod_params_alive
    
    def pipe_run_gen_mock_galaxy(self, icosmo, irlz, ihod, ihod_param):
        cosmo = self.get_cosmo_instance(icosmo, irlz, otype='ccl')
        hod_halocat = self.load_hod_halocat(icosmo, irlz, cosmo=cosmo, ofmt='hod')
        indx = get_hod_seed_offset(icosmo, irlz, ihod, NRLZS_PER_COSMO=self.config.nrlzs_per_cosmo, NHOD_PER_COSMO=self.config.nhod_per_cosmo)

        galcone = self.gen_mock_gal(cosmo, hod_halocat, indx, ihod_param, return_box=False)

        self._save_fits(self.gfmt.format(icosmo, irlz, ihod), galcone)
    
    def pipe_run_gen_mock_void(self, icosmo, irlz, ihod, rank=0):
        gfname = self.gfmt.format(icosmo, irlz, ihod)
        vfname = self.vfmt.format(icosmo, irlz, ihod)

        cosmo = self.get_cosmo_instance(icosmo, irlz, otype='ccl')
        galcone = self._read_fits(gfname)
        
        voidcone = self.gen_mock_void(cosmo, galcone, 
                                      self.dive_input_fmt.format(rank),
                                      self.dive_output_fmt.format(rank))
        
        self._save_fits(vfname, voidcone)

    def pipe_run_gen_mock_void_bs(self):
        raise NotImplementedError
    
    def pipe_run_gen_mock_shape(self, icosmo, irlz):
        ntomo = len(self.tomo_labels_dict)
        seed_off = get_shape_seed(icosmo, irlz, self.config.nrlzs_noise, ntomo)
        seed_Phz_off = seed_off + self.config.seed_Phz_ini
        seed_SN_off  = seed_off + self.config.seed_SN_ini

        shear_maps_curr = self.load_shear_maps(icosmo, irlz)

        shapecone = self.gen_mock_shape(shear_maps_curr, seed_Phz_off, seed_SN_off)

        self._save_fits(self.spfmt.format(icosmo, irlz), shapecone)
    
    # >>> ============================================================== <<<

class CosmoGridRunner(BaseRunner):

    def set_file_fmts(self, 
                      sim_fmt=None, halo_fmt=None, lb_z_file=None,
                      gfmt=None, vfmt=None,
                      dive_input_fmt='tmp_input_rank{}.tmp',
                      dive_output_fmt='tmp_output_rank{}.tmp',
                      redshift_src_list=None,
                      smfmt=None, # shear maps format
                      spfmt=None  # shape cats format
                      ):
        self.sim_fmt = sim_fmt
        self.halo_fmt = halo_fmt
        self.lb_z_file = lb_z_file
        self.gfmt = gfmt
        self.vfmt = vfmt
        self.dive_input_fmt = dive_input_fmt
        self.dive_output_fmt = dive_output_fmt
        self.redshift_src_list = redshift_src_list
        self.smfmt = smfmt
        self.spfmt = spfmt

    def get_cosmo_instance(self, icosmo, irlz, otype="ccl") -> Union[dict, ccl.Cosmology]:
        ''' get ccl cosmology instance '''
        fname = self.get_cosmo_fname(icosmo, irlz)
        cosmo_par = {}
        with open(fname, "r") as f:
            for line in f.readlines():
                items = line.split(":")
                cosmo_par[items[0]] = float(items[1])
        
        if otype == "ccl":
            outputs = ccl.Cosmology(
                h=cosmo_par["H0"]/100, 
                Omega_b=cosmo_par["Ob"], 
                Omega_c=cosmo_par["O_cdm"], 
                sigma8=cosmo_par["s8"], 
                n_s=cosmo_par["ns"], 
                w0=cosmo_par["w0"], 
                wa=cosmo_par["wa"],
                m_nu=cosmo_par["m_nu"]*3
            )
        elif otype == "dict":
            outputs = cosmo_par
        else:
            raise NotImplementedError(f"Output type {otype} not implemented!")

        return outputs

    def get_redshift_label(self, redshift, lb_z_file):
        label_z_table = np.loadtxt(lb_z_file) # col0: label; col1: redshift
        zdiff = np.abs(label_z_table[:,1] - redshift)
        nearest_idx = np.argmin(zdiff)
        redshift_label = int(label_z_table[nearest_idx][0])

        print("Nearest redshift: {:.4f}".format(label_z_table[redshift_label, 1]))

        return redshift_label
    
    def get_cosmo_fname(self, icosmo, irlz):
        return os.path.join(self.sim_fmt.format(icosmo, irlz), "params.yml")

    def get_halo_fname(self, icosmo, irlz):

        redshift_label = self.get_redshift_label(self.config.redshift, self.lb_z_file)
        return os.path.join(
            self.sim_fmt.format(icosmo, irlz),
            self.halo_fmt.format(redshift_label),
        )


    def load_hod_halocat(self, icosmo, irlz, ofmt='hod', cosmo=None, clean=False):
        '''
        Load halo catalog, and transfer to HOD type
        '''

        Lbox = self.config.Lbox
        redshift = self.config.redshift
        Npart = self.config.Npart

        halo_fname = self.get_halo_fname(icosmo, irlz)
        
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
            if cosmo is None:
                cosmo = self.get_cosmo_instance(icosmo, irlz, otype='ccl')
            OmegaM = cosmo.omega_x(a=1.0, species='matter')
            pmass = rhoc0*OmegaM*(Lbox/Npart)**3 # Msun/h
            ## Initialize HOD model class
            halocat = pkd_to_hod_type(pkd_halo_infos, cosmo=cosmo, pmass=pmass, boxsize=Lbox, redshift=redshift)
            
            return halocat

    def load_shear_maps(self, icosmo, irlz=0):

        print("Load shear maps", flush=True)

        shear_map_dict = {}
        for ishell in range(len(self.redshift_src_list)):
            redshift_src = self.redshift_src_list[ishell]

            shear_map_dict[f"shell{ishell}"] = {}
            shear_map_dict[f"shell{ishell}"]['redshift'] = redshift_src

            shear_sim_fname = self.smfmt.format(icosmo, redshift_src)
            with h5py.File(shear_sim_fname, 'r') as f:
                A = np.array(f["Distortion_matrix"]["Raytraced"])

            gamma1 = -(A[0][0] - A[1][1]) / 2
            gamma2 = -(A[0][1] + A[1][0]) / 2

            shear_map_dict[f"shell{ishell}"]['gamma1'] = gamma1
            shear_map_dict[f"shell{ishell}"]['gamma2'] = gamma2

        return shear_map_dict

class FastPMRunner(BaseRunner):

    def set_file_fmts(self, 
                      cosmo_file=None, halo_fmt=None,
                      gfmt=None, vfmt=None,
                      dive_input_fmt='tmp_input_rank{}.tmp',
                      dive_output_fmt='tmp_output_rank{}.tmp',
                      redshift_src_list=None,
                      smfmt=None, # shear maps format
                      spfmt=None  # shape cats format
                      ):
        self.cosmo_file = cosmo_file
        self.halo_fmt = halo_fmt
        self.gfmt = gfmt
        self.vfmt = vfmt
        self.dive_input_fmt = dive_input_fmt
        self.dive_output_fmt = dive_output_fmt
        self.redshift_src_list = redshift_src_list
        self.smfmt = smfmt
        self.spfmt = spfmt

    def get_cosmo_instance(self, icosmo, irlz=0, otype="ccl"):
        fname = self.cosmo_file

        params = {}
        with open(fname, "r") as file_obj:
            fixed_line = file_obj.readline()
            fixed_line = fixed_line.strip("# ").strip("\n")
            fixed_line = fixed_line.split(" ")
            fixed_param_names = [part.split("=")[0] for part in fixed_line]
            fixed_param_vals  = [float(part.split("=")[1]) for part in fixed_line]
            for key, val in zip(fixed_param_names, fixed_param_vals):
                params[key] = val

            varied_line = file_obj.readline()
            varied_param_names = varied_line.strip("# ").strip("\n").split(" ")

        varied_params = np.loadtxt(fname, comments="#")
        varied_params = np.atleast_2d(varied_params)
        varied_params_curr = varied_params[icosmo]
        for j in range(len(varied_param_names)):
            params[varied_param_names[j]] = varied_params_curr[j]

        if "OmegaM" in (fixed_param_names + varied_param_names):
            params['Omegac'] = params['OmegaM'] - params['Omegab']

        if "S8" in (fixed_param_names + varied_param_names):
            params['sigma8'] = params['S8']/np.sqrt(params['OmegaM']/0.3)

        if otype == "dict":
            return params
        if otype == "ccl":
            return ccl.Cosmology(
                h=params["hubble"],
                Omega_b=params["Omegab"],
                Omega_c=params["Omegac"],
                sigma8=params["sigma8"],
                n_s=params["ns"],
                w0=-1.0,
                wa=0.0,
                m_nu=0.0,
            )
        raise NotImplementedError(f"Output type {otype} not implemented")

    def get_halo_fname(self, icosmo: int, redshift) -> str:
        scale_factor = 1./(1+redshift)
        halo_fname = self.halo_fmt.format(icosmo, scale_factor)
        return halo_fname

    def load_hod_halocat(
            self, icosmo, irlz=0,
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

        halo_fname = self.get_halo_fname(icosmo, redshift)

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
            if cosmo is None:
                cosmo = self.get_cosmo_instance(icosmo, otype='ccl')
            OmegaM = cosmo.omega_x(a=1.0, species='matter')
            pmass = rhoc0*OmegaM*(Lbox/Npart)**3  # Msun/h
            halocat = rstar_to_hod_type(
                rstar_halo_infos,
                pmass=pmass,
                boxsize=Lbox,
                redshift=redshift,
            )

            return halocat

    def load_shear_maps(self, icosmo, irlz=0):
        if self.smfmt is None:
            raise ValueError("shear_sim_fmt is required to load shear maps")
        shear_fname = self.smfmt.format(icosmo)

        shear_map_dict = {}
        with np.load(shear_fname) as file_obj:
            zgrid = file_obj['zgrid']
            gamma1s = file_obj['gamma1s']
            gamma2s = file_obj['gamma2s']

            for ishell in range(len(zgrid)-1):
                redshift_src = zgrid[ishell+1]

                shear_map_dict[f"shell{ishell}"] = {}
                shear_map_dict[f"shell{ishell}"]['redshift'] = redshift_src

                shear_map_dict[f"shell{ishell}"]['gamma1'] = gamma1s[ishell]
                shear_map_dict[f"shell{ishell}"]['gamma2'] = gamma2s[ishell]

        return shear_map_dict

# class AbacusRunner:
#     '''
#     Runner that generates shape (background source) catalogs from Abacus
#     raytracing shear maps. Only the shape pipeline is supported: no
#     galaxy/HOD or void generation is available.

#     Abacus shear products are stored in ``/data2/suchen/Abacus/shear_maps``
#     as pairs of FITS binary tables (``gamma1_rt_z{z:.2f}.fits`` /
#     ``gamma2_rt_z{z:.2f}.fits``). Each file contains one full-sky map
#     flattened into 196608 rows x 1024 float32 pixels, i.e. a HEALPix map
#     with RING ordering at NSIDE=4096 in celestial coordinates.

#     Parameters
#     ----------
#     config: PipeConfig
#         Pipeline configuration (shape noise ``sigma_e`` / ``seed_SN``,
#         photo-z error ``sigma_phz`` / ``seed_Phz``, ...).
#     shear_map_fmt: str
#         Format string of the Abacus shear maps with two placeholders: the
#         gamma component index (1 or 2) and the source redshift, e.g.
#         ``"/data2/suchen/Abacus/shear_maps/gamma{:d}_rt_z{:.2f}.fits"``.
#     redshift_src_list: list
#         Source-plane redshifts (shell centers) of the maps to load, in
#         ascending order. Each shell costs ~1.6 GB (gamma1 + gamma2 at
#         NSIDE=4096), so only request the shells that the source n(z)
#         actually covers.
#     back_mask_fnames_dict, back_nofz_fnames_dict, back_survey_labels_dict,
#     back_ngals_dict, tomo_labels_dict: dict
#         Background survey configuration (same as ``CosmoGridRunner``).
#     shear_ofmt: str
#         Output file path of the shape catalog (used as-is, no formatting).
#     '''

#     def __init__(
#             self, config=None, shear_map_fmt=None,
#             back_mask_fnames_dict=None, back_nofz_fnames_dict=None,
#             back_survey_labels_dict=None, back_ngals_dict=None,
#             tomo_labels_dict=None, redshift_src_list=None, shear_ofmt=None,
#             mass_map_fmt=None, z_to_mass_label=None,
#             position_method="random", bias=1.0):
#         self.config = config
#         self.shear_map_fmt = shear_map_fmt
#         self.back_mask_fnames_dict = back_mask_fnames_dict
#         self.back_nofz_fnames_dict = back_nofz_fnames_dict
#         self.back_survey_labels_dict = back_survey_labels_dict
#         self.back_ngals_dict = back_ngals_dict
#         self.tomo_labels_dict = tomo_labels_dict
#         self.redshift_src_list = redshift_src_list
#         self.shear_ofmt = shear_ofmt
#         self.mass_map_fmt = mass_map_fmt
#         self.z_to_mass_label = z_to_mass_label
#         self.position_method = position_method
#         self.bias = bias
#         self.shear_assigner = None

#         if position_method not in ("random", "density"):
#             raise ValueError(
#                 f"unsupported position_method: {position_method}"
#             )
#         self._validate_shape_parameters("build_shape_runner")
#         self._initialize_shear_assigner()

#     @classmethod
#     def build_shape_runner(
#             cls, config=None, shear_map_fmt=None,
#             back_mask_fnames_dict=None, back_nofz_fnames_dict=None,
#             back_survey_labels_dict=None, back_ngals_dict=None,
#             tomo_labels_dict=None, redshift_src_list=None, shear_ofmt=None,
#             mass_map_fmt=None, z_to_mass_label=None,
#             position_method="random", bias=1.0):
#         return cls(
#             config=config,
#             shear_map_fmt=shear_map_fmt,
#             back_mask_fnames_dict=back_mask_fnames_dict,
#             back_nofz_fnames_dict=back_nofz_fnames_dict,
#             back_survey_labels_dict=back_survey_labels_dict,
#             back_ngals_dict=back_ngals_dict,
#             tomo_labels_dict=tomo_labels_dict,
#             redshift_src_list=redshift_src_list,
#             shear_ofmt=shear_ofmt,
#             mass_map_fmt=mass_map_fmt,
#             z_to_mass_label=z_to_mass_label,
#             position_method=position_method,
#             bias=bias,
#         )

#     def _validate_shape_parameters(self, context):
#         _require_parameters(
#             context,
#             config=self.config,
#             shear_map_fmt=self.shear_map_fmt,
#             back_mask_fnames_dict=self.back_mask_fnames_dict,
#             back_nofz_fnames_dict=self.back_nofz_fnames_dict,
#             back_survey_labels_dict=self.back_survey_labels_dict,
#             back_ngals_dict=self.back_ngals_dict,
#             tomo_labels_dict=self.tomo_labels_dict,
#             redshift_src_list=self.redshift_src_list,
#         )
#         if set(self.back_survey_labels_dict) != set(self.back_mask_fnames_dict):
#             raise ValueError(
#                 "background survey labels and mask keys must match"
#             )
#         if set(self.back_ngals_dict) != set(self.back_nofz_fnames_dict):
#             raise ValueError(
#                 "background number-density and n(z) keys must match"
#             )
#         if set(self.tomo_labels_dict) != set(self.back_nofz_fnames_dict):
#             raise ValueError("tomographic labels and n(z) keys must match")
#         for tomo_name, tomo_label in self.tomo_labels_dict.items():
#             expected_name = f"tomo{tomo_label}"
#             if tomo_name != expected_name:
#                 raise ValueError(
#                     f"tomographic key {tomo_name} must be {expected_name}"
#                 )
#         redshifts = np.asarray(self.redshift_src_list, dtype=float)
#         if redshifts.size < 2:
#             raise ValueError(
#                 "redshift_src_list must contain at least two shells"
#             )
#         if np.any(redshifts[1:] <= redshifts[:-1]):
#             raise ValueError("redshift_src_list must be strictly ascending")
#         if self.position_method == "density":
#             _require_parameters(
#                 "build_shape_runner (method='density')",
#                 mass_map_fmt=self.mass_map_fmt,
#                 z_to_mass_label=self.z_to_mass_label,
#             )

#     def _load_mass_maps(self):
#         '''
#         Load Abacus mass maps (``shell_{:d}.fits``, ``SIGNAL`` column) and
#         convert them to full-sky overdensity maps: delta = Sigma/<Sigma> - 1.
#         '''
#         print("Load Abacus mass maps", flush=True)

#         labels = sorted(set(self.z_to_mass_label.values()))
#         mass_maps = {}
#         for label in labels:
#             fname = self.mass_map_fmt.format(label)
#             try:
#                 hdus = fits.open(fname, memmap=True)
#             except OSError as err:
#                 raise ValueError(
#                     f"{fname} is not a valid FITS file ({err}); "
#                     "it may be corrupted or incompletely written"
#                 ) from err
#             with hdus:
#                 if len(hdus) < 2 or not isinstance(
#                         hdus[1], fits.BinTableHDU):
#                     raise ValueError(
#                         f"{fname} must contain a binary table extension"
#                     )
#                 header = hdus[1].header
#                 if header.get("PIXTYPE", "").strip() != "HEALPIX":
#                     raise ValueError(f"{fname} is not a HEALPix map")
#                 if header.get("ORDERING", "").strip() != "RING":
#                     raise ValueError(
#                         f"{fname} uses {header.get('ORDERING')} "
#                         "ordering; only RING is supported"
#                     )
#                 nside = int(header["NSIDE"])
#                 signal = np.asarray(hdus[1].data["SIGNAL"]).ravel()
#             npix = 12 * nside * nside
#             if signal.size != npix:
#                 raise ValueError(
#                     f"{fname} has {signal.size} pixels, expected "
#                     f"{npix} for NSIDE={nside}"
#                 )
#             mean = float(signal.mean(dtype=np.float64))
#             if not np.isfinite(mean) or mean <= 0:
#                 raise ValueError(
#                     f"{fname} has non-finite or non-positive mean: {mean}"
#                 )
#             delta = signal.astype(np.float32) / np.float32(mean)
#             delta -= np.float32(1.0)
#             mass_maps[label] = delta

#         return mass_maps

#     def _initialize_shear_assigner(self):
#         if self.shear_assigner is not None:
#             return
#         back_masks = self._prepare_back_masks(self.back_mask_fnames_dict)
#         back_nofzs = self._prepare_back_nofzs(self.back_nofz_fnames_dict)
#         mass_maps = None
#         if self.mass_map_fmt is not None and self.z_to_mass_label is not None:
#             mass_maps = self._load_mass_maps()
#         self.shear_assigner = ShearAssigner(
#             config=self.config, masks=back_masks, nofzs=back_nofzs,
#             mass_maps=mass_maps, z_to_mass_label=self.z_to_mass_label,
#         )

#     def _prepare_back_masks(self, mask_fnames):
#         masks = {}
#         for survey_name, mask_fname in mask_fnames.items():
#             if survey_name in {"KiDS1000-North", "KiDS1000-South"}:
#                 mask = loadFitsMaps(mask_fname)
#                 mask = mask[0]
#                 mask = np.where(mask > 0, 1, 0)
#             elif survey_name == "DES-Y3":
#                 with fits.open(mask_fname, memmap=True) as hdus:
#                     if len(hdus) < 2 or not isinstance(
#                             hdus[1], fits.BinTableHDU):
#                         raise ValueError(
#                             f"{mask_fname} must contain a binary table"
#                         )
#                     header = hdus[1].header
#                     if header.get("PIXTYPE", "").strip() != "HEALPIX":
#                         raise ValueError(f"{mask_fname} is not a HEALPix map")
#                     if header.get("COORDSYS", "").strip() != "C":
#                         raise ValueError(
#                             f"{mask_fname} must use celestial coordinates"
#                         )
#                     if "SCIENCE" not in hdus[1].columns.names:
#                         raise ValueError(
#                             f"{mask_fname} must contain a SCIENCE column"
#                         )
#                     ordering = header.get("ORDERING", "").strip()
#                     mask = np.array(hdus[1].data["SCIENCE"]).reshape(-1)
#                 if ordering == "NESTED":
#                     mask = hp.reorder(mask, n2r=True)
#                 elif ordering != "RING":
#                     raise ValueError(
#                         f"{mask_fname} uses unsupported {ordering} ordering"
#                     )
#                 mask = np.where(mask > 0, 1, 0).astype(np.uint8)
#             elif survey_name == "FullSky":
#                 nside = 1024
#                 mask = np.ones(12 * nside * nside)
#             elif survey_name == "boss_cmass_ngc":
#                 mask = pymangle.Mangle(mask_fname)
#             else:
#                 raise ValueError(
#                     f"Unsupported background survey mask: {survey_name}"
#                 )
#             masks[survey_name] = mask
#         return masks

#     def _prepare_back_nofzs(self, nofz_fnames):
#         nofzs = {}
#         for tomo_name, nofz_fname in nofz_fnames.items():
#             with open(nofz_fname, "r") as stream:
#                 first_row = stream.readline().split()
#             skiprows = 1 if len(first_row) != 2 else 0
#             tmp = np.loadtxt(nofz_fname, skiprows=skiprows, ndmin=2)
#             if tmp.shape[1] != 2:
#                 raise ValueError(
#                     f"{nofz_fname} must contain redshift and n(z) columns"
#                 )
#             nofzs[tomo_name] = make_nofz(tmp[:, 0], tmp[:, 1])
#         return nofzs

#     def _load_shear_maps(self):
#         '''
#         Load Abacus gamma1/gamma2 HEALPix maps into the ``shear_map_dict``
#         consumed by ``ShearAssigner.assign_shear``.
#         '''
#         print("Load Abacus shear maps", flush=True)

#         shear_map_dict = {}
#         for ishell, redshift_src in enumerate(self.redshift_src_list):
#             shear_map_dict[f"shell{ishell}"] = {"redshift": redshift_src}
#             for comp in (1, 2):
#                 fname = self.shear_map_fmt.format(comp, redshift_src)
#                 try:
#                     hdus = fits.open(fname, memmap=True)
#                 except OSError as err:
#                     raise ValueError(
#                         f"{fname} is not a valid FITS file ({err}); "
#                         "it may be corrupted or incompletely written"
#                     ) from err
#                 with hdus:
#                     if len(hdus) < 2 or not isinstance(
#                             hdus[1], fits.BinTableHDU):
#                         raise ValueError(
#                             f"{fname} must contain a binary table extension"
#                         )
#                     header = hdus[1].header
#                     if header.get("PIXTYPE", "").strip() != "HEALPIX":
#                         raise ValueError(f"{fname} is not a HEALPix map")
#                     if header.get("ORDERING", "").strip() != "RING":
#                         raise ValueError(
#                             f"{fname} uses {header.get('ORDERING')} "
#                             "ordering; only RING is supported"
#                         )
#                     nside = int(header["NSIDE"])
#                     map_data = np.asarray(hdus[1].data["T"]).reshape(-1)
#                 npix = 12 * nside * nside
#                 if map_data.size != npix:
#                     raise ValueError(
#                         f"{fname} has {map_data.size} pixels, expected "
#                         f"{npix} for NSIDE={nside}"
#                     )
#                 shear_map_dict[f"shell{ishell}"][f"gamma{comp}"] = map_data

#         return shear_map_dict

#     def gen_mock_shear(self, save=True):
#         ''' Generate mock shape catalog pipeline from Abacus shear maps '''
#         if self.shear_assigner is None:
#             raise ValueError(
#                 "AbacusRunner.build_shape_runner() is required; "
#                 "missing components: shear_assigner"
#             )
#         if save and self.shear_ofmt is None:
#             raise ValueError("shear_ofmt is required when save=True")

#         # >>> =========   1. Load shear maps   =========== <<<
#         shear_maps_curr = self._load_shear_maps()

#         # >>> =========   2. Generate background positions & assign shear   =========== <<<
#         shapecone_survey = []
#         ### Loop of surveys
#         for isurvey_name, isurvey_label in self.back_survey_labels_dict.items():

#             print(f"\nMaking {isurvey_name}-like shape mock", flush=True)

#             ### Loop of tomographic bins
#             for itomo_name, itomo_label in self.tomo_labels_dict.items():

#                 shapecone_curr = self.shear_assigner.gen_gal_positions(
#                     ngal=self.back_ngals_dict[itomo_name],
#                     tomo_label=itomo_label,
#                     survey_name=isurvey_name,
#                     survey_label=isurvey_label,
#                     method=self.position_method,
#                     bias=self.bias,
#                     seed=self.config.seed_pos)

#                 ### Drop sources outside shear-map coverage BEFORE assigning
#                 ### shear; assign_shear_vals would otherwise drop rows
#                 ### silently and desynchronize the catalog.
#                 source_redshifts = np.asarray(shapecone_curr["z_true"])
#                 shell_redshifts = np.asarray([
#                     shear_maps_curr[f"shell{index}"]["redshift"]
#                     for index in range(len(shear_maps_curr))
#                 ])
#                 max_redshift = (
#                     shell_redshifts[-1]
#                     + shell_redshifts[-1]
#                     - shell_redshifts[-2]
#                 )
#                 zcover = (
#                     np.isfinite(source_redshifts)
#                     & (source_redshifts >= 0)
#                     & (source_redshifts < max_redshift)
#                 )
#                 ndropped = int((~zcover).sum())
#                 if ndropped:
#                     print(
#                         f"Tomo {itomo_label}: drop {ndropped} sources "
#                         f"outside shear-map coverage [0, {max_redshift})",
#                         flush=True,
#                     )
#                 shapecone_curr = shapecone_curr[zcover]
#                 if len(shapecone_curr) == 0:
#                     raise ValueError(
#                         "all sources are outside shear-map coverage "
#                         f"[0, {max_redshift})"
#                     )

#                 shapecone_curr = self.shear_assigner.assign_shear(
#                     shapecone_curr, shear_maps_curr)
#                 ### FIXME: Support assigning weights by weight map in the future
#                 shapecone_curr = self.shear_assigner.assign_weights(
#                     shapecone_curr, weight_type='unity')

#                 print(f"Tomo {itomo_label}: {len(shapecone_curr)}", flush=True)

#                 shapecone_survey.append(shapecone_curr)

#         shapecone_survey = np.concatenate(shapecone_survey)

#         # >>> =========   3. (Optional) Save to file   =========== <<<
#         if save:
#             shapecone_survey_tb = Table(shapecone_survey)
#             shapecone_survey_tb.write(self.shear_ofmt)
#             del shapecone_survey_tb

#         return shapecone_survey
