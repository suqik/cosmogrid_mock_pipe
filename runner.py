import os
import numpy as np
from astropy.table import Table

from handler import *

class CosmoGridRunner:
    def __init__(self, config:PipeConfig,
                 sim_fmt:str,
                 halo_fmt:str,
                 shear_sim_fmt:str,
                 lb_z_file:str,
                 fore_mask_fnames_dict:dict,
                 fore_nofz_fnames_dict:dict,
                 fore_survey_labels_dict:dict,
                 back_mask_fnames_dict:dict,
                 back_nofz_fnames_dict:dict,
                 back_survey_labels_dict:dict,
                 back_ngals_dict: dict,
                 tomo_labels_dict: dict,
                 redshift_src_list: list,
                 gal_ofmt:str = None,
                 void_ofmt:str = None,
                 shear_ofmt:str = None
                 ):

        self.config = config

        self.sim_fmt = sim_fmt
        self.halo_fmt = halo_fmt
        self.lb_z_file = lb_z_file
        self.fore_survey_labels_dict = fore_survey_labels_dict
        assert list(fore_survey_labels_dict.keys()) == list(fore_nofz_fnames_dict.keys())

        self.shear_sim_fmt = shear_sim_fmt
        self.redshift_src_list = redshift_src_list
        self.back_ngals_dict = back_ngals_dict
        self.tomo_labels_dict = tomo_labels_dict
        self.back_survey_labels_dict = back_survey_labels_dict
        assert list(back_ngals_dict.keys()) == list(back_nofz_fnames_dict.keys())
        assert list(tomo_labels_dict.keys()) == list(back_nofz_fnames_dict.keys())

        self.gal_ofmt = gal_ofmt
        self.void_ofmt = void_ofmt
        self.shear_ofmt = shear_ofmt

        ### initializations

        self.redshift_label = self._get_redshift_label(config.redshift)
        fore_masks = self._prepare_fore_masks(fore_mask_fnames_dict)
        fore_nofzs = self._prepare_fore_nofzs(fore_nofz_fnames_dict)
        back_masks = self._prepare_back_masks(back_mask_fnames_dict=back_mask_fnames_dict)
        back_nofzs = self._prepare_back_nofzs(back_nofz_fnames_dict=back_nofz_fnames_dict)

        self.cata_loader = CatalogLoader(config=config)
        self.hod_populator = HODPopulator(config=config)
        self.survey_generator = SurveyGenerator(config=config, masks=fore_masks, nofzs=fore_nofzs)
        self.void_finder = VoidFinder(config=config)
        self.shear_assigner = ShearAssigner(config=config, 
                                            masks=back_masks, nofzs=back_nofzs)
        
    def _get_cosmo_instance(self, fname:str, otype="ccl") -> Union[dict, ccl.Cosmology]:
        ''' get ccl cosmology instance '''
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

    def _get_redshift_label(self, redshift):
        label_z_table = np.loadtxt(self.lb_z_file) # col0: label; col1: redshift
        zdiff = np.abs(label_z_table[:,1] - redshift)
        nearest_idx = np.argmin(zdiff)
        redshift_label = int(label_z_table[nearest_idx][0])

        print("Nearest redshift: {:.4f}".format(label_z_table[redshift_label, 1]))

        return redshift_label
    
    def _get_fnames(self, icosmo, irlz):
        cpar_fname = os.path.join(self.sim_fmt.format(icosmo, irlz), "params.yml")
        halo_fname = os.path.join(self.sim_fmt.format(icosmo, irlz), self.halo_fmt.format(self.redshift_label))
        return cpar_fname, halo_fname
    
    def _make_hod_param_dict(self, hod_param:np.ndarray):
        model_param_names = self.config.model_params_names
        hod_param_dict = dict(zip(model_param_names, hod_param))
        return hod_param_dict

    def _gsample_dict_to_array(self, dict_of_gsamples:dict):
        gcat_key = list(dict_of_gsamples.keys())[0]
        gsample_arr = dict_of_gsamples[gcat_key]
        return gsample_arr
    
    def _get_hod_seed_offset(self, icosmo, irlz, ihod):
        NRLZS_PER_COSMO = self.config.nrlzs_per_cosmo
        NHOD_PER_COSMO = self.config.nhod_per_cosmo

        offset = icosmo * NRLZS_PER_COSMO * NHOD_PER_COSMO + irlz * NHOD_PER_COSMO + ihod
        return offset
    
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
        if 'boss_cmass' in nofz_info.keys():
            nofz_info['boss_cmass']['nz_ref'] *= 0.93 

        return nofz_info
    
    def _prepare_back_masks(self, back_mask_fnames_dict:dict):
        masks = {}

        print("Load background mask files.", flush=True)

        for isurvey_name, mask_fname in back_mask_fnames_dict.items():
            match isurvey_name:
                case "KiDS1000-North":
                    mask = loadFitsMaps(mask_fname)
                    mask = mask[0]

                    # ### FIXME: For test, KiDS-North mask was downloaded from kids-sbi repository
                    # ### https://github.com/mwiet/kids_sbi . However, I cannot read the binary type
                    # ### file, which may consider some effects depending on position. So I just read
                    # ### the mask as a float64 array and convert it to boolean
                    mask = np.where(mask > 0, 1, 0)
                case "KiDS1000-South":
                    mask = loadFitsMaps(mask_fname)
                    mask = mask[0]

                    # ### FIXME: For test, KiDS-North mask was downloaded from kids-sbi repository
                    # ### https://github.com/mwiet/kids_sbi . However, I cannot read the binary type
                    # ### file, which may consider some effects depending on position. So I just read
                    # ### the mask as a float64 array and convert it to boolean
                    mask = np.where(mask > 0, 1, 0)
                case "FullSky":
                    nside = 1024
                    mask = np.ones_like(12*nside*nside)

                case "boss_cmass_ngc":
                    mask = pymangle.Mangle(mask_fname)

            masks[isurvey_name] = mask
        
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

    def _load_shear_maps(self, icosmo, irlz=0):
        shear_map_dict = {}
        for ishell in range(len(self.redshift_src_list)):
            redshift_src = self.redshift_src_list[ishell]

            shear_map_dict[f"shell{ishell}"] = {}
            shear_map_dict[f"shell{ishell}"]['redshift'] = redshift_src

            shear_sim_fname = self.shear_sim_fmt.format(icosmo, redshift_src)
            with h5py.File(shear_sim_fname, 'r') as f:
                A = np.array(f["Distortion_matrix"]["Raytraced"])

            gamma1 = -(A[0][0] - A[1][1]) / 2
            gamma2 = -(A[0][1] + A[1][0]) / 2

            shear_map_dict[f"shell{ishell}"]['gamma1'] = gamma1
            shear_map_dict[f"shell{ishell}"]['gamma2'] = gamma2

        return shear_map_dict

    def sample_hod_params(self, icosmo, irlz):
        cpar_fname, halo_fname = self._get_fnames(icosmo, irlz)
        cosmo = self._get_cosmo_instance(cpar_fname, otype='ccl')
        hod_halocat = self.cata_loader.load_pkd_halocat(halo_fname, cosmo, 
                                                        ofmt='hod', clean=False)
        hod_params_alive = self.hod_populator.find_hod_params(hod_halocat, seed_offset=icosmo)
        
        return hod_params_alive
    
    def gen_mock_gal(self, icosmo, irlz, ihod, ihod_param:np.ndarray, save=False):
        ''' Generate mock catalog pipeline '''

        # >>> =========   1. Load halo catalog   =========== <<<
        cpar_fname, halo_fname = self._get_fnames(icosmo, irlz)
        cosmo = self._get_cosmo_instance(cpar_fname, otype='ccl')
        hod_halocat = self.cata_loader.load_pkd_halocat(halo_fname, cosmo, 
                                                        ofmt='hod', clean=False)
        
        # >>> =========   2. Populate galaxies via HOD   =========== <<<
        OmegaM = cosmo.omega_x(a=1.0, species='matter')
        ihod_param_dict = self._make_hod_param_dict(ihod_param)
        indx = self._get_hod_seed_offset(icosmo, irlz, ihod)
        dict_of_gsamples = self.hod_populator.populate_galaxies(hod_halocat, ihod_param_dict, indx=indx, OmegaM=OmegaM)
        gsample_arr = self._gsample_dict_to_array(dict_of_gsamples)
        gal_pos, gal_vel = self.hod_populator.get_galaxy_features(gsample_arr, features=["pos", "vel"])

        # >>> =========   3. Box to lightcone and apply survey geometry & n(z)   =========== <<<
        galcone_fullsky = self.survey_generator.box_to_lightcone(cosmo, gal_pos=gal_pos, gal_adj_props={'gal_vel': gal_vel})
        galcone_survey = []
        for isurvey_name, isurvey_label in self.fore_survey_labels_dict.items():

            print(f"\nMaking {isurvey_name}-like galaxy mock")

            gen_func = self._pick_gen_mock_func(isurvey_name)
            galcone_curr = gen_func(galcone_fullsky, isurvey_name, isurvey_label)
            galcone_survey.append(galcone_curr)

        galcone_survey = np.concatenate(galcone_survey)

        # >>> =========   4. (Optional) Save to file   =========== <<<
        if save:
            galcone_survey_tb = Table(galcone_survey)
            galcone_survey_tb.write(self.gal_ofmt.format(icosmo, irlz, ihod))
            del galcone_survey_tb

        return galcone_survey

    def gen_mock_void(self, icosmo, irlz, ihod, galcone_survey, dive_input, dive_output, save=False):
        ''' Generate mock void pipeline '''

        # >>> =========   1. Load cosmology   =========== <<<
        cpar_fname, _ = self._get_fnames(icosmo, 0)
        cosmo = self._get_cosmo_instance(cpar_fname, otype='ccl')
        
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

        # >>> =========   3. (Optional) Save to file   =========== <<<
        if save:
            voidcone_survey_tb = Table(voidcone_survey)
            voidcone_survey_tb.write(self.void_ofmt.format(icosmo, irlz, ihod))
            del voidcone_survey_tb

        return voidcone_survey
    
    def gen_mock_shear(self, icosmo, irlz=0, save=True):
        ''' Generate mock shape catalog pipeline '''
        # >>> =========   1. Load shear maps   =========== <<<
        shear_maps_curr = self._load_shear_maps(icosmo)
        # >>> =========   2. Generate background positions & assign shear   =========== <<<
        shapecone_survey = []
        ### Loop of surveys
        for isurvey_name, isurvey_label in self.back_survey_labels_dict.items():

            print(f"\nMaking {isurvey_name}-like shape mock", flush=True)

            ### Loop of tomographic bins
            for itomo_name, itomo_label in self.tomo_labels_dict.items():

                shapecone_curr = self.shear_assigner.gen_gal_positions(ngal=self.back_ngals_dict[itomo_name], tomo_label=itomo_label, 
                                                                       survey_name=isurvey_name, survey_label=isurvey_label)
                shapecone_curr = self.shear_assigner.assign_shear(shapecone_curr, shear_maps_curr)
                ### FIXME: Support assigning weights by weight map in the future
                shapecone_curr = self.shear_assigner.assign_weights(shapecone_curr, weight_type='unity')

                print(f"Tomo {itomo_label}: {len(shapecone_curr)}", flush=True)

                shapecone_survey.append(shapecone_curr)

        shapecone_survey = np.concatenate(shapecone_survey)

        # >>> =========   3. (Optional) Save to file   =========== <<<
        if save:
            shapecone_survey_tb = Table(shapecone_survey)
            shapecone_survey_tb.write(self.shear_ofmt.format(icosmo))
            del shapecone_survey_tb
        
        return shapecone_survey


class FastPMRunner:
    def __init__(
            self, config: PipeConfig,
            halo_fmt: str,
            cosmo_par_fname: str,
            fore_mask_fnames_dict: dict,
            fore_nofz_fnames_dict: dict,
            fore_survey_labels_dict: dict,
            gal_ofmt: str = None,
            void_ofmt: str = None):
        self.config = config
        self.halo_fmt = halo_fmt
        self.cosmo_par_fname = str(cosmo_par_fname)
        self.fore_survey_labels_dict = fore_survey_labels_dict
        self.gal_ofmt = gal_ofmt
        self.void_ofmt = void_ofmt
        self.scale_factor = 1.0 / (1.0 + config.redshift)

        if set(fore_survey_labels_dict) != set(fore_nofz_fnames_dict):
            raise ValueError("foreground survey labels and n(z) keys must match")

        fore_masks = self._prepare_fore_masks(fore_mask_fnames_dict)
        fore_nofzs = self._prepare_fore_nofzs(fore_nofz_fnames_dict)
        self.cata_loader = CatalogLoader(config=config)
        self.hod_populator = HODPopulator(config=config)
        self.survey_generator = SurveyGenerator(
            config=config, masks=fore_masks, nofzs=fore_nofzs
        )
        self.void_finder = VoidFinder(config=config)

    def _prepare_fore_masks(self, mask_fnames):
        if "boss_veto" not in mask_fnames:
            raise ValueError("fore_mask_fnames_dict must contain boss_veto")

        survey_names = [name for name in mask_fnames if name != "boss_veto"]
        have_boss = any("boss" in name for name in survey_names)
        have_2dflens = any("2dflens" in name for name in survey_names)
        masks = {}

        if have_boss:
            if not mask_fnames["boss_veto"]:
                raise ValueError("Must provide BOSS veto mask files")
            if any("lowze2_ngc" in name or "lowze3_ngc" in name
                   for name in survey_names):
                if "boss_lowz_ngc" not in survey_names:
                    raise ValueError("Must provide LOWZ NGC geometry")
            if any("lowze2_sgc" in name or "lowze3_sgc" in name
                   for name in survey_names):
                if "boss_lowz_sgc" not in survey_names:
                    raise ValueError("Must provide LOWZ SGC geometry")
            masks["boss_geom"] = {}
            masks["boss_masks"] = [
                pymangle.Mangle(path) for path in mask_fnames["boss_veto"]
            ]

        if have_2dflens:
            masks["2dflens_geom"] = {}

        for survey_name in survey_names:
            if "boss" in survey_name:
                masks["boss_geom"][survey_name] = pymangle.Mangle(
                    mask_fnames[survey_name]
                )
            elif "2dflens" in survey_name:
                masks["2dflens_geom"][survey_name] = loadFitsMaps(
                    mask_fnames[survey_name]
                )
            else:
                raise ValueError(
                    f"Unsupported foreground mask: {survey_name}"
                )
        return masks

    def _prepare_fore_nofzs(self, nofz_fnames):
        nofz_info = {}
        for survey_name, nofz_fname in nofz_fnames.items():
            if "boss" in survey_name:
                nofz = np.loadtxt(nofz_fname, usecols=(1, 2, 3, 5))
            elif "2dflens" in survey_name:
                nofz = np.loadtxt(nofz_fname, usecols=(1, 2, 3, 4))
            else:
                raise ValueError(
                    f"Unsupported foreground n(z): {survey_name}"
                )
            nofz_info = make_nofz_info(
                nofz_info,
                survey_name,
                np.append(nofz[:, 0], nofz[-1, 1]),
                nofz[:, 3],
                nofz[:, 2],
            )

        if "boss_cmass" in nofz_info:
            nofz_info["boss_cmass"]["nz_ref"] *= 0.93
        return nofz_info

    def _get_cosmo_instance(self, icosmo: int, otype="ccl"):
        with open(self.cosmo_par_fname, "r") as stream:
            fixed_line = stream.readline().strip()
            varying_line = stream.readline().strip()

        if not fixed_line.startswith("#") or not varying_line.startswith("#"):
            raise ValueError("FastPM cosmology file must start with two headers")

        fixed = {}
        for item in fixed_line[1:].split():
            if "=" not in item:
                raise ValueError("malformed fixed cosmology header")
            name, value = item.split("=", 1)
            fixed[name] = float(value)

        varying_names = varying_line[1:].split()
        required_fixed = {"hubble", "Omegab", "ns"}
        required_varying = {"OmegaM", "S8"}
        if not required_fixed.issubset(fixed):
            raise ValueError("missing fixed FastPM cosmology parameters")
        if not required_varying.issubset(varying_names):
            raise ValueError("missing varying FastPM cosmology parameters")

        rows = np.loadtxt(self.cosmo_par_fname, comments="#", ndmin=2)
        if rows.shape[1] != len(varying_names):
            raise ValueError("cosmology header and data column counts differ")
        if icosmo < 0 or icosmo >= len(rows):
            raise IndexError(f"cosmology label {icosmo} is out of range")

        selected = dict(zip(varying_names, rows[icosmo]))
        OmegaM = float(selected["OmegaM"])
        if OmegaM <= fixed["Omegab"]:
            raise ValueError("OmegaM must be larger than Omegab")

        params = {
            **fixed,
            **{name: float(value) for name, value in selected.items()},
            "Omega_c": OmegaM - fixed["Omegab"],
            "sigma8": float(selected["S8"]) / np.sqrt(OmegaM / 0.3),
        }
        if otype == "dict":
            return params
        if otype == "ccl":
            return ccl.Cosmology(
                h=params["hubble"],
                Omega_b=params["Omegab"],
                Omega_c=params["Omega_c"],
                sigma8=params["sigma8"],
                n_s=params["ns"],
                w0=-1.0,
                wa=0.0,
                m_nu=0.0,
            )
        raise NotImplementedError(f"Output type {otype} not implemented")

    def _get_halo_fname(self, icosmo: int) -> str:
        halo_fname = self.halo_fmt.format(icosmo, self.scale_factor)
        if os.path.basename(halo_fname) != "out_0_wPID.list":
            raise ValueError(
                "FastPM halo catalog must be out_0_wPID.list: "
                f"{halo_fname}"
            )
        if not os.path.isfile(halo_fname):
            raise FileNotFoundError(
                f"Parent-processed Rockstar catalog not found: {halo_fname}"
            )
        with open(halo_fname, "r") as stream:
            columns = stream.readline().lstrip("#").split()
        if "PID" not in columns:
            raise ValueError(f"Rockstar catalog has no PID column: {halo_fname}")
        return halo_fname

    def _get_sampling_seed_offset(self, icosmo, irlz):
        return icosmo * self.config.nrlzs_per_cosmo + irlz

    def _load_hod_halocat(self, icosmo):
        cosmo = self._get_cosmo_instance(icosmo, otype="ccl")
        halo_fname = self._get_halo_fname(icosmo)
        halo_catalog = self.cata_loader.load_rstar_halocat(
            halo_fname,
            cosmo=cosmo,
            ofmt="hod",
            clean=False,
            host_only=True,
        )
        return cosmo, halo_catalog

    def sample_hod_params(self, icosmo, irlz=0):
        _, halo_catalog = self._load_hod_halocat(icosmo)
        return self.hod_populator.find_hod_params(
            halo_catalog,
            seed_offset=self._get_sampling_seed_offset(icosmo, irlz),
        )

    def _get_hod_seed_offset(self, icosmo, irlz, ihod):
        return (
            icosmo
            * self.config.nrlzs_per_cosmo
            * self.config.nhod_per_cosmo
            + irlz * self.config.nhod_per_cosmo
            + ihod
        )

    def _make_hod_param_dict(self, hod_param):
        return dict(zip(self.config.model_params_names, hod_param))

    def _gsample_dict_to_array(self, dict_of_gsamples):
        return dict_of_gsamples[next(iter(dict_of_gsamples))]

    def _pick_gen_mock_func(self, survey_name):
        boss_like = {
            "boss_lowz_ngc", "boss_cmass_ngc",
            "boss_lowz_sgc", "boss_cmass_sgc",
        }
        boss_trim = {
            "boss_lowze2_ngc", "boss_lowze3_ngc",
            "boss_lowze2_sgc", "boss_lowze3_sgc",
        }
        if survey_name in boss_like:
            return self.survey_generator.gen_boss_like
        if survey_name in boss_trim:
            return self.survey_generator.gen_boss_like_trim
        if survey_name == "2dflens_south":
            return self.survey_generator.gen_2dflens_like
        raise ValueError(f"Unsupported foreground survey: {survey_name}")

    def gen_mock_gal(self, icosmo, irlz, ihod, ihod_param: np.ndarray,
                     save=False):
        if save and self.gal_ofmt is None:
            raise ValueError("gal_ofmt is required when save=True")

        cosmo, halo_catalog = self._load_hod_halocat(icosmo)
        OmegaM = cosmo.omega_x(a=1.0, species="matter")
        model_params = self._make_hod_param_dict(ihod_param)
        seed_offset = self._get_hod_seed_offset(icosmo, irlz, ihod)
        samples = self.hod_populator.populate_galaxies(
            halo_catalog,
            model_params,
            indx=seed_offset,
            OmegaM=OmegaM,
        )
        galaxies = self._gsample_dict_to_array(samples)
        gal_pos, gal_vel = self.hod_populator.get_galaxy_features(
            galaxies, features=["pos", "vel"]
        )
        fullsky = self.survey_generator.box_to_lightcone(
            cosmo, gal_pos=gal_pos, gal_adj_props={"gal_vel": gal_vel}
        )

        survey_catalogs = []
        for survey_name, survey_label in self.fore_survey_labels_dict.items():
            generator = self._pick_gen_mock_func(survey_name)
            survey_catalogs.append(generator(fullsky, survey_name, survey_label))
        if not survey_catalogs:
            raise ValueError("No foreground surveys configured")
        result = np.concatenate(survey_catalogs)

        if save:
            Table(result).write(self.gal_ofmt.format(icosmo, irlz, ihod))
        return result
