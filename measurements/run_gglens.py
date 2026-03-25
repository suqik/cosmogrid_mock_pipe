import warnings
import numpy as np
from astropy.table import Table
from matplotlib import pyplot as plt

import sys
sys.path.append("/home/suchen/Program/CosmoGrid/")

from utils.io_func import *
from utils.mkfore_utils import *

from astropy.cosmology import FlatwCDM
from dsigma.precompute import precompute

from dsigma.stacking import excess_surface_density
from dsigma.jackknife import compute_jackknife_fields, jackknife_resampling

from loguru import logger

class Dsigma_Runner(object):
    def __init__(self, cosmo_file_fmt, lens_file_fmt, srcs_file_fmt, 
                 ran_radec_fname=None, ran_cat_fname=None,
                 ls_cut=0.1, rp_edges_mpch=np.logspace(0,2,15)):
        
        self.cosmo_file_fmt = cosmo_file_fmt
        self.lens_file_fmt = lens_file_fmt
        self.srcs_file_fmt = srcs_file_fmt
        self.ran_radec_fname = ran_radec_fname
        self.ran_cat_fname = ran_cat_fname
        self.lens_source_cut = ls_cut
        self.rp_edges_mpch = rp_edges_mpch # in Mpc/h
  
        if self.ran_cat_fname is not None:
            self.ran_cat = np.load(self.ran_cat_fname)
            self.FLAG_RAN = True # do not need to estimate (z,Rv) from data
        elif self.ran_radec_fname is not None:
            self.ran_radec = np.load(self.ran_radec_fname)
            self.FLAG_RAN = False

        self.FLAG_COSMO = False
        self.FLAG_RP_WOH = False
        self.FLAG_LEN = False
        self.FLAG_SRC = False
        self.FLAG_MATCH = False

    def __convert_table_dtype(self, tb, target_dtype):
        for col in tb.colnames:
            tb[col] = tb[col].astype(target_dtype)
        return tb
    
    def __get_min_max(self, cat:np.ndarray, keys:str):
        results = []
        for key in keys:
            try:
                results += [cat[key].min(), cat[key].max()]
            except:
                raise ValueError("Column {} does not exist".format(key))
        
        return tuple(results)
    
    def current_state(self):
        print("="*28)
        print("    Cosmology: {}            ".format(self.FLAG_COSMO))
        print("    Lens catalog: {}         ".format(self.FLAG_LEN))
        print("    Source catalog: {}       ".format(self.FLAG_SRC))
        print("    Random catalog: {}       ".format(self.FLAG_RAN))
        print("    Matched catalog: {}      ".format(self.FLAG_MATCH))
        print("    rp in Mpc: {}            ".format(self.FLAG_RP_WOH))
        print("="*28)

    def load_cosmology(self, cosmo_label):
        cosmo_dict = get_cosmo_from_file(self.cosmo_file_fmt.format(cosmo_label), otype='dict')
        H0 = cosmo_dict['H0']
        w0 = cosmo_dict['w0']
        Om0 = cosmo_dict['Ob'] + cosmo_dict['O_cdm']
        self.cosmology = FlatwCDM(H0=H0, Om0=Om0, w0=w0)
        self.FLAG_COSMO = True
        
        return None
    
    def set_cosmology(self, Om0, H0, w0=-1):
        if self.FLAG_COSMO:
            warnings.warn("Will ignore previous set cosmology")

        self.cosmology = FlatwCDM(H0=H0, Om0=Om0, w0=w0)
        self.FLAG_COSMO = True

        return None
    
    def set_phys_sep_bin(self):
        if not self.FLAG_COSMO:
            raise AttributeError("Cosmology not set!")
        
        hubble = self.cosmology.H(0).value / 100
        self.rp_edges_mpc = self.rp_edges_mpch / hubble 
        self.FLAG_RP_WOH = True

        return None
    
    def mk_srcs_cat(self, srcs_filename, flip_g1=False, flip_g2=False, wphz=False, wSN=True, logger=None):
        if not isinstance(srcs_filename, list):
            srcs_filename = [srcs_filename]
        
        srcs_cat = []
        for i_srcs_cat_fname in srcs_filename:
            if logger is not None:
                logger.info("Load source catalog from {}".format(i_srcs_cat_fname))
            
            i_srcs_cat = np.load(i_srcs_cat_fname)
            srcs_cat.append(i_srcs_cat)

        srcs_cat = np.concatenate(srcs_cat)

        self.srcs_cat = Table(srcs_cat)
        self.srcs_cat = self.__convert_table_dtype(self.srcs_cat, np.double)
        if wSN:
            self.srcs_cat.rename_columns(['g1', 'g2'], ['e_1', 'e_2'])
        else:
            self.srcs_cat.rename_columns(['g1_pure', 'g2_pure'], ['e_1', 'e_2'])

        if not wphz:
            logger.info("Use true z")
            self.srcs_cat.remove_column('z')
            self.srcs_cat.rename_column('z_true', 'z')

        if flip_g1:
            self.srcs_cat['e_1'] = -self.srcs_cat['e_1']
        if flip_g2:
            self.srcs_cat['e_2'] = -self.srcs_cat['e_2']

        logger.debug(f"zmin = {self.srcs_cat['z'].min():.2f}, zmax = {self.srcs_cat['z'].max():.2f}")

        self.FLAG_SRC = True

        return None
    
    def mk_lens_cat(self, lens_filename, cut:dict={'Rv': [18,25]}, wrsd=False, logger=None):
        if logger is not None:
            logger.info("Load lens catalog from {}".format(lens_filename))
        lens_cat = np.load(lens_filename)
        for key, value in cut.items():
            lens_cat = lens_cat[(lens_cat[key] >= value[0]) & (lens_cat[key] <= value[1])]
        self.lens_cat = Table(lens_cat)
        self.lens_cat = self.__convert_table_dtype(self.lens_cat, np.double)
        self.lens_cat.rename_column('w', 'w_sys')
        ### only galaxy catalogs have 'z' and 'zrsd'
        if wrsd and "zrsd" in self.lens_cat.colnames:
            self.lens_cat.remove_column('z')
            self.lens_cat.rename_column('zrsd', 'z')

        self.FLAG_LEN = True

        return None
    
    def mk_rand_cat(self, logger=None):
        if not self.FLAG_RAN:
            north_cut = (self.lens_cat['survey'] != 3)
            ran_cat_north = self.__mk_rand_cat_intermidiate(self.lens_cat[north_cut], logger)
            ran_cat_south = self.__mk_rand_cat_intermidiate(self.lens_cat[np.logical_not(north_cut)], logger)
            self.ran_cat = np.concatenate([ran_cat_north, ran_cat_south])
            self.ran_cat = Table(self.ran_cat)
            self.ran_cat = self.__convert_table_dtype(self.ran_cat, np.double)
            self.ran_cat.rename_column('w', 'w_sys')

            self.FLAG_RAN = True

        return None
    
    def __mk_rand_cat_intermidiate(self, catalog:np.ndarray, logger):
        zmin, zmax, Rvmin, Rvmax = self.__get_min_max(catalog, ['z', 'Rv'])
        z_rv_bounds = [(zmin, zmax), (Rvmin, Rvmax)]
        if logger is not None:
            logger.info("Build KDE of p(z,Rv)")
        zRv_KDE = bounded_kde_transform(np.c_[catalog['z'], catalog['Rv']], z_rv_bounds)
        ran_cat = self.ran_radec.copy()
        if logger is not None:
            logger.info("Resampling (z,Rv)")
        ran_cat['z'], ran_cat['Rv'] = resample_bounded(zRv_KDE, len(ran_cat), z_rv_bounds)

        return ran_cat

    def make_srcs_catalog(self, cosmo_label, tomo_label, flip_g1=False, flip_g2=False, wSN=False, logger=None):
        srcs_filename = self.srcs_file_fmt.format(cosmo_label, tomo_label+1)
        self.mk_srcs_cat(srcs_filename, flip_g1=flip_g1, flip_g2=flip_g2, wSN=wSN, logger=logger)
    
    def make_lens_catalog(self, cosmo_label, hod_label, cut:dict={'Rv': [18,25]}, logger=None):
        lens_filename = self.lens_file_fmt.format(cosmo_label, hod_label)
        self.mk_lens_cat(lens_filename, cut=cut, logger=logger)
    
    def match_back_fore_catalogs(self, nside_x=500, nside_y=300, logger=None):
        if logger is not None:
            logger.info("Matching lens and source catalogs")
        self.lens_cat_matched = find_matching_samples_table(self.srcs_cat, self.lens_cat, nside_x=nside_x, nside_y=nside_y)
        if self.FLAG_RAN:
            if logger is not None:
                logger.info("Matching rand and source catalogs")
            self.ran_cat_matched = find_matching_samples_table(self.srcs_cat, self.ran_cat, nside_x=nside_x, nside_y=nside_y)
        else:
            self.ran_cat_matched = None
        
        self.FLAG_MATCH = True

        if logger is not None:
            logger.info(f"Original Lens: {(self.lens_cat['survey'] != 3).sum()} + {(self.lens_cat['survey'] == 3).sum()}")
            logger.info(f"Matched Lens: {(self.lens_cat_matched['survey'] != 3).sum()} + {(self.lens_cat_matched['survey'] == 3).sum()}")
            if self.FLAG_RAN:
                logger.info(f"Original Rand: {len(self.ran_cat)}")
                logger.info(f"Matched Rand: {len(self.ran_cat_matched)}")

        return None
    
    def check_catalog(self, odir="./"):
        fig = plt.figure(figsize=(13,13))
        ax = fig.add_subplot(projection="mollweide")
        ax.set_title('MOCK')
        ax.scatter(np.deg2rad(180-self.srcs_cat["ra"]), np.deg2rad(self.srcs_cat["dec"]), s=0.01, alpha=0.1, marker='.')
        ax.scatter(np.deg2rad(180-self.lens_cat_matched["ra"]), np.deg2rad(self.lens_cat_matched["dec"]), s=0.01, alpha=1, marker='.')
        ax.grid()
        fig.savefig(odir + "check_lens_catalog.png", dpi=300)

        fig = plt.figure(figsize=(13,13))
        ax = fig.add_subplot(projection="mollweide")
        ax.set_title('RANDOM')
        ax.scatter(np.deg2rad(180-self.srcs_cat["ra"]), np.deg2rad(self.srcs_cat["dec"]), s=0.01, alpha=0.1, marker='.')
        ax.scatter(np.deg2rad(180-self.ran_cat_matched["ra"]), np.deg2rad(self.ran_cat_matched["dec"]), s=0.01, alpha=1, marker='.')
        ax.grid()
        fig.savefig(odir + "check_rand_catalog.png", dpi=300)

        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.hist(self.lens_cat_matched['Rv'], bins=50, histtype='step', lw=2, label='lens', density=True)
        # ax.hist(self.ran_cat_matched['Rv'], bins=50, histtype='step', lw=2, label='rand', density=True)
        # ax.legend()
        # fig.savefig(odir + "check_Rv.png", dpi=300)

        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.hist(self.lens_cat_matched['z'], bins=50, histtype='step', lw=2, label='lens', density=True)
        # ax.hist(self.ran_cat_matched['z'], bins=50, histtype='step', lw=2, label='rand', density=True)
        # ax.legend()
        # fig.savefig(odir + "check_z.png", dpi=300)


    def __stack_signals(self, lens_catalog, njk, esd_kwargs):
        esd = excess_surface_density(lens_catalog,
                                    return_table=True,
                                    **esd_kwargs
                                    )
        if njk > 2:    
            if logger is not None:
                logger.info("Estimating jackknife errors")

            esd_cov = jackknife_resampling(
                    excess_surface_density, 
                    lens_catalog,
                    return_table=False, 
                    **esd_kwargs)
        
            esd['ds_err'] = np.sqrt(np.diag(esd_cov))

            return esd, esd_cov
        else:
            return esd
    
    def run(self, njk=1, n_jobs=32, logger=None):
        if not self.FLAG_COSMO:
            raise AttributeError("Cosmology not set!")
        if not self.FLAG_RP_WOH:
            raise AttributeError("Physical separation bin not set!")
        if not self.FLAG_LEN:
            raise AttributeError("Lens catalog not found!")
        if not self.FLAG_SRC:
            raise AttributeError("Source catalog not found!")
        if not self.FLAG_MATCH:
            warnings.warn("Lens and source catalogs not matched! If it is fullsky simulation, ignore this.")
            self.lens_cat_matched = self.lens_cat
            if self.FLAG_RAN:
                self.ran_cat_matched = self.ran_cat

        logger.info("Precomputing lens-source pairs")
        precompute(self.lens_cat_matched, self.srcs_cat, self.rp_edges_mpc, cosmology=self.cosmology,
                            comoving=True, lens_source_cut=self.lens_source_cut,
                            progress_bar=True, n_jobs=n_jobs)
        
        if logger is not None:
            logger.info("Separating jackknife fields")

        if njk > 2:
            centers = compute_jackknife_fields(
                    self.lens_cat_matched, njk, 
                    weights=np.sum(self.lens_cat_matched['sum 1'], axis=1)
            )

        if self.FLAG_RAN:
            if logger is not None:
                logger.info("Precomputing rand-source pairs")
            
            precompute(self.ran_cat_matched, self.srcs_cat, self.rp_edges_mpc, cosmology=self.cosmology,
                            comoving=True, lens_source_cut=self.lens_source_cut,
                            progress_bar=True, n_jobs=n_jobs)
            
            if logger is not None:
                logger.info("Separating jackknife fields")

            if njk > 2:
                compute_jackknife_fields(self.ran_cat_matched, centers)
            
        if logger is not None:
            logger.info("Stacking lensing signals")

        north_slt = (self.lens_cat_matched['survey'] != 3)
        north_rslt = (self.ran_cat_matched['survey'] != 3)

        #### stack north        
        if self.FLAG_RAN:
            kwargs = {
                'table_r': self.ran_cat_matched[north_rslt],
                'random_subtraction': True
            }
        else:
            kwargs = {}
        esd_n, esd_cov_n = self.__stack_signals(self.lens_cat_matched[north_slt], njk, kwargs)

        #### stack south        
        if self.FLAG_RAN:
            kwargs = {
                'table_r': self.ran_cat_matched[np.logical_not(north_rslt)],
                'random_subtraction': True
            }
        else:
            kwargs = {}
        esd_s, esd_cov_s = self.__stack_signals(self.lens_cat_matched[np.logical_not(north_slt)], njk, kwargs)

        #### combine        
        if self.FLAG_RAN:
            kwargs = {
                'table_r': self.ran_cat_matched,
                'random_subtraction': True
            }
        else:
            kwargs = {}
        esd, esd_cov = self.__stack_signals(self.lens_cat_matched, njk, kwargs)
        
        return esd_n, esd_cov_n, esd_s, esd_cov_s, esd, esd_cov

def cal_bin_interval(start, end, num, bintype='lin'):
    if bintype == 'lin':
        dr = (end - start) / (num - 1)
    if bintype == 'log':
        lg_start = np.log10(start)
        lg_end = np.log10(end)
        dr = cal_bin_interval(lg_start, lg_end, num, bintype='lin')

    return dr

def gen_rp_bins(start, end, dr, bintype='lin'):
    if bintype == 'lin':
        rp_bins = np.arange(start, end+dr/2., dr)
    if bintype == 'log':
        lg_start = np.log10(start)
        lg_end = np.log10(end)
        lg_rp_bins = gen_rp_bins(lg_start, lg_end, dr, bintype='lin')
        rp_bins = 10**lg_rp_bins

    return rp_bins  

def bin2edge(bin_arr, bintype='lin'):
    if bintype == 'lin':
        dr = cal_bin_interval(bin_arr[0], bin_arr[-1], len(bin_arr), bintype=bintype)
        edge_start = bin_arr[0] - dr/2
        nedge = len(bin_arr) + 1
        if edge_start < 0:
            edge_start = bin_arr[0] + dr/2
            nedge = len(bin_arr)
        edge_end   = bin_arr[-1] + dr/2
        edge_arr = np.linspace(edge_start, edge_end, nedge)
    if bintype == 'log':
        lgbin_arr = np.log10(bin_arr)
        lgedge_arr = bin2edge(lgbin_arr, bintype='lin')
        edge_arr = 10**lgedge_arr

    return edge_arr

def find_matching_samples_table(ref_cat, match_cat, nside_x=64, nside_y=64):
    
    ref_xy = np.vstack([ref_cat['ra'], ref_cat['dec']]).T
    match_xy = np.vstack([match_cat['ra'], match_cat['dec']]).T

    all_points = np.vstack([ref_xy, match_xy])
    xmin, ymin = np.min(all_points, axis=0)
    xmax, ymax = np.max(all_points, axis=0)

    x_edges = np.linspace(xmin, xmax, nside_x + 1)
    y_edges = np.linspace(ymin, ymax, nside_y + 1)

    def compute_indices(data):
        ix = np.digitize(data[:, 0], x_edges) - 1
        iy = np.digitize(data[:, 1], y_edges) - 1
        return ix, iy

    ref_ix, ref_iy = compute_indices(ref_xy)
    match_ix, match_iy = compute_indices(match_xy)

    def valid_mask(ix, iy, n_x, n_y):
        return (ix >= 0) & (ix < n_x) & (iy >= 0) & (iy < n_y)

    ref_mask = valid_mask(ref_ix, ref_iy, nside_x, nside_y)
    match_mask = valid_mask(match_ix, match_iy, nside_x, nside_y)

    ref_cells = set(zip(ref_ix[ref_mask], ref_iy[ref_mask]))
    match_cells = set(zip(match_ix[match_mask], match_iy[match_mask]))
    common_cells = ref_cells & match_cells

    match_selected = [i for i, (ix, iy) in enumerate(zip(match_ix, match_iy)) if (ix, iy) in common_cells]

    return match_cat[match_selected]

if __name__ == "__main__":
    n_jobs = 16

    ''' --------- set up separation bins -------- '''
    rvmin = 15.0 # Mpc/h
    rvmax = 30.0 # Mpc/h
    rvmean = 0.5*(rvmin + rvmax)

    bintype = 'log'
    dr = cal_bin_interval(0.15, 1, 10, bintype=bintype)
    rp_bins_inRv = gen_rp_bins(0.15, 3.0, dr, bintype=bintype)

    rp_edges_inRv = bin2edge(rp_bins_inRv, bintype=bintype)
    rp_edges = rp_edges_inRv * rvmean # Mpc/h

    ''' ---- set up file name formats ---- '''
    cparnamebase = "/data3/suchen/CosmoGridV1/grid/cosmo_{:06d}/run_0/params.yml"

    ### directly set lens/srcs filenames
    cosmo_label = 1
    survey_name = "cmass"
    wrsd = False
    if wrsd:
        lens_cat_fname = f"/data2/suchen/CosmoGrid/high_ngal_suits_wrsd/Void_{survey_name}_wrsd/cosmo_{cosmo_label:06d}_run_0_HOD_0_run_0_boss_north_2dflens_south.npy"
    else:
        lens_cat_fname = f"/data2/suchen/CosmoGrid/high_ngal_suits_wrsd/Void_{survey_name}/cosmo_{cosmo_label:06d}_run_0_HOD_0_run_0_boss_north_2dflens_south.npy"

    #### srcs fname can be single string
    # src_tomo_label = 4
    # srcs_cat_fname = f"/data2/suchen/CosmoGrid/Shape/KiDS_ngal_suits/cosmo_{cosmo_label:06d}_run_0_kids_north_tomo{src_tomo_label}.npy"
    #### or a list
    src_tomo_label = [4]
    ##### KiDS north
    srcs_cat_fbase = f"/data2/suchen/CosmoGrid/Shape/KiDS_ngal_suits_wphzerr/cosmo_{cosmo_label:06d}" + "_run_0_kids_north_tomo{:d}.npy"
    srcs_cat_fname = [srcs_cat_fbase.format(itomo) for itomo in src_tomo_label]
    ##### KiDS south
    srcs_cat_fbase = f"/data2/suchen/CosmoGrid/Shape/KiDS_ngal_suits_wphzerr/cosmo_{cosmo_label:06d}" + "_run_0_kids_south_tomo{:d}.npy"
    srcs_cat_fname += [srcs_cat_fbase.format(itomo) for itomo in src_tomo_label]

    wSN = False
    wphz = False
    rand_radec_fname = f"/data2/suchen/CosmoGrid/Rand/boss{survey_name}_north_2dflens_south_radec.npy"

    # ### or filename formats, used in batch runs
    # vnamebase = "/data2/suchen/CosmoGrid/fix_HOD_suits/old/Void/cosmo_{:06d}_run_0_HOD_{:d}_run_0_boss_north.npy"
    # ran_radec_fname = "/data2/suchen/CosmoGrid/Rand/boss_lowztot_bin1_north_radec.npy"
    # shape_noise_flag = "pure"
    # bgnamebase = f"/data2/suchen/CosmoGrid/Shape/{shape_noise_flag:s}_kids_ngal/" + "cosmo_{:06d}_run_0_kids_north_tomo{:d}.npy"
    # # ofilename = f'aux/results/vlens/{shape_noise_flag}/cosmo_{cosmo_label:06d}_hod_{hod_label:d}_tomo{tomo_label+1}_rv{rvmin:.0f}{rvmax:.0f}_{bintype}bin.fits'
    # ofilename = f'aux/results/vlens/tomobins/test_{shape_noise_flag}_cosmo_{cosmo_label:06d}_hod_{hod_label:d}_tomo{tomo_label+1}_rerv{scaled_rvmin:.0f}{scaled_rvmax:.0f}_{bintype}bin.fits'


    ''' ---- output file name ---- '''
    ofilebase = f"aux/results/vlens/cosmo_{cosmo_label:06d}_{survey_name}_2dflens_kids_"
    tomobase = "_".join([f"tomo{itomo}" for itomo in src_tomo_label])
    ftypebase = ".fits"

    ofilename = ofilebase + tomobase

    if wrsd:
        ofilename += "_wrsd"
    if wSN:
        ofilename += "_wSN"
    if wphz:
        ofilename += "_wphzerr"
    
    ofilename += ftypebase


    ''' ----------- Main routines ------------ '''


    logger.info("Initialize dsigma runner")

    runner = Dsigma_Runner(
        cosmo_file_fmt=cparnamebase,
        lens_file_fmt=None, 
        srcs_file_fmt=None,
        ran_radec_fname=rand_radec_fname,
        ran_cat_fname=None,
        ls_cut=0.2, # lens source cut
        rp_edges_mpch=rp_edges
    )

    logger.info("Processing measurements")

    logger.info("Load cosmology")

    runner.load_cosmology(cosmo_label)
    runner.set_phys_sep_bin()

    logger.info("Load catalogs")

    # runner.make_srcs_catalog(cosmo_label, tomo_label, flip_g1=True, logger=logger)
    # runner.make_lens_catalog(cosmo_label, hod_label, cut={'Rv': [rvmin, rvmax]}, logger=logger)
    runner.mk_srcs_cat(srcs_cat_fname, flip_g1=True, wSN=wSN, wphz=wphz, logger=logger)
    runner.mk_lens_cat(lens_cat_fname, cut={'Rv': [rvmin, rvmax]}, wrsd=wrsd, logger=logger)
    runner.mk_rand_cat(logger=logger)
    runner.match_back_fore_catalogs(nside_x=500, nside_y=300, logger=logger)

    # runner.check_catalog()
    logger.info("Run")

    esdn, esdn_cov, esds, esds_cov, esd, esd_cov = runner.run(njk=100, n_jobs=n_jobs, logger=logger)

    # logger.info(f"Save to file {ofilename}")

    # esdn.write(ofilename.replace('.fits', '_oN.fits'), overwrite=True)
    # np.save(ofilename.replace('.fits', '_oN.cov.npy'), esdn_cov)

    # esds.write(ofilename.replace('.fits', '_oS.fits'), overwrite=True)
    # np.save(ofilename.replace('.fits', '_oS.cov.npy'), esds_cov)

    # esd.write(ofilename, overwrite=True)
    # np.save(ofilename.replace('.fits', '.cov.npy'), esd_cov)

    # esd = runner.run(njk=1, n_jobs=n_jobs, logger=logger)
    # logger.info(f"Save to file {ofilename}")
    # esd.write(ofilename, overwrite=True)