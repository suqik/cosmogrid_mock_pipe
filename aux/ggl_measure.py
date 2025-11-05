import os
import pickle
import numpy as np
from astropy.table import Table
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
    def __init__(self, cosmo_file_fmt, lens_file_fmt, srcs_file_fmt, ran_radec_fname=None, zRv_file_fmt=None, 
                 zlmin=0.2, zlmax=0.4, rvlmin=0., rvlmax=40.,
                 ls_cut=0.1, rp_edges_mpch=np.logspace(0,2,15)):
        
        self.cosmo_file_fmt = cosmo_file_fmt
        self.lens_file_fmt = lens_file_fmt
        self.srcs_file_fmt = srcs_file_fmt
        self.ran_radec_fname = ran_radec_fname
        self.zRv_file_fmt = zRv_file_fmt
        self.zRv_bound = [(zlmin, zlmax), (rvlmin, rvlmax)]
        self.lens_source_cut = ls_cut
        self.rp_edges_mpch = rp_edges_mpch # in Mpc/h

        if self.ran_radec_fname is not None:
            self.ran_radec = np.load(self.ran_radec_fname)

        self.FLAG_COSMO = False
        self.FLAG_RP_WOH = False
        self.FLAG_LEN = False
        self.FLAG_SRC = False
        self.FLAG_RAN = False
        self.FLAG_MATCH = False

    def __convert_table_dtype(self, tb, target_dtype):
        for col in tb.colnames:
            tb[col] = tb[col].astype(target_dtype)
        return tb

    def current_state(self):
        print("="*28)
        print("    Cosmology: {}        ".format(self.FLAG_COSMO))
        print("    Lens catalog: {}     ".format(self.FLAG_LEN))
        print("    Source catalog: {}   ".format(self.FLAG_SRC))
        print("    Random catalog: {}   ".format(self.FLAG_RAN))
        print("    Matched catalog: {}  ".format(self.FLAG_MATCH))
        print("    rp in Mpc: {}        ".format(self.FLAG_RP_WOH))
        print("="*28)

    def load_cosmology(self, cosmo_label):
        cosmo_dict = get_cosmo_from_file(self.cosmo_file_fmt.format(cosmo_label), otype='dict')
        H0 = cosmo_dict['H0']
        w0 = cosmo_dict['w0']
        Om0 = cosmo_dict['Ob'] + cosmo_dict['O_cdm']
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
    
    def make_srcs_catalog(self, cosmo_label, tomo_label, flip_g1=False, flip_g2=False, logger=None):
        srcs_filename = self.srcs_file_fmt.format(cosmo_label, tomo_label+1)
        if logger is not None:
            logger.info("Load source catalog from {}".format(srcs_filename))
        srcs_cat = np.load(srcs_filename)
        self.srcs_cat = Table(srcs_cat)
        self.srcs_cat = self.__convert_table_dtype(self.srcs_cat, np.double)
        self.srcs_cat.rename_columns(['g1', 'g2'], ['e_1', 'e_2'])

        if flip_g1:
            self.srcs_cat['e_1'] = -self.srcs_cat['e_1']
        if flip_g2:
            self.srcs_cat['e_2'] = -self.srcs_cat['e_2']

        self.FLAG_SRC = True
        
        return None
    
    def make_lens_catalog(self, cosmo_label, hod_label, cut:dict={'Rv': [18,25]}, logger=None):
        lens_filename = self.lens_file_fmt.format(cosmo_label, hod_label)
        if logger is not None:
            logger.info("Load lens catalog from {}".format(lens_filename))
        lens_cat = np.load(lens_filename)
        for key, value in cut.items():
            lens_cat = lens_cat[(lens_cat[key] >= value[0]) & (lens_cat[key] <= value[1])]
        self.lens_cat = Table(lens_cat)
        self.lens_cat = self.__convert_table_dtype(self.lens_cat, np.double)
        self.lens_cat.rename_column('w', 'w_sys')

        self.FLAG_LEN = True

        return None

    def make_rand_catalog(self, cosmo_label, hod_label, cut:dict={'Rv': [18,25]}, logger=None):
        zRv_filename = self.zRv_file_fmt.format(cosmo_label, hod_label)
        if logger is not None:
            logger.info("Load p(z,R) from {}".format(zRv_filename))
        with open(self.zRv_file_fmt.format(cosmo_label, hod_label), 'rb') as f:
            zRv_KDE = pickle.load(f)
        
        ran_cat = self.ran_radec.copy()
        ran_cat['z'], ran_cat['Rv'] = resample_bounded(zRv_KDE, len(ran_cat), self.zRv_bound)
        for key, value in cut.items():
            ran_cat = ran_cat[(ran_cat[key] >= value[0]) & (ran_cat[key] <= value[1])]
        self.ran_cat = Table(ran_cat)
        self.ran_cat = self.__convert_table_dtype(self.ran_cat, np.double)
        self.ran_cat.rename_column('w', 'w_sys')

        self.FLAG_RAN = True

        return None
    
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

        return None
    
    def run(self, n_jk=100, n_jobs=32, logger=None):
        if not self.FLAG_COSMO:
            raise AttributeError("Cosmology not set!")
        if not self.FLAG_RP_WOH:
            raise AttributeError("Physical separation bin not set!")
        if not self.FLAG_LEN:
            raise AttributeError("Lens catalog not found!")
        if not self.FLAG_SRC:
            raise AttributeError("Source catalog not found!")
        if not self.FLAG_MATCH:
            raise AttributeError("Lens and source catalogs not matched!")
        
        if logger is not None:
            logger.info("Precomputing lens-source pairs")

        precompute(self.lens_cat_matched, self.srcs_cat, self.rp_edges_mpc, cosmology=self.cosmology,
                            comoving=True, lens_source_cut=self.lens_source_cut,
                            progress_bar=True, n_jobs=n_jobs)
        
        if logger is not None:
            logger.info("Separating jackknife fields")

        centers = compute_jackknife_fields(
                self.lens_cat_matched, n_jk, 
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

            compute_jackknife_fields(self.ran_cat_matched, centers)
            
        if logger is not None:
            logger.info("Stacking lensing signals")

        esd = excess_surface_density(self.lens_cat_matched,
                                    table_r=self.ran_cat_matched, 
                                    return_table=True, 
                                    random_subtraction=True
                                    )
        
        if logger is not None:
            logger.info("Estimating jackknife errors")

        esd['ds_err'] = np.sqrt(np.diag(jackknife_resampling(
                excess_surface_density, self.lens_cat_matched, return_table=False)))
        
        return esd

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
    
    # 1. 提取 ra/dec 并转换为 numpy 数组
    ref_xy = np.vstack([ref_cat['ra'], ref_cat['dec']]).T
    match_xy = np.vstack([match_cat['ra'], match_cat['dec']]).T

    # 2. 合并所有坐标，确定边界
    all_points = np.vstack([ref_xy, match_xy])
    xmin, ymin = np.min(all_points, axis=0)
    xmax, ymax = np.max(all_points, axis=0)

    # 3. 网格边界（可异向）
    x_edges = np.linspace(xmin, xmax, nside_x + 1)
    y_edges = np.linspace(ymin, ymax, nside_y + 1)

    # 4. 计算网格索引
    def compute_indices(data):
        ix = np.digitize(data[:, 0], x_edges) - 1
        iy = np.digitize(data[:, 1], y_edges) - 1
        return ix, iy

    ref_ix, ref_iy = compute_indices(ref_xy)
    match_ix, match_iy = compute_indices(match_xy)

    # 5. 有效索引掩码
    def valid_mask(ix, iy, n_x, n_y):
        return (ix >= 0) & (ix < n_x) & (iy >= 0) & (iy < n_y)

    ref_mask = valid_mask(ref_ix, ref_iy, nside_x, nside_y)
    match_mask = valid_mask(match_ix, match_iy, nside_x, nside_y)

    # 6. 找出重合网格
    ref_cells = set(zip(ref_ix[ref_mask], ref_iy[ref_mask]))
    match_cells = set(zip(match_ix[match_mask], match_iy[match_mask]))
    common_cells = ref_cells & match_cells

    # 7. 选择落入重合网格的样本
    match_selected = [i for i, (ix, iy) in enumerate(zip(match_ix, match_iy)) if (ix, iy) in common_cells]

    return match_cat[match_selected]

if __name__ == "__main__":
    n_jobs = 28

    ### basic setups

    cosmo_label = 1
    hod_label = 0

    tomo_label = 4 # TOMO {tomo_label+1}

    rvmin = 18
    rvmax = 25
    rvmean = 0.5*(rvmin + rvmax)

    bintype = 'log'
    dr = cal_bin_interval(0.15, 1, 10, bintype=bintype)
    rp_bins_inRv = gen_rp_bins(0.15, 3.0, dr, bintype=bintype)
    rp_edges_inRv = bin2edge(rp_bins_inRv, bintype=bintype)
    rp_edges = rp_edges_inRv * rvmean # Mpc/h

    ### file basenames
    cparnamebase = "/data3/suchen/CosmoGridV1/grid/cosmo_{:06d}/run_0/params.yml"
    vnamebase = "/data2/suchen/CosmoGrid/Void/cosmo_{:06d}_run_0_HOD_{:d}_run_0_boss_north.npy"
    ran_radec_fname = "/data2/suchen/CosmoGrid/Rand/boss_cmasslowztot_north_radec.npy"
    zRvnamebase = "/data2/suchen/CosmoGrid/NofZ/Void/cosmo_{:06d}_run_0_HOD_{:d}_run_0_boss_north.pkl"
    shape_noise_flag = "sigma0.3"
    bgnamebase = f"/data2/suchen/CosmoGrid/Shape/{shape_noise_flag:s}_kids_ngal/" + "cosmo_{:06d}_run_0_kids_north_tomo{:d}.npy"

    logger.info("Initialize dsigma runner")

    runner = Dsigma_Runner(
        cosmo_file_fmt=cparnamebase,
        lens_file_fmt=vnamebase, 
        srcs_file_fmt=bgnamebase,
        ran_radec_fname=ran_radec_fname,
        zRv_file_fmt=zRvnamebase,
        ls_cut=0.1, # lens source cut
        zlmin=0.2, zlmax=0.4, rvlmin=0, rvlmax=40, # bounds of KDE of p(z, Rv)
        rp_edges_mpch=rp_edges
    )

    logger.info("Processing measurements")

    logger.info("Load cosmology")

    runner.load_cosmology(cosmo_label)
    runner.set_phys_sep_bin()

    logger.info("Load catalogs")

    runner.make_srcs_catalog(cosmo_label, tomo_label, flip_g1=True, logger=logger)
    runner.make_lens_catalog(cosmo_label, hod_label, cut={'Rv': [rvmin, rvmax]}, logger=logger)
    runner.make_rand_catalog(cosmo_label, hod_label, cut={'Rv': [rvmin, rvmax]}, logger=logger)
    runner.match_back_fore_catalogs(nside_x=500, nside_y=300, logger=logger)

    logger.info("Run")

    esd = runner.run(n_jk=100, n_jobs=n_jobs, logger=logger)

    ofilename = f'aux/results/vlens/{shape_noise_flag}/cosmo_{cosmo_label:06d}_hod_{hod_label:d}_tomo{tomo_label+1}_rv{rvmin:.0f}{rvmax:.0f}_{bintype}bin.fits'

    logger.info(f"Save to file {ofilename}")

    esd.write(ofilename, overwrite=True)