import numpy as np
from loguru import logger
from astropy.cosmology import FlatwCDM
from dsigma.precompute import precompute

from dsigma.stacking import excess_surface_density
from dsigma.jackknife import compute_jackknife_fields, jackknife_resampling

from container import *
from dataclasses import dataclass

@dataclass
class GGLConfig:
    rp_min: float
    rp_max: float
    rp_bins: int
    rp_unit: str = 'mpc_h'
    bin_type: str = 'log'
    flip_g1: bool = False
    flip_g2: bool = False
    lens_source_cut: float = 0.2
    wRSD: bool = True
    wPhZ: bool = False
    wSN: bool = False
    njk: int = 1

class GGLCalculator:
    def __init__(self, config:GGLConfig):
        self.config = config
    
    def get_cosmo(self, cosmo_dict:dict):
        H0 = cosmo_dict["H0"]
        Om0 = cosmo_dict["Om0"]
        w0 = cosmo_dict["w0"]
        cosmology = FlatwCDM(H0=H0, Om0=Om0, w0=w0)
        return cosmology
    
    def _rps_to_mpc(self, hubble, Rvoid_mpch=1.0):
        rp_unit = self.config.rp_unit
        rp_min_ori = self.config.rp_min
        rp_max_ori = self.config.rp_max

        match rp_unit.lower():
            case 'mpc':
                factor = 1.0
            case 'mpc_h':
                factor = 1. / hubble
            case 'kpc':
                factor = 1e-3
            case 'kpc_h':
                factor = 1e-3 / hubble
            case 'gpc':
                factor = 1e3
            case 'gpc_h':
                factor = 1e3 / hubble
            case 'rv':
                factor = Rvoid_mpch / hubble

        rp_min_mpc = rp_min_ori * factor
        rp_max_mpc = rp_max_ori * factor
        
        return rp_min_mpc, rp_max_mpc
    
    def get_bin_edges(self, hubble, Rvoid_mpch=1.0):
        rp_bins = self.config.rp_bins
        bin_type = self.config.bin_type
        rp_min_mpc, rp_max_mpc = self._rps_to_mpc(hubble, Rvoid_mpch)

        if bin_type == "linear":
            bin_edges = np.linspace(rp_min_mpc, rp_max_mpc, rp_bins+1)
            # bin_ctrs = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        
        if bin_type == "log":
            bin_edges = np.geomspace(rp_min_mpc, rp_max_mpc, rp_bins+1)
            # bin_ctrs = np.sqrt(bin_edges[1:] * bin_edges[:-1])
    
        return bin_edges
    
    def _convert_table_dtype(self, tb, dtype_tbc, dtype_target):
        for col in tb.colnames:
            if isinstance(tb[col][0], dtype_tbc):
                tb[col] = tb[col].astype(dtype_target)
        return tb
    
    def mk_srcs_cat(self, cat:SurveyData):
        flip_g1 = self.config.flip_g1
        flip_g2 = self.config.flip_g2
        lens_source_cut = self.config.lens_source_cut

        srcs_cat = cat.to_astropy_table()
        srcs_cat = self._convert_table_dtype(srcs_cat, np.float32, np.double)

        if self.config.wSN:
            srcs_cat.rename_columns(['g1', 'g2'], ['e_1', 'e_2'])
        else:
            srcs_cat.rename_columns(['g1_pure', 'g2_pure'], ['e_1', 'e_2'])
        if not self.config.wPhZ:
            srcs_cat.remove_column('z')
            srcs_cat.rename_column('z_true', 'z')

        if flip_g1:
            srcs_cat['e_1'] = -srcs_cat['e_1']
        if flip_g2:
            srcs_cat['e_2'] = -srcs_cat['e_2']

        srcs_cat['z_l_max'] = srcs_cat['z'] - lens_source_cut

        return srcs_cat
    
    def mk_lens_cat(self, cat:SurveyData):
        logger.info("Catalog size: {}".format(len(cat)))
        lens_cat = cat.to_astropy_table()
        lens_cat = self._convert_table_dtype(lens_cat, np.float32, np.double)
        lens_cat.rename_column('w', 'w_sys')
        ### only galaxy catalogs have 'z' and 'zrsd'
        if self.config.wRSD and "zrsd" in lens_cat.colnames:
            lens_cat.remove_column('z')
            lens_cat.rename_column('zrsd', 'z')
        return lens_cat

    def get_Rv_mean_mpch(self, lens_cat:SurveyData):
        self.Rv_mean_mpch = np.median(lens_cat.Rv)
        return self.Rv_mean_mpch

    def compute_pairs(self, cosmo_dict:dict, 
                lens_table:Table, srcs_table:Table,
                n_jobs=32):
        
        cosmology = self.get_cosmo(cosmo_dict)
        hubble = cosmology.H(0).value / 100
        if self.config.rp_unit == "Rv":
            self.Rv_mean_mpc = self.Rv_mean_mpch / hubble
            rp_edges_mpc = self.get_bin_edges(hubble, self.Rv_mean_mpch)
        else:
            rp_edges_mpc = self.get_bin_edges(hubble)

        logger.info("Precomputing lens-source pairs")
        precompute(lens_table, srcs_table, rp_edges_mpc, 
                   cosmology=cosmology, comoving=True,
                   progress_bar=True, n_jobs=n_jobs)
        
        lens_table = lens_table[np.sum(lens_table['sum 1'], axis=1) > 0]

        return lens_table
    
    def stack_signals(self, lens_table, rand_table=None):
        if rand_table is not None:
            esd_kwargs = {'table_r': rand_table,
                          'random_subtraction': True}
        else:
            esd_kwargs = {'random_subtraction': False}

        esd = excess_surface_density(lens_table,
                                    return_table=True,
                                    **esd_kwargs
                                    )

        if self.config.rp_unit == "Rv":
            esd['rp_min_Rv'] = esd['rp_min'] / self.Rv_mean_mpc
            esd['rp_max_Rv'] = esd['rp_max'] / self.Rv_mean_mpc
        
        return esd
    
    def estimate_jackknife_cov(self, lens_table, rand_table=None, return_samples=False):
        njk = self.config.njk
        if njk <= 2:
            raise ValueError("Jackknife subsamples should be larger than 2!")
        centers = compute_jackknife_fields(
            lens_table, njk, weights=np.sum(lens_table['sum 1'], axis=1))
        
        if rand_table is not None:
            compute_jackknife_fields(rand_table, centers)
            esd_kwargs = {'table_r': rand_table,
                          'random_subtraction': True}
        else:
            esd_kwargs = {'random_subtraction': False}

        if return_samples:
            jk_cov, jk_samples = jackknife_resampling(excess_surface_density, 
                                      lens_table, return_samples=return_samples, 
                                      **esd_kwargs)
            
            return jk_cov, jk_samples
        else:
            jk_cov = jackknife_resampling(excess_surface_density, 
                                      lens_table, return_samples=return_samples, 
                                      **esd_kwargs)
        
            return jk_cov