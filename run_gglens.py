import numpy as np
import pickle
from loguru import logger
from astropy.table import Table
from utils.mkfore_utils import bounded_kde_transform, resample_bounded

from astropy.cosmology import FlatLambdaCDM
from dsigma.precompute import precompute
from dsigma.stacking import excess_surface_density
from dsigma.jackknife import compute_jackknife_fields, jackknife_resampling

from utils.io_func import *

rng_list = [np.random.default_rng(iseed+1427) for iseed in range(5)]

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

with open("/data3/suchen/CosmoGridV1/grid/dirnames.txt", "r") as f:
    dirnames = f.readlines()
    cosmo_labels_tot = [int(i.strip("\n").split("_")[1]) for i in dirnames]

bintype = 'log'
Rvmean = 21.5
dr = cal_bin_interval(0.15, 1, 10, bintype=bintype)
rp_bins_inRv = gen_rp_bins(0.15, 3.0, dr, bintype=bintype)
rp_edges_inRv = bin2edge(rp_bins_inRv, bintype=bintype)
# rp_edges = rp_edges_inRv * Rvmean # Mpc/h
lens_source_cut = 0.2

ngal_list = np.load("/data2/suchen/CosmoGrid/fix_HOD/ngals.npy")

sim_void_rand_radec = np.load("/data2/suchen/CosmoGrid/Rand/boss_cmasslowztot_north_radec.npy")

Rvmean_list = []
for idx, icosmo in enumerate(cosmo_labels_tot):
    if idx % 10 == 0:
        logger.info(f"Processing cosmo_{icosmo:06d}")
    cosmo_ccl = get_cosmo_from_file(f"/data3/suchen/CosmoGridV1/grid/cosmo_{icosmo:06d}/run_0/params.yml")
    hubble = cosmo_ccl.to_dict()['h']
    Om0 = cosmo_ccl.omega_x(1, 'matter')
    
    cosmology = FlatLambdaCDM(H0=100*hubble, Om0=Om0)

    ### shear catalog
    try:
        sim_shear_cat = np.load(f"/data2/suchen/CosmoGrid/Shape/sigma0.3_kids_ngal/cosmo_{icosmo:06d}_run_0_kids_north_tomo4.npy")
        sim_shear_cat['g1'] = -sim_shear_cat['g1']
        sim_shear_cat = Table(sim_shear_cat)
        sim_shear_cat.rename_columns(['g1', 'g2'], ['e_1', 'e_2'])
    except:
        continue
    ### lens catalog
    sim_void_cat = np.load(f"/data2/suchen/CosmoGrid/fix_HOD_Void/cosmo_{icosmo:06d}_run_0_HOD_0_run_0_boss_north.npy")

    zmin = np.minimum(sim_void_cat['z'].min(), 0.2)
    zmax = np.maximum(sim_void_cat['z'].max(), 0.4)
    Rvmin = sim_void_cat['Rv'].min()
    Rvmax = sim_void_cat['Rv'].max()

    z_rv_bounds = [(zmin, zmax), (Rvmin, Rvmax)]

    curr_ngal = ngal_list[idx]
    scaled_Rv = sim_void_cat['Rv']*np.cbrt(curr_ngal*1e-4) # Rv * n^(1/3)
    slt = (scaled_Rv > 1.2) & (scaled_Rv < 1.8)
    sim_void_cat = sim_void_cat[slt]

    Rvmean = np.mean(sim_void_cat['Rv'])
    Rvmean_list.append(Rvmean)
    rp_edges = rp_edges_inRv * Rvmean # Mpc/h
    curr_rp_edges = rp_edges / hubble

    sim_void_cat = Table(sim_void_cat)
    sim_void_cat.rename_column("w", "w_sys")

    ### rand catalog
    ### first build p(z, Rv), then sampling from it
    z_rv_bounds = [(zmin, zmax), (Rvmin, Rvmax)]
    zRv_KDE = bounded_kde_transform(np.c_[sim_void_cat['z'], sim_void_cat['Rv']], z_rv_bounds)

    sim_void_rancat = sim_void_rand_radec.copy()
    sim_void_rancat['z'], sim_void_rancat['Rv'] = resample_bounded(zRv_KDE, len(sim_void_rancat), z_rv_bounds)
    sim_void_rancat = Table(sim_void_rancat)
    sim_void_rancat.rename_column("w", "w_sys")

    precompute(sim_void_cat, sim_shear_cat, curr_rp_edges, cosmology=cosmology,
                        comoving=True, lens_source_cut=lens_source_cut,
                        progress_bar=True, n_jobs=28)
    
    precompute(sim_void_rancat, sim_shear_cat, curr_rp_edges, cosmology=cosmology,
                        comoving=True, lens_source_cut=lens_source_cut,
                        progress_bar=True, n_jobs=28)    

    esd = excess_surface_density(sim_void_cat,
                                 table_r=sim_void_rancat,
                                 random_subtraction=True,
                                 return_table=True)

    esd.write(f"results/vlens/fix_HOD/cosmo_{icosmo:06d}_tomo4_vlens.fits", overwrite=True)

np.save("results/vlens/fix_HOD/Rvmean_list.npy", Rvmean_list)