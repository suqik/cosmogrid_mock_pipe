import os
import numpy as np
from astropy.table import Table
import sys
sys.path.append("/home/suchen/Program/CosmoGrid/")

from utils.io_func import *

from astropy.cosmology import FlatwCDM
from dsigma.precompute import precompute

from dsigma.stacking import excess_surface_density
from dsigma.jackknife import compute_jackknife_fields, jackknife_resampling

from loguru import logger

def cal_bin_interval(start, end, num, bintype='lin'):
    if bintype == 'lin':
        dr = (end - start) / (num - 1)
    if bintype == 'log':
        lg_start = np.log10(start)
        lg_end = np.log10(end)
        dr = cal_bin_interval(lg_start, lg_end, num, bintype='lin')

    return dr

def bin2edge(bin_arr, bintype='lin'):
    if bintype == 'lin':
        dr = cal_bin_interval(bin_arr[0], bin_arr[-1], len(bin_arr), bintype=bintype)
        edge_start = bin_arr[0] - dr/2
        if edge_start < 0:
            edge_start = bin_arr[0] + dr/2
        edge_end   = bin_arr[-1] + dr/2
        edge_arr = np.linspace(edge_start, edge_end, len(bin_arr)+1)
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
    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    print(n_jobs)

    ### load cosmology and hod params

    hod_param_fname = "cfgs/hod/hod_5params_dict_Nsat_7.5.json"
    # hod_param_fname = "cfgs/hod/hod_5params_dict.json"

    hod_params_dict = get_hod_params(hod_param_fname)
    cosmo_labels_tot = []

    for icosmo_str in hod_params_dict.keys():
        if len(hod_params_dict[icosmo_str]) > 0:
            cosmo_labels_tot.append(int(icosmo_str[5:]))

    cosmo_labels_tot = cosmo_labels_tot[33:]

    ### setup cosmology
    ncosmos = len(cosmo_labels_tot)
    nchunk = ncosmos // 2

    rvmin = 18
    rvmax = 25
    rvmean = 0.5*(rvmin + rvmax)

    dr = cal_bin_interval(0.15, 1, 5, bintype='lin')
    rp_bins_inRv = np.arange(0.15, 3.0, dr)
    rp_edges_inRv = bin2edge(rp_bins_inRv, bintype='lin')
    rp_edges = rp_edges_inRv * rvmean # Mpc/h

    ### basename of catalogs
    vnamebase = "/data2/suchen/CosmoGrid/Void/Nsat_7.5/cosmo_{:06d}_run_0_HOD_{:d}_run_0_boss_north.npy"
    rnamebase = "/data2/suchen/CosmoGrid/Rand/Nsat_7.5/cosmo_{:06d}_run_0_HOD_{:d}_run_0_boss_north.npy"
    # vnamebase = "/data2/suchen/CosmoGrid/Void/cosmo_{:06d}_run_0_HOD_{:d}_run_0_boss_north.npy"
    # rnamebase = "/data2/suchen/CosmoGrid/Rand/cosmo_{:06d}_run_0_HOD_{:d}_run_0_boss_north.npy"
    bg_gal_fbase = "/data2/suchen/CosmoGrid/Shape/cosmo_{:06d}_run_0_kids_north_tomo{:d}_sigma0.3.txt"

    # for cosmo_label in cosmo_labels_tot:
    # for cosmo_label in cosmo_labels_tot[0:nchunk]:
    for cosmo_label in cosmo_labels_tot[nchunk:]:
    # for cosmo_label in cosmo_labels_tot[nchunk*2:nchunk*3]:
    # for cosmo_label in cosmo_labels_tot[nchunk*3:]:

        logger.info("Initialize")

        cosmo_dict = get_cosmo_from_file(f"/data3/suchen/CosmoGridV1/grid/cosmo_{cosmo_label:06d}/run_0/params.yml", otype='dict')
        H0 = cosmo_dict['H0']
        w0 = cosmo_dict['w0']
        Om0 = cosmo_dict['Ob'] + cosmo_dict['O_cdm']

        # rp_bins_inRv = np.logspace(0.1, 3, 16)
        # rp_bins = np.linspace(0.1, 200, 32) # Mpc
        cosmology = FlatwCDM(H0=H0, Om0=Om0, w0=w0)
        lens_source_cut = 0.1

        rp_edges = rp_edges/(H0/100.) # Mpc

        logger.info("Load shear catalog")

        itomo = 4 # Actually TOMO5
        bg_galcat = np.loadtxt(bg_gal_fbase.format(cosmo_label, itomo+1), dtype=bgal_type)

        bg_galcat_tb = Table(bg_galcat)
        bg_galcat_tb.rename_column('g1', 'e_1')
        bg_galcat_tb.rename_column('g2', 'e_2')

        bg_galcat_tb['e_1'] = -bg_galcat_tb['e_1']

        for hod_label in range(10):

            ### load void catalog
            logger.info("Load void catalog")

            mock_data_north = np.load(vnamebase.format(cosmo_label, hod_label))
            rvcut = (mock_data_north['Rv'] > rvmin) & (mock_data_north['Rv'] < rvmax)
            mock_data_north = mock_data_north[rvcut]

            ### load random catalog
            logger.info("Load random catalog")
            
            mock_rand_north = np.load(rnamebase.format(cosmo_label, hod_label))
            rvcut = (mock_rand_north['Rv'] > rvmin) & (mock_rand_north['Rv'] < rvmax)
            mock_rand_north = mock_rand_north[rvcut]

            logger.info(f"Data: {len(mock_data_north)} Random: {len(mock_rand_north)}")

            mock_gal_tb = Table(mock_data_north)
            mock_gal_tb['ra'] = mock_data_north['ra']
            mock_gal_tb['dec'] = mock_data_north['dec']
            mock_gal_tb['z'] = mock_data_north['z']
            mock_gal_tb['w_sys'] = mock_data_north['w']
            # mock_gal_tb['survey'] = mock_data_north['survey']

            mock_rand_north_tb = Table()
            mock_rand_north_tb['ra'] = mock_rand_north['ra']
            mock_rand_north_tb['dec'] = mock_rand_north['dec']
            mock_rand_north_tb['z'] = mock_rand_north['z']
            mock_rand_north_tb['w_sys'] = mock_rand_north['w']
            # mock_rand_north_tb['survey'] = mock_rand_north['survey']

            logger.info("Matching lens source catalogs")

            dcat_bossmock_matched = find_matching_samples_table(
                bg_galcat, mock_gal_tb, nside_x=500, nside_y=300
            )

            rcat_bossmock_matched = find_matching_samples_table(
                bg_galcat, mock_rand_north_tb, nside_x=500, nside_y=300
            )

            logger.info(f"Mathched data: {len(dcat_bossmock_matched)} Matched random: {len(rcat_bossmock_matched)}")

            logger.info("Begin measurements")

            logger.info("Precompute")

            precompute(dcat_bossmock_matched, bg_galcat_tb, rp_edges, cosmology=cosmology,
                                comoving=True, lens_source_cut=lens_source_cut,
                                progress_bar=True, n_jobs=n_jobs)

            precompute(rcat_bossmock_matched, bg_galcat_tb, rp_edges, cosmology=cosmology,
                                comoving=True, lens_source_cut=lens_source_cut,
                                progress_bar=True, n_jobs=n_jobs)

            logger.info("Setup jackknife centers")

            centers = compute_jackknife_fields(
                dcat_bossmock_matched, 100, weights=np.sum(dcat_bossmock_matched['sum 1'], axis=1))
            compute_jackknife_fields(rcat_bossmock_matched, centers)

            logger.info("Get ESD")

            esd = excess_surface_density(dcat_bossmock_matched,
                                        table_r=rcat_bossmock_matched, 
                                        return_table=True, 
                                        random_subtraction=True
                                        )

            logger.info("Evaluate jackknife errors")

            esd['ds_err'] = np.sqrt(np.diag(jackknife_resampling(
                    excess_surface_density, dcat_bossmock_matched, return_table=False)))

            logger.info("Save to file")

            esd.write(f'results/vlens/Nsat_7.5/cosmo_{cosmo_label:06d}_hod_{hod_label:d}_tomo{itomo+1}_rv{rvmin:.0f}{rvmax:.0f}.fits', overwrite=True)
            # esd.write(f'results/vlens/cosmo_{cosmo_label:06d}_hod_{hod_label:d}_tomo{itomo+1}_rv{rvmin:.0f}{rvmax:.0f}.fits', overwrite=True)
