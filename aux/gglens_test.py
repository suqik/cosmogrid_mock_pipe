import numpy as np
from astropy.table import Table

import sys
sys.path.append("/home/suchen/Program/CosmoGrid/")
from src.io_func import get_cosmo_from_file

from loguru import logger

cosmo_label = 1

cosmo_ccl = get_cosmo_from_file(f"/data3/suchen/CosmoGridV1/grid/cosmo_{cosmo_label:06d}/run_0/params.yml")

catname = 'wb'
mock_gal = np.load(f"aux/catalogs/boss_lowze3_void_{catname}.npy")

mock_gal_north = mock_gal[mock_gal['dec']>-15]

rand_gal = np.load(f"aux/catalogs/boss_lowze3_void_rand_{catname}.npy")

logger.info(f"Data: {len(mock_gal_north)} Random: {len(rand_gal)}")

bgal_type = np.dtype(
    [
        ("ra", "f8"), 
        ("dec", "f8"), 
        ("z", "f8"), 
        ("sigz", "f8"),
        ("g1", "f8"), 
        ("g2", "f8"), 
        ("w", "f8")
    ]
)

bg_gal_ensemble = []
for tomo_bin in range(5):
    bg_galcat = np.loadtxt(f"catalogs/Shape/cosmo_{cosmo_label:06d}/cosmo_{cosmo_label:06d}_run_0_kids_north_tomo{tomo_bin+1}_wo_noise.txt", dtype=bgal_type)
    logger.info(f"Load mock source galaxy catalogs from `catalogs/Shape/cosmo_{cosmo_label:06d}/cosmo_{cosmo_label:06d}_run_0_kids_north_tomo{tomo_bin+1}_wo_noise.txt`")
    bg_gal_ensemble.append(bg_galcat)

itomo = 4 # Actually TOMO5
bg_galcat = bg_gal_ensemble[itomo]

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

mock_gal_tb = Table()
mock_gal_tb['ra'] = mock_gal_north['ra']
mock_gal_tb['dec'] = mock_gal_north['dec']
mock_gal_tb['z'] = mock_gal_north['z']
mock_gal_tb['w_sys'] = mock_gal_north['w']

rand_gal_tb = Table()
rand_gal_tb['ra'] = rand_gal['ra']
rand_gal_tb['dec'] = rand_gal['dec']
rand_gal_tb['z'] = rand_gal['z']
rand_gal_tb['w_sys'] = rand_gal['w']

logger.info("Matching lens source catalogs")

gcat_bossmock_matched = find_matching_samples_table(
    bg_galcat, mock_gal_tb, nside_x=500, nside_y=300
)

rcat_bossmock_matched = find_matching_samples_table(
    bg_galcat, rand_gal_tb, nside_x=500, nside_y=300
)

logger.debug(f"{len(gcat_bossmock_matched)}")

from astropy.cosmology import FlatLambdaCDM
from dsigma.precompute import precompute

from dsigma.stacking import excess_surface_density
from dsigma.jackknife import compute_jackknife_fields, jackknife_resampling

logger.info("Begin measurements")

hubble = cosmo_ccl.to_dict()['h']
Om0 = cosmo_ccl.omega_x(1, 'matter')

rp_bins = np.logspace(-1, 2, 16)
cosmology = FlatLambdaCDM(H0=100*hubble, Om0=Om0)
lens_source_cut = 0.1

bg_galcat_tb = Table(bg_galcat)
bg_galcat_tb.rename_column('g1', 'e_1')
bg_galcat_tb.rename_column('g2', 'e_2')

bg_galcat_tb['e_1'] = -bg_galcat_tb['e_1']

logger.info("Precompute")

precompute(gcat_bossmock_matched, bg_galcat_tb, rp_bins, cosmology=cosmology,
                       comoving=True, lens_source_cut=lens_source_cut,
                       progress_bar=True, n_jobs=64)

precompute(rcat_bossmock_matched, bg_galcat_tb, rp_bins, cosmology=cosmology,
                       comoving=True, lens_source_cut=lens_source_cut,
                       progress_bar=True, n_jobs=64)

logger.info("Setup jackknife centers")

centers = compute_jackknife_fields(
    gcat_bossmock_matched, 100, weights=np.sum(gcat_bossmock_matched['sum 1'], axis=1))
compute_jackknife_fields(rcat_bossmock_matched, centers)

logger.info("Get ESD")

esd = excess_surface_density(gcat_bossmock_matched,
                             table_r=rcat_bossmock_matched, 
                             return_table=True, 
                             random_subtraction=True
                             )

for key in esd.colnames:
    esd[key].format='.4f'

logger.info("Evaluate jackknife errors")

esd['ds_err'] = np.sqrt(np.diag(jackknife_resampling(
        excess_surface_density, gcat_bossmock_matched, return_table=False)))

logger.info("Save to file")

np.savetxt(f'aux/results/vlens/cosmo_000001_void_{catname}_tomo5_wrand.txt', np.c_[esd['rp'], esd['ds'], esd['ds_err']])