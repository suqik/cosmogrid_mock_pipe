'''
Script to generate void random catalog
'''

import numpy as np
import pymangle
from loguru import logger
import sys
sys.path.append('/home/suchen/Program/CosmoGrid/')

from utils.io_func import *
from utils.mkfore_utils import bounded_kde_transform, resample_bounded

# >>> ========================     For test    =========================
cattype = "bossdata" # `boss` for simulation and `bossdata` for observational data

# 1: randomly sample (RA,DEC) and generate p(z) from void catalog
# 2: shuffle (RA,DEC) and (z,R) separately
method = 1

if cattype == "boss":
    catname = 'wb'
    survey_part = "lowze3"
    fnamebase = f"aux/catalogs/cosmo_000001/{cattype}_{survey_part}_void_{catname}"

if cattype == "bossdata":
    fnamebase = f"aux/catalogs/Data/{cattype}_lowzcmasstot_void"

logger.info(f"Load void catalog from {fnamebase}.npy")

vcat = np.load(fnamebase+".npy")

# >>> ===================================================================

logger.info(f"Use method {method} to generate void random catalog")

zmin = 0.2
zmax = 0.4
Rvmin = 15.
Rvmax = 25.

if method == 1:
    # >>> ========================     Method 1    ========================= <<<
    # >>> == randomly sample (RA,DEC) and generate p(z) from void catalog == <<<

    area_ratio_dict = {
        'lowzcmass': 0.62,
        'lowze2': 0.013,
        'lowze3': 0.078
    }

    def sample_radec(survey_part, mask_boss_fdir, nrand):
        if survey_part == "lowzcmass":
            geoms = pymangle.Mangle(mask_boss_fdir + "mask_DR12v5_CMASSLOWZ_North.ply")
            ra_rand, dec_rand = geoms.genrand(nrand)
            select = (geoms.weight(ra_rand, dec_rand) != 0)
            ra_rand = ra_rand[select]
            dec_rand = dec_rand[select]
            survey_rand = np.zeros(len(ra_rand))

        if survey_part == "lowze2":
            geoms = [
                pymangle.Mangle(mask_boss_fdir + "mask_DR12v5_LOWZE2_North.ply"), 
                pymangle.Mangle(mask_boss_fdir + "mask_DR12v5_LOWZ_North.ply")
            ]
            ra_rand, dec_rand = geoms[0].genrand(nrand)
            select = (geoms[0].weight(ra_rand, dec_rand) != 0)
            ra_rand = ra_rand[select]
            dec_rand = dec_rand[select]
            idx_out_mask = (geoms[1].weight(ra_rand, dec_rand) == 0)

            ra_rand = ra_rand[idx_out_mask]
            dec_rand = dec_rand[idx_out_mask]
            survey_rand = np.ones(len(ra_rand))

        if survey_part == "lowze3":
            geoms = [
                pymangle.Mangle(mask_boss_fdir + "mask_DR12v5_LOWZE3_North.ply"), 
                pymangle.Mangle(mask_boss_fdir + "mask_DR12v5_LOWZ_North.ply")
            ]
            ra_rand, dec_rand = geoms[0].genrand(nrand)
            select = (geoms[0].weight(ra_rand, dec_rand) != 0)
            ra_rand = ra_rand[select]
            dec_rand = dec_rand[select]
            idx_out_mask = (geoms[1].weight(ra_rand, dec_rand) == 0)

            ra_rand = ra_rand[idx_out_mask]
            dec_rand = dec_rand[idx_out_mask]
            survey_rand = np.ones(len(ra_rand))*2

        return ra_rand, dec_rand, survey_rand

    nrand_to_ndata = 10

    logger.info("Load observational masks")

    ### mask files corresponding to observational effects
    mask_boss_fdir = "catalogs/masks/boss_geom/"
    mask_boss_fname_list = [
        mask_boss_fdir + "badfield_mask_postprocess_pixs8.ply",
        mask_boss_fdir + "badfield_mask_unphot_seeing_extinction_pixs8_dr12.ply",
        mask_boss_fdir + "allsky_bright_star_mask_pix.ply",
        mask_boss_fdir + "bright_object_mask_rykoff_pix.ply", 
        mask_boss_fdir + "collision_priority_mask_dr12.ply", 
        mask_boss_fdir + "centerpost_mask_dr12.ply"
    ]

    masks = []
    for mask_file in mask_boss_fname_list:
        masks.append(pymangle.Mangle(mask_file))

    logger.info("Load survey geometry and randomly sample RA DEC")

    ra_rand = []
    dec_rand = []
    survey_rand = []
    for lb, survey_part in enumerate(["lowzcmass", "lowze2", "lowze3"]):
        nvoid = (vcat['survey'] == lb).sum()
        nrand = int(nvoid * nrand_to_ndata)

        logger.info(f"Survey part: {survey_part}, Nrand={nrand}")

        nrand = int(nrand/area_ratio_dict[survey_part])
        ra_rand_, dec_rand_, survey_rand_ = sample_radec(survey_part, mask_boss_fdir, nrand)
        ra_rand.append(ra_rand_)
        dec_rand.append(dec_rand_)
        survey_rand.append(survey_rand_)
    
    ra_rand = np.concatenate(ra_rand)
    dec_rand = np.concatenate(dec_rand)
    survey_rand = np.concatenate(survey_rand)

    logger.info("Apply boss observational geometry")
    
    for ipoly in range(len(masks)):
        masked_idx = masks[ipoly].contains(ra_rand, dec_rand)
        total_masked_idx = masked_idx if ipoly == 0 else total_masked_idx | masked_idx

    ra_rand = ra_rand[~total_masked_idx]
    dec_rand = dec_rand[~total_masked_idx]
    survey_rand = survey_rand[~total_masked_idx]

    rand_cat = np.empty(len(ra_rand), dtype=fvoid_type)
    rand_cat["ra"] = ra_rand
    rand_cat["dec"] = dec_rand
    rand_cat["w"] = 1
    rand_cat["survey"] = survey_rand

    logger.info("Generate redshifts that follows void redshifts distribution")

    z_rv_bounds = [(zmin, zmax), (Rvmin, Rvmax)]

    bounded_kde = bounded_kde_transform(np.c_[vcat['z'], vcat['Rv']], z_rv_bounds)
    rand_cat['z'], rand_cat['Rv'] = resample_bounded(bounded_kde, len(ra_rand), z_rv_bounds)

    logger.info(f"Save to file {fnamebase}_rand.npy")

    np.save(fnamebase+"_rand.npy", rand_cat)
    # >>> =================================================================== <<<

# if method == 2:
#     # >>> ========================     Method 2    ========================= <<<
#     # >>> ========   shuffle (RA,DEC) and (z,R) of void catalog   ========== <<<

#     def shuffle_void_catalog(catalog, seed=None):
#         '''
#         Shuffle the void catalog to obtain a random catalog.
#         '''
#         if seed is not None:
#             rng = np.random.default_rng(seed)
#         else:
#             rng = np.random.default_rng()

#         n = len(catalog)

#         # 生成两个独立的 shuffle 索引
#         idx1 = np.arange(n)
#         idx2 = np.arange(n)
#         rng.shuffle(idx1)
#         rng.shuffle(idx2)

#         # 创建输出数组，dtype 保持一致
#         shuffled = np.empty(n, dtype=catalog.dtype)

#         # (ra, dec) 用 idx1 打乱
#         shuffled['ra'] = catalog['ra'][idx1]
#         shuffled['dec'] = catalog['dec'][idx1]

#         # (z, Rv) 用 idx2 打乱
#         shuffled['z'] = catalog['z'][idx2]
#         shuffled['Rv'] = catalog['Rv'][idx2]

#         # 其他字段保持不变，直接拷贝
#         shuffled['w'] = catalog['w']
#         shuffled['survey'] = catalog['survey']

#         return shuffled

#     zbin_edges = [0.2,0.3,0.4]
#     rv_edges = np.append(np.arange(15,21),25)

#     vrand = []
#     for izbin in range(len(zbin_edges)-1):
#         for irvbin in range(len(rv_edges)-1):
#             select = ((vcat['z'] >= zbin_edges[izbin]) & (vcat['z'] < zbin_edges[izbin+1]) &
#                     (vcat['Rv'] >= rv_edges[irvbin]) & (vcat['Rv'] < rv_edges[irvbin+1]))
#             logger.info(f"Current subcatalog have {select.sum()} voids")
#             vrand.append(shuffle_void_catalog(vcat[select]))
        
#     vrand = np.concatenate(vrand)
#     np.save(fnamebase+f"_rand{method}.npy", vrand)
#     # >>> =================================================================== <<<
