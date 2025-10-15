'''
Script to generate void random catalog
'''

import numpy as np
import pymangle
from tqdm import tqdm
from loguru import logger

from utils.io_func import *
from utils.mkfore_utils import bounded_kde_transform, resample_bounded

hod_param_fname = "cfgs/hod/hod_5params_dict.json"

vfmt = "/data2/suchen/CosmoGrid/Void/cosmo_{:06d}_run_0_HOD_{}_run_0_boss_north.npy"
rfile = "/data2/suchen/CosmoGrid/Rand/boss_cmasslowztot_north_radec.npy"

def sample_radec(survey_part, nrand):
    if survey_part == "lowzcmass":
        # geoms = pymangle.Mangle(mask_boss_fdir + "mask_DR12v5_CMASSLOWZ_North.ply")
        geoms = geoms_dict["lowzcmass"]
        ra_rand, dec_rand = geoms.genrand(nrand)
        select = (geoms.weight(ra_rand, dec_rand) != 0)
        ra_rand = ra_rand[select]
        dec_rand = dec_rand[select]
        survey_rand = np.zeros(len(ra_rand))

    if survey_part == "lowze2":
        # geoms = [
        #     pymangle.Mangle(mask_boss_fdir + "mask_DR12v5_LOWZE2_North.ply"), 
        #     pymangle.Mangle(mask_boss_fdir + "mask_DR12v5_LOWZ_North.ply")
        # ]
        geoms = [
            geoms_dict["lowze2"],
            geoms_dict["lowz"]
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
        # geoms = [
        #     pymangle.Mangle(mask_boss_fdir + "mask_DR12v5_LOWZE3_North.ply"), 
        #     pymangle.Mangle(mask_boss_fdir + "mask_DR12v5_LOWZ_North.ply")
        # ]
        geoms = [
            geoms_dict["lowze3"],
            geoms_dict["lowz"]
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

if __name__ == "__main__":

    logger.info("Get 3 parts number ratio")

    hod_params_dict = get_hod_params(hod_param_fname)
    cosmo_labels_tot = []

    for icosmo_str in hod_params_dict.keys():
        if len(hod_params_dict[icosmo_str]) > 0:
            cosmo_labels_tot.append(int(icosmo_str[5:]))

    count = 0

    void_num = {}
    void_num['lowzcmass'] = 0
    void_num['lowze2'] = 0
    void_num['lowze3'] = 0

    for icosmo in tqdm(cosmo_labels_tot, desc='processing'):
        for ihod in range(10): # we already know each cosmology has 10 hod rlzs
            vcat = np.load(vfmt.format(icosmo, ihod))
            void_num["lowzcmass"] += (vcat['survey'] == 0).sum()
            void_num["lowze2"] += (vcat['survey'] == 1).sum()
            void_num["lowze3"] += (vcat['survey'] == 2).sum()

            count += 1


    void_num["lowzcmass"] /= count
    void_num["lowze2"] /= count
    void_num["lowze3"] /= count

    # tot_num = lowzcmass_num + lowze2_num + lowze3_num

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

    geoms_dict = {}
    geoms_dict["lowzcmass"] = pymangle.Mangle(mask_boss_fdir + "mask_DR12v5_CMASSLOWZ_North.ply")
    geoms_dict["lowze2"] = pymangle.Mangle(mask_boss_fdir + "mask_DR12v5_LOWZE2_North.ply")
    geoms_dict["lowze3"] = pymangle.Mangle(mask_boss_fdir + "mask_DR12v5_LOWZE3_North.ply")
    geoms_dict["lowz"] = pymangle.Mangle(mask_boss_fdir + "mask_DR12v5_LOWZ_North.ply")


    rv_bounds = [0, 40]
    zmin = 0.2
    zmax = 0.4
    Rvmin = 0
    Rvmax = 40

    area_ratio_dict = {
        'lowzcmass': 0.62,
        'lowze2': 0.013,
        'lowze3': 0.078
    }

    nrand_to_ndata = 5

    logger.info("randomly sample RA DEC")

    ra_rand = []
    dec_rand = []
    survey_rand = []
    for lb, survey_part in enumerate(["lowzcmass", "lowze2", "lowze3"]):
        nvoid = void_num[survey_part]
        nrand = int(nvoid * nrand_to_ndata)

        logger.info(f"Survey part: {survey_part}, Nrand={nrand}")

        nrand = int(nrand/area_ratio_dict[survey_part])
        ra_rand_, dec_rand_, survey_rand_ = sample_radec(survey_part, nrand)
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

    np.save(rfile, rand_cat)

    # >>> =================================================================== <<<
