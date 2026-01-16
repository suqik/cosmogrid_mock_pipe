'''
Script to generate void random catalog
Only generate (RA, DEC)
'''

import numpy as np
import pymangle
from tqdm import tqdm
from loguru import logger

from utils.io_func import *

def sample_radec(survey_part, nrand):
    if survey_part == "lowz":
        geoms = geoms_dict["lowz"]
        ra_rand, dec_rand = geoms.genrand(nrand)
        select = (geoms.weight(ra_rand, dec_rand) != 0)
        ra_rand = ra_rand[select]
        dec_rand = dec_rand[select]
        survey_rand = np.zeros(len(ra_rand))

    if survey_part == "lowze2":
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

    if survey_part == "cmass":
        geoms = geoms_dict["cmass"]
        ra_rand, dec_rand = geoms.genrand(nrand)
        select = (geoms.weight(ra_rand, dec_rand) != 0)
        ra_rand = ra_rand[select]
        dec_rand = dec_rand[select]

        survey_rand = np.ones(len(ra_rand))*4
  
    return ra_rand, dec_rand, survey_rand

if __name__ == "__main__":

    hod_param_fname = "cfgs/hod/hod_5params_dict_high_ngal_wcosmo2.json"

    bin_lb = 2
    vfmt = f"/data2/suchen/CosmoGrid/high_ngal_suits/Void_bin{bin_lb}/" + "cosmo_{:06d}_run_0_HOD_{}_run_0_boss_north_2dflens_south.npy"
    rfile = f"/data2/suchen/CosmoGrid/Rand/boss_cmass_bin{bin_lb}_north_radec.npy"

    survey_name_label_dict = {
        'lowz': 0,
        'lowze2': 1,
        'lowze3': 2,
        'cmass': 4
    }

    survey_name_list = ["cmass"]

    logger.info("Get 3 parts number ratio")

    cosmo_labels_tot = get_cosmo_name_list_process(hod_param_fname)

    count = 0
    void_num = {}  
    
    for iname, survey_name in enumerate(survey_name_list):
        void_num[survey_name] = 0

    vcat = np.load(vfmt.format(1, 0)) # only use 1 rlz to estimate the number of voids

    for iname, survey_name in enumerate(survey_name_list):
        void_num[survey_name] += (vcat['survey'] == survey_name_label_dict[survey_name]).sum()

    print(void_num)

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
    geoms_dict["lowz"] = pymangle.Mangle(mask_boss_fdir + "mask_DR12v5_LOWZ_North.ply")
    geoms_dict["lowze2"] = pymangle.Mangle(mask_boss_fdir + "mask_DR12v5_LOWZE2_North.ply")
    geoms_dict["lowze3"] = pymangle.Mangle(mask_boss_fdir + "mask_DR12v5_LOWZE3_North.ply")
    geoms_dict["cmass"] = pymangle.Mangle(mask_boss_fdir + "mask_DR12v5_CMASS_North.ply")

    area_ratio_dict = {
        'lowz': 0.62,
        'lowze2': 0.013,
        'lowze3': 0.078,
        'cmass': 1.0
    }

    nrand_to_ndata = 5

    logger.info("randomly sample RA DEC")

    ra_rand = []
    dec_rand = []
    survey_rand = []

    for lb, survey_part in enumerate(survey_name_list):
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

    logger.info("Save random catalog")

    np.save(rfile, rand_cat)

    # >>> =================================================================== <<<
