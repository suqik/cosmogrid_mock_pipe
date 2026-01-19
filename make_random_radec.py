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

    ADD_Z_RV = True
    SURVEY_NAME = "lowz"

    if ADD_Z_RV:
        from utils.mkfore_utils import bounded_kde_transform, resample_bounded

    survey_name_label_dict = {
        'lowz': 0,
        'lowze2': 1,
        'lowze3': 2,
        'cmass': 4
    }

    area_ratio_dict = {
        'lowz': 0.62,
        'lowze2': 0.013,
        'lowze3': 0.078,
        'cmass': 1.0
    }

    nrand_to_ndata = 5

    if SURVEY_NAME == "cmass":
        survey_name_list = ["cmass"]

        vfile = "catalogs/bossdata_cmass_void.npy"
        vcat = np.load(vfile) 

        z_rv_bounds = [
            (0.4, 0.6),
            (0.0, 60.0)
        ]
        ### output file name
        rfile = f"/data2/suchen/CosmoGrid/Rand/boss_cmass_north_radec.npy"    

    if SURVEY_NAME == "lowz":
        survey_name_list = ["lowz", "lowze2", "lowze3"]

        vfile = "catalogs/bossdata_lowz_void.npy"
        vcat = np.load(vfile) 

        z_rv_bounds = [
            (0.2, 0.4),
            (0.0, 60.0)
        ]
        ### output file name
        rfile = f"/data2/suchen/CosmoGrid/Rand/boss_lowz_north_radec.npy"   

    ### Announcements
    logger.info("Info")
    logger.info(f"Survey name: {SURVEY_NAME}") 
    logger.info(f"Sample p(z,Rv): {ADD_Z_RV}")
    logger.info(f"Nrand/Ndata = {nrand_to_ndata:d}")
    logger.info(f"Survey name list: {survey_name_list}")
    
    ### Get number of voids
    logger.info("Get void number ratio")

    void_num = {}  
    for iname, survey_name in enumerate(survey_name_list):
        void_num[survey_name] = (vcat['survey'] == survey_name_label_dict[survey_name]).sum()

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
    if "lowz" in survey_name_list:
        geoms_dict["lowz"] = pymangle.Mangle(mask_boss_fdir + "mask_DR12v5_LOWZ_North.ply")
    if "lowze2" in survey_name_list:
        geoms_dict["lowze2"] = pymangle.Mangle(mask_boss_fdir + "mask_DR12v5_LOWZE2_North.ply")
    if "lowze3" in survey_name_list:
        geoms_dict["lowze3"] = pymangle.Mangle(mask_boss_fdir + "mask_DR12v5_LOWZE3_North.ply")
    if "cmass" in survey_name_list:
        geoms_dict["cmass"] = pymangle.Mangle(mask_boss_fdir + "mask_DR12v5_CMASS_North.ply")

    logger.info("randomly sample RA DEC")

    ra_rand = []
    dec_rand = []
    if ADD_Z_RV:
        z_rand = []
        Rv_rand = []
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

        ### if sampling p(z,Rv) from data
        if ADD_Z_RV:
            bounded_kde = bounded_kde_transform(np.c_[vcat['z'], vcat['Rv']], z_rv_bounds)
            z_rand_, Rv_rand_ = resample_bounded(bounded_kde, len(ra_rand_), z_rv_bounds)
            z_rand.append(z_rand_)
            Rv_rand.append(Rv_rand_)
    
    ra_rand = np.concatenate(ra_rand)
    dec_rand = np.concatenate(dec_rand)
    survey_rand = np.concatenate(survey_rand)
    
    if ADD_Z_RV:
        z_rand = np.concatenate(z_rand)
        Rv_rand = np.concatenate(Rv_rand)

    logger.info("Apply boss observational geometry")
    
    for ipoly in range(len(masks)):
        masked_idx = masks[ipoly].contains(ra_rand, dec_rand)
        total_masked_idx = masked_idx if ipoly == 0 else total_masked_idx | masked_idx

    ra_rand = ra_rand[~total_masked_idx]
    dec_rand = dec_rand[~total_masked_idx]
    survey_rand = survey_rand[~total_masked_idx]
    
    if ADD_Z_RV:
        z_rand = z_rand[~total_masked_idx]
        Rv_rand = Rv_rand[~total_masked_idx]

    rand_cat = np.empty(len(ra_rand), dtype=fvoid_type)
    rand_cat["ra"] = ra_rand
    rand_cat["dec"] = dec_rand
    rand_cat["w"] = 1
    rand_cat["survey"] = survey_rand

    if ADD_Z_RV:
        rand_cat["z"] = z_rand
        rand_cat["Rv"] = Rv_rand

    logger.info("Save random catalog")

    np.save(rfile, rand_cat)