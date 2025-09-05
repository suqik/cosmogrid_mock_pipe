'''
Script to generate void random catalog
'''

import numpy as np
import pymangle
from loguru import logger
import sys
sys.path.append('/home/suchen/Program/CosmoGrid/')

from utils.mkfore_utils import make_nofz_from_sample, sample_from_histogram

fgal_type = np.dtype(
    [
        ("ra", "f4"), 
        ("dec", "f4"), 
        ("z", "f4"), 
        ("w", "f4")
    ]
)

cattype = "boss" # `boss` for simulation and `bossdata` for observational data
survey_part = "lowzcmass"

if cattype == "boss":
    catname = 'wb'
    fnamebase = f"aux/catalogs/{cattype}_{survey_part}_void_{catname}"
if cattype == "bossdata":
    fnamebase = f"aux/catalogs/{cattype}_{survey_part}_void"

logger.info(f"Load void catalog from {fnamebase}.npy")

vcat = np.load(fnamebase+".npy")
nvoid = len(vcat)
zmin = vcat['z'].min()
zmax = vcat['z'].max()

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
nrand = 20*len(vcat)

if survey_part == "lowzcmass":
    geoms = pymangle.Mangle(mask_boss_fdir + "mask_DR12v5_CMASSLOWZ_North.ply")
    ra_rand, dec_rand = geoms[0].genrand(nrand)
    select = (geoms[0].weight(ra_rand, dec_rand) != 0)
    ra_rand = ra_rand[select]
    dec_rand = dec_rand[select]

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

logger.info("Apply boss observational geometry")

for ipoly in range(len(masks)):
    masked_idx = masks[ipoly].contains(ra_rand, dec_rand)
    total_masked_idx = masked_idx if ipoly == 0 else total_masked_idx | masked_idx

ra_rand = ra_rand[~total_masked_idx]
dec_rand = dec_rand[~total_masked_idx]

logger.info("Generate redshifts that follows void redshifts distribution")

z, nofz = make_nofz_from_sample(vcat['z'], bins=50)
z_rand = sample_from_histogram(len(ra_rand), z, nofz)
w_rand = np.ones(len(ra_rand))

rand_cat = np.empty(len(ra_rand), dtype=fgal_type)
rand_cat["ra"] = ra_rand
rand_cat["dec"] = dec_rand
rand_cat["z"] = z_rand
rand_cat["w"] = w_rand

logger.info("Save to file")

np.save(fnamebase+"_rand.npy", rand_cat)