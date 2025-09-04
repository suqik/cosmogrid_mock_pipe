'''
Script to match foreground and
background galaxies
'''

import numpy as np
import healpy as hp
import datetime
from loguru import logger

from .io_func import *
from .mkfore_utils import cat2shell

def match_fore_back_cat(fcat, bcat, nside=256):
    ### make occupation maps for foreground and background catalogs
    fshell = cat2shell(np.c_[fcat['ra'], fcat['dec']], Nside=nside, coord="lonlat")
    bshell = cat2shell(np.c_[bcat['ra'], bcat['dec']], Nside=nside, coord="lonlat")

    ### make intersection of two occupation maps
    fshell = fshell * bshell

    ### select galaxies in the intersection
    selected_pix = np.argwhere(fshell != 0)
    fcat_pix = hp.ang2pix(nside, fcat["ra"], fcat["dec"], lonlat=True)
    fcat_selected = fcat[np.isin(fcat_pix, selected_pix)]
    del fcat_pix, fcat
    bcat_pix = hp.ang2pix(nside, bcat["ra"], bcat["dec"], lonlat=True)
    bcat_selected = bcat[np.isin(bcat_pix, selected_pix)]
    del bcat_pix, bcat

    return fcat_selected, bcat_selected

''' main routine '''

wdir = "/home/suchen/Program/CosmoGrid"
fore_catalog_fname_fmt = wdir + "/catalogs/HOD/cosmo_000001_run_0_HOD_0_run_0_lowzcmass_part{}.txt"
back_catalog_fname = f"{wdir}/catalogs/Shape/bg_gal_flip_g1.txt"
nside_occ_map = 256

start = datetime.datetime.now()
logger.info("Load catalogs")
### load foreground catalog
fore_catalog = []
for i in range(4):
    tmp = np.loadtxt(fore_catalog_fname_fmt.format(i+1), dtype=make_survey_type)
    fore_catalog.append(tmp)
fore_catalog = np.concatenate(fore_catalog)
### load background catalog
back_catalog = np.loadtxt(back_catalog_fname, dtype=bgal_type)

logger.info("Matching catalogs")
### match foreground and background catalogs
fore_catalog_selected, back_catalog_selected = match_fore_back_cat(fore_catalog, back_catalog, nside=nside_occ_map)

logger.info("Save to files")
### save matched catalogs
np.savetxt("catalogs/Matched/fore_gal.txt", fore_catalog_selected[['ra','dec','z','w']], fmt="%.3f %.3f %.3f %.8f")
np.savetxt("catalogs/Matched/back_gal.txt", back_catalog_selected, fmt="%.3f %.3f %.3f %.3f %.8f %.8f %.3f")

end = datetime.datetime.now()
logger.info(f"Time elapsed: {end-start}")