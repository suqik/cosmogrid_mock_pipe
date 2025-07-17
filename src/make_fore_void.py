'''
Read voids files, separate size bins 
and transform to swot input format.
'''

import os
import healpy as hp
import pyccl as ccl
import numpy as np
import datetime
from loguru import logger
from io_func import *

''' Setup '''
### fiducial cosmology
### Note this has nothing to do with the cosmology of the simulation
cosmo_ccl = ccl.CosmologyVanillaLCDM()

### void size parameters
tot_rv_min = 15. # Mpc/h
tot_rv_max = 26. # Mpc/h
dRv = 1.0 # Mpc/h

### file fmts
basedir = '/home/suchen/Program/CosmoGrid/catalogs/Matched/'
ifile = basedir + 'fore_gal.txt'
dive_input = "tmp_fore_gal_cart.txt"
dive_output = "tmp_fore_void_cart.txt"
ofile_fmt = '/home/suchen/Program/CosmoGrid/catalogs/Matched/fore_voids_rvbin{:d}.txt'

''' Main '''
start = datetime.datetime.now()
### Convert ra-dec to x-y-z
logger.info("Convert ra-dec to x-y-z")

tmp = np.loadtxt(ifile, dtype=fgal_type)
chi_radial = ccl.comoving_radial_distance(cosmo_ccl, 1./(1+tmp['z'])) # Mpc
chi_radial *= cosmo_ccl.to_dict()["h"] # Mpc/h
pos = hp.ang2vec(tmp['ra'], tmp['dec'], lonlat=True) # Actually norm of position
pos = (pos.T * chi_radial).T
np.savetxt(dive_input, pos, fmt="%.3f %.3f %.3f")
del chi_radial, pos

### Find voids
logger.info("Find voids")
exec_path = "/home/suchen/applications/DIVE/DIVE"
cmd = exec_path + " -i " + dive_input + " -o " + dive_output
print(cmd)
os.system(cmd)
print(f"rm {dive_input}")
os.system(f"rm {dive_input}")

### Binning voids and convert to ra-dec
logger.info("Binning voids")
### initial void size bins
rv_edges = np.arange(tot_rv_min, tot_rv_max+0.1*dRv, dRv)
rv_mins = rv_edges[:-1]
rv_maxs = rv_edges[1:]

### load void catalogs
tmp = np.loadtxt(dive_output, dtype=dive_void_type)

catalogs = []
for rv_min, rv_max in zip(rv_mins, rv_maxs):
    mask = (tmp['Rv'] >= rv_min) & (tmp['Rv'] < rv_max)
    catalogs.append(tmp[mask])

del tmp

logger.info("Transform to RADEC and save to file")
### transform to swot input format and save to file
for i, catalog in enumerate(catalogs):
    chi_radial = np.linalg.norm(catalog['pos'], axis=1) # Mpc/h
    chi_radial /= cosmo_ccl.to_dict()["h"] # Mpc
    redshifts = 1./ccl.scale_factor_of_chi(cosmo_ccl, chi_radial) - 1.
    ra, dec = hp.vec2ang(catalog['pos'], lonlat=True)
    sigmaz = np.ones(len(ra))*0.001
    np.savetxt(ofile_fmt.format(i), np.c_[ra, dec, redshifts, sigmaz], fmt="%.3f %.3f %.3f %.3f")

print(f"rm {dive_output}")
os.system(f"rm {dive_output}")

end = datetime.datetime.now()
logger.info(f"Time elapsed: {end - start}")