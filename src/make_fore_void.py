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
cosmo_label = 1
sim_fmt = "/data3/suchen/CosmoGridV1/grid/cosmo_{:06d}/run_0/"
cosmo_ccl = get_cosmo_from_file(sim_fmt.format(cosmo_label) + "params.yml", otype='ccl')

### file fmts
basedir = '/home/suchen/Program/CosmoGrid/aux/'
ifile = basedir + 'test_halo_lcone.npy'
dive_input = "tmp_fore_gal_cart.txt"
dive_output = "tmp_fore_void_cart.txt"
ofile = '/home/suchen/Program/CosmoGrid/aux/test_void_lcone_w_boundary.npy'

''' Main '''
start = datetime.datetime.now()
### Convert ra-dec to x-y-z
logger.info("Convert ra-dec to x-y-z")

# tmp = np.loadtxt(ifile, dtype=fgal_type)

##### for test #####
tmp = np.load(ifile)
####################

chi_radial = ccl.comoving_radial_distance(cosmo_ccl, 1./(1+tmp['z'])) # Mpc
chi_min_mpc = chi_radial.min()
chi_max_mpc = chi_radial.max()
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

### load void catalogs
catalog = np.loadtxt(dive_output, dtype=dive_void_type)
chi_radial = np.linalg.norm(catalog['pos'], axis=1) # Mpc/h
chi_radial /= cosmo_ccl.to_dict()["h"] # Mpc

cut = (chi_radial < chi_max_mpc) & (chi_radial > chi_min_mpc)
catalog = catalog[cut]
chi_radial = chi_radial[cut]

redshifts = 1./ccl.scale_factor_of_chi(cosmo_ccl, chi_radial) - 1.
ra, dec = hp.vec2ang(catalog['pos'], lonlat=True)
weight = np.ones(len(ra))
np.save(ofile, np.c_[ra, dec, redshifts, weight])

print(f"rm {dive_output}")
os.system(f"rm {dive_output}")

end = datetime.datetime.now()
logger.info(f"Time elapsed: {end - start}")