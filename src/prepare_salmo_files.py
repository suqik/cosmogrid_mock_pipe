'''
Prepare necessary files for SALMO code.
Including density fields, shear fields and redshift distribution.
'''

import numpy as np
import h5py
import healpy as hp

from io_func import *
from mkback_utils import *

''' simulation info '''
sim_fmt = "/data3/suchen/CosmoGridV1/raytrace/cosmo_{:06d}/nside0512/raytracing_z{:.2f}_nufft.hdf5"
cosmo_label = 1
redshift_src = 1.0

''' main routine '''
### read shear map
sim_fname = sim_fmt.format(cosmo_label, redshift_src)
with h5py.File(sim_fname, 'r') as f:
    A = np.array(f["Distortion_matrix"]["Raytraced"])

kappa  = (A[0][0] + A[1][1]) / 2
gamma1 = -(A[0][0] - A[1][1]) / 2
gamma2 = -(A[0][1] + A[1][0]) / 2

saveFitsFullMap("cfgs/salmo/lenMap_run0_f2z1.fits", [kappa, gamma1, gamma2], comments=["kappa", "gamma1", "gamma2"])

### read density field
map_nside = np.npix2nside(len(kappa))
dens = np.zeros(len(kappa)).astype(np.float32)

saveFitsFullMap("cfgs/salmo/lenMap_run0_f1z1_dens.fits", [dens], comments=["density"])