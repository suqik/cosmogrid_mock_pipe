import sys
sys.path.append('/home/suchen/Program/CosmoGrid/')

import numpy as np
import healpy as hp
from matplotlib import pyplot as plt

from utils.io_func import *
from utils.mkback_utils import *

import pymaster as nmt


sim_fmt = "/data3/suchen/CosmoGridV1/raytrace/cosmo_{:06d}/nside0512/raytracing_z{:.2f}_nufft.hdf5"
def read_shear_maps(cosmo_label:int, redshift_src_list:list) -> dict:
    shear_map_dict = {}
    for ishell in range(len(redshift_src_list)):
        redshift_src = redshift_src_list[ishell]

        shear_map_dict[f"shell{ishell}"] = {}
        shear_map_dict[f"shell{ishell}"]['redshift'] = redshift_src

        sim_fname = sim_fmt.format(cosmo_label, redshift_src)
        with h5py.File(sim_fname, 'r') as f:
            A = np.array(f["Distortion_matrix"]["Raytraced"])

        gamma1 = -(A[0][0] - A[1][1]) / 2
        gamma2 = -(A[0][1] + A[1][0]) / 2
        kappa  = -(A[0][0] + A[1][1]) / 2 + 1

        shear_map_dict[f"shell{ishell}"]['kappa'] = kappa
        shear_map_dict[f"shell{ishell}"]['gamma1'] = gamma1
        shear_map_dict[f"shell{ishell}"]['gamma2'] = gamma2

    return shear_map_dict

mask_file = "/home/suchen/Program/CosmoGrid/catalogs/masks/mask_KiDS_North_1024.fits"
mask = loadFitsMaps(mask_file)
mask = mask[0]

mask_dsample = hp.ud_grade(mask, nside_out=512, order_out='RING')
mask_dsample = np.where(mask_dsample > 0.25, 1, 0)
# mask = np.where(mask > 0, 1, 0)

nside = hp.npix2nside(len(mask_dsample))
rot_degrees_list = [
    [0,50,0],
    [90,0,-50],
    [180,-50,0],
    [270,0,50],
    [0,-50,0],
    [90,0,50],
    [180,50,0],
    [270,0,-50],
]

cosmo_label = 1

shear_map_dict = read_shear_maps(cosmo_label, [1.0])

# ====  for one realization   ====
kappa_map = shear_map_dict['shell0']['kappa']
kappa_map = kappa_map*mask_dsample

nside = hp.npix2nside(len(kappa_map))
f_0 = nmt.NmtField(mask_dsample, [kappa_map])

# Initialize binning scheme with 4 ells per bandpower
b = nmt.NmtBin.from_nside_linear(nside, 32)

# Compute MASTER estimator
# spin-0 x spin-0
cl_00 = nmt.compute_full_master(f_0, f_0, b)

# Plot results
ell_arr = b.get_effective_ells()

np.savetxt(f"aux/results/cl_kk/cl_part_sky_z1.0_band32.txt", np.c_[ell_arr, cl_00[0]])

# ====  for eight realizations   ====
# new_mask_list = []
# for rot_degrees in rot_degrees_list:
#     new_mask = np.zeros_like(mask_dsample)
#     new_mask_pix = rotate_pix(np.argwhere(mask_dsample!=0).flatten(), nside=nside, rot_degrees=rot_degrees)
#     new_mask[new_mask_pix] = 1
#     new_mask_list.append(new_mask)

# for idx, imask in enumerate(new_mask_list):
#     print(f"Process part {idx}")
#     kappa_map = shear_map_dict['shell0']['kappa']
#     kappa_map = kappa_map*imask

#     nside = hp.npix2nside(len(kappa_map))
#     f_0 = nmt.NmtField(imask, [kappa_map])

#     # Initialize binning scheme with 4 ells per bandpower
#     b = nmt.NmtBin.from_nside_linear(nside, 32)

#     # Compute MASTER estimator
#     # spin-0 x spin-0
#     cl_00 = nmt.compute_full_master(f_0, f_0, b)

#     # Plot results
#     ell_arr = b.get_effective_ells()

#     np.savetxt(f"aux/results/cl_kk/cl_part_sky_z1.0_part{idx}_band32.txt", np.c_[ell_arr, cl_00[0]])