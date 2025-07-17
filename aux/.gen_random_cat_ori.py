'''
Script to generate random catalog
'''

import numpy as np
import pymangle
import datetime
from loguru import logger

fgal_type = np.dtype(
    [
        ("ra", "f4"), 
        ("dec", "f4"), 
        ("z", "f4"), 
        ("w", "f4")
    ]
)

def apply_boss_geometry(galcone, masks):
    for ipoly in range(len(masks)):
        mask = masks[ipoly].contains(galcone["ra"], galcone["dec"])
        tot_mask = mask if ipoly == 0 else tot_mask | mask
    galcone = galcone[~tot_mask]

    return galcone

def generate_z_from_nz(nofz, N, seed=None):
    rng = np.random.default_rng(seed)
    z_lows = nofz[:,0]
    z_highs = nofz[:,1]
    nz = nofz[:,2]
    
    probs = nz/nz.sum()

    bin_indices = rng.choice(len(nofz), size=N, p=probs)
    z_sampled = rng.uniform(low=z_lows[bin_indices], high=z_highs[bin_indices])

    return z_sampled

def make_nofz_info(zedges, shell_vol, nz_ref):
    '''
    Parameters
    ----------
    zedges : array
        z edges
    shell_vol : float
        volume of the shell
    nz_ref : float
        reference number density
    '''

    nofz_info = {}
    nofz_info['zedges'] = zedges
    nofz_info['shell_vol'] = shell_vol
    nofz_info['nz_ref'] = nz_ref

    return nofz_info

def apply_nz_downsample(galcone, nofz_info):
    zedges = nofz_info['zedges']
    shell_vol = nofz_info['shell_vol']
    nz_ref = nofz_info['nz_ref']

    z_mock = galcone["z"]
    Nz_mock, _ = np.histogram(z_mock, zedges)
    downsample_rate = nz_ref/Nz_mock*shell_vol
    downsample_rate = np.clip(downsample_rate, 0, 1)
    
    galcone_dsampled = []
    number_in_bin = []
    for ibin in range(len(zedges)-1):
        zmin, zmax = zedges[ibin], zedges[ibin+1]
        in_bin = (z_mock >= zmin) & (z_mock < zmax)
        gal_in_bin = galcone[in_bin]

        mask = np.random.choice(np.arange(len(gal_in_bin)), size=int(downsample_rate[ibin]*len(gal_in_bin)), replace=False)
        galcone_dsampled.append(gal_in_bin[mask])
        number_in_bin.append(len(gal_in_bin[mask]))

    galcone_dsampled = np.concatenate(galcone_dsampled)

    return galcone_dsampled

mask_boss_fdir = "catalogs/masks/boss_geom/"
### geometry files
geom_boss_fname_list = [
    mask_boss_fdir + "mask_DR12v5_CMASSLOWZ_North.ply",
    mask_boss_fdir + "mask_DR12v5_LOWZE2_North.ply",
    mask_boss_fdir + "mask_DR12v5_LOWZE3_North.ply"
]
### mask files corresponding to observational effects
mask_boss_fname_list = [
    mask_boss_fdir + "badfield_mask_postprocess_pixs8.ply",
    mask_boss_fdir + "badfield_mask_unphot_seeing_extinction_pixs8_dr12.ply",
    mask_boss_fdir + "allsky_bright_star_mask_pix.ply",
    mask_boss_fdir + "bright_object_mask_rykoff_pix.ply", 
    mask_boss_fdir + "collision_priority_mask_dr12.ply", 
    mask_boss_fdir + "centerpost_mask_dr12.ply"
]

### n(z) files
nz_boss_fname_list = [
    "catalogs/NOfZ/nbar_DR12v5_CMASSLOWZ_North_om0p31_Pfkp10000.dat",
    "catalogs/NOfZ/nbar_DR12v5_LOWZE2_North_om0p31_Pfkp10000.dat",
    "catalogs/NOfZ/nbar_DR12v5_LOWZE3_North_om0p31_Pfkp10000.dat"
]

start = datetime.datetime.now()
logger.info("Load masks")

geoms = []
for geom_file in geom_boss_fname_list:
    geoms.append(pymangle.Mangle(geom_file))

masks = []
for mask_file in mask_boss_fname_list:
    masks.append(pymangle.Mangle(mask_file))

logger.info("Generate random RA DEC and apply weights")

galcone_tot = []
part_num = [] # cumulative number

for geometry in geoms:
    gal_num = int(1e5)
    galcone = np.empty((int(gal_num),), dtype=fgal_type)
    galcone['ra'], galcone['dec'] = geometry.genrand(int(gal_num))
    galcone['w'] = geometry.weight(galcone['ra'], galcone['dec'])
    galcone = galcone[galcone['w'] > 0]
    
    galcone_tot.append(galcone)
    part_num.append(np.sum([len(arr) for arr in galcone_tot]))

part_num = np.append(0, part_num)
galcone_tot = np.concatenate(galcone_tot)

logger.info("Apply survey masks")
galcone_tot = apply_boss_geometry(galcone_tot, masks)
np.savetxt("test_rand.txt", galcone_tot, fmt="%.5f")

logger.info("Generate random n(z)")
zmin = 0.2
zmax = 0.4
rng = np.random.default_rng(seed=9971)

galcone_tot['z'] = rng.uniform(low=zmin, high=zmax, size=len(galcone_tot))

boss_part_names = ['boss_lowzcmass', 'boss_lowze2', 'boss_lowze3']
galcone_dsampled_tot = []
for ipart, nz_boss_fname in enumerate(nz_boss_fname_list):
    nofz = np.loadtxt(nz_boss_fname, usecols=(1,2,3,5)) # zmin, zmax, nbar, shell_vol
    argstart = np.argwhere(nofz[:,0] == zmin)[0,0]
    argend = np.argwhere(nofz[:,1] == zmax)[0,0]

    nofz_info = make_nofz_info(nofz[argstart:argend+2,0], nofz[argstart:argend+1,3], nofz[argstart:argend+1,2])

    curr_galcone = galcone_tot[part_num[ipart]:part_num[ipart+1]]
    logger.debug(len(curr_galcone))
    galcone_dsampled = apply_nz_downsample(curr_galcone, nofz_info)
    galcone_dsampled_tot.append(galcone_dsampled)

galcone_dsampled_tot = np.concatenate(galcone_dsampled_tot)

logger.info("Save catalog")
np.savetxt("catalogs/Random/boss_lowzcmass_rand.txt", galcone_dsampled_tot, fmt="%.5f")

end = datetime.datetime.now()
logger.info(f"Time taken: {end - start}")