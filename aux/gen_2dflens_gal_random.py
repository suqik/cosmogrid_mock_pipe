'''
Script to generate random catalog
'''

import sys
sys.path.append("/home/suchen/Program/CosmoGrid/")
import numpy as np
import healpy as hp
from src.io_func import loadFitsMaps
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

def generate_radecw_from_mask_weight(mask_map, weight_map, Ngal):
    nside = hp.npix2nside(len(mask_map))
    selected_pix = np.argwhere(mask_map > 0).flatten()
    slt_ra, slt_dec = hp.pix2ang(nside, selected_pix, lonlat=True)
    slt_ra_rdf = np.where(slt_ra>180, slt_ra-360, slt_ra)

    t_ra_rdf = np.random.uniform(low=slt_ra_rdf.min(), high=slt_ra_rdf.max(), size=Ngal)
    t_ra = np.where(t_ra_rdf < 0, t_ra_rdf + 360, t_ra_rdf)
    t_cos_dec = np.random.uniform(low=np.cos(np.deg2rad(slt_dec.min())), high=np.cos(np.deg2rad(slt_dec.max())), size=Ngal)
    t_dec = np.rad2deg(-np.arccos(t_cos_dec))

    t_pix = hp.ang2pix(nside, t_ra, t_dec, lonlat=True)
    pick = np.isin(t_pix, selected_pix)
    t_picked_ra = t_ra[pick]
    t_picked_dec = t_dec[pick]
    t_picked_weight = hp.get_interp_val(weight_map, t_picked_ra, t_picked_dec, lonlat=True)

    Ngal_final = len(t_picked_ra)
    galcone = np.empty(Ngal_final, dtype=fgal_type)
    galcone["ra"] = t_picked_ra
    galcone["dec"] = t_picked_dec
    galcone["z"] = np.zeros(Ngal_final)
    galcone["w"] = t_picked_weight

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
    Ndata_max = (nz_ref*shell_vol).max()

    z_mock = galcone["z"]
    Nz_mock, _ = np.histogram(z_mock, zedges)
    k = Nz_mock.mean()/Ndata_max
    downsample_rate = nz_ref/Nz_mock*shell_vol*k
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

### geometry files
geom_2dflens_file = "catalogs/masks/2dflens_geom/2dFLens_mask_weight.fits"

### n(z) files
nz_2dflens_fname = "catalogs/NOfZ/nbar_2dFLens_south_random.dat"

start = datetime.datetime.now()
logger.info("Load masks")

mask_weight_2dflens = loadFitsMaps(geom_2dflens_file)
mask_map = mask_weight_2dflens[0]
weight_map = mask_weight_2dflens[1]

logger.info("Generate random RA DEC")
gal_num = 350000
### random sampling `gal_num` galaxies in the non-zero region of healpix map `mask_map`
### where `mask_map` is 1 for the region of interest and 0 otherwise

galcone = generate_radecw_from_mask_weight(mask_map, weight_map, gal_num)

logger.info("Generate n(z)")
nofz_info = {}
nofz = np.loadtxt(nz_2dflens_fname, usecols=(1,2,3,4)) # zmin, zmax, nz, shell_vol
zmin = 0.2
zmax = 0.4
argstart = np.argwhere(nofz[:,0] == zmin)[0,0]
argend = np.argwhere(nofz[:,1] == zmax)[0,0]
nofz_info = make_nofz_info(nofz[argstart:argend+2,0], nofz[argstart:argend+1,3], nofz[argstart:argend+1,2])

rng = np.random.default_rng(seed=9971)

galcone['z'] = rng.uniform(low=zmin, high=zmax, size=len(galcone))
galcone = apply_nz_downsample(galcone, nofz_info)

logger.info("Save catalog")
np.savetxt("catalogs/Random/2dflens_random_cat.txt", galcone, fmt="%.5f")

end = datetime.datetime.now()
logger.info("Time taken: {}".format(end-start))