'''
Script to apply different kinds of measurements.
1. Angular power spectrum.
2. Two point correlation functions (xi(r), wp(r), xi_(0,2,4)).
3. Cosmic shear two point correlation function (xi+, xi-).
...
'''

import numpy as np
from astropy.table import Table
import healpy as hp
import pymaster as nmt
import treecorr

def measure_angular_power(map1, ell_nbins, map2=None, mask=None):
    nside = hp.npix2nside(len(map1))

    if mask is not None:
        if mask1 is None:
            mask1 = np.ones_like(map1)
        if mask2 is None:
            mask2 = np.ones_like(map2)

        f1 = nmt.NmtField(mask1, [map1])

        b = nmt.NmtBin.from_nside_linear(nside, ell_nbins)
        if map2 is not None:
            f2 = nmt.NmtField(mask2, [map2])
            cl = nmt.compute_full_master(f1, f2,b)
        else:
            cl = nmt.compute_full_master(f1, f1, b)
        ell_arr = b.get_effective_ells()

    else:

        cl = hp.anafast(map1, map2)
        ell_arr = np.arange(len(cl))
    
    return cl, ell_arr

def measure_cosmic_shear(ra, dec, g1, g2, 
                         min_sep=1, max_sep=500., nbins=20,
                         sep_units='arcmin'
                         ):
    tc_cat = treecorr.Catalog(ra=ra, dec=dec, g1=g1, g2=g2, ra_units='degrees', dec_units='degrees', npatch=128)
    gg = treecorr.GGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, sep_units=sep_units, var_method='jackknife')
    gg.process(tc_cat)

    rnom = gg.rnom
    xip = gg.xip
    xim = gg.xim
    var_xip = gg.varxip
    var_xim = gg.varxim

    return rnom, xip, xim, var_xip, var_xim

if __name__ == "__main__":

    input_file  = "/data2/suchen/CosmoGrid/Shape/KiDS_ngal_suits/cosmo_000001_run_0_kids_north_tomo5.npy"
    output_file = "aux/results/xip_xim/cosmo_000001_tomo5_wSN"
    stats_type  = "cosmic_shear"
    wSN = True # only use in cosmic shear

    print(input_file)
    print(output_file)
    print(stats_type)
    if stats_type == "cosmic_shear":
        print(f"with Shape Noise: {wSN}")

    if stats_type == "cosmic_shear":
        cat = np.load(input_file)

        if wSN:
            rnom, xip, xim, var_xip, var_xim = measure_cosmic_shear(
                cat['ra'], cat['dec'], cat['g1'], -cat['g2'], 
                min_sep = 1.0, max_sep = 100.0, nbins=15)
        else:
            rnom, xip, xim, var_xip, var_xim = measure_cosmic_shear(
                cat['ra'], cat['dec'], cat['g1_pure'], -cat['g2_pure'],
                min_sep = 1.0, max_sep = 100.0, nbins=15)
            
        np.savez(output_file, theta=rnom, xip=xip, xim=xim, varxip=var_xip, varxim = var_xim)
