import sys
sys.path.append('/home/suchen/Program/CosmoGrid/')
import numpy as np
from matplotlib import pyplot as plt
import pyccl as ccl
from astropy.table import Table
import treecorr
from loguru import logger

from utils.io_func import get_cosmo_from_file

''' general setup '''
cosmo_label = 1
cosmo_par_fmt = "/data3/suchen/CosmoGridV1/grid/cosmo_{:06d}/run_0/params.yml"
tomo_bin = 1
nz_file_fmt = "/home/suchen/Program/CosmoGrid/catalogs/NOfZ/srcs/K1000_NS_V1.0.0A_ugriZYJHKs_photoz_SG_mask_LF_svn_309c_2Dbins_v2_SOMcols_Fid_blindC_TOMO{}_Nz.asc"

''' cosmic shear theory '''
logger.info("Get cosmology and n(z)")

cosmo_ccl = get_cosmo_from_file(cosmo_par_fmt.format(cosmo_label), otype='ccl')

filename = nz_file_fmt.format(tomo_bin)
nofz = np.loadtxt(filename)

logger.info("Initialize CCL")

lens = ccl.WeakLensingTracer(cosmo_ccl, dndz=(nofz[:,0], nofz[:,1]))

logger.info("Calculate C_ell")

ell = np.geomspace(1, 10000, 5000)
C_ell = ccl.angular_cl(cosmo_ccl, lens, lens, ell)

logger.info("Calculate xi+/xi-")

theta = np.geomspace(10, 1000, 500) # arcmin
xi_p = ccl.correlation(cosmo_ccl, ell=ell, C_ell=C_ell, theta=theta/60., type='GG+')
xi_m = ccl.correlation(cosmo_ccl, ell=ell, C_ell=C_ell, theta=theta/60., type='GG-')

logger.info("Save to file")

np.savetxt(f"aux/gg_cosmic_shear_tomo{tomo_bin}_wo_noise_theory.txt", np.c_[theta, xi_p, xi_m])

''' cosmic shear measurements '''
for ipart in range(8):
    logger.info(f"Process part {ipart + 1}")

    ### Load catalog
    logger.info("Load data")

    fname = f"catalogs/Shape/cosmo_{cosmo_label:06d}_run_0_kids_north_tomo{tomo_bin}_wo_noise_part{ipart+1}.txt"
    cat = Table.read(fname, format='ascii.no_header', names=['ra', 'dec', 'z', 'sigz', 'g1', 'g2', 'weight'])

    ### Define the treecorr catalog
    logger.info("Initial treecorr catalog")

    Npatch = 32
    cat = treecorr.Catalog(ra=cat['ra'], dec=cat['dec'], g1=-cat['g1'], g2=cat['g2'], w=cat['weight'], 
                        ra_units='degrees', dec_units='degrees',
                        npatch=Npatch)

    ### Measure the correlation function
    logger.info("Measure the correlation function")

    gg = treecorr.GGCorrelation(min_sep=10, max_sep=1000, nbins=20, sep_units='arcmin', var_method='jackknife', cross_patch_weight='match')
    gg.process(cat)

    ### Save the correlation function
    logger.info("Save to file")

    gg.write(f'aux/results/xip_xim/gg_cosmic_shear_tomo{tomo_bin}_wo_noise_part{ipart+1}.txt')