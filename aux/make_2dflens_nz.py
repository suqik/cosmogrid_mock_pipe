import numpy as np
from astropy.table import Table, vstack
import pyccl as ccl

from tqdm import trange
from loguru import logger

cosmo = ccl.Cosmology(
    h=0.6711,
    Omega_c=0.26,
    Omega_b=0.05,
    n_s=0.9667,
    sigma8=0.83
)

south_area = 533.476 # deg^2

# logger.info("Load data")

# lowz_2dflens = Table.read("/data2/suchen/2dFLenS/data_2dfloz_kidss/data_loz_atlas_kidss_160105_ntar.dat", 
#                           format='ascii',
#                           header_start=0, data_start=1)

# higz_2dflens = Table.read("/data2/suchen/2dFLenS/data_2dfhiz_kidss/data_hiz_atlas_kidss_160105_ntar.dat",
#                           format='ascii',
#                           header_start=0, data_start=1)

logger.info("Load random")

tmp_list = []
for i in trange(1,41):
    tmp = Table.read(f"/data2/suchen/2dFLenS/data_2dfloz_kidss/rand{i:03d}_loz_atlas_kidss_160105_ntar.dat", 
                    format='ascii',
                    header_start=0, data_start=1)
    tmp_list.append(tmp)

lowz_2dflens = vstack(tmp_list)

tmp_list = []
for i in trange(1,41):
    tmp = Table.read(f"/data2/suchen/2dFLenS/data_2dfhiz_kidss/rand{i:03d}_hiz_atlas_kidss_160105_ntar.dat", 
                    format='ascii',
                    header_start=0, data_start=1)
    tmp_list.append(tmp)

higz_2dflens = vstack(tmp_list)

total_2dflens = vstack([lowz_2dflens, higz_2dflens])
select = total_2dflens['type'] != 2
total_2dflens = total_2dflens[select]

logger.info("Total number of galaxies: %d", len(total_2dflens))

logger.info("Binning data")

z = total_2dflens['redshift']
zedges = np.arange(0.15, 0.7+1e-5, 0.005)
zctrs  = 0.5 * (zedges[1:] + zedges[:-1])
chi_edges = ccl.comoving_radial_distance(cosmo, 1./(1+zedges)) # Mpc
chi_edges *= 0.6711 # Mpc/h
chi_ctrs = 0.5 * (chi_edges[1:] + chi_edges[:-1]) # Mpc/h
volume = south_area * (np.pi/180)**2 * chi_ctrs**2 * (chi_edges[1:] - chi_edges[:-1]) # (Mpc/h)^3
Nz, _ = np.histogram(z, bins=zedges)
nbar = Nz / volume

logger.info("Save n(z)")

with open("catalogs/NOfZ/nbar_2dFLens_south_random.dat", "w+") as f:
    f.write(f"# effective area (deg^2), effective volume (Mpc/h)^3: {south_area:.6e} {volume.sum():.6e}\n")
    f.write("# zcen,zlow,zhigh,nbar,shell_vol,total gals\n")
    np.savetxt(f, np.c_[zctrs, zedges[:-1], zedges[1:], nbar, volume, Nz], 
               fmt="%.6f %.6f %.6f %.6e %.6e %.6e")
