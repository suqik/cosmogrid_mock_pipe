''' Generate random catalog for FastPM LOWZ-NGC mock w_p (FCFC) '''

import numpy as np
import pymangle
from astropy.table import Table

mangle_dir = "/public/home/suchen/Programs/cosmogrid_mock_pipe/extras/masks/boss_geom/"

geom = pymangle.Mangle(mangle_dir + "mask_DR12v5_LOWZ_North.ply")
veto_fnames = [
    "badfield_mask_postprocess_pixs8.ply",
    "badfield_mask_unphot_seeing_extinction_pixs8_dr12.ply",
    "allsky_bright_star_mask_pix.ply",
    "bright_object_mask_rykoff_pix.ply",
    "collision_priority_mask_dr12.ply",
    "centerpost_mask_dr12.ply",
]
vetos = [pymangle.Mangle(mangle_dir + f) for f in veto_fnames]

Ntarget = 500_000
ra_keep, dec_keep = [], []
nkeep = 0
while nkeep < Ntarget:
    ra, dec = geom.genrand(200000)
    w = geom.weight(ra, dec)
    for v in vetos:
        w -= v.weight(ra, dec)
    sel = w > 0
    ra_keep.append(ra[sel])
    dec_keep.append(dec[sel])
    nkeep = sum(len(a) for a in ra_keep)

ra = np.concatenate(ra_keep)[:Ntarget].astype(np.float64)
dec = np.concatenate(dec_keep)[:Ntarget].astype(np.float64)

# bootstrap rand redshifts from the mock's own (RSD) redshift distribution
data = Table.read(
    "/public/share/ace66so15x/suchen/FastPM/MockCatalogs/Gals/"
    "cosmo_000000_realization_0000_HOD_0_boss_lowz_north_2dflens_south.fits"
)
data_lowz = data[data["survey"] == 0]
rng = np.random.default_rng(42)
zrsd = rng.choice(np.asarray(data_lowz["zrsd"]), size=Ntarget, replace=True)

rand = Table([ra, dec, zrsd, np.ones(Ntarget)],
             names=("ra", "dec", "zrsd", "w"))
out = "/public/home/suchen/Programs/cosmogrid_mock_pipe/measurements/wp_fcfc/rand_lowz_500k.fits"
rand.write(out, overwrite=True)
print(f"saved {out}: {len(rand)} rand points; "
      f"data LOWZ n(z) z range {data_lowz['zrsd'].min():.3f}-{data_lowz['zrsd'].max():.3f}")
