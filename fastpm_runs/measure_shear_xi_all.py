''' Measure shear-shear auto-correlation (xi+/xi-) for all 1000 shape catalogs '''

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import treecorr
from astropy.table import Table
from loguru import logger

TOMOS = [3, 4, 5]
NBINS = 10
MIN_SEP = 5.0
MAX_SEP = 300.0

SHAPE_FMT = (
    "/public/share/ace66so15x/suchen/FastPM/MockCatalogs/Shapes/"
    "cosmo_{:06d}_realization_{:04d}_kids1000_north_3tomos.fits"
)
OUT_FNAME = Path(__file__).resolve().parent / "shear_xi_all.npz"

# even chunking over ranks (same recipe as the other run scripts)
def divide_MPI_chunks(data, size):
    k, m = divmod(len(data), size)
    return [
        data[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]
        for i in range(size)
    ]


def measure_one(fname, tomo):
    cat = Table.read(fname)
    sel = cat[cat["tomo"] == tomo]
    tc = treecorr.Catalog(
        ra=sel["ra"], dec=sel["dec"], g1=sel["g1"], g2=sel["g2"], w=sel["w"],
        ra_units="deg", dec_units="deg",
    )
    gg = treecorr.GGCorrelation(
        nbins=NBINS, min_sep=MIN_SEP, max_sep=MAX_SEP, sep_units="arcmin"
    )
    gg.process(tc)
    return gg.meanr, gg.xip, gg.xim


if __name__ == "__main__":

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    cosmo_indices = np.arange(1000)
    chunks = divide_MPI_chunks(cosmo_indices, size) if rank == 0 else None
    chunks = comm.scatter(chunks, root=0)

    n_cosmo = len(chunks)
    xip_local = np.full((n_cosmo, len(TOMOS), NBINS), np.nan)
    xim_local = np.full((n_cosmo, len(TOMOS), NBINS), np.nan)

    for i, icosmo in enumerate(chunks):
        fname = SHAPE_FMT.format(icosmo, 0)
        logger.info(f"Rank {rank}: cosmo_{icosmo:06d}")
        try:
            for j, tomo in enumerate(TOMOS):
                theta, xip, xim = measure_one(fname, tomo)
                if i == 0 and j == 0:
                    theta_out = theta.copy()
                xip_local[i, j] = xip
                xim_local[i, j] = xim
        except Exception as err:
            logger.error(f"Rank {rank}: cosmo_{icosmo:06d} failed: {err}")

    xip_g = comm.gather(xip_local, root=0)
    xim_g = comm.gather(xim_local, root=0)

    if rank == 0:
        xip_all = np.concatenate(xip_g, axis=0)  # [1000, 3, 10]
        xim_all = np.concatenate(xim_g, axis=0)
        failed = [
            int(i) for i in range(1000)
            if not np.all(np.isfinite(xip_all[i])) or not np.all(np.isfinite(xim_all[i]))
        ]
        np.savez(
            OUT_FNAME,
            cosmo_indices=np.arange(1000),
            tomos=np.array(TOMOS),
            theta_arcmin=theta_out,
            xip=xip_all,
            xim=xim_all,
        )
        logger.info(
            f"saved {OUT_FNAME}; failed cosmologies ({len(failed)}): {failed[:20]}"
        )
