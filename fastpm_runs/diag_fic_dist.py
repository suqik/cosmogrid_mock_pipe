''' Diagnose f_ic distribution for HOD parameter pool draws '''

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from loguru import logger

from handler import PipeConfig
from runner import FastPMRunner
from utils.mkfore_utils import get_ngal, compute_HMF

if __name__ == "__main__":

    icosmo = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    n_draws = int(sys.argv[2]) if len(sys.argv) > 2 else 2000

    fastpm_config = PipeConfig(
        Lbox = 1000.0,
        Npart = 1024,
        redshift = 0.3,
        model = 2,
        model_params_names = ('logMcut', 'sigma_logM', 'logM1', 'k', 'alpha', 'fic'),
        nhod_per_cosmo = 10,
        Num_ptcl_requirement = 12,
        verbose = True,
        num_seeds = 1,
        init_seed = 33000,
        ngal_ref = 3.5e-4,
        z_space = False,
        param_prior_low  = np.array([13, 0.1, 13, 0.00, 0.0]),
        param_prior_high = np.array([13.6, 0.6, 15.0, 10.0, 1.5]),
        zmin_lightcone = 0.4,
        zmax_lightcone = 0.6,
        ctr_lightcone = [0,0,0],
        rsd_lightcone = True,
        nofz_method = "const",
    )

    cosmo_par_fname = (
        "/public/home/suchen/Programs/Simtool/Pipeline/"
        "cfgs/fiducial/cosmo_list.txt"
    )
    halo_fmt = (
        "/public/share/ace66so15x/suchen/FastPM/Cosmology/"
        "L1000_N1024_1000cosmo/cosmo{:d}/"
        "a_{:5.4f}/rstar/out_0_wsub.list"
    )

    fastpm_runner = FastPMRunner.build_hod_runner(
        config=fastpm_config,
        halo_fmt=halo_fmt,
        cosmo_par_fname=cosmo_par_fname,
    )

    _, halo_catalog = fastpm_runner._load_hod_halocat(icosmo)
    halo_mass = halo_catalog.halo_table["halo_mvir"].value
    logger.info(f"cosmo{icosmo}: {len(halo_mass)} host halos, "
                f"Mmin={halo_mass.min():.3e} Mmax={halo_mass.max():.3e}")

    pop = fastpm_runner.hod_populator
    pool = np.asarray(pop._open_params_pool(n_draws, 9782 + icosmo))

    massbin, NM = compute_HMF(halo_mass, fastpm_config.Lbox)

    fics = np.empty(n_draws)
    ngals = np.empty(n_draws)
    for i, p in enumerate(pool):
        ngal_mock, _ = get_ngal(
            halo_mass=halo_mass, Lbox=fastpm_config.Lbox,
            redshift=fastpm_config.redshift,
            model_lb=fastpm_config.model,
            model_params_names=fastpm_config.model_params_names,
            hod_param_vals=p, massbin=massbin, NM=NM,
        )
        ngals[i] = ngal_mock
        fics[i] = fastpm_config.ngal_ref / ngal_mock

    ok = (fics > 0) & (fics <= 0.3)
    logger.info(f"ngal_mock: min={ngals.min():.4e} median={np.median(ngals):.4e} "
                f"max={ngals.max():.4e} (ngal_ref={fastpm_config.ngal_ref:.4e})")
    logger.info(f"f_ic: min={fics.min():.4e} median={np.median(fics):.4e} "
                f"max={fics.max():.4e}")
    logger.info(f"0 < f_ic <= 0.3: {ok.sum()}/{n_draws} "
                f"({100.0*ok.mean():.2f}%)")
    logger.info("f_ic percentiles: " + ", ".join(
        f"p{pp}={np.percentile(fics, pp):.3e}" for pp in [1, 5, 25, 50, 75, 95, 99]
    ))
