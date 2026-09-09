''' Test FastPM HOD parameter sampling on a single cosmology '''

import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from loguru import logger

from handler import PipeConfig
from runner import FastPMRunner

if __name__ == "__main__":

    icosmo = int(sys.argv[1]) if len(sys.argv) > 1 else 1

    fastpm_config = PipeConfig(
        ### fixed siminfo
        Lbox = 1000.0,
        Npart = 1024,
        redshift = 0.3,
        # ### HOD model params
        model = 2, # label of model name.
        model_params_names = ('logMcut', 'sigma_logM', 'logM1', 'k', 'alpha', 'fic'),
        nhod_per_cosmo = 10, # Number of varied HOD parameter values per cosmology
        Num_ptcl_requirement = 12,
        verbose = True,
        num_seeds = 1,
        init_seed = 33000, ## initial seed for generating galaxy catalog
        ngal_ref = 3.5e-4,
        z_space = False, ## RSD in box. Note if need RSD in survey-like, do not open this.

        ### HOD param sampling
        param_prior_low  = np.array([12.0, 0.1, 13, 0.00, 0.0]),
        param_prior_high = np.array([13.6, 0.6, 15.0, 10.0, 1.5]),

        ### lightcone redshift range
        zmin_lightcone = 0.4,
        zmax_lightcone = 0.6,
        ctr_lightcone = [0,0,0],
        rsd_lightcone = True,

        ### nofz
        nofz_method = "const", # can be `rank`, `downsample`, or `const`
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
    hod_samples_output = Path(__file__).resolve().parent / (
        f"cosmo_hod_pairs_test_{icosmo}.json"
    )

    fastpm_runner = FastPMRunner.build_hod_runner(
        config=fastpm_config,
        halo_fmt=halo_fmt,
        cosmo_par_fname=cosmo_par_fname,
    )

    logger.info(f"start sampling HOD params for cosmo_{icosmo}")

    hod_params_alive = fastpm_runner.sample_hod_params(icosmo, 0)

    cosmo_hod_pairs = {
        f"cosmo_{icosmo:06d}": {
            f"HOD{ihod}": hod_params_alive[ihod].tolist()
            for ihod in range(len(hod_params_alive))
        }
    }

    hod_samples_output.parent.mkdir(parents=True, exist_ok=True)
    with open(hod_samples_output, "w+") as f:
        json.dump(cosmo_hod_pairs, f)

    logger.info(f"saved {len(hod_params_alive)} HOD samples to {hod_samples_output}")
