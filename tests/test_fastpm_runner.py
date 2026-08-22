import tempfile
import unittest
from pathlib import Path

import numpy as np
from halotools.sim_manager import UserSuppliedHaloCatalog

from handler import PipeConfig
import runner as runner_module
from tests.test_io_func import RSTAR_HEADER


RSTAR_HOST_ROW = (
    "10 -1 2.0e12 200 100 200 40 50 1 2 3 10 20 30 "
    "0 0 0 0.1 40 2.0e12 0 0 0 0 0 0 0 1 1 0 0 0 "
    "1 1 0 0 0 0.5 0 0 50 -1\n"
)
RSTAR_SUBHALO_ROW = (
    "11 -1 5.0e11 120 60 100 25 20 4 5 6 40 50 60 "
    "0 0 0 0.2 25 5.0e11 0 0 0 0 0 0 0 1 1 0 0 0 "
    "1 1 0 0 0 0.6 0 0 200 10\n"
)


class SamplingHODPopulator:
    def find_hod_params(self, halo_catalog, seed_offset):
        return {
            "halo_count": len(halo_catalog.halo_table),
            "seed_offset": seed_offset,
        }


class FastPMRunnerCoreTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(
            hasattr(runner_module, "FastPMRunner"),
            "runner.FastPMRunner must be implemented",
        )
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.cosmo_file = self.root / "cosmo_list.txt"
        self.cosmo_file.write_text(
            "# hubble=0.6727 Omegab=0.0491 ns=0.9667\n"
            "# OmegaM S8\n"
            "0.200614 0.842526\n"
            "0.306602 0.636991\n"
        )
        self.halo_fmt = str(
            self.root
            / "cosmo{:d}"
            / "a_{:5.4f}"
            / "rstar"
            / "out_0_wPID.list"
        )
        self.config = PipeConfig(Lbox=1000.0, Npart=1024, redshift=0.3)
        self.runner = runner_module.FastPMRunner(
            config=self.config,
            halo_fmt=self.halo_fmt,
            cosmo_par_fname=self.cosmo_file,
            fore_mask_fnames_dict={"boss_veto": []},
            fore_nofz_fnames_dict={},
            fore_survey_labels_dict={},
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def write_rstar_catalog(self, icosmo=0):
        path = (
            self.root
            / f"cosmo{icosmo}"
            / "a_0.7692"
            / "rstar"
            / "out_0_wPID.list"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(RSTAR_HEADER + RSTAR_HOST_ROW + RSTAR_SUBHALO_ROW)
        return path

    def test_cosmo0_parses_fixed_and_varying_parameters(self):
        result = self.runner._get_cosmo_instance(0, otype="dict")
        self.assertEqual(result["hubble"], 0.6727)
        self.assertEqual(result["Omegab"], 0.0491)
        self.assertEqual(result["ns"], 0.9667)
        self.assertEqual(result["OmegaM"], 0.200614)
        self.assertEqual(result["S8"], 0.842526)
        self.assertAlmostEqual(result["Omega_c"], 0.151514)
        self.assertAlmostEqual(
            result["sigma8"],
            0.842526 / np.sqrt(0.200614 / 0.3),
        )

    def test_ccl_cosmology_uses_derived_parameters(self):
        cosmo = self.runner._get_cosmo_instance(0, otype="ccl")
        params = cosmo.to_dict()
        self.assertAlmostEqual(params["h"], 0.6727)
        self.assertAlmostEqual(params["Omega_b"], 0.0491)
        self.assertAlmostEqual(params["Omega_c"], 0.151514)
        self.assertAlmostEqual(
            params["sigma8"],
            0.842526 / np.sqrt(0.200614 / 0.3),
        )

    def test_cosmology_label_uses_zero_based_data_row(self):
        result = self.runner._get_cosmo_instance(1, otype="dict")
        self.assertEqual(result["OmegaM"], 0.306602)
        self.assertEqual(result["S8"], 0.636991)

    def test_halo_path_uses_cosmology_label_and_scale_factor(self):
        expected = (
            self.root / "cosmo7" / "a_0.7692" / "rstar" / "out_0_wPID.list"
        )
        expected.parent.mkdir(parents=True)
        expected.write_text("#ID PID\n1 -1\n")
        self.assertEqual(self.runner._get_halo_fname(7), str(expected))

    def test_seed_offsets_retain_realization_index(self):
        self.assertEqual(self.runner._get_sampling_seed_offset(2, 3), 5)
        self.assertEqual(self.runner._get_hod_seed_offset(2, 3, 4), 54)

    def test_invalid_cosmology_label_is_rejected(self):
        with self.assertRaisesRegex(IndexError, "cosmology label"):
            self.runner._get_cosmo_instance(2)

    def test_invalid_matter_density_is_rejected(self):
        self.cosmo_file.write_text(
            "# hubble=0.6727 Omegab=0.0491 ns=0.9667\n"
            "# OmegaM S8\n"
            "0.040000 0.842526\n"
        )
        with self.assertRaisesRegex(ValueError, "OmegaM"):
            self.runner._get_cosmo_instance(0)

    def test_unsupported_cosmology_output_is_rejected(self):
        with self.assertRaisesRegex(NotImplementedError, "yaml"):
            self.runner._get_cosmo_instance(0, otype="yaml")

    def test_missing_cosmology_parameter_is_rejected(self):
        self.cosmo_file.write_text(
            "# hubble=0.6727 Omegab=0.0491 ns=0.9667\n"
            "# OmegaM\n"
            "0.200614\n"
        )
        with self.assertRaisesRegex(ValueError, "varying"):
            self.runner._get_cosmo_instance(0)

    def test_missing_parent_catalog_is_rejected(self):
        with self.assertRaisesRegex(FileNotFoundError, "out_0_wPID"):
            self.runner._get_halo_fname(0)

    def test_catalog_without_pid_is_rejected(self):
        path = self.root / "cosmo0" / "a_0.7692" / "rstar" / "out_0_wPID.list"
        path.parent.mkdir(parents=True)
        path.write_text("#ID Mvir\n1 1e12\n")
        with self.assertRaisesRegex(ValueError, "PID"):
            self.runner._get_halo_fname(0)

    def test_non_parent_processed_catalog_is_rejected_with_pid(self):
        runner = runner_module.FastPMRunner(
            config=self.config,
            halo_fmt=str(
                self.root
                / "cosmo{:d}"
                / "a_{:5.4f}"
                / "rstar"
                / "out_0.list"
            ),
            cosmo_par_fname=self.cosmo_file,
            fore_mask_fnames_dict={"boss_veto": []},
            fore_nofz_fnames_dict={},
            fore_survey_labels_dict={},
        )
        path = self.root / "cosmo0" / "a_0.7692" / "rstar" / "out_0.list"
        path.parent.mkdir(parents=True)
        path.write_text("#ID PID\n1 -1\n")
        with self.assertRaisesRegex(ValueError, "out_0_wPID.list"):
            runner._get_halo_fname(0)

    def test_load_hod_halocat_uses_only_hosts_without_rhalf_cut(self):
        self.assertTrue(
            hasattr(self.runner, "_load_hod_halocat"),
            "FastPMRunner._load_hod_halocat must be implemented",
        )
        self.write_rstar_catalog()
        cosmo, catalog = self.runner._load_hod_halocat(0)
        table = catalog.halo_table
        params = cosmo.to_dict()
        self.assertAlmostEqual(params["Omega_c"] + params["Omega_b"], 0.200614)
        self.assertIsInstance(catalog, UserSuppliedHaloCatalog)
        self.assertEqual(len(table), 1)
        np.testing.assert_array_equal(table["halo_id"], [10])
        np.testing.assert_array_equal(table["halo_upid"], [-1])
        np.testing.assert_allclose(table["halo_rhalf"], [0.05])
        np.testing.assert_allclose(table["halo_nfw_conc"], [5.0])

    def test_sample_hod_params_uses_realization_seed_and_host_catalog(self):
        self.assertTrue(
            hasattr(self.runner, "sample_hod_params"),
            "FastPMRunner.sample_hod_params must be implemented",
        )
        self.write_rstar_catalog()
        self.runner.hod_populator = SamplingHODPopulator()
        result = self.runner.sample_hod_params(0, irlz=3)
        self.assertEqual(result, {"halo_count": 1, "seed_offset": 3})
