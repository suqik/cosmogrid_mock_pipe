import gc
import inspect
import json
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import Mock, patch, sentinel

import numpy as np
from astropy.io import fits
from astropy.table import Table
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


class GalaxyHODPopulator:
    def populate_galaxies(self, halo_catalog, model_params_dict, indx, OmegaM):
        galaxies = np.zeros(
            1,
            dtype=[
                ("x", "f8"), ("y", "f8"), ("z", "f8"),
                ("vx", "f8"), ("vy", "f8"), ("vz", "f8"),
                ("gal_type", "i4"), ("halo_mvir", "f8"),
            ],
        )
        galaxies["x"] = indx
        galaxies["vx"] = OmegaM
        galaxies["gal_type"] = 1
        galaxies["halo_mvir"] = 2.5e13
        return {"dummy": galaxies}

    def get_galaxy_features(self, galaxies, features):
        pos = np.column_stack([galaxies["x"], galaxies["y"], galaxies["z"]])
        vel = np.column_stack([galaxies["vx"], galaxies["vy"], galaxies["vz"]])
        outputs = [pos]
        if "vel" in features:
            outputs.append(vel)
        if "gal_type" in features:
            outputs.append(galaxies["gal_type"])
        if "host_halo_mvir" in features:
            outputs.append(galaxies["halo_mvir"])
        return tuple(outputs)


class GalaxySurveyGenerator:
    def box_to_lightcone(self, cosmo, gal_pos, gal_adj_props):
        result = np.zeros(
            1,
            dtype=[
                ("marker", "f8"),
                ("survey", "i4"),
                ("gal_type", "i4"),
                ("host_halo_mvir", "f8"),
            ],
        )
        result["marker"] = gal_pos[0, 0] + gal_adj_props["gal_vel"][0, 0]
        result["gal_type"] = gal_adj_props.get("gal_type", [-1])
        result["host_halo_mvir"] = gal_adj_props.get(
            "host_halo_mvir", [-1.0]
        )
        return result

    def gen_boss_like(self, galcone, survey_name, survey_label, make_nz=True):
        result = galcone.copy()
        result["survey"] = survey_label
        return result


class FakeVoidFinder:
    def galcone_to_voidcone(
            self, galcone, cosmo, survey, dive_input, dive_output):
        result = np.zeros(
            2,
            dtype=[("z", "f8"), ("survey", "i4"), ("marker", "i4")],
        )
        result["z"] = [0.2, 1.2]
        result["survey"] = survey
        result["marker"] = [1, 2]
        return result


class VoidSurveyGenerator:
    def gen_boss_like(self, voids, survey_name, survey_label, make_nz=True):
        result = voids.copy()
        result["survey"] = survey_label
        return result


class FakeShearAssigner:
    def __init__(self, redshifts=(0.2,)):
        self.redshifts = np.asarray(redshifts)

    def gen_gal_positions(self, ngal, survey_name, tomo_label, survey_label):
        result = np.zeros(
            len(self.redshifts),
            dtype=[
                ("survey", "i4"),
                ("tomo", "i4"),
                ("ngal", "f8"),
                ("z_true", "f8"),
                ("g1", "f8"),
                ("map_count", "i4"),
            ],
        )
        result["survey"] = survey_label
        result["tomo"] = tomo_label
        result["ngal"] = ngal
        result["z_true"] = self.redshifts
        return result

    def assign_shear(self, catalog, shear_maps):
        result = catalog.copy()
        result["g1"] = shear_maps["shell0"]["gamma1"][0]
        result["map_count"] = len(shear_maps)
        return result

    def assign_weights(self, catalog, weight_type):
        return catalog


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

    def write_shear_product(
            self, icosmo=0, irlz=0,
            metadata_icosmo=None, metadata_irlz=None,
            ordering="RING", coordinate_system="C",
            map_size=12, include_near_gamma2=True,
            cosmology_overrides=None,
            near_redshift=0.2, far_redshift=0.8,
            omit_near_redshift=False):
        path = (
            self.root
            / "products"
            / f"cosmo_{icosmo:06d}"
            / f"realization_{irlz:04d}.npz"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        if metadata_icosmo is None:
            metadata_icosmo = icosmo
        if metadata_irlz is None:
            metadata_irlz = irlz
        cosmologies = {
            0: {
                "OmegaM": 0.200614,
                "S8": 0.842526,
                "hubble": 0.6727,
                "Omegab": 0.0491,
                "ns": 0.9667,
            },
            1: {
                "OmegaM": 0.306602,
                "S8": 0.636991,
                "hubble": 0.6727,
                "Omegab": 0.0491,
                "ns": 0.9667,
            },
        }
        cosmology = dict(cosmologies[icosmo])
        if cosmology_overrides is not None:
            cosmology.update(cosmology_overrides)
        near_metadata = (
            {} if omit_near_redshift
            else {"effective_redshift": near_redshift}
        )
        metadata = {
            "cosmology_index": metadata_icosmo,
            "realization_index": metadata_irlz,
            "cosmology": cosmology,
            "map": {
                "coordinate_system": coordinate_system,
                "ordering": ordering,
                "nside": 1,
            },
            "sources": {
                "source_far": {"effective_redshift": far_redshift},
                "source_near": near_metadata,
            },
        }
        payload = {
            "source_far__gamma1": np.full(map_size, 8.0),
            "source_far__gamma2": np.full(map_size, -8.0),
            "source_far__kappa": np.full(map_size, 0.8),
            "source_near__gamma1": np.full(map_size, 2.0),
            "source_near__kappa": np.full(map_size, 0.2),
            "metadata_json": np.asarray(json.dumps(metadata)),
        }
        if include_near_gamma2:
            payload["source_near__gamma2"] = np.full(map_size, -2.0)
        np.savez(path, **payload)
        return path

    def write_background_inputs(self):
        mask_path = self.root / "background_mask.fits"
        mask_hdu = fits.BinTableHDU.from_columns([
            fits.Column(
                name="VALUE",
                format="12E",
                array=np.ones((1, 12), dtype=np.float32),
            )
        ])
        fits.HDUList([fits.PrimaryHDU(), mask_hdu]).writeto(mask_path)

        nofz_path = self.root / "background_nz.txt"
        np.savetxt(
            nofz_path,
            np.array([
                [0.10, 0.0],
                [0.20, 1.0],
                [0.30, 1.0],
                [0.40, 0.0],
            ]),
        )
        return mask_path, nofz_path

    def test_constructor_and_builder_parameters_default_to_none(self):
        callables = [
            runner_module.FastPMRunner.__init__,
            runner_module.FastPMRunner.build_hod_runner,
            runner_module.FastPMRunner.build_gal_runner,
            runner_module.FastPMRunner.build_void_runner,
            runner_module.FastPMRunner.build_shape_runner,
        ]
        for callable_object in callables:
            with self.subTest(callable=callable_object.__name__):
                for name, parameter in inspect.signature(
                        callable_object).parameters.items():
                    if name in {"self", "cls"}:
                        continue
                    self.assertIsNone(parameter.default, name)

    def test_each_builder_initializes_exact_component_set(self):
        foreground = {
            "fore_mask_fnames_dict": {"boss_veto": []},
            "fore_nofz_fnames_dict": {},
            "fore_survey_labels_dict": {},
        }
        background = {
            "back_mask_fnames_dict": {},
            "back_nofz_fnames_dict": {},
            "back_survey_labels_dict": {},
            "back_ngals_dict": {},
            "tomo_labels_dict": {},
        }
        replacements = {
            "CatalogLoader": Mock(return_value=sentinel.catalog_loader),
            "HODPopulator": Mock(return_value=sentinel.hod_populator),
            "SurveyGenerator": Mock(return_value=sentinel.survey_generator),
            "VoidFinder": Mock(return_value=sentinel.void_finder),
            "ShearAssigner": Mock(return_value=sentinel.shear_assigner),
        }
        with patch.multiple(runner_module, **replacements):
            hod = runner_module.FastPMRunner.build_hod_runner(
                config=self.config, halo_fmt=self.halo_fmt,
                cosmo_par_fname=self.cosmo_file,
            )
            gal = runner_module.FastPMRunner.build_gal_runner(
                config=self.config, halo_fmt=self.halo_fmt,
                cosmo_par_fname=self.cosmo_file, **foreground,
            )
            void = runner_module.FastPMRunner.build_void_runner(
                config=self.config, cosmo_par_fname=self.cosmo_file,
                **foreground,
            )
            shape = runner_module.FastPMRunner.build_shape_runner(
                config=self.config, cosmo_par_fname=self.cosmo_file,
                shear_sim_fmt="product_{:06d}_{:04d}.npz",
                **background,
            )

        expected = {
            "hod": (sentinel.catalog_loader, sentinel.hod_populator,
                    None, None, None),
            "gal": (sentinel.catalog_loader, sentinel.hod_populator,
                    sentinel.survey_generator, None, None),
            "void": (None, None, sentinel.survey_generator,
                     sentinel.void_finder, None),
            "shape": (None, None, None, None, sentinel.shear_assigner),
        }
        for name, runner in {
                "hod": hod, "gal": gal, "void": void, "shape": shape}.items():
            self.assertEqual(
                (
                    runner.cata_loader, runner.hod_populator,
                    runner.survey_generator, runner.void_finder,
                    runner.shear_assigner,
                ),
                expected[name],
            )

    def test_void_builder_requires_no_halo_path(self):
        runner = runner_module.FastPMRunner.build_void_runner(
            config=self.config,
            cosmo_par_fname=self.cosmo_file,
            fore_mask_fnames_dict={"boss_veto": []},
            fore_nofz_fnames_dict={},
            fore_survey_labels_dict={},
        )
        self.assertIsNone(runner.halo_fmt)
        self.assertIsNone(runner.cata_loader)
        self.assertIsNone(runner.hod_populator)

    def test_partial_direct_background_group_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "back_nofz_fnames_dict"):
            runner_module.FastPMRunner(
                config=self.config,
                cosmo_par_fname=self.cosmo_file,
                back_mask_fnames_dict={},
            )

    def test_builder_lists_all_missing_required_parameters(self):
        with self.assertRaisesRegex(
                ValueError, "build_shape_runner.*shear_sim_fmt.*tomo_labels_dict"):
            runner_module.FastPMRunner.build_shape_runner(config=self.config)

    def test_direct_constructor_infers_each_supplied_group(self):
        foreground = {
            "fore_mask_fnames_dict": {"boss_veto": []},
            "fore_nofz_fnames_dict": {},
            "fore_survey_labels_dict": {},
        }
        background = {
            "back_mask_fnames_dict": {},
            "back_nofz_fnames_dict": {},
            "back_survey_labels_dict": {},
            "back_ngals_dict": {},
            "tomo_labels_dict": {},
        }
        hod = runner_module.FastPMRunner(
            config=self.config,
            halo_fmt=self.halo_fmt,
            cosmo_par_fname=self.cosmo_file,
        )
        foreground_runner = runner_module.FastPMRunner(
            config=self.config,
            cosmo_par_fname=self.cosmo_file,
            **foreground,
        )
        background_runner = runner_module.FastPMRunner(
            config=self.config,
            cosmo_par_fname=self.cosmo_file,
            shear_sim_fmt="product_{:06d}_{:04d}.npz",
            **background,
        )
        self.assertIsNotNone(hod.hod_populator)
        self.assertIsNone(hod.survey_generator)
        self.assertIsNotNone(foreground_runner.survey_generator)
        self.assertIsNotNone(foreground_runner.void_finder)
        self.assertIsNone(foreground_runner.hod_populator)
        self.assertIsNotNone(background_runner.shear_assigner)
        self.assertIsNone(background_runner.survey_generator)

    def test_incompatible_task_method_names_required_builder(self):
        runner = runner_module.FastPMRunner()
        calls = [
            ("build_hod_runner", lambda: runner.sample_hod_params(0)),
            ("build_gal_runner", lambda: runner.gen_mock_gal(0, 0, 0, [1.0])),
            (
                "build_void_runner",
                lambda: runner.gen_mock_void(
                    0, 0, 0, np.zeros(0), "input", "output"
                ),
            ),
            ("build_shape_runner", lambda: runner.gen_mock_shear(0)),
        ]
        for builder_name, call in calls:
            with self.subTest(builder=builder_name):
                with self.assertRaisesRegex(ValueError, builder_name):
                    call()

    def test_component_initializers_reuse_existing_instances(self):
        runner = runner_module.FastPMRunner.build_hod_runner(
            config=self.config,
            halo_fmt=self.halo_fmt,
            cosmo_par_fname=self.cosmo_file,
        )
        components = (runner.cata_loader, runner.hod_populator)
        runner._initialize_hod_components()
        self.assertEqual(
            components,
            (runner.cata_loader, runner.hod_populator),
        )

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

    def test_cosmology_file_without_two_headers_is_rejected(self):
        self.cosmo_file.write_text(
            "hubble=0.6727 Omegab=0.0491 ns=0.9667\n"
            "# OmegaM S8\n"
            "0.200614 0.842526\n"
        )
        with self.assertRaisesRegex(ValueError, "two headers"):
            self.runner._get_cosmo_instance(0)

    def test_malformed_fixed_cosmology_header_is_rejected(self):
        self.cosmo_file.write_text(
            "# hubble=0.6727 Omegab=0.0491 ns\n"
            "# OmegaM S8\n"
            "0.200614 0.842526\n"
        )
        with self.assertRaisesRegex(ValueError, "malformed fixed"):
            self.runner._get_cosmo_instance(0)

    def test_missing_fixed_cosmology_parameter_is_rejected(self):
        self.cosmo_file.write_text(
            "# hubble=0.6727 Omegab=0.0491\n"
            "# OmegaM S8\n"
            "0.200614 0.842526\n"
        )
        with self.assertRaisesRegex(ValueError, "missing fixed"):
            self.runner._get_cosmo_instance(0)

    def test_cosmology_header_and_data_column_mismatch_is_rejected(self):
        self.cosmo_file.write_text(
            "# hubble=0.6727 Omegab=0.0491 ns=0.9667\n"
            "# OmegaM S8 w0\n"
            "0.200614 0.842526\n"
        )
        with self.assertRaisesRegex(ValueError, "column counts"):
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

    def test_gen_mock_gal_returns_survey_catalog_with_fastpm_seed(self):
        self.assertTrue(
            hasattr(self.runner, "gen_mock_gal"),
            "FastPMRunner.gen_mock_gal must be implemented",
        )
        self.write_rstar_catalog()
        self.runner.fore_survey_labels_dict = {"boss_lowz_ngc": 7}
        self.runner.hod_populator = GalaxyHODPopulator()
        self.runner.survey_generator = GalaxySurveyGenerator()
        result = self.runner.gen_mock_gal(
            icosmo=0,
            irlz=2,
            ihod=3,
            ihod_param=np.array([13.2, 0.3, 14.0, 1.0, 0.8, 1.0]),
            save=False,
        )
        self.assertEqual(len(result), 1)
        np.testing.assert_array_equal(result["survey"], [7])
        self.assertAlmostEqual(result["marker"][0], 23.200614)
        np.testing.assert_array_equal(result["gal_type"], [1])
        np.testing.assert_allclose(result["host_halo_mvir"], [2.5e13])

    def test_gen_mock_gal_requires_output_format_when_saving(self):
        self.assertTrue(
            hasattr(self.runner, "gen_mock_gal"),
            "FastPMRunner.gen_mock_gal must be implemented",
        )
        with self.assertRaisesRegex(ValueError, "gal_ofmt"):
            self.runner.gen_mock_gal(0, 0, 0, np.ones(6), save=True)

    def test_gen_mock_gal_saves_astropy_readable_catalog(self):
        self.write_rstar_catalog()
        output = self.root / "gal_0_2_3.fits"
        self.runner.gal_ofmt = str(self.root / "gal_{:d}_{:d}_{:d}.fits")
        self.runner.fore_survey_labels_dict = {"boss_lowz_ngc": 7}
        self.runner.hod_populator = GalaxyHODPopulator()
        self.runner.survey_generator = GalaxySurveyGenerator()

        result = self.runner.gen_mock_gal(
            icosmo=0,
            irlz=2,
            ihod=3,
            ihod_param=np.array([13.2, 0.3, 14.0, 1.0, 0.8, 1.0]),
            save=True,
        )

        self.assertTrue(output.is_file())
        saved = Table.read(output)
        self.assertEqual(
            saved.colnames,
            ["marker", "survey", "gal_type", "host_halo_mvir"],
        )
        np.testing.assert_allclose(saved["marker"], [23.200614])
        np.testing.assert_array_equal(saved["survey"], [7])
        np.testing.assert_array_equal(saved["gal_type"], [1])
        np.testing.assert_allclose(saved["host_halo_mvir"], [2.5e13])
        np.testing.assert_array_equal(saved.as_array(), result)

    def test_gen_mock_void_filters_redshift_and_preserves_survey(self):
        self.assertTrue(
            hasattr(self.runner, "gen_mock_void"),
            "FastPMRunner.gen_mock_void must be implemented",
        )
        self.runner.config.zmin_lightcone = 0.0
        self.runner.config.zmax_lightcone = 1.0
        self.runner.fore_survey_labels_dict = {"boss_lowz_ngc": 7}
        self.runner.void_finder = FakeVoidFinder()
        self.runner.survey_generator = VoidSurveyGenerator()
        galaxies = np.zeros(1, dtype=[("survey", "i4")])
        galaxies["survey"] = 7
        result = self.runner.gen_mock_void(
            icosmo=0,
            irlz=2,
            ihod=3,
            galcone_survey=galaxies,
            dive_input="input.dat",
            dive_output="output.dat",
            save=False,
        )
        self.assertEqual(len(result), 1)
        np.testing.assert_allclose(result["z"], [0.2])
        np.testing.assert_array_equal(result["survey"], [7])

    def test_gen_mock_void_requires_output_format_when_saving(self):
        self.assertTrue(
            hasattr(self.runner, "gen_mock_void"),
            "FastPMRunner.gen_mock_void must be implemented",
        )
        galaxies = np.zeros(1, dtype=[("survey", "i4")])
        with self.assertRaisesRegex(ValueError, "void_ofmt"):
            self.runner.gen_mock_void(
                0, 0, 0, galaxies, "input", "output", save=True
            )

    def test_gen_mock_void_saves_astropy_readable_catalog(self):
        self.runner.config.zmin_lightcone = 0.0
        self.runner.config.zmax_lightcone = 1.0
        output = self.root / "void_0_2_3.fits"
        self.runner.void_ofmt = str(self.root / "void_{:d}_{:d}_{:d}.fits")
        self.runner.fore_survey_labels_dict = {"boss_lowz_ngc": 7}
        self.runner.void_finder = FakeVoidFinder()
        self.runner.survey_generator = VoidSurveyGenerator()
        galaxies = np.zeros(1, dtype=[("survey", "i4")])
        galaxies["survey"] = 7

        result = self.runner.gen_mock_void(
            icosmo=0,
            irlz=2,
            ihod=3,
            galcone_survey=galaxies,
            dive_input="input.dat",
            dive_output="output.dat",
            save=True,
        )

        self.assertTrue(output.is_file())
        saved = Table.read(output)
        self.assertEqual(saved.colnames, ["z", "survey", "marker"])
        np.testing.assert_allclose(saved["z"], [0.2])
        np.testing.assert_array_equal(saved["survey"], [7])
        np.testing.assert_array_equal(saved["marker"], [1])
        np.testing.assert_array_equal(saved.as_array(), result)

    def test_unsupported_foreground_survey_is_rejected(self):
        self.assertTrue(
            hasattr(self.runner, "_pick_gen_mock_func"),
            "FastPMRunner._pick_gen_mock_func must be implemented",
        )
        with self.assertRaisesRegex(ValueError, "unknown"):
            self.runner._pick_gen_mock_func("unknown")

    def test_load_shear_maps_uses_cosmology_realization_and_redshift_order(self):
        self.assertTrue(
            hasattr(self.runner, "_load_shear_maps"),
            "FastPMRunner._load_shear_maps must be implemented",
        )
        self.write_shear_product(icosmo=1, irlz=2)
        self.runner.shear_sim_fmt = str(
            self.root
            / "products"
            / "cosmo_{:06d}"
            / "realization_{:04d}.npz"
        )

        result = self.runner._load_shear_maps(icosmo=1, irlz=2)

        self.assertEqual(list(result), ["shell0", "shell1"])
        self.assertEqual(result["shell0"]["redshift"], 0.2)
        self.assertEqual(result["shell1"]["redshift"], 0.8)
        np.testing.assert_array_equal(result["shell0"]["gamma1"], [2.0] * 12)
        np.testing.assert_array_equal(result["shell1"]["gamma2"], [-8.0] * 12)

    def test_load_shear_maps_rejects_mismatched_product_identity(self):
        self.write_shear_product(
            icosmo=1,
            irlz=2,
            metadata_icosmo=4,
            metadata_irlz=7,
        )
        self.runner.shear_sim_fmt = str(
            self.root
            / "products"
            / "cosmo_{:06d}"
            / "realization_{:04d}.npz"
        )

        with self.assertRaisesRegex(ValueError, "cosmology_index"):
            self.runner._load_shear_maps(icosmo=1, irlz=2)

    def test_load_shear_maps_rejects_mismatched_cosmology_parameters(self):
        self.write_shear_product(
            icosmo=0,
            irlz=0,
            cosmology_overrides={"OmegaM": 0.9},
        )
        self.runner.shear_sim_fmt = str(
            self.root
            / "products"
            / "cosmo_{:06d}"
            / "realization_{:04d}.npz"
        )

        with self.assertRaisesRegex(ValueError, "OmegaM"):
            self.runner._load_shear_maps(icosmo=0, irlz=0)

    def test_load_shear_maps_rejects_incompatible_healpix_convention(self):
        self.write_shear_product(
            icosmo=1,
            irlz=2,
            ordering="NESTED",
            coordinate_system="G",
        )
        self.runner.shear_sim_fmt = str(
            self.root
            / "products"
            / "cosmo_{:06d}"
            / "realization_{:04d}.npz"
        )

        with self.assertRaisesRegex(ValueError, "RING.*celestial"):
            self.runner._load_shear_maps(icosmo=1, irlz=2)

    def test_load_shear_maps_rejects_unpaired_gamma_components(self):
        self.write_shear_product(
            icosmo=1,
            irlz=2,
            include_near_gamma2=False,
        )
        self.runner.shear_sim_fmt = str(
            self.root
            / "products"
            / "cosmo_{:06d}"
            / "realization_{:04d}.npz"
        )

        try:
            self.runner._load_shear_maps(icosmo=1, irlz=2)
        except ValueError as error:
            self.assertRegex(str(error), "gamma1.*gamma2")
        except Exception as error:
            self.fail(f"invalid gamma components need a clear ValueError: {error}")
        else:
            self.fail("unpaired gamma components must be rejected")

    def test_load_shear_maps_rejects_invalid_healpix_array_size(self):
        self.write_shear_product(icosmo=1, irlz=2, map_size=10)
        self.runner.shear_sim_fmt = str(
            self.root
            / "products"
            / "cosmo_{:06d}"
            / "realization_{:04d}.npz"
        )

        with self.assertRaisesRegex(ValueError, "HEALPix"):
            self.runner._load_shear_maps(icosmo=1, irlz=2)

    def test_load_shear_maps_rejects_duplicate_source_redshifts(self):
        self.write_shear_product(
            icosmo=1,
            irlz=2,
            near_redshift=0.5,
            far_redshift=0.5,
        )
        self.runner.shear_sim_fmt = str(
            self.root
            / "products"
            / "cosmo_{:06d}"
            / "realization_{:04d}.npz"
        )

        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            self.runner._load_shear_maps(icosmo=1, irlz=2)

    def test_load_shear_maps_rejects_source_without_effective_redshift(self):
        self.write_shear_product(
            icosmo=1,
            irlz=2,
            omit_near_redshift=True,
        )
        self.runner.shear_sim_fmt = str(
            self.root
            / "products"
            / "cosmo_{:06d}"
            / "realization_{:04d}.npz"
        )

        try:
            self.runner._load_shear_maps(icosmo=1, irlz=2)
        except ValueError as error:
            self.assertRegex(str(error), "source_near.*effective_redshift")
        except Exception as error:
            self.fail(f"invalid source metadata needs a clear ValueError: {error}")
        else:
            self.fail("missing effective_redshift must be rejected")

    def test_gen_mock_shear_returns_and_saves_all_tomographic_catalogs(self):
        self.assertTrue(
            hasattr(self.runner, "gen_mock_shear"),
            "FastPMRunner.gen_mock_shear must be implemented",
        )
        self.write_shear_product(icosmo=1, irlz=2)
        self.runner.shear_sim_fmt = str(
            self.root
            / "products"
            / "cosmo_{:06d}"
            / "realization_{:04d}.npz"
        )
        self.runner.shear_ofmt = str(self.root / "shape_{:d}_{:d}.fits")
        self.runner.back_survey_labels_dict = {"survey_a": 9}
        self.runner.back_ngals_dict = {"tomo_low": 1.25, "tomo_high": 2.5}
        self.runner.tomo_labels_dict = {"tomo_low": 1, "tomo_high": 2}
        self.runner.shear_assigner = FakeShearAssigner()

        result = self.runner.gen_mock_shear(icosmo=1, irlz=2, save=True)

        np.testing.assert_array_equal(result["survey"], [9, 9])
        np.testing.assert_array_equal(result["tomo"], [1, 2])
        np.testing.assert_allclose(result["ngal"], [1.25, 2.5])
        np.testing.assert_allclose(result["g1"], [2.0, 2.0])
        np.testing.assert_array_equal(result["map_count"], [2, 2])
        saved = Table.read(self.root / "shape_1_2.fits")
        np.testing.assert_array_equal(saved.as_array(), result)

    def test_gen_mock_shear_requires_shear_map_format(self):
        try:
            self.runner.gen_mock_shear(icosmo=0, irlz=0, save=False)
        except ValueError as error:
            self.assertRegex(str(error), "build_shape_runner")
        except Exception as error:
            self.fail(f"missing shape builder needs a clear ValueError: {error}")
        else:
            self.fail("gen_mock_shear must require a shape builder")

    def test_gen_mock_shear_requires_background_configuration(self):
        self.write_shear_product(icosmo=0, irlz=0)
        self.runner.shear_sim_fmt = str(
            self.root
            / "products"
            / "cosmo_{:06d}"
            / "realization_{:04d}.npz"
        )

        with self.assertRaisesRegex(ValueError, "build_shape_runner"):
            self.runner.gen_mock_shear(icosmo=0, irlz=0, save=False)

    def test_gen_mock_shear_rejects_sources_beyond_map_coverage(self):
        self.write_shear_product(icosmo=0, irlz=0)
        self.runner.shear_sim_fmt = str(
            self.root
            / "products"
            / "cosmo_{:06d}"
            / "realization_{:04d}.npz"
        )
        self.runner.back_survey_labels_dict = {"survey_a": 9}
        self.runner.back_ngals_dict = {"tomo1": 1.25}
        self.runner.tomo_labels_dict = {"tomo1": 1}
        self.runner.shear_assigner = FakeShearAssigner(
            redshifts=(0.4, 0.8, 1.5)
        )

        with self.assertRaisesRegex(ValueError, "redshift.*coverage"):
            self.runner.gen_mock_shear(icosmo=0, irlz=0, save=False)

    def test_gen_mock_shear_requires_output_format_when_saving(self):
        self.write_shear_product(icosmo=0, irlz=0)
        self.runner.shear_sim_fmt = str(
            self.root
            / "products"
            / "cosmo_{:06d}"
            / "realization_{:04d}.npz"
        )
        self.runner.back_survey_labels_dict = {"survey_a": 9}
        self.runner.back_ngals_dict = {"tomo_low": 1.25}
        self.runner.tomo_labels_dict = {"tomo_low": 1}
        self.runner.shear_assigner = FakeShearAssigner()

        try:
            self.runner.gen_mock_shear(icosmo=0, irlz=0, save=True)
        except ValueError as error:
            self.assertRegex(str(error), "shear_ofmt")
        except Exception as error:
            self.fail(f"missing shear_ofmt needs a clear ValueError: {error}")
        else:
            self.fail("save=True must require shear_ofmt")

    def test_constructor_shear_configuration_generates_real_shape_catalog(self):
        self.write_shear_product(icosmo=0, irlz=0)
        mask_path, nofz_path = self.write_background_inputs()
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", ResourceWarning)
            try:
                runner = runner_module.FastPMRunner(
                    config=PipeConfig(
                        Lbox=1000.0,
                        Npart=1024,
                        redshift=0.3,
                        sigma_e=0.0,
                        sigma_phz=0.0,
                    ),
                    halo_fmt=self.halo_fmt,
                    cosmo_par_fname=self.cosmo_file,
                    fore_mask_fnames_dict={"boss_veto": []},
                    fore_nofz_fnames_dict={},
                    fore_survey_labels_dict={},
                    shear_sim_fmt=str(
                        self.root
                        / "products"
                        / "cosmo_{:06d}"
                        / "realization_{:04d}.npz"
                    ),
                    back_mask_fnames_dict={
                        "KiDS1000-North": str(mask_path)
                    },
                    back_nofz_fnames_dict={"tomo1": str(nofz_path)},
                    back_survey_labels_dict={"KiDS1000-North": 4},
                    back_ngals_dict={"tomo1": 1e-6},
                    tomo_labels_dict={"tomo1": 1},
                    shear_ofmt=str(self.root / "shape_{:d}_{:d}.fits"),
                )
            except TypeError as error:
                self.fail(
                    f"FastPMRunner must accept shear configuration: {error}"
                )
            gc.collect()
        resource_warnings = [
            warning for warning in caught_warnings
            if issubclass(warning.category, ResourceWarning)
        ]
        self.assertEqual(resource_warnings, [])

        result = runner.gen_mock_shear(icosmo=0, irlz=0, save=False)

        self.assertGreater(len(result), 0)
        np.testing.assert_array_equal(result["survey"], 4)
        np.testing.assert_array_equal(result["tomo"], 1)
        np.testing.assert_allclose(result["g1_pure"], 2.0)
        np.testing.assert_allclose(result["g2_pure"], -2.0)
        np.testing.assert_allclose(result["g1"], 2.0)
        np.testing.assert_allclose(result["g2"], -2.0)
        np.testing.assert_allclose(result["w"], 1.0)

    def test_constructor_rejects_tomographic_key_label_mismatch(self):
        mask_path, nofz_path = self.write_background_inputs()

        with self.assertRaisesRegex(ValueError, "tomo_low.*tomo1"):
            runner_module.FastPMRunner(
                config=self.config,
                halo_fmt=self.halo_fmt,
                cosmo_par_fname=self.cosmo_file,
                fore_mask_fnames_dict={"boss_veto": []},
                fore_nofz_fnames_dict={},
                fore_survey_labels_dict={},
                shear_sim_fmt="unused_{:06d}_{:04d}.npz",
                back_mask_fnames_dict={
                    "KiDS1000-North": str(mask_path)
                },
                back_nofz_fnames_dict={"tomo_low": str(nofz_path)},
                back_survey_labels_dict={"KiDS1000-North": 4},
                back_ngals_dict={"tomo_low": 1e-6},
                tomo_labels_dict={"tomo_low": 1},
            )

    def test_fastpm_background_masks_reject_unsupported_full_sky(self):
        with patch.object(runner_module.hp, "nside2npix", return_value=12):
            with self.assertRaisesRegex(ValueError, "FullSky"):
                self.runner._prepare_back_masks({"FullSky": None})
