import ast
import json
import inspect
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, sentinel

import numpy as np

from handler import HODPopulator, PipeConfig, SurveyGenerator, VoidFinder
from runner import CosmoGridRunner
from utils.mkfore_utils import apply_nz, find_void


class _HaloMassColumn:
    def __init__(self, values):
        self.value = np.asarray(values, dtype=float)


class _HaloTable:
    def __init__(self, masses):
        self._masses = _HaloMassColumn(masses)

    def __getitem__(self, name):
        if name != "halo_mvir":
            raise KeyError(name)
        return self._masses


class _HODHaloCatalog:
    def __init__(self, masses=(1.0e13,)):
        self.halo_table = _HaloTable(masses)


class HODSamplingRegressionTests(unittest.TestCase):
    def setUp(self):
        self.config = PipeConfig(
            Lbox=1000.0,
            Npart=1024,
            redshift=0.3,
            model=2,
            nhod_per_cosmo=2,
        )
        self.populator = HODPopulator(self.config)
        self.halo_catalog = _HODHaloCatalog()

    def test_find_hod_params_returns_configured_number_of_rows(self):
        pool = np.arange(20.0).reshape(4, 5)
        self.populator._open_params_pool = lambda size, seed: pool

        result = self.populator.find_hod_params(
            self.halo_catalog,
            num_pool=len(pool),
        )

        result = np.asarray(result)
        self.assertEqual(result.shape, (2, 6))
        np.testing.assert_allclose(result[:, :5], pool[:2])
        np.testing.assert_allclose(result[:, -1], 1.0)

    def test_find_hod_params_rejects_pool_smaller_than_requested_count(self):
        with self.assertRaisesRegex(ValueError, "nhod_per_cosmo"):
            self.populator.find_hod_params(self.halo_catalog, num_pool=1)

    def test_cosmogrid_sampling_seed_retains_realization_index(self):
        class CatalogLoader:
            def load_pkd_halocat(self, *args, **kwargs):
                return self

        class SamplingPopulator:
            def find_hod_params(self, halo_catalog, seed_offset):
                return seed_offset

        runner = CosmoGridRunner.__new__(CosmoGridRunner)
        runner.config = PipeConfig(
            Lbox=1000.0,
            Npart=1024,
            redshift=0.3,
            nrlzs_per_cosmo=3,
        )
        runner._get_fnames = lambda icosmo, irlz: ("params", "halos")
        runner._get_cosmo_instance = lambda fname, otype: object()
        runner.cata_loader = CatalogLoader()
        runner.hod_populator = SamplingPopulator()

        result = runner.sample_hod_params(icosmo=2, irlz=1)

        self.assertEqual(result, 7)

    def test_model3_parameter_pool_is_reproducible_for_the_same_seed(self):
        config = PipeConfig(
            Lbox=1000.0,
            Npart=1024,
            redshift=0.3,
            model=3,
        )
        populator = HODPopulator(config)

        first = populator._open_params_pool(num_pool=16, seed=123)
        repeated = populator._open_params_pool(num_pool=16, seed=123)
        different = populator._open_params_pool(num_pool=16, seed=124)

        np.testing.assert_array_equal(first, repeated)
        self.assertFalse(np.array_equal(first, different))


class ScriptHODIORegressionTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = Path(self.tempdir.name) / "hod.json"

    def tearDown(self):
        self.tempdir.cleanup()

    def test_sampling_output_matches_all_mock_loaders(self):
        import run_mock_gal
        import run_mock_shape
        import run_mock_void
        import run_sampling_hod

        first = {
            "cosmo_000001": run_sampling_hod.get_hod_params_container(
                np.arange(12.0).reshape(2, 6)
            ),
        }
        second = {
            "cosmo_000002": run_sampling_hod.get_hod_params_container(
                np.arange(6.0).reshape(1, 6)
            ),
        }
        samples = run_sampling_hod.merge_hod_sample_parts([first, second])
        run_sampling_hod.save_hod_samples(self.path, samples)

        expected = {
            "cosmo_000001": {
                "HOD0": np.arange(6.0).tolist(),
                "HOD1": np.arange(6.0, 12.0).tolist(),
            },
            "cosmo_000002": {
                "HOD0": np.arange(6.0).tolist(),
            },
        }
        self.assertEqual(json.loads(self.path.read_text()), expected)
        for module in (run_mock_gal, run_mock_void, run_mock_shape):
            with self.subTest(module=module.__name__):
                self.assertEqual(module.load_hod_samples(self.path), expected)

    def test_all_mock_scripts_read_underscore_cosmology_keys(self):
        import run_mock_gal
        import run_mock_shape
        import run_mock_void

        self.path.write_text(json.dumps({
            "cosmo_000001": {"HOD0": [1.0]},
            "cosmo_000023": {"HOD0": [2.0]},
        }))

        for module in (run_mock_gal, run_mock_void, run_mock_shape):
            with self.subTest(module=module.__name__):
                self.assertEqual(
                    module.get_cosmo_labels_processed(self.path),
                    [1, 23],
                )


class DriverBuilderRegressionTests(unittest.TestCase):
    def _assert_runner_builder_call(
            self, tree, *, builder_name, expected_keywords):
        runner_assignment = next(
            (
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name)
                    and target.id == "cosmogrid_runner"
                    for target in node.targets
                )
            ),
            None,
        )
        self.assertIsNotNone(runner_assignment)
        call = runner_assignment.value
        self.assertIsInstance(call, ast.Call)
        self.assertIsInstance(call.func, ast.Attribute)
        self.assertIsInstance(call.func.value, ast.Name)
        self.assertEqual(call.func.value.id, "CosmoGridRunner")
        self.assertEqual(call.func.attr, builder_name)
        self.assertEqual(
            {keyword.arg for keyword in call.keywords},
            expected_keywords,
        )

    def test_runner_assignment_rejects_unrelated_expected_builder_call(self):
        tree = ast.parse(
            "CosmoGridRunner.build_hod_runner(config=config)\n"
            "cosmogrid_runner = CosmoGridRunner.build_gal_runner(\n"
            "    config=config\n"
            ")\n"
        )

        with self.assertRaisesRegex(AssertionError, "build_hod_runner"):
            self._assert_runner_builder_call(
                tree,
                builder_name="build_hod_runner",
                expected_keywords={"config"},
            )

    def test_each_driver_uses_its_task_specific_builder(self):
        expected = {
            "run_sampling_hod.py": (
                "build_hod_runner",
                {"config", "sim_fmt", "halo_fmt", "lb_z_file"},
            ),
            "run_mock_gal.py": (
                "build_gal_runner",
                {
                    "config", "sim_fmt", "halo_fmt", "lb_z_file",
                    "fore_mask_fnames_dict", "fore_nofz_fnames_dict",
                    "fore_survey_labels_dict", "gal_ofmt",
                },
            ),
            "run_mock_void.py": (
                "build_void_runner",
                {
                    "config", "sim_fmt", "lb_z_file",
                    "fore_mask_fnames_dict", "fore_nofz_fnames_dict",
                    "fore_survey_labels_dict", "void_ofmt",
                },
            ),
            "run_mock_shape.py": (
                "build_shape_runner",
                {
                    "config", "shear_sim_fmt", "back_mask_fnames_dict",
                    "back_nofz_fnames_dict", "back_survey_labels_dict",
                    "back_ngals_dict", "tomo_labels_dict",
                    "redshift_src_list", "shear_ofmt",
                },
            ),
        }
        root = Path(__file__).resolve().parents[1]
        for filename, (builder_name, expected_keywords) in expected.items():
            with self.subTest(filename=filename):
                tree = ast.parse((root / filename).read_text())
                called_attributes = {
                    node.func.attr
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                }
                self._assert_runner_builder_call(
                    tree,
                    builder_name=builder_name,
                    expected_keywords=expected_keywords,
                )
                self.assertNotIn("for_foreground", called_attributes)


class CosmoGridRunnerBuilderTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.label_file = Path(self.tempdir.name) / "label_z.txt"
        self.label_file.write_text("0 0.3\n1 0.4\n")
        self.config = PipeConfig(Lbox=1000.0, Npart=1024, redshift=0.3)
        self.foreground = {
            "fore_mask_fnames_dict": {"boss_veto": []},
            "fore_nofz_fnames_dict": {},
            "fore_survey_labels_dict": {},
        }
        self.background = {
            "back_mask_fnames_dict": {},
            "back_nofz_fnames_dict": {},
            "back_survey_labels_dict": {},
            "back_ngals_dict": {},
            "tomo_labels_dict": {},
        }

    def tearDown(self):
        self.tempdir.cleanup()

    def test_constructor_and_builder_parameters_default_to_none(self):
        callables = [
            CosmoGridRunner.__init__,
            CosmoGridRunner.build_hod_runner,
            CosmoGridRunner.build_gal_runner,
            CosmoGridRunner.build_void_runner,
            CosmoGridRunner.build_shape_runner,
        ]
        for callable_object in callables:
            with self.subTest(callable=callable_object.__name__):
                parameters = inspect.signature(callable_object).parameters
                for name, parameter in parameters.items():
                    if name in {"self", "cls"}:
                        continue
                    self.assertIsNone(parameter.default, name)

    def test_each_builder_initializes_exact_component_set(self):
        replacements = {
            "CatalogLoader": Mock(return_value=sentinel.catalog_loader),
            "HODPopulator": Mock(return_value=sentinel.hod_populator),
            "SurveyGenerator": Mock(return_value=sentinel.survey_generator),
            "VoidFinder": Mock(return_value=sentinel.void_finder),
            "ShearAssigner": Mock(return_value=sentinel.shear_assigner),
        }
        with patch.multiple("runner", **replacements):
            hod = CosmoGridRunner.build_hod_runner(
                config=self.config,
                sim_fmt="sim/{:d}/{:d}",
                halo_fmt="halo.{:d}",
                lb_z_file=self.label_file,
            )
            gal = CosmoGridRunner.build_gal_runner(
                config=self.config,
                sim_fmt="sim/{:d}/{:d}",
                halo_fmt="halo.{:d}",
                lb_z_file=self.label_file,
                **self.foreground,
            )
            void = CosmoGridRunner.build_void_runner(
                config=self.config,
                sim_fmt="sim/{:d}/{:d}",
                lb_z_file=self.label_file,
                **self.foreground,
            )
            shape = CosmoGridRunner.build_shape_runner(
                config=self.config,
                shear_sim_fmt="shear_{:d}_{:.2f}.hdf5",
                redshift_src_list=[0.2, 0.4],
                **self.background,
            )

        self.assertEqual(
            (hod.cata_loader, hod.hod_populator, hod.survey_generator,
             hod.void_finder, hod.shear_assigner),
            (sentinel.catalog_loader, sentinel.hod_populator,
             None, None, None),
        )
        self.assertEqual(hod.runner_type, "hod")
        self.assertEqual(
            (gal.cata_loader, gal.hod_populator, gal.survey_generator,
             gal.void_finder, gal.shear_assigner),
            (sentinel.catalog_loader, sentinel.hod_populator,
             sentinel.survey_generator, None, None),
        )
        self.assertEqual(
            (void.cata_loader, void.hod_populator, void.survey_generator,
             void.void_finder, void.shear_assigner),
            (None, None, sentinel.survey_generator,
             sentinel.void_finder, None),
        )
        self.assertEqual(
            (shape.cata_loader, shape.hod_populator, shape.survey_generator,
             shape.void_finder, shape.shear_assigner),
            (None, None, None, None, sentinel.shear_assigner),
        )

    def test_void_builder_requires_no_halo_path(self):
        runner = CosmoGridRunner.build_void_runner(
            config=self.config,
            sim_fmt="sim/{:d}/{:d}",
            lb_z_file=self.label_file,
            **self.foreground,
        )
        self.assertIsNone(runner.halo_fmt)
        self.assertIsNone(runner.cata_loader)
        self.assertIsNone(runner.hod_populator)

    def test_builder_lists_all_missing_required_parameters(self):
        with self.assertRaisesRegex(
                ValueError, "build_shape_runner.*shear_sim_fmt.*tomo_labels_dict"):
            CosmoGridRunner.build_shape_runner(config=self.config)

    def test_partial_direct_foreground_group_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "fore_nofz_fnames_dict"):
            CosmoGridRunner(
                config=self.config,
                fore_mask_fnames_dict={"boss_veto": []},
            )

    def test_direct_constructor_can_initialize_both_groups(self):
        runner = CosmoGridRunner(
            config=self.config,
            sim_fmt="sim/{:d}/{:d}",
            halo_fmt="halo.{:d}",
            lb_z_file=self.label_file,
            shear_sim_fmt="shear_{:d}_{:.2f}.hdf5",
            redshift_src_list=[0.2, 0.4],
            **self.foreground,
            **self.background,
        )
        self.assertIsNotNone(runner.hod_populator)
        self.assertIsNotNone(runner.survey_generator)
        self.assertIsNotNone(runner.void_finder)
        self.assertIsNotNone(runner.shear_assigner)

    def test_incompatible_task_method_names_required_builder(self):
        runner = CosmoGridRunner()
        calls = [
            ("build_hod_runner", lambda: runner.sample_hod_params(0, 0)),
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
        runner = CosmoGridRunner.build_gal_runner(
            config=self.config,
            sim_fmt="sim/{:d}/{:d}",
            halo_fmt="halo.{:d}",
            lb_z_file=self.label_file,
            **self.foreground,
        )
        components = (
            runner.cata_loader,
            runner.hod_populator,
            runner.survey_generator,
        )
        runner._initialize_hod_components()
        runner._initialize_survey_generator()
        self.assertEqual(
            components,
            (runner.cata_loader, runner.hod_populator, runner.survey_generator),
        )

    def test_void_cosmology_path_does_not_need_halo_format(self):
        runner = CosmoGridRunner.build_void_runner(
            config=self.config,
            sim_fmt="sim/cosmo_{:d}/run_{:d}",
            lb_z_file=self.label_file,
            **self.foreground,
        )
        self.assertEqual(
            runner._get_cosmo_fname(2, 3),
            "sim/cosmo_2/run_3/params.yml",
        )

    def test_void_generation_uses_requested_realization_cosmology_path(self):
        class StopAfterCosmologyLookup(Exception):
            pass

        runner = CosmoGridRunner.build_void_runner(
            config=self.config,
            sim_fmt="sim/cosmo_{:d}/run_{:d}",
            lb_z_file=self.label_file,
            **self.foreground,
        )

        def stop_after_cosmology_lookup(fname, otype):
            self.assertEqual(fname, "sim/cosmo_2/run_3/params.yml")
            self.assertEqual(otype, "ccl")
            raise StopAfterCosmologyLookup

        runner._get_cosmo_instance = stop_after_cosmology_lookup

        with self.assertRaises(StopAfterCosmologyLookup):
            runner.gen_mock_void(2, 3, 0, np.zeros(0), "input", "output")

    def test_direct_constructor_infers_each_supplied_group(self):
        hod = CosmoGridRunner(
            config=self.config,
            sim_fmt="sim/{:d}/{:d}",
            halo_fmt="halo.{:d}",
            lb_z_file=self.label_file,
        )
        foreground = CosmoGridRunner(
            config=self.config,
            sim_fmt="sim/{:d}/{:d}",
            lb_z_file=self.label_file,
            **self.foreground,
        )
        background = CosmoGridRunner(
            config=self.config,
            shear_sim_fmt="shear_{:d}_{:.2f}.hdf5",
            redshift_src_list=[0.2, 0.4],
            **self.background,
        )
        self.assertIsNotNone(hod.hod_populator)
        self.assertIsNone(hod.survey_generator)
        self.assertIsNotNone(foreground.survey_generator)
        self.assertIsNotNone(foreground.void_finder)
        self.assertIsNone(foreground.hod_populator)
        self.assertIsNotNone(background.shear_assigner)
        self.assertIsNone(background.survey_generator)


class ApplyNzRegressionTests(unittest.TestCase):
    def setUp(self):
        self.catalog = np.zeros(
            3,
            dtype=[
                ("z", "f8"),
                ("zrsd", "f8"),
                ("host_halo_mvir", "f8"),
                ("GID", "i4"),
            ],
        )
        self.catalog["z"] = [0.2, 0.8, 1.2]
        self.catalog["zrsd"] = self.catalog["z"]
        self.catalog["host_halo_mvir"] = [1.0e13, 2.0e13, 3.0e13]
        self.catalog["GID"] = [1, 2, 3]
        self.nofz = {
            "zedges": np.array([0.0, 1.0]),
            "shell_vol": np.array([1.0]),
            "nz_ref": np.array([1.0]),
        }

    def _apply(self, catalog, method):
        try:
            return apply_nz(catalog, self.nofz, method)
        except Exception as error:
            self.fail(
                f"apply_nz({method!r}) raised {type(error).__name__}: {error}"
            )

    def test_apply_nz_const_keeps_catalog_inside_edges(self):
        result = self._apply(self.catalog, "const")

        np.testing.assert_array_equal(result["GID"], [1, 2])

    def test_apply_nz_rank_uses_integer_target_count(self):
        result = self._apply(self.catalog, "rank")

        np.testing.assert_array_equal(result["GID"], [2])

    def test_apply_nz_empty_selection_preserves_dtype(self):
        result = self._apply(self.catalog[:0], "const")

        self.assertEqual(result.dtype, self.catalog.dtype)
        self.assertEqual(len(result), 0)

    def test_apply_nz_rejects_unknown_method(self):
        try:
            apply_nz(self.catalog, self.nofz, "unknown")
        except ValueError as error:
            self.assertIn("nofz_method", str(error))
        except Exception as error:
            self.fail(
                f"unknown method raised {type(error).__name__}, not ValueError"
            )
        else:
            self.fail("unknown nofz_method was accepted")


class GalaxyFieldPropagationRegressionTests(unittest.TestCase):
    def test_cosmogrid_runner_preserves_galaxy_type_and_host_mass(self):
        class Cosmology:
            def omega_x(self, a, species):
                return 0.3

        class CatalogLoader:
            def load_pkd_halocat(self, *args, **kwargs):
                return object()

        class GalaxyPopulator:
            def __init__(self, config):
                self.config = config

            def populate_galaxies(self, *args, **kwargs):
                galaxies = np.zeros(
                    1,
                    dtype=[
                        ("x", "f8"), ("y", "f8"), ("z", "f8"),
                        ("vx", "f8"), ("vy", "f8"), ("vz", "f8"),
                        ("gal_type", "i4"), ("halo_mvir", "f8"),
                    ],
                )
                galaxies["gal_type"] = 1
                galaxies["halo_mvir"] = 4.0e13
                return {"sample": galaxies}

            get_galaxy_features = HODPopulator.get_galaxy_features

        class SurveyGenerator:
            def box_to_lightcone(self, cosmo, gal_pos, gal_adj_props):
                result = np.zeros(
                    1,
                    dtype=[
                        ("survey", "i4"),
                        ("gal_type", "i4"),
                        ("host_halo_mvir", "f8"),
                    ],
                )
                result["gal_type"] = gal_adj_props.get("gal_type", [-1])
                result["host_halo_mvir"] = gal_adj_props.get(
                    "host_halo_mvir", [-1.0]
                )
                return result

            def gen_boss_like(
                    self, galcone, survey_name, survey_label, make_nz=True):
                result = galcone.copy()
                result["survey"] = survey_label
                return result

        config = PipeConfig(Lbox=1000.0, Npart=1024, redshift=0.3)
        runner = CosmoGridRunner.__new__(CosmoGridRunner)
        runner.config = config
        runner.fore_survey_labels_dict = {"boss_lowz_ngc": 7}
        runner.gal_ofmt = None
        runner._get_fnames = lambda icosmo, irlz: ("params", "halos")
        runner._get_cosmo_instance = lambda fname, otype: Cosmology()
        runner.cata_loader = CatalogLoader()
        runner.hod_populator = GalaxyPopulator(config)
        runner.survey_generator = SurveyGenerator()

        result = runner.gen_mock_gal(
            icosmo=0,
            irlz=0,
            ihod=0,
            ihod_param=np.ones(6),
        )

        np.testing.assert_array_equal(result["gal_type"], [1])
        np.testing.assert_allclose(result["host_halo_mvir"], [4.0e13])

    def test_real_survey_generator_preserves_adjacent_galaxy_fields(self):
        config = PipeConfig(
            Lbox=1000.0,
            Npart=1024,
            redshift=0.3,
            zmin_lightcone=0.0,
            zmax_lightcone=1.0,
            rsd_lightcone=False,
        )
        generator = SurveyGenerator(config, masks={}, nofzs={})
        generator._calc_radial_dist = lambda cosmo, zs: (0.0, 100.0)
        gal_pos = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        gal_vel = np.zeros((2, 3))
        gal_type = np.array([1, 0])
        host_mass = np.array([4.0e13, 2.0e13])

        def make_tiles(
                positions, boxsize, chi_min, chi_max, ctr, other_props):
            gids = np.arange(len(positions), dtype=float)[:, None]
            return (
                np.column_stack([positions, gids]),
                [np.asarray(values).copy() for values in other_props],
            )

        with patch("handler.make_lightcone_tiles", side_effect=make_tiles), patch(
            "handler.Cart2Sph",
            return_value=(
                np.array([10.0, 20.0]),
                np.array([30.0, 40.0]),
                np.array([0.2, 0.8]),
                np.array([True, True]),
            ),
        ):
            result = generator.box_to_lightcone(
                cosmo=object(),
                gal_pos=gal_pos,
                gal_adj_props={
                    "gal_vel": gal_vel,
                    "gal_type": gal_type,
                    "host_halo_mvir": host_mass,
                },
            )

        np.testing.assert_array_equal(result["gal_type"], [1, 0])
        np.testing.assert_allclose(
            result["host_halo_mvir"],
            [4.0e13, 2.0e13],
            rtol=1e-6,
        )

    def test_real_survey_generator_rejects_misaligned_adjacent_fields(self):
        config = PipeConfig(
            Lbox=1000.0,
            Npart=1024,
            redshift=0.3,
            rsd_lightcone=False,
        )
        generator = SurveyGenerator(config, masks={}, nofzs={})
        generator._calc_radial_dist = lambda cosmo, zs: (0.0, 100.0)

        with self.assertRaisesRegex(ValueError, "gal_type.*1 rows.*2"):
            generator.box_to_lightcone(
                cosmo=object(),
                gal_pos=np.zeros((2, 3)),
                gal_adj_props={"gal_type": np.array([1])},
            )


class VoidExecutionRegressionTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.input_path = self.root / "dive_input.txt"
        self.output_path = self.root / "dive_output.txt"

    def tearDown(self):
        self.tempdir.cleanup()

    def _make_fake_dive(
            self, name="fake_dive", exit_code=0,
            output_text="4 5 6 7\n"):
        executable = self.root / name
        executable.write_text(
            "#!/usr/bin/env python3\n"
            "import pathlib\n"
            "import sys\n"
            f"exit_code = {exit_code}\n"
            "if exit_code:\n"
            "    raise SystemExit(exit_code)\n"
            "output = pathlib.Path(sys.argv[sys.argv.index('-o') + 1])\n"
            f"output.write_text({output_text!r})\n"
        )
        executable.chmod(0o755)
        return executable

    def _find_void(self, executable):
        try:
            return find_void(
                np.array([[1.0, 2.0, 3.0]]),
                exec_path=executable,
                dive_input=self.input_path,
                dive_output=self.output_path,
            )
        except Exception as error:
            self.fail(
                f"find_void raised {type(error).__name__}: {error}"
            )

    def test_find_void_handles_executable_path_with_spaces_and_one_row(self):
        executable = self._make_fake_dive("fake DIVE")

        positions, radii = self._find_void(executable)

        np.testing.assert_allclose(positions, [[4.0, 5.0, 6.0]])
        np.testing.assert_allclose(radii, [7.0])
        self.assertFalse(self.input_path.exists())
        self.assertFalse(self.output_path.exists())

    def test_find_void_cleans_temporary_files_when_executable_fails(self):
        executable = self._make_fake_dive(exit_code=3)

        try:
            find_void(
                np.array([[1.0, 2.0, 3.0]]),
                exec_path=executable,
                dive_input=self.input_path,
                dive_output=self.output_path,
            )
        except subprocess.CalledProcessError:
            pass
        except Exception as error:
            self.fail(
                "failed DIVE run raised "
                f"{type(error).__name__}, not CalledProcessError"
            )
        else:
            self.fail("failed DIVE executable was treated as successful")

        self.assertFalse(self.input_path.exists())
        self.assertFalse(self.output_path.exists())

    def test_void_finder_uses_configured_dive_executable(self):
        executable = self._make_fake_dive()
        config = PipeConfig(
            Lbox=1000.0,
            Npart=1024,
            redshift=0.3,
            dive_exec_path=str(executable),
        )
        finder = VoidFinder(config)
        galaxy_catalog = np.zeros(
            1,
            dtype=[("ra", "f8"), ("dec", "f8"), ("zrsd", "f8")],
        )

        with patch(
            "handler.Sph2Cart",
            return_value=np.array([[1.0, 2.0, 3.0]]),
        ), patch(
            "handler.Cart2Sph",
            return_value=(
                np.array([10.0]),
                np.array([20.0]),
                np.array([0.5]),
                np.array([True]),
            ),
        ):
            try:
                result = finder.galcone_to_voidcone(
                    galaxy_catalog,
                    cosmo_ccl=object(),
                    survey=4,
                    dive_input=self.input_path,
                    dive_output=self.output_path,
                )
            except Exception as error:
                self.fail(
                    "VoidFinder did not use the configured executable: "
                    f"{type(error).__name__}: {error}"
                )

        np.testing.assert_allclose(result["Rv"], [7.0])
        np.testing.assert_array_equal(result["survey"], [4])

    def test_find_void_rejects_malformed_output_and_cleans_files(self):
        executable = self._make_fake_dive(output_text="4 5 6\n")

        with self.assertRaisesRegex(ValueError, "x, y, z, and radius"):
            find_void(
                np.array([[1.0, 2.0, 3.0]]),
                exec_path=executable,
                dive_input=self.input_path,
                dive_output=self.output_path,
            )

        self.assertFalse(self.input_path.exists())
        self.assertFalse(self.output_path.exists())


if __name__ == "__main__":
    unittest.main()
