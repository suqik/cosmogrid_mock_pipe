# Task-Specific Runner Builders Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `CosmoGridRunner` and `FastPMRunner` four readable task-specific builders while retaining direct construction and initializing only the runtime components required by each task.

**Architecture:** Every constructor argument defaults to `None`. Builders pass an internal `runner_type` that selects eager, one-time initialization of HOD, galaxy, void, or shape components; direct construction with `runner_type=None` infers complete supplied groups for backward compatibility. Shared validation reports missing parameters and incompatible task-method calls explicitly.

**Tech Stack:** Python 3, NumPy, Astropy, unittest, unittest.mock, inspect, ast.

**Spec:** `docs/superpowers/specs/2026-08-28-task-specific-runner-builders-design.md`

## Global Constraints

- Apply the design to both `CosmoGridRunner` and `FastPMRunner`.
- Every argument of `__init__` and every public builder has the default value `None`.
- Keep direct `Runner(...)` construction supported.
- Builders initialize components eagerly once; generation methods never rebuild them.
- Preserve scientific calculations, catalog formats, output naming, and FastPM shear validation.
- Preserve all pre-existing working-tree changes and keep them out of builder-specific commits.
- Use `None` to mean “not supplied”; empty dictionaries and empty lists count as supplied values.
- Do not restore `utils/hod_io.py` or legacy non-underscore HOD keys.

---

### Task 0: Establish a Clean Regression-Fix Baseline

**Files:**
- Verify and commit: `handler.py`
- Verify and commit: `runner.py`
- Verify and commit: `utils/mkfore_utils.py`
- Verify and commit: `run_sampling_hod.py`
- Verify and commit: `run_mock_gal.py`
- Verify and commit: `run_mock_void.py`
- Verify and commit: `run_mock_shape.py`
- Verify and commit: `tests/test_fastpm_runner.py`
- Verify and commit: `tests/test_pipeline_regressions.py`
- Verify and commit: `docs/superpowers/specs/2026-08-28-pipeline-regression-fixes-design.md`
- Verify and commit: `docs/superpowers/plans/2026-08-28-pipeline-regression-fixes.md`
- Exclude from this commit: `docs/superpowers/plans/2026-08-30-task-specific-runner-builders.md`

**Interfaces:**
- Consumes: the already-developed pipeline regression fixes in the dirty worktree.
- Produces: a clean, tested baseline commit before constructor refactoring begins.

- [ ] **Step 1: Run the existing complete regression suite**

Run:

```bash
env MPLCONFIGDIR=/tmp/cosmogrid_mock_pipe_matplotlib \
/Users/suqikuai777/miniforge3/envs/mock_pipe/bin/python -m unittest discover -v
```

Expected: 65 tests pass with zero failures and zero errors.

- [ ] **Step 2: Compile the current modified Python files**

Run:

```bash
/Users/suqikuai777/miniforge3/envs/mock_pipe/bin/python -m py_compile \
handler.py runner.py utils/mkfore_utils.py run_sampling_hod.py \
run_mock_gal.py run_mock_void.py run_mock_shape.py \
tests/test_fastpm_runner.py tests/test_pipeline_regressions.py
```

Expected: exit code 0.

- [ ] **Step 3: Confirm diff hygiene and the deleted HOD utility**

Run:

```bash
git diff --check
test ! -e utils/hod_io.py
git status --short
```

Expected: no whitespace errors, `utils/hod_io.py` is absent, and only the files listed above plus this plan are pending.

- [ ] **Step 4: Commit only the regression-fix baseline**

Run:

```bash
git add handler.py runner.py utils/mkfore_utils.py \
run_sampling_hod.py run_mock_gal.py run_mock_void.py run_mock_shape.py \
tests/test_fastpm_runner.py tests/test_pipeline_regressions.py \
docs/superpowers/specs/2026-08-28-pipeline-regression-fixes-design.md \
docs/superpowers/plans/2026-08-28-pipeline-regression-fixes.md
git commit -m "fix: repair pipeline regressions"
```

Expected: the new commit contains the existing regression fixes and does not contain this implementation plan.

---

### Task 1: Add CosmoGrid Task Builders and Conditional Initialization

**Files:**
- Modify: `runner.py:9-140`
- Modify: `runner.py:322-460`
- Test: `tests/test_pipeline_regressions.py`

**Interfaces:**
- Consumes: `PipeConfig`, `CatalogLoader`, `HODPopulator`, `SurveyGenerator`, `VoidFinder`, and `ShearAssigner` from `handler.py`.
- Produces: `CosmoGridRunner.build_hod_runner(...)`, `build_gal_runner(...)`, `build_void_runner(...)`, `build_shape_runner(...)`.
- Produces: module helpers `_require_parameters(context, **parameters)`, `_group_requested(*values)`, and `_require_components(runner, builder_name, *component_names)`.
- Preserves: direct `CosmoGridRunner(...)` and compatibility method `_get_fnames(icosmo, irlz)`.

- [ ] **Step 1: Replace the obsolete foreground-constructor test with failing builder-contract tests**

Add `inspect`, `Mock`, and `sentinel` imports and replace `ForegroundRunnerRegressionTests` with tests shaped as follows:

```python
import inspect
from unittest.mock import Mock, patch, sentinel


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
```

- [ ] **Step 2: Run the CosmoGrid builder tests and verify the intended failures**

Run:

```bash
env MPLCONFIGDIR=/tmp/cosmogrid_mock_pipe_matplotlib \
/Users/suqikuai777/miniforge3/envs/mock_pipe/bin/python -m unittest \
tests.test_pipeline_regressions.CosmoGridRunnerBuilderTests -v
```

Expected: failures report missing `build_hod_runner`, `build_gal_runner`, `build_void_runner`, and `build_shape_runner`, and the constructor-default assertion fails.

- [ ] **Step 3: Add shared validation and component-guard helpers**

Add these functions above `CosmoGridRunner` in `runner.py`:

```python
def _require_parameters(context, **parameters):
    missing = [name for name, value in parameters.items() if value is None]
    if missing:
        raise ValueError(
            f"{context} requires: {', '.join(missing)}"
        )


def _group_requested(*values):
    return any(value is not None for value in values)


def _require_components(runner, builder_name, *component_names):
    missing = [
        name for name in component_names
        if getattr(runner, name, None) is None
    ]
    if missing:
        raise ValueError(
            f"{type(runner).__name__}.{builder_name}() is required; "
            f"missing components: {', '.join(missing)}"
        )
```

- [ ] **Step 4: Replace the CosmoGrid constructor and `for_foreground()` with task-aware initialization**

Give every parameter this exact defaulted shape and remove `for_foreground()`:

```python
def __init__(
        self, config=None, sim_fmt=None, halo_fmt=None,
        shear_sim_fmt=None, lb_z_file=None,
        fore_mask_fnames_dict=None, fore_nofz_fnames_dict=None,
        fore_survey_labels_dict=None, back_mask_fnames_dict=None,
        back_nofz_fnames_dict=None, back_survey_labels_dict=None,
        back_ngals_dict=None, tomo_labels_dict=None,
        redshift_src_list=None, gal_ofmt=None, void_ofmt=None,
        shear_ofmt=None, runner_type=None):
```

Store every input on `self`, set `redshift_label`, `cata_loader`,
`hod_populator`, `survey_generator`, `void_finder`, and `shear_assigner` to
`None`, then call `_initialize_runner(runner_type)`.

Implement task and direct-constructor selection exactly as follows:

```python
def _initialize_runner(self, runner_type):
    valid_types = {None, "hod", "gal", "void", "shape"}
    if runner_type not in valid_types:
        raise ValueError(f"unsupported runner_type: {runner_type}")

    if runner_type == "hod":
        self._validate_hod_parameters("build_hod_runner")
        self._initialize_hod_components()
        return
    if runner_type == "gal":
        self._validate_hod_parameters("build_gal_runner")
        self._validate_foreground_parameters("build_gal_runner")
        self._initialize_hod_components()
        self._initialize_survey_generator()
        return
    if runner_type == "void":
        _require_parameters(
            "build_void_runner",
            config=self.config,
            sim_fmt=self.sim_fmt,
            lb_z_file=self.lb_z_file,
        )
        self._validate_foreground_parameters("build_void_runner")
        self._initialize_survey_generator()
        self._initialize_void_finder()
        return
    if runner_type == "shape":
        self._validate_shape_parameters("build_shape_runner")
        self._initialize_shear_assigner()
        return

    if self.halo_fmt is not None:
        self._validate_hod_parameters("CosmoGridRunner")
        self._initialize_hod_components()
    foreground_values = (
        self.fore_mask_fnames_dict,
        self.fore_nofz_fnames_dict,
        self.fore_survey_labels_dict,
    )
    if _group_requested(*foreground_values):
        _require_parameters(
            "CosmoGridRunner",
            config=self.config,
            sim_fmt=self.sim_fmt,
            lb_z_file=self.lb_z_file,
        )
        self._validate_foreground_parameters("CosmoGridRunner")
        self._initialize_survey_generator()
        self._initialize_void_finder()
    background_values = (
        self.back_mask_fnames_dict,
        self.back_nofz_fnames_dict,
        self.back_survey_labels_dict,
        self.back_ngals_dict,
        self.tomo_labels_dict,
    )
    if _group_requested(*background_values):
        self._validate_shape_parameters("CosmoGridRunner")
        self._initialize_shear_assigner()
```

Implement the validation methods with these exact groups:

```python
def _validate_hod_parameters(self, context):
    _require_parameters(
        context,
        config=self.config,
        sim_fmt=self.sim_fmt,
        halo_fmt=self.halo_fmt,
        lb_z_file=self.lb_z_file,
    )

def _validate_foreground_parameters(self, context):
    _require_parameters(
        context,
        fore_mask_fnames_dict=self.fore_mask_fnames_dict,
        fore_nofz_fnames_dict=self.fore_nofz_fnames_dict,
        fore_survey_labels_dict=self.fore_survey_labels_dict,
    )
    if list(self.fore_survey_labels_dict) != list(self.fore_nofz_fnames_dict):
        raise ValueError("foreground survey labels and n(z) keys must match")

def _validate_shape_parameters(self, context):
    _require_parameters(
        context,
        config=self.config,
        shear_sim_fmt=self.shear_sim_fmt,
        back_mask_fnames_dict=self.back_mask_fnames_dict,
        back_nofz_fnames_dict=self.back_nofz_fnames_dict,
        back_survey_labels_dict=self.back_survey_labels_dict,
        back_ngals_dict=self.back_ngals_dict,
        tomo_labels_dict=self.tomo_labels_dict,
        redshift_src_list=self.redshift_src_list,
    )
    if list(self.back_ngals_dict) != list(self.back_nofz_fnames_dict):
        raise ValueError("background number-density and n(z) keys must match")
    if list(self.tomo_labels_dict) != list(self.back_nofz_fnames_dict):
        raise ValueError("tomographic labels and n(z) keys must match")
```

The initialization methods must be idempotent by returning immediately when
their component already exists. HOD initialization sets `redshift_label` from
`lb_z_file`; shape initialization prepares only background masks and n(z).

- [ ] **Step 5: Add the four CosmoGrid builders**

Use explicit parameters, all defaulting to `None`, and delegate to the
constructor with the matching `runner_type`:

```python
@classmethod
def build_hod_runner(
        cls, config=None, sim_fmt=None, halo_fmt=None, lb_z_file=None):
    return cls(
        config=config, sim_fmt=sim_fmt, halo_fmt=halo_fmt,
        lb_z_file=lb_z_file, runner_type="hod",
    )

@classmethod
def build_gal_runner(
        cls, config=None, sim_fmt=None, halo_fmt=None, lb_z_file=None,
        fore_mask_fnames_dict=None, fore_nofz_fnames_dict=None,
        fore_survey_labels_dict=None, gal_ofmt=None):
    return cls(
        config=config, sim_fmt=sim_fmt, halo_fmt=halo_fmt,
        lb_z_file=lb_z_file,
        fore_mask_fnames_dict=fore_mask_fnames_dict,
        fore_nofz_fnames_dict=fore_nofz_fnames_dict,
        fore_survey_labels_dict=fore_survey_labels_dict,
        gal_ofmt=gal_ofmt, runner_type="gal",
    )

@classmethod
def build_void_runner(
        cls, config=None, sim_fmt=None, lb_z_file=None,
        fore_mask_fnames_dict=None, fore_nofz_fnames_dict=None,
        fore_survey_labels_dict=None, void_ofmt=None):
    return cls(
        config=config, sim_fmt=sim_fmt, lb_z_file=lb_z_file,
        fore_mask_fnames_dict=fore_mask_fnames_dict,
        fore_nofz_fnames_dict=fore_nofz_fnames_dict,
        fore_survey_labels_dict=fore_survey_labels_dict,
        void_ofmt=void_ofmt, runner_type="void",
    )

@classmethod
def build_shape_runner(
        cls, config=None, shear_sim_fmt=None,
        back_mask_fnames_dict=None, back_nofz_fnames_dict=None,
        back_survey_labels_dict=None, back_ngals_dict=None,
        tomo_labels_dict=None, redshift_src_list=None, shear_ofmt=None):
    return cls(
        config=config, shear_sim_fmt=shear_sim_fmt,
        back_mask_fnames_dict=back_mask_fnames_dict,
        back_nofz_fnames_dict=back_nofz_fnames_dict,
        back_survey_labels_dict=back_survey_labels_dict,
        back_ngals_dict=back_ngals_dict,
        tomo_labels_dict=tomo_labels_dict,
        redshift_src_list=redshift_src_list,
        shear_ofmt=shear_ofmt, runner_type="shape",
    )
```

- [ ] **Step 6: Separate CosmoGrid cosmology and halo paths and guard task methods**

Replace the coupled implementation with:

```python
def _get_cosmo_fname(self, icosmo, irlz):
    return os.path.join(self.sim_fmt.format(icosmo, irlz), "params.yml")

def _get_halo_fname(self, icosmo, irlz):
    return os.path.join(
        self.sim_fmt.format(icosmo, irlz),
        self.halo_fmt.format(self.redshift_label),
    )

def _get_fnames(self, icosmo, irlz):
    return (
        self._get_cosmo_fname(icosmo, irlz),
        self._get_halo_fname(icosmo, irlz),
    )
```

At the first executable line of each task method add:

```python
_require_components(self, "build_hod_runner", "cata_loader", "hod_populator")
_require_components(
    self, "build_gal_runner",
    "cata_loader", "hod_populator", "survey_generator",
)
_require_components(
    self, "build_void_runner", "survey_generator", "void_finder",
)
_require_components(self, "build_shape_runner", "shear_assigner")
```

Use the matching guard in `sample_hod_params`, `gen_mock_gal`,
`gen_mock_void`, and `gen_mock_shear`. Change `gen_mock_void` to obtain
`cpar_fname` from `_get_cosmo_fname(icosmo, irlz)` so it never formats a halo
path.

- [ ] **Step 7: Add failing-then-green coverage for method guards and one-time initialization**

Add these tests to `CosmoGridRunnerBuilderTests` before the implementation in
Steps 3–6, verify they fail, then rerun them after the implementation:

```python
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
```

Run:

```bash
env MPLCONFIGDIR=/tmp/cosmogrid_mock_pipe_matplotlib \
/Users/suqikuai777/miniforge3/envs/mock_pipe/bin/python -m unittest \
tests.test_pipeline_regressions.CosmoGridRunnerBuilderTests \
tests.test_pipeline_regressions.HODSamplingRegressionTests \
tests.test_pipeline_regressions.GalaxyFieldPropagationRegressionTests -v
```

Expected after implementation: all selected tests pass.

- [ ] **Step 8: Commit the CosmoGrid builder implementation**

Run:

```bash
git add runner.py tests/test_pipeline_regressions.py
git commit -m "refactor: add task-specific CosmoGrid runners"
```

---

### Task 2: Add FastPM Task Builders and Conditional Initialization

**Files:**
- Modify: `runner.py:466-1050`
- Test: `tests/test_fastpm_runner.py`

**Interfaces:**
- Consumes: `_require_parameters`, `_group_requested`, and `_require_components` from Task 1.
- Produces: `FastPMRunner.build_hod_runner(...)`, `build_gal_runner(...)`, `build_void_runner(...)`, `build_shape_runner(...)`.
- Preserves: direct `FastPMRunner(...)`, real Rockstar HOD loading, and FastPM NPZ shear validation.

- [ ] **Step 1: Add failing FastPM constructor and builder tests**

Add `inspect`, `Mock`, and `sentinel` imports to `tests/test_fastpm_runner.py`,
then add these tests to `FastPMRunnerCoreTests`:

```python
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
```

- [ ] **Step 2: Run the FastPM builder tests and verify the intended failures**

Run:

```bash
env MPLCONFIGDIR=/tmp/cosmogrid_mock_pipe_matplotlib \
/Users/suqikuai777/miniforge3/envs/mock_pipe/bin/python -m unittest \
tests.test_fastpm_runner.FastPMRunnerCoreTests.test_constructor_and_builder_parameters_default_to_none \
tests.test_fastpm_runner.FastPMRunnerCoreTests.test_each_builder_initializes_exact_component_set \
tests.test_fastpm_runner.FastPMRunnerCoreTests.test_void_builder_requires_no_halo_path \
tests.test_fastpm_runner.FastPMRunnerCoreTests.test_partial_direct_background_group_is_rejected \
tests.test_fastpm_runner.FastPMRunnerCoreTests.test_builder_lists_all_missing_required_parameters \
tests.test_fastpm_runner.FastPMRunnerCoreTests.test_direct_constructor_infers_each_supplied_group -v
```

Expected: failures identify missing builder methods and non-default constructor parameters.

- [ ] **Step 3: Refactor the FastPM constructor into stored state plus exact component initialization**

Use this constructor signature:

```python
def __init__(
        self, config=None, halo_fmt=None, cosmo_par_fname=None,
        fore_mask_fnames_dict=None, fore_nofz_fnames_dict=None,
        fore_survey_labels_dict=None, gal_ofmt=None, void_ofmt=None,
        shear_sim_fmt=None, back_mask_fnames_dict=None,
        back_nofz_fnames_dict=None, back_survey_labels_dict=None,
        back_ngals_dict=None, tomo_labels_dict=None,
        shear_ofmt=None, runner_type=None):
```

Store the inputs, initialize `scale_factor`, `cata_loader`, `hod_populator`,
`survey_generator`, `void_finder`, and `shear_assigner` to `None`, and dispatch
through `_initialize_runner(runner_type)`.

Implement these FastPM validation methods:

```python
def _validate_hod_parameters(self, context):
    _require_parameters(
        context,
        config=self.config,
        halo_fmt=self.halo_fmt,
        cosmo_par_fname=self.cosmo_par_fname,
    )

def _validate_foreground_parameters(self, context):
    _require_parameters(
        context,
        fore_mask_fnames_dict=self.fore_mask_fnames_dict,
        fore_nofz_fnames_dict=self.fore_nofz_fnames_dict,
        fore_survey_labels_dict=self.fore_survey_labels_dict,
    )
    if set(self.fore_survey_labels_dict) != set(self.fore_nofz_fnames_dict):
        raise ValueError("foreground survey labels and n(z) keys must match")

def _validate_shape_parameters(self, context):
    _require_parameters(
        context,
        config=self.config,
        cosmo_par_fname=self.cosmo_par_fname,
        shear_sim_fmt=self.shear_sim_fmt,
        back_mask_fnames_dict=self.back_mask_fnames_dict,
        back_nofz_fnames_dict=self.back_nofz_fnames_dict,
        back_survey_labels_dict=self.back_survey_labels_dict,
        back_ngals_dict=self.back_ngals_dict,
        tomo_labels_dict=self.tomo_labels_dict,
    )
    if set(self.back_survey_labels_dict) != set(self.back_mask_fnames_dict):
        raise ValueError("background survey labels and mask keys must match")
    if set(self.back_ngals_dict) != set(self.back_nofz_fnames_dict):
        raise ValueError("background number-density and n(z) keys must match")
    if set(self.tomo_labels_dict) != set(self.back_nofz_fnames_dict):
        raise ValueError("tomographic labels and n(z) keys must match")
    for tomo_name, tomo_label in self.tomo_labels_dict.items():
        expected_name = f"tomo{tomo_label}"
        if tomo_name != expected_name:
            raise ValueError(
                f"tomographic key {tomo_name} must be {expected_name}"
            )
```

HOD initialization computes `scale_factor` and constructs only
`CatalogLoader` and `HODPopulator`. Galaxy additionally creates
`SurveyGenerator`; void creates only `SurveyGenerator` and `VoidFinder`; shape
creates only `ShearAssigner`. Keep all existing key-matching and tomographic
key/label checks in the corresponding foreground/background validation paths.

Implement exact task dispatch and direct-constructor inference:

```python
def _initialize_runner(self, runner_type):
    valid_types = {None, "hod", "gal", "void", "shape"}
    if runner_type not in valid_types:
        raise ValueError(f"unsupported runner_type: {runner_type}")

    if runner_type == "hod":
        self._validate_hod_parameters("build_hod_runner")
        self._initialize_hod_components()
        return
    if runner_type == "gal":
        self._validate_hod_parameters("build_gal_runner")
        self._validate_foreground_parameters("build_gal_runner")
        self._initialize_hod_components()
        self._initialize_survey_generator()
        return
    if runner_type == "void":
        _require_parameters(
            "build_void_runner",
            config=self.config,
            cosmo_par_fname=self.cosmo_par_fname,
        )
        self._validate_foreground_parameters("build_void_runner")
        self._initialize_survey_generator()
        self._initialize_void_finder()
        return
    if runner_type == "shape":
        self._validate_shape_parameters("build_shape_runner")
        self._initialize_shear_assigner()
        return

    if self.halo_fmt is not None:
        self._validate_hod_parameters("FastPMRunner")
        self._initialize_hod_components()
    if _group_requested(
            self.fore_mask_fnames_dict,
            self.fore_nofz_fnames_dict,
            self.fore_survey_labels_dict):
        _require_parameters(
            "FastPMRunner",
            config=self.config,
            cosmo_par_fname=self.cosmo_par_fname,
        )
        self._validate_foreground_parameters("FastPMRunner")
        self._initialize_survey_generator()
        self._initialize_void_finder()
    if _group_requested(
            self.back_mask_fnames_dict,
            self.back_nofz_fnames_dict,
            self.back_survey_labels_dict,
            self.back_ngals_dict,
            self.tomo_labels_dict):
        self._validate_shape_parameters("FastPMRunner")
        self._initialize_shear_assigner()
```

- [ ] **Step 4: Add the four FastPM builders**

Implement these classmethods:

```python
@classmethod
def build_hod_runner(
        cls, config=None, halo_fmt=None, cosmo_par_fname=None):
    return cls(
        config=config,
        halo_fmt=halo_fmt,
        cosmo_par_fname=cosmo_par_fname,
        runner_type="hod",
    )

@classmethod
def build_gal_runner(
        cls, config=None, halo_fmt=None, cosmo_par_fname=None,
        fore_mask_fnames_dict=None, fore_nofz_fnames_dict=None,
        fore_survey_labels_dict=None, gal_ofmt=None):
    return cls(
        config=config,
        halo_fmt=halo_fmt,
        cosmo_par_fname=cosmo_par_fname,
        fore_mask_fnames_dict=fore_mask_fnames_dict,
        fore_nofz_fnames_dict=fore_nofz_fnames_dict,
        fore_survey_labels_dict=fore_survey_labels_dict,
        gal_ofmt=gal_ofmt,
        runner_type="gal",
    )

@classmethod
def build_void_runner(
        cls, config=None, cosmo_par_fname=None,
        fore_mask_fnames_dict=None, fore_nofz_fnames_dict=None,
        fore_survey_labels_dict=None, void_ofmt=None):
    return cls(
        config=config,
        cosmo_par_fname=cosmo_par_fname,
        fore_mask_fnames_dict=fore_mask_fnames_dict,
        fore_nofz_fnames_dict=fore_nofz_fnames_dict,
        fore_survey_labels_dict=fore_survey_labels_dict,
        void_ofmt=void_ofmt,
        runner_type="void",
    )

@classmethod
def build_shape_runner(
        cls, config=None, cosmo_par_fname=None, shear_sim_fmt=None,
        back_mask_fnames_dict=None, back_nofz_fnames_dict=None,
        back_survey_labels_dict=None, back_ngals_dict=None,
        tomo_labels_dict=None, shear_ofmt=None):
    return cls(
        config=config,
        cosmo_par_fname=cosmo_par_fname,
        shear_sim_fmt=shear_sim_fmt,
        back_mask_fnames_dict=back_mask_fnames_dict,
        back_nofz_fnames_dict=back_nofz_fnames_dict,
        back_survey_labels_dict=back_survey_labels_dict,
        back_ngals_dict=back_ngals_dict,
        tomo_labels_dict=tomo_labels_dict,
        shear_ofmt=shear_ofmt,
        runner_type="shape",
    )
```

Do not pass unrelated foreground, background, or halo parameters from a
builder.

- [ ] **Step 5: Guard all FastPM task methods and test component reuse**

Use `_require_components` at the beginning of the four methods as follows:

```python
# sample_hod_params
_require_components(self, "build_hod_runner", "cata_loader", "hod_populator")

# gen_mock_gal
_require_components(
    self, "build_gal_runner",
    "cata_loader", "hod_populator", "survey_generator",
)

# gen_mock_void
_require_components(
    self, "build_void_runner", "survey_generator", "void_finder",
)

# gen_mock_shear
_require_components(self, "build_shape_runner", "shear_assigner")
```

Add:

```python
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
```

- [ ] **Step 6: Run focused and complete FastPM tests**

Run:

```bash
env MPLCONFIGDIR=/tmp/cosmogrid_mock_pipe_matplotlib \
/Users/suqikuai777/miniforge3/envs/mock_pipe/bin/python -m unittest \
tests.test_fastpm_runner.FastPMRunnerCoreTests -v
```

Expected: every `FastPMRunnerCoreTests` test passes, including direct-constructor
and real temporary-file catalog tests.

- [ ] **Step 7: Commit the FastPM builder implementation**

Run:

```bash
git add runner.py tests/test_fastpm_runner.py
git commit -m "refactor: add task-specific FastPM runners"
```

---

### Task 3: Migrate Driver Scripts to Explicit Builders

**Files:**
- Modify: `run_sampling_hod.py:156-165`
- Modify: `run_mock_gal.py:144-154`
- Modify: `run_mock_void.py:150-160`
- Modify: `run_mock_shape.py:182-199`
- Test: `tests/test_pipeline_regressions.py`

**Interfaces:**
- Consumes: the eight builder classmethods from Tasks 1 and 2.
- Produces: top-level CosmoGrid scripts whose construction call communicates the exact task.

- [ ] **Step 1: Write a failing AST-based driver routing test**

Add `ast` to the imports in `tests/test_pipeline_regressions.py`, then add:

```python
class DriverBuilderRegressionTests(unittest.TestCase):
    def test_each_driver_uses_its_task_specific_builder(self):
        expected = {
            "run_sampling_hod.py": "build_hod_runner",
            "run_mock_gal.py": "build_gal_runner",
            "run_mock_void.py": "build_void_runner",
            "run_mock_shape.py": "build_shape_runner",
        }
        root = Path(__file__).resolve().parents[1]
        for filename, builder_name in expected.items():
            with self.subTest(filename=filename):
                tree = ast.parse((root / filename).read_text())
                called_attributes = {
                    node.func.attr
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                }
                self.assertIn(builder_name, called_attributes)
                self.assertNotIn("for_foreground", called_attributes)
```

- [ ] **Step 2: Run the routing test and verify all four drivers fail**

Run:

```bash
/Users/suqikuai777/miniforge3/envs/mock_pipe/bin/python -m unittest \
tests.test_pipeline_regressions.DriverBuilderRegressionTests -v
```

Expected: each subtest reports that its expected builder is absent.

- [ ] **Step 3: Change each script to its matching builder**

Use these construction calls:

```python
# run_sampling_hod.py
cosmogrid_runner = CosmoGridRunner.build_hod_runner(
    config=cosmogridV1_config,
    sim_fmt=sim_fmt,
    halo_fmt=halo_fmt,
    lb_z_file=lb_z_file,
)

# run_mock_gal.py
cosmogrid_runner = CosmoGridRunner.build_gal_runner(
    config=cosmogridV1_config,
    sim_fmt=sim_fmt,
    halo_fmt=halo_fmt,
    lb_z_file=lb_z_file,
    fore_mask_fnames_dict=mask_fnames_dict,
    fore_nofz_fnames_dict=nofz_fnames_dict,
    fore_survey_labels_dict=survey_labels_dict,
    gal_ofmt=galcone_fmt,
)

# run_mock_void.py
cosmogrid_runner = CosmoGridRunner.build_void_runner(
    config=cosmogridV1_config,
    sim_fmt=sim_fmt,
    lb_z_file=lb_z_file,
    fore_mask_fnames_dict=mask_fnames_dict,
    fore_nofz_fnames_dict=nofz_fnames_dict,
    fore_survey_labels_dict=survey_labels_dict,
    void_ofmt=voidcone_fmt,
)

# run_mock_shape.py
cosmogrid_runner = CosmoGridRunner.build_shape_runner(
    config=cosmogridV1_config,
    shear_sim_fmt=shear_sim_fmt,
    back_mask_fnames_dict=back_mask_fnames_dict,
    back_nofz_fnames_dict=back_nofz_fnames_dict,
    back_survey_labels_dict=back_survey_labels_dict,
    back_ngals_dict=back_ngals_dict,
    tomo_labels_dict=tomo_labels_dict,
    redshift_src_list=redshift_src_list,
    shear_ofmt=shapecone_fmt,
)
```

Do not pass foreground dictionaries to the HOD or shape builders. Do not pass
`halo_fmt` to the void or shape builders.

- [ ] **Step 4: Run routing, HOD JSON, and compilation checks**

Run:

```bash
/Users/suqikuai777/miniforge3/envs/mock_pipe/bin/python -m unittest \
tests.test_pipeline_regressions.DriverBuilderRegressionTests \
tests.test_pipeline_regressions.ScriptHODIORegressionTests -v
/Users/suqikuai777/miniforge3/envs/mock_pipe/bin/python -m py_compile \
run_sampling_hod.py run_mock_gal.py run_mock_void.py run_mock_shape.py
```

Expected: all selected tests pass and all four scripts compile.

- [ ] **Step 5: Commit the driver migration**

Run:

```bash
git add run_sampling_hod.py run_mock_gal.py run_mock_void.py \
run_mock_shape.py tests/test_pipeline_regressions.py
git commit -m "refactor: route scripts through task runners"
```

---

### Task 4: Full Regression and Real-Data Verification

**Files:**
- Verify: `runner.py`
- Verify: `run_sampling_hod.py`
- Verify: `run_mock_gal.py`
- Verify: `run_mock_void.py`
- Verify: `run_mock_shape.py`
- Verify: `tests/test_pipeline_regressions.py`
- Verify: `tests/test_fastpm_runner.py`

**Interfaces:**
- Consumes: completed task-specific builders and migrated scripts.
- Produces: fresh evidence that all repository behavior and the real FastPM shear product remain valid.

- [ ] **Step 1: Run the complete test suite**

Run:

```bash
env MPLCONFIGDIR=/tmp/cosmogrid_mock_pipe_matplotlib \
/Users/suqikuai777/miniforge3/envs/mock_pipe/bin/python -m unittest discover -v
```

Expected: zero failures and zero errors.

- [ ] **Step 2: Compile every changed Python file**

Run:

```bash
/Users/suqikuai777/miniforge3/envs/mock_pipe/bin/python -m py_compile \
runner.py run_sampling_hod.py run_mock_gal.py run_mock_void.py \
run_mock_shape.py tests/test_pipeline_regressions.py tests/test_fastpm_runner.py
```

Expected: exit code 0.

- [ ] **Step 3: Load the real FastPM shear product**

Run:

```bash
env MPLCONFIGDIR=/tmp/cosmogrid_mock_pipe_matplotlib \
/Users/suqikuai777/miniforge3/envs/mock_pipe/bin/python -c '
from runner import FastPMRunner
runner = FastPMRunner.build_shape_runner(
    config=__import__("handler").PipeConfig(
        Lbox=1000.0, Npart=1024, redshift=0.3
    ),
    cosmo_par_fname=(
        "/Users/suqikuai777/Dataspace/FastPM/Cosmology/cosmo_list.txt"
    ),
    shear_sim_fmt=(
        "/Users/suqikuai777/Workspace/fast_shear_map/outputs/"
        "dz_tomography_acceptance_v2/products/cosmo_{:06d}/"
        "realization_{:04d}.npz"
    ),
    back_mask_fnames_dict={},
    back_nofz_fnames_dict={},
    back_survey_labels_dict={},
    back_ngals_dict={},
    tomo_labels_dict={},
)
maps = runner._load_shear_maps(0, 0)
assert len(maps) == 20
assert all(
    shell["gamma1"].shape == shell["gamma2"].shape
    for shell in maps.values()
)
print(f"shells={len(maps)} pixels={maps['"'"'shell0'"'"']['"'"'gamma1'"'"'].size}")
'
```

Expected: `shells=20 pixels=192` and exit code 0.

- [ ] **Step 4: Check API shape and diff hygiene**

Run:

```bash
/Users/suqikuai777/miniforge3/envs/mock_pipe/bin/python -c '
import inspect
from runner import CosmoGridRunner, FastPMRunner
for cls in (CosmoGridRunner, FastPMRunner):
    for method_name in (
        "build_hod_runner", "build_gal_runner",
        "build_void_runner", "build_shape_runner",
    ):
        assert hasattr(cls, method_name)
    for parameter in inspect.signature(cls.__init__).parameters.values():
        if parameter.name != "self":
            assert parameter.default is None
print("runner builder API verified")
'
git diff --check
git status --short
```

Expected: API check prints `runner builder API verified`, diff check is clean,
and status contains only intentional plan or implementation files.

- [ ] **Step 5: Commit any verification-only test adjustments**

If Step 1 exposes a test-only mismatch, update only the affected test to the
approved builder contract, rerun Steps 1–4, then commit the verified adjustment:

```bash
git add tests/test_pipeline_regressions.py tests/test_fastpm_runner.py
git commit -m "test: verify task-specific runner builders"
```

If no test-only adjustment is needed, do not create an empty commit.
