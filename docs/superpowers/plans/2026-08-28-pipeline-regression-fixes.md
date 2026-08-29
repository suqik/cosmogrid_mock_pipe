# Pipeline Regression Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the HOD, foreground catalog, DIVE, and legacy driver regressions found during the FastPM runner review.

**Architecture:** Establish explicit contracts at the shared boundaries. HOD JSON logic stays local to the original driver scripts, both runners propagate the complete galaxy payload, and foreground-only scripts construct `CosmoGridRunner` through one supported classmethod.

**Tech Stack:** Python 3, NumPy, Astropy, SciPy, mpi4py driver scripts, `unittest`, `subprocess`.

**Spec:** `docs/superpowers/specs/2026-08-28-pipeline-regression-fixes-design.md`

## Global Constraints

- Preserve the committed FastPM shear-map format and validation behavior.
- HOD JSON keys use only `cosmo_000001`.
- Do not rewrite the production data locations embedded in the driver scripts.
- Write and run each regression test before changing its production behavior.
- Do not commit the bug-fix changes unless the user explicitly requests it.

---

### Task 1: HOD Sampling Contract

**Files:**
- Modify: `handler.py:174-232`
- Modify: `runner.py:111-129`
- Create: `tests/test_pipeline_regressions.py`

**Interfaces:**
- Consumes: `PipeConfig.nhod_per_cosmo`, `_open_params_pool(num_pool, seed)`.
- Produces: `HODPopulator.find_hod_params(...) -> np.ndarray` with exactly one row per configured HOD and deterministic realization-aware sampling seeds.

- [ ] **Step 1: Write failing tests for the HOD row count**

```python
def test_find_hod_params_returns_configured_number_of_rows(self):
    populator = HODPopulator(self.config)
    populator._open_params_pool = lambda size, seed: np.arange(20.0).reshape(4, 5)
    result = populator.find_hod_params(self.halo_catalog, num_pool=4)
    self.assertEqual(result.shape, (2, 6))
    np.testing.assert_allclose(result[:, -1], 1.0)

def test_find_hod_params_rejects_pool_smaller_than_requested_count(self):
    with self.assertRaisesRegex(ValueError, "nhod_per_cosmo"):
        self.populator.find_hod_params(self.halo_catalog, num_pool=1)
```

- [ ] **Step 2: Run the focused tests and verify the current single-vector return fails**

Run: `python -m unittest tests.test_pipeline_regressions.HODSamplingRegressionTests -v`

Expected: the shape assertion fails because the current result is a six-element list.

- [ ] **Step 3: Collect accepted candidates and stop at the configured count**

```python
target_count = self.config.nhod_per_cosmo
accepted = []
for candidate in hod_params_pool:
    if self.config.model in {2, 3, 4}:
        accepted.append([*candidate, 1.0])
    elif self.config.model == 0 and candidate_matches:
        accepted.append(candidate.tolist())
    else:
        raise NotImplementedError(...)
    if len(accepted) == target_count:
        break
if len(accepted) != target_count:
    raise RuntimeError(...)
return np.asarray(accepted, dtype=float)
```

- [ ] **Step 4: Make CosmoGrid sampling seeds depend on cosmology and realization**

```python
def _get_sampling_seed_offset(self, icosmo, irlz):
    return icosmo * self.config.nrlzs_per_cosmo + irlz
```

Pass this value to `find_hod_params()`.

- [ ] **Step 5: Run the focused tests until green**

Run: `python -m unittest tests.test_pipeline_regressions.HODSamplingRegressionTests -v`

Expected: all HOD sampling regression tests pass.

### Task 2: HOD JSON and Foreground Runner Contract

**Files:**
- Modify: `runner.py` near `CosmoGridRunner.__init__`
- Modify: `run_sampling_hod.py`
- Modify: `run_mock_gal.py`
- Modify: `run_mock_void.py`
- Modify: `run_mock_shape.py`
- Modify: `tests/test_pipeline_regressions.py`

**Interfaces:**
- Produces in `run_sampling_hod.py`: `get_hod_params_container(params)`, `merge_hod_sample_parts(parts)`, and `save_hod_samples(path, samples)`.
- Produces in each `run_mock_*` script: `load_hod_samples(path)` and underscore-key cosmology-label parsing.
- Produces: `CosmoGridRunner.for_foreground(...) -> CosmoGridRunner`.

- [ ] **Step 1: Write failing tests for the local JSON writer/loader contract**

```python
def test_sampling_output_matches_all_mock_loaders(self):
    samples = merge_hod_sample_parts([
        {"cosmo_000001": get_hod_params_container(params)},
    ])
    save_hod_samples(self.path, samples)
    for module in (run_mock_gal, run_mock_void, run_mock_shape):
        self.assertEqual(module.load_hod_samples(self.path), samples)
```

- [ ] **Step 2: Run the JSON tests and verify the local interfaces are missing**

Run: `python -m unittest tests.test_pipeline_regressions.ScriptHODIORegressionTests -v`

Expected: failures naming the missing local helper functions and underscore-key parsing.

- [ ] **Step 3: Implement the original-style local HOD JSON helpers**

```python
def merge_hod_sample_parts(parts):
    merged = {}
    for part in parts:
        merged.update(part)
    return merged
```

Convert sampled NumPy rows with `.tolist()`, merge MPI rank dictionaries before
saving, and use `cosmo_{icosmo:06d}` everywhere. Do not add legacy-format
compatibility or validation.

- [ ] **Step 4: Write a failing test for the foreground-only runner constructor**

```python
def test_foreground_runner_constructor_needs_no_background_configuration(self):
    runner = CosmoGridRunner.for_foreground(
        config=self.config,
        sim_fmt="sim/{}/{}",
        halo_fmt="halo.{}",
        lb_z_file=self.label_file,
        fore_mask_fnames_dict={"boss_veto": []},
        fore_nofz_fnames_dict={},
        fore_survey_labels_dict={},
    )
    self.assertIsNone(runner.shear_sim_fmt)
```

- [ ] **Step 5: Add the classmethod and update the driver scripts**

```python
@classmethod
def for_foreground(cls, config, sim_fmt, halo_fmt, lb_z_file,
                   fore_mask_fnames_dict, fore_nofz_fnames_dict,
                   fore_survey_labels_dict, gal_ofmt=None, void_ofmt=None):
    return cls(
        config=config, sim_fmt=sim_fmt, halo_fmt=halo_fmt,
        shear_sim_fmt=None, lb_z_file=lb_z_file,
        fore_mask_fnames_dict=fore_mask_fnames_dict,
        fore_nofz_fnames_dict=fore_nofz_fnames_dict,
        fore_survey_labels_dict=fore_survey_labels_dict,
        back_mask_fnames_dict={}, back_nofz_fnames_dict={},
        back_survey_labels_dict={}, back_ngals_dict={},
        tomo_labels_dict={}, redshift_src_list=[],
        gal_ofmt=gal_ofmt, void_ofmt=void_ofmt,
    )
```

Use local HOD helpers in the scripts, merge `comm.gather()` on rank zero, pass
`gal_ofmt`/`void_ofmt`, and read galaxy catalogs with
`galcone_fmt.format(icosmo, irlz, ihod)`.

- [ ] **Step 6: Run JSON, constructor, and script compilation checks**

Run: `python -m unittest tests.test_pipeline_regressions.ScriptHODIORegressionTests tests.test_pipeline_regressions.ForegroundRunnerRegressionTests -v`

Run: `python -m py_compile run_sampling_hod.py run_mock_gal.py run_mock_void.py run_mock_shape.py`

Expected: tests pass and all scripts compile.

### Task 3: Foreground Catalog Fields and n(z)

**Files:**
- Modify: `handler.py:270-410`
- Modify: `utils/mkfore_utils.py:613-655`
- Modify: `runner.py` in both `gen_mock_gal` implementations
- Modify: `tests/test_fastpm_runner.py`
- Modify: `tests/test_pipeline_regressions.py`

**Interfaces:**
- Consumes: HOD array fields `gal_type` and `halo_mvir`.
- Produces: foreground catalogs with initialized `gal_type` and `host_halo_mvir` fields.

- [ ] **Step 1: Write failing tests for `const`, `rank`, empty output, and invalid modes**

```python
def test_apply_nz_const_keeps_catalog_inside_edges(self):
    result = apply_nz(self.catalog, self.nofz, "const")
    np.testing.assert_array_equal(result["GID"], [1, 2])

def test_apply_nz_rank_uses_an_integer_target_count(self):
    result = apply_nz(self.catalog, self.nofz, "rank")
    self.assertEqual(len(result), 1)

def test_apply_nz_empty_selection_preserves_dtype(self):
    result = apply_nz(self.catalog[:0], self.nofz, "const")
    self.assertEqual(result.dtype, self.catalog.dtype)

def test_apply_nz_rejects_unknown_method(self):
    with self.assertRaises(ValueError):
        apply_nz(self.catalog, self.nofz, "unknown")
```

- [ ] **Step 2: Run the n(z) tests and observe the current undefined-mask and slicing failures**

Run: `python -m unittest tests.test_pipeline_regressions.ApplyNzRegressionTests -v`

Expected: `const` raises `UnboundLocalError`, `rank` raises a slice-index error,
and the empty input raises from `np.concatenate`.

- [ ] **Step 3: Implement explicit n(z) branches**

```python
if nofz_method not in {"const", "downsample", "rank"}:
    raise ValueError(...)
if nofz_method == "const":
    return galcone[(z_mock >= zedges[0]) & (z_mock < zedges[-1])].copy()
```

Convert target counts to bounded integers for other modes and return
`galcone[:0].copy()` when no bin contributes rows.

- [ ] **Step 4: Extend the FastPM behavior fake and add a failing field-propagation assertion**

```python
self.assertEqual(result["gal_type"][0], 1)
self.assertEqual(result["host_halo_mvir"][0], 2.5e13)
```

The fake HOD output must mirror the real structured array, including
`gal_type` and `halo_mvir`; the survey fake returns the adjacent properties so
the assertion checks runner behavior rather than the fake itself.

- [ ] **Step 5: Propagate complete galaxy features through both runners**

```python
gal_pos, gal_vel, gal_type, host_mass = self.hod_populator.get_galaxy_features(
    gsample_arr,
    features=["pos", "vel", "gal_type", "host_halo_mvir"],
)
gal_adj_props = {
    "gal_vel": gal_vel,
    "gal_type": gal_type,
    "host_halo_mvir": host_mass,
}
```

Accept `host_halo_mvir` and the legacy `gal_host_halo_mvir` name in
`get_galaxy_features()`. Initialize `fgal_type` with `np.zeros` and reject
adjacent arrays whose length differs from `gal_pos`.

- [ ] **Step 6: Run the foreground and FastPM tests until green**

Run: `python -m unittest tests.test_pipeline_regressions.ApplyNzRegressionTests tests.test_fastpm_runner.FastPMRunnerCoreTests -v`

Expected: all selected tests pass.

### Task 4: Configured and Safe DIVE Execution

**Files:**
- Modify: `handler.py:485-506`
- Modify: `utils/mkfore_utils.py:767-791`
- Modify: `tests/test_pipeline_regressions.py`

**Interfaces:**
- Consumes: `PipeConfig.dive_exec_path`, tracer positions, input/output paths.
- Produces: `(void_pos, void_radius)` arrays or a checked subprocess/format error, always with temporary cleanup.

- [ ] **Step 1: Write a failing end-to-end test with a temporary fake DIVE executable**

```python
def test_find_void_runs_configured_executable_and_cleans_files(self):
    executable = self.make_fake_dive("fake DIVE")
    positions, radii = find_void(
        np.array([[1.0, 2.0, 3.0]]),
        exec_path=executable,
        dive_input=self.input_path,
        dive_output=self.output_path,
    )
    np.testing.assert_allclose(positions, [[4.0, 5.0, 6.0]])
    np.testing.assert_allclose(radii, [7.0])
    self.assertFalse(self.input_path.exists())
    self.assertFalse(self.output_path.exists())
```

- [ ] **Step 2: Run the DIVE tests and verify the shell-built command fails for the spaced executable path**

Run: `python -m unittest tests.test_pipeline_regressions.VoidExecutionRegressionTests -v`

Expected: the current `os.system` command cannot execute the fake path and the
output read fails.

- [ ] **Step 3: Replace shell commands with checked subprocess execution**

```python
command = [str(exec_path), "-i", str(input_path), "-o", str(output_path)]
if boxsize is not None:
    command.extend(["-u", str(boxsize)])
try:
    np.savetxt(input_path, tracer_pos, fmt="%.3f")
    subprocess.run(command, check=True)
    void_info = np.loadtxt(output_path, ndmin=2)
finally:
    input_path.unlink(missing_ok=True)
    output_path.unlink(missing_ok=True)
```

Validate that DIVE output has four columns before splitting it.

- [ ] **Step 4: Pass the configured executable from `VoidFinder`**

```python
find_void(
    galpos_cart,
    exec_path=self.config.dive_exec_path,
    dive_input=dive_input,
    dive_output=dive_output,
)
```

- [ ] **Step 5: Run DIVE regression tests until green**

Run: `python -m unittest tests.test_pipeline_regressions.VoidExecutionRegressionTests -v`

Expected: configured path, one-row parsing, and cleanup tests pass.

### Task 5: Full Verification

**Files:**
- Verify all files modified in Tasks 1-4.

**Interfaces:**
- Consumes: the completed regression fixes.
- Produces: fresh evidence that the repository and real FastPM shear product remain usable.

- [ ] **Step 1: Run the complete test suite**

Run: `MPLCONFIGDIR=/tmp/codex-matplotlib python -m unittest discover -v`

Expected: every test passes with zero failures and zero errors.

- [ ] **Step 2: Compile all modified Python files**

Run: `python -m py_compile handler.py runner.py utils/mkfore_utils.py run_sampling_hod.py run_mock_gal.py run_mock_void.py run_mock_shape.py tests/test_pipeline_regressions.py tests/test_fastpm_runner.py`

Expected: exit code 0.

- [ ] **Step 3: Re-run the real FastPM shear NPZ acceptance check**

Load `/Users/suqikuai777/Workspace/fast_shear_map/outputs/dz_tomography_acceptance_v2/products/cosmo_000000/realization_0000.npz` through `FastPMRunner._load_shear_maps(0, 0)` using the real cosmology table and confirm the shell count and HEALPix array sizes.

Expected: 20 ordered shells with valid paired gamma arrays.

- [ ] **Step 4: Check the final patch**

Run: `git diff --check`

Run: `git status --short`

Expected: no whitespace errors and only the planned bug-fix files are modified.
