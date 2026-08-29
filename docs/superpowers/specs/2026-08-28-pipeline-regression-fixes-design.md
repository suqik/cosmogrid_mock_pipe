# Pipeline Regression Fixes Design

## Goal

Repair the repository-wide regressions found during the FastPM runner review
without changing the scientific meaning of the configured HOD priors, survey
masks, redshift distributions, or shear products.

## Scope

The repair covers four shared data boundaries and the legacy CosmoGrid driver
scripts:

- HOD sampling must return exactly `PipeConfig.nhod_per_cosmo` parameter
  vectors instead of the final vector from the candidate pool.
- HOD JSON written after MPI gather must be one dictionary, and cosmology keys
  use the single supported spelling `cosmo_000001`.
- Foreground catalog generation must preserve `gal_type` and
  `host_halo_mvir`, and every `apply_nz` mode must have defined behavior.
- Void generation must use `PipeConfig.dive_exec_path`, execute DIVE without a
  shell, detect failures, and clean temporary files.
- `run_sampling_hod.py`, `run_mock_gal.py`, `run_mock_void.py`, and
  `run_mock_shape.py` must use the underscore HOD key format. The foreground
  scripts must call the current `CosmoGridRunner` API and format catalog paths
  with `(icosmo, irlz, ihod)` where applicable.

Hard-coded production data locations remain script configuration and are not
rewritten in this change. The FastPM shear-map format and its validation remain
unchanged.

## Architecture

### HOD sampling and persistence

`HODPopulator.find_hod_params()` will collect candidates until it has exactly
`nhod_per_cosmo` rows. Models 2, 3, and 4 append `fic=1.0` to each accepted
five-parameter candidate. Model 0 retains its existing number-density and
satellite-fraction filter. Exhausting the pool before reaching the configured
count raises an explicit error.

`run_sampling_hod.py` will keep its original local helper style for converting
sample rows, merging MPI parts, and writing JSON. Each `run_mock_*` script will
keep a local `load_hod_samples()` function. These helpers support only the
underscore key format and do not add compatibility or validation layers.

### Foreground catalogs

Both runners will request position, velocity, galaxy type, and host-halo mass
from `HODPopulator`. The last three arrays will cross the box-to-lightcone
boundary as adjacent properties with names matching `fgal_type`.

`SurveyGenerator.box_to_lightcone()` will initialize output rows with zeros and
validate adjacent-property lengths. `apply_nz(..., "const")` keeps all objects
inside the supplied redshift edges; `downsample` and `rank` use integer target
counts; empty selections return an empty catalog of the original dtype.

### Void execution

`VoidFinder` passes the configured executable path to `find_void()`. The helper
uses `subprocess.run([...], check=True)`, reads one-row and multi-row output
uniformly, and removes temporary input/output files in a `finally` block.

### Driver scripts

`CosmoGridRunner.for_foreground()` will provide a narrow constructor for the
three foreground-only scripts and supply empty background/shear configuration.
The scripts will use their local HOD JSON helpers, corrected output keyword
names, and the complete catalog format arguments.

## Error Handling

- Invalid HOD pool size and unsupported HOD sampling models fail before catalog
  work.
- Invalid `nofz_method` values raise `ValueError`.
- DIVE non-zero exits propagate as `subprocess.CalledProcessError`; malformed
  output raises `ValueError` after cleanup.
- Saving through a driver always supplies the appropriate runner output format.

## Verification

Each regression gets a test that first fails against the current code. After
the focused fixes pass, run the complete unittest suite, compile all modified
Python files, run `git diff --check`, and exercise the real FastPM shear NPZ
loader to guard the already-completed feature.
