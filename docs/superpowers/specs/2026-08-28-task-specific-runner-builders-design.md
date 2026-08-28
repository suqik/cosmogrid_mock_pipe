# Task-Specific Runner Builders Design

## Goal

Refactor `CosmoGridRunner` and `FastPMRunner` so every constructor parameter
defaults to `None`, while task-specific classmethod builders initialize only
the components needed by HOD sampling, galaxy generation, void generation, or
shape-catalog generation.

The refactor must preserve direct construction for existing callers and must
not change catalog formats or scientific calculations.

## Public Construction API

Both runner classes will expose the same four classmethods:

```python
Runner.build_hod_runner(...)
Runner.build_gal_runner(...)
Runner.build_void_runner(...)
Runner.build_shape_runner(...)
```

Every argument of `__init__` and of these builders will have a default value of
`None`. A builder will validate its required arguments before returning a
runner. Output paths remain optional and are required only when a generation
method is called with `save=True`.

The existing `CosmoGridRunner.for_foreground()` classmethod will be removed.

## Component Boundaries

Each builder initializes exactly the following runtime components:

| Builder | Runtime components |
| --- | --- |
| `build_hod_runner()` | `CatalogLoader`, `HODPopulator` |
| `build_gal_runner()` | `CatalogLoader`, `HODPopulator`, `SurveyGenerator` |
| `build_void_runner()` | `SurveyGenerator`, `VoidFinder` |
| `build_shape_runner()` | `ShearAssigner` |

The constructor first stores its configuration and sets all five component
attributes to `None`. It then initializes components according to an internal
`runner_type` value supplied by a builder. Initialization is eager and occurs
once during construction; repeated calls to a generation method reuse the same
component instances.

### CosmoGrid builder inputs

- HOD: `config`, `sim_fmt`, `halo_fmt`, `lb_z_file`.
- Galaxy: the HOD inputs plus `fore_mask_fnames_dict`,
  `fore_nofz_fnames_dict`, and `fore_survey_labels_dict`.
- Void: `config`, `sim_fmt`, `lb_z_file`, and the three foreground
  dictionaries. A halo path is not required.
- Shape: `config`, `shear_sim_fmt`, `back_mask_fnames_dict`,
  `back_nofz_fnames_dict`, `back_survey_labels_dict`, `back_ngals_dict`,
  `tomo_labels_dict`, and `redshift_src_list`.

### FastPM builder inputs

- HOD: `config`, `halo_fmt`, `cosmo_par_fname`.
- Galaxy: the HOD inputs plus the three foreground dictionaries.
- Void: `config`, `cosmo_par_fname`, and the three foreground dictionaries. A
  halo path is not required.
- Shape: `config`, `cosmo_par_fname`, `shear_sim_fmt`, and the five background
  survey/tomography dictionaries.

`gal_ofmt`, `void_ofmt`, and `shear_ofmt` are accepted only by the corresponding
builders and remain optional.

## Direct-Constructor Compatibility

Calling `CosmoGridRunner(...)` or `FastPMRunner(...)` directly remains
supported.

- Passing `runner_type` explicitly uses the same exact component selection as
  a builder.
- Leaving `runner_type=None` enables compatibility inference from the supplied
  parameter groups.
- A complete HOD/path group initializes `CatalogLoader` and `HODPopulator`.
- A complete foreground group initializes `SurveyGenerator` and `VoidFinder`.
- A complete background group initializes `ShearAssigner`.
- Supplying both foreground and background groups initializes both sides.
- Supplying no groups creates an inert runner whose component attributes remain
  `None`.

Compatibility inference can initialize a broader foreground component set than
a task builder because a direct constructor does not declare whether the caller
will generate galaxies or voids. The builders are the preferred API when exact
component selection matters.

## Initialization Helpers

Each runner will use small private helpers so builder and direct-constructor
paths share implementation:

```python
_initialize_hod_components()
_initialize_survey_generator()
_initialize_void_finder()
_initialize_shear_assigner()
```

Validation will be centralized in a helper that receives a task name and a
mapping of required parameters. Missing values raise `ValueError` listing the
missing parameter names.

Foreground configuration consists of three dictionaries and background
configuration consists of five dictionaries. If direct construction supplies
any member of a group, the whole relevant group must be supplied. An empty
dictionary is a supplied value; `None` means the group was not supplied.

## Path Separation

`CosmoGridRunner._get_fnames()` currently couples cosmology-parameter and halo
paths. It will be split into:

```python
_get_cosmo_fname(icosmo, irlz)
_get_halo_fname(icosmo, irlz)
```

`_get_fnames()` may remain as a compatibility wrapper returning both values.
Void generation will request only the cosmology filename, allowing
`build_void_runner()` to omit `halo_fmt`.

## Task-Method Guards

Task methods will check their required initialized components at entry:

- `sample_hod_params()` requires `cata_loader` and `hod_populator`.
- `gen_mock_gal()` requires `cata_loader`, `hod_populator`, and
  `survey_generator`.
- `gen_mock_void()` requires `survey_generator` and `void_finder`.
- `gen_mock_shear()` requires `shear_assigner` and the background mappings.

Calling a task on an incompatible runner raises `ValueError` naming the builder
that should have been used. This replaces accidental `AttributeError` and
`NoneType` failures.

Existing save-path guards remain unchanged.

## Driver Migration

The top-level scripts will use the explicit builders:

- `run_sampling_hod.py` uses `build_hod_runner()`.
- `run_mock_gal.py` uses `build_gal_runner()`.
- `run_mock_void.py` uses `build_void_runner()`.
- `run_mock_shape.py` uses `build_shape_runner()`.

The script configuration values and generated catalog formats remain
unchanged.

## Testing

Regression tests will cover both runner classes:

1. Every constructor parameter has a `None` default.
2. Each of the four builders initializes exactly its documented components.
3. Repeated generation calls retain the same component instances.
4. A builder reports all missing required parameters clearly.
5. Partial foreground and background groups are rejected.
6. Direct construction works with HOD-only, foreground-only, background-only,
   and combined parameter groups.
7. Calling a task method on an incompatible runner names the correct builder.
8. Each top-level driver script uses its corresponding builder.

After focused tests pass, verification will run the complete unittest suite,
compile all changed Python files, run `git diff --check`, and load the real
FastPM shear NPZ product to ensure the existing shear workflow is unchanged.
