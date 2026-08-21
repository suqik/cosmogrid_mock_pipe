# FastPMRunner Design

Date: 2026-08-21
Status: Approved in chat; awaiting written-spec review

## Context

The repository currently provides `CosmoGridRunner` for PKD halo catalogs and
the complete galaxy, void, and shear workflows. FastPM simulations use
Rockstar ASCII halo catalogs and a different cosmology parameter source and
snapshot path convention.

`FastPMRunner` will be added alongside `CosmoGridRunner` in `runner.py`.
It will not inherit from or modify `CosmoGridRunner`. It will reuse the
existing handlers in `handler.py`, especially
`CatalogLoader.load_rstar_halocat`, `HODPopulator`, `SurveyGenerator`, and
`VoidFinder`.

## Scope

The implementation will support:

- loading FastPM cosmologies from `cosmo_list.txt`;
- resolving a Rockstar catalog from cosmology label and scale factor;
- sampling HOD parameters;
- generating foreground galaxy catalogs;
- generating void catalogs from foreground galaxy catalogs;
- optional galaxy and void catalog output through the existing Astropy Table
  conventions.

The following are outside this change:

- shear and background shape catalogs;
- changes to `CosmoGridRunner` behavior or its public API;
- automatic compilation or execution of Rockstar `find_parents` from inside
  `FastPMRunner`;
- a complete DIVE and observational-mask end-to-end integration test.

## Rockstar Parent Preprocessing

The supplied raw FastPM Rockstar file has no `PID` column. The official
Rockstar source will be downloaded from
`https://bitbucket.org/gfcstanford/rockstar/src/main/` into a temporary build
directory and compiled with:

```bash
make parents
```

The example catalog will then be preprocessed outside the Runner:

```bash
find_parents <halo_fname> 1000 > out_0_wPID.list
```

The generated file will be stored beside the raw catalog. Preprocessing is a
separate, one-time operation so repeated or concurrent pipeline runs do not
rebuild Rockstar or overwrite the same file. `FastPMRunner` only consumes
`out_0_wPID.list` and raises a clear error when it is missing or has no `PID`
column.

The preprocessing verification will compare raw and processed data-row
counts, confirm that the processed header contains `PID`, and report host and
subhalo counts.

## Class Location and Construction

`FastPMRunner` will be defined in `runner.py` after `CosmoGridRunner` as an
independent class. Its constructor will be:

```python
FastPMRunner(
    config,
    halo_fmt,
    cosmo_par_fname,
    fore_mask_fnames_dict,
    fore_nofz_fnames_dict,
    fore_survey_labels_dict,
    gal_ofmt=None,
    void_ofmt=None,
)
```

`halo_fmt` is a full format string with two positional fields, for example:

```python
(
    "/Users/suqikuai777/Dataspace/FastPM/Cosmology/"
    "L1000_N1024_1000cosmo/cosmo{:d}/"
    "a_{:5.4f}/rstar/out_0_wPID.list"
)
```

The constructor computes and stores:

```python
scale_factor = 1.0 / (1.0 + config.redshift)
```

It prepares foreground masks and n(z) data and creates only these handlers:

- `CatalogLoader`;
- `HODPopulator`;
- `SurveyGenerator`;
- `VoidFinder`.

It does not accept or initialize a lightcone-label table, shear maps,
background masks, background n(z), tomography labels, or `ShearAssigner`.

## Cosmology Parsing

The cosmology source has this structure:

```text
# hubble=0.6727 Omegab=0.0491 ns=0.9667
# OmegaM S8
0.200614 0.842526
...
```

The first line supplies fixed parameters. The second line supplies the names
of the varying columns. Data rows are indexed from zero, so `icosmo=0`
selects the first data row and corresponds to the `cosmo0` directory.

`FastPMRunner._get_cosmo_instance(icosmo, otype="ccl")` derives:

```python
Omega_c = OmegaM - Omegab
sigma8 = S8 / np.sqrt(OmegaM / 0.3)
```

It constructs:

```python
ccl.Cosmology(
    h=hubble,
    Omega_b=Omegab,
    Omega_c=Omega_c,
    sigma8=sigma8,
    n_s=ns,
    w0=-1.0,
    wa=0.0,
    m_nu=0.0,
)
```

For `otype="dict"`, it returns a dictionary containing the fixed parameters,
the selected `OmegaM` and `S8`, and the derived `Omega_c` and `sigma8`.

The parser rejects unsupported output types, out-of-range cosmology labels,
missing required parameter names, and `OmegaM <= Omegab`.

## Halo Path and Loading

The input halo filename is resolved as:

```python
halo_fname = halo_fmt.format(icosmo, scale_factor)
```

There is no lightcone-label table and no redshift-label lookup. The retained
`irlz` argument does not participate in the input halo path.

The HOD catalog is loaded with:

```python
CatalogLoader.load_rstar_halocat(
    halo_fname,
    cosmo,
    ofmt="hod",
    host_only=True,
)
```

`find_parents` supplies `PID`; host-only loading keeps `PID == -1` halos so
the HOD population step creates satellite galaxies rather than repopulating
Rockstar subhalos.

## Public Workflows

### HOD parameter sampling

```python
sample_hod_params(icosmo, irlz=0)
```

This loads the FastPM cosmology and host halo catalog and delegates parameter
selection to `HODPopulator.find_hod_params`. The sampling seed offset is:

```python
icosmo * config.nrlzs_per_cosmo + irlz
```

### Galaxy catalog generation

```python
gen_mock_gal(icosmo, irlz, ihod, ihod_param, save=False)
```

This performs the same post-loading workflow as `CosmoGridRunner`:

1. load the FastPM cosmology and Rockstar HOD catalog;
2. populate galaxies with `HODPopulator`;
3. convert populated galaxies to position and velocity arrays;
4. transform the box to a lightcone;
5. apply each configured foreground survey geometry and n(z);
6. concatenate and optionally write the output.

The galaxy population seed offset is:

```python
(
    icosmo * config.nrlzs_per_cosmo * config.nhod_per_cosmo
    + irlz * config.nhod_per_cosmo
    + ihod
)
```

### Void catalog generation

```python
gen_mock_void(
    icosmo,
    irlz,
    ihod,
    galcone_survey,
    dive_input,
    dive_output,
    save=False,
)
```

This loads the selected FastPM cosmology, runs `VoidFinder` independently for
each non-empty foreground survey, applies the configured lightcone redshift
range and survey geometry, concatenates results, and optionally writes the
output. `irlz` is retained for output-name and call compatibility but does not
affect the FastPM input path.

## Foreground Survey Behavior

Foreground masks, foreground n(z), and survey-function routing will preserve
the existing `CosmoGridRunner` behavior. The FastPM class will contain its own
methods rather than modifying or inheriting from `CosmoGridRunner`. Supported
routes remain the existing BOSS LOWZ/E2/E3/CMASS NGC/SGC and 2dFLenS South
routes.

## Error Handling

The implementation will fail early and clearly for:

- an out-of-range `icosmo`;
- malformed or incomplete cosmology headers;
- `OmegaM <= Omegab`;
- an unsupported cosmology output type;
- a missing `out_0_wPID.list`;
- a processed catalog without `PID`;
- an unsupported foreground survey name;
- `save=True` without a corresponding output format.

## Testing and Acceptance Criteria

Automated tests will cover:

- parsing fixed and varying cosmology parameters;
- zero-based cosmology row selection;
- the standard `S8` to `sigma8` conversion;
- rejection of invalid cosmology labels and parameters;
- exact `cosmo{:d}/a_{:5.4f}` halo path formatting;
- missing processed-catalog handling;
- loading a processed Rockstar catalog as a host-only
  `UserSuppliedHaloCatalog`;
- HOD and galaxy seed-offset behavior;
- public method presence and output contracts for HOD sampling, galaxy, and
  void workflows.

Real-data verification will use:

- cosmology file:
  `/Users/suqikuai777/Dataspace/FastPM/Cosmology/cosmo_list.txt`;
- raw catalog:
  `/Users/suqikuai777/Dataspace/FastPM/Cosmology/L1000_N1024_1000cosmo/cosmo0/a_0.7692/rstar/out_0.list`;
- processed catalog in the same directory:
  `out_0_wPID.list`.

Acceptance requires successful parsing of `cosmo0`, resolution of
`a_0.7692`, successful host-only HOD conversion from the processed catalog,
and passing automated tests. A full DIVE execution and complete observational
mask run are explicitly not required for acceptance.

## Planned Files

- Modify `runner.py` to add `FastPMRunner` without changing
  `CosmoGridRunner`.
- Add or extend tests under `tests/` for FastPM cosmology, path, catalog, and
  workflow behavior.
- Generate `out_0_wPID.list` beside the supplied real catalog as external
  test/preprocessing data; it is not a repository file.

