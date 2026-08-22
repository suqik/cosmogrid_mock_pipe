# FastPMRunner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an independent `FastPMRunner` to `runner.py` that reads FastPM cosmologies and parent-processed Rockstar catalogs and supports HOD sampling, foreground galaxy catalogs, and void catalogs without changing `CosmoGridRunner`.

**Architecture:** `FastPMRunner` is a sibling of `CosmoGridRunner`, not a subclass. It owns FastPM-specific cosmology and halo-path resolution while reusing `CatalogLoader.load_rstar_halocat`, `HODPopulator`, `SurveyGenerator`, and `VoidFinder`; Rockstar parent finding remains an external preprocessing step.

**Tech Stack:** Python 3.12, NumPy, pyccl, Halotools, Astropy, pymangle, unittest, Rockstar `find_parents`

**Spec:** `docs/superpowers/specs/2026-08-21-fastpm-runner-design.md`

## Global Constraints

- Define `FastPMRunner` in `runner.py` without modifying `CosmoGridRunner` behavior or public API.
- Do not add shear/background-catalog arguments, handlers, or methods to `FastPMRunner`.
- Resolve halos with a full format string containing `cosmo{:d}` and `a_{:5.4f}`, where `a = 1 / (1 + config.redshift)`.
- Index `cosmo_list.txt` data rows from zero, so row zero maps to `cosmo0`.
- Convert `S8` using `sigma8 = S8 / sqrt(OmegaM / 0.3)` and use `Omega_c = OmegaM - Omegab`.
- Use `w0=-1.0`, `wa=0.0`, and `m_nu=0.0` for FastPM CCL cosmologies.
- Read only parent-processed `out_0_wPID.list`; never invoke `find_parents` from `FastPMRunner`.
- Load the HOD catalog with `host_only=True`; do not apply an `rHalf` cut.
- Retain `irlz` for seed and output-name compatibility, but never include it in the FastPM input halo path.
- Preserve existing, unrelated working-tree changes in `handler.py`, `utils/io_func.py`, and `tests/`.
- Run repository tests with `/Users/suqikuai777/miniforge3/envs/mock_pipe/bin/python` and set `MPLCONFIGDIR=/tmp/cosmogrid_mock_pipe_mpl`.

## File Structure

- Modify: `runner.py` — append independent `FastPMRunner`; do not edit `CosmoGridRunner`.
- Create: `tests/test_fastpm_runner.py` — cosmology, path, HOD, sampling, galaxy, and void behavior.
- External generated data: `/Users/suqikuai777/Dataspace/FastPM/Cosmology/L1000_N1024_1000cosmo/cosmo0/a_0.7692/rstar/out_0_wPID.list` — parent-processed real catalog, not committed.

---

### Task 1: Build Rockstar Parent Finder and Prepare the Real Catalog

**Files:**
- Read: `/Users/suqikuai777/Dataspace/FastPM/Cosmology/L1000_N1024_1000cosmo/cosmo0/a_0.7692/rstar/out_0.list`
- Create externally: `/Users/suqikuai777/Dataspace/FastPM/Cosmology/L1000_N1024_1000cosmo/cosmo0/a_0.7692/rstar/out_0_wPID.list`
- Repository files: none

**Interfaces:**
- Consumes: Rockstar raw ASCII catalog and box size `1000` Mpc/h.
- Produces: Same-row-count Rockstar ASCII catalog whose first header line ends with `PID`, suitable for `get_rstar_halo_attrs(..., host_only=True)`.

- [ ] **Step 1: Confirm the output does not already exist**

Run:

```bash
test ! -e /Users/suqikuai777/Dataspace/FastPM/Cosmology/L1000_N1024_1000cosmo/cosmo0/a_0.7692/rstar/out_0_wPID.list
```

Expected: exit 0. If the file exists, do not overwrite it; validate it with Step 5 and reuse it only if validation passes.

- [ ] **Step 2: Download the official Rockstar source into a temporary directory**

Run with network approval:

```bash
set -euo pipefail
rockstar_source_dir=/private/tmp/cosmogrid-rockstar-parents-20260821
test ! -e "$rockstar_source_dir"
git clone https://bitbucket.org/gfcstanford/rockstar.git "$rockstar_source_dir"
test -f "$rockstar_source_dir/Makefile"
```

Expected: all commands exit 0 and `$rockstar_source_dir/Makefile` exists.

- [ ] **Step 3: Compile `util/find_parents`**

Run:

```bash
set -euo pipefail
rockstar_source_dir=/private/tmp/cosmogrid-rockstar-parents-20260821
make -C "$rockstar_source_dir" parents
test -x "$rockstar_source_dir/util/find_parents"
```

Expected: both commands exit 0.

- [ ] **Step 4: Generate the processed catalog without partial-file exposure**

Run with approval to write beside the supplied catalog:

```bash
set -euo pipefail
raw_halo=/Users/suqikuai777/Dataspace/FastPM/Cosmology/L1000_N1024_1000cosmo/cosmo0/a_0.7692/rstar/out_0.list
processed_halo=/Users/suqikuai777/Dataspace/FastPM/Cosmology/L1000_N1024_1000cosmo/cosmo0/a_0.7692/rstar/out_0_wPID.list
rockstar_source_dir=/private/tmp/cosmogrid-rockstar-parents-20260821
temporary_processed="$(mktemp "${processed_halo}.tmp.XXXXXX")"
"$rockstar_source_dir/util/find_parents" "$raw_halo" 1000 > "$temporary_processed"
test ! -e "$processed_halo"
mv "$temporary_processed" "$processed_halo"
```

Expected: `util/find_parents` and both safety checks exit 0; the final move is atomic on completion.

- [ ] **Step 5: Validate output schema and row counts**

Run:

```bash
set -euo pipefail
raw_halo=/Users/suqikuai777/Dataspace/FastPM/Cosmology/L1000_N1024_1000cosmo/cosmo0/a_0.7692/rstar/out_0.list
processed_halo=/Users/suqikuai777/Dataspace/FastPM/Cosmology/L1000_N1024_1000cosmo/cosmo0/a_0.7692/rstar/out_0_wPID.list
head -1 "$processed_halo"
raw_rows="$(rg -v '^#' "$raw_halo" | wc -l | tr -d ' ')"
processed_rows="$(rg -v '^#' "$processed_halo" | wc -l | tr -d ' ')"
test "$raw_rows" = "$processed_rows"
```

Expected: first line contains `PID`; data-row counts are identical.

Run the semantic check:

```bash
env MPLCONFIGDIR=/tmp/cosmogrid_mock_pipe_mpl \
/Users/suqikuai777/miniforge3/envs/mock_pipe/bin/python -c '
from utils.io_func import get_rstar_halo_attrs
path = "/Users/suqikuai777/Dataspace/FastPM/Cosmology/L1000_N1024_1000cosmo/cosmo0/a_0.7692/rstar/out_0_wPID.list"
data = get_rstar_halo_attrs(path, attrs=["ID", "PID"], host_only=False)
nhost = int((data["PID"] == -1).sum())
nsub = len(data["PID"]) - nhost
assert nhost > 0
assert nsub > 0
print(
    "rows={} hosts={} subhalos={}".format(
        len(data["PID"]), nhost, nsub
    )
)
'
```

Expected: exit 0 and positive host and subhalo counts.

- [ ] **Step 6: Record the external-artifact checkpoint**

No repository commit is made because the generated 1+ GB catalog and temporary Rockstar source are external artifacts. Record the exact output path and validation counts in the task handoff.

---

### Task 2: Add FastPM Cosmology Parsing and Halo Path Resolution

**Files:**
- Modify: `runner.py:405` (append `FastPMRunner` after `CosmoGridRunner`)
- Create: `tests/test_fastpm_runner.py`

**Interfaces:**
- Consumes: `PipeConfig`, `halo_fmt: str`, `cosmo_par_fname: str`, foreground file dictionaries, foreground survey labels, optional galaxy/void output formats.
- Produces: `FastPMRunner`, `_get_cosmo_instance(icosmo, otype)`, `_get_halo_fname(icosmo)`, `_get_sampling_seed_offset(icosmo, irlz)`, and `_get_hod_seed_offset(icosmo, irlz, ihod)`.

- [ ] **Step 1: Write failing cosmology and path tests**

Create `tests/test_fastpm_runner.py` with a temporary cosmology table and a minimal no-survey Runner constructor:

```python
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
```

With `nrlzs_per_cosmo=1` and `nhod_per_cosmo=10`, the literal seed expectations are independently derived: sampling `2*1+3=5` and galaxy `2*1*10+3*10+4=54`.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
env MPLCONFIGDIR=/tmp/cosmogrid_mock_pipe_mpl \
/Users/suqikuai777/miniforge3/envs/mock_pipe/bin/python -m unittest \
tests.test_fastpm_runner.FastPMRunnerCoreTests -v
```

Expected: assertion failure stating that `runner.FastPMRunner` must be implemented.

- [ ] **Step 3: Add the minimal constructor and foreground setup**

Append an independent class to `runner.py`:

```python
class FastPMRunner:
    def __init__(
            self, config: PipeConfig,
            halo_fmt: str,
            cosmo_par_fname: str,
            fore_mask_fnames_dict: dict,
            fore_nofz_fnames_dict: dict,
            fore_survey_labels_dict: dict,
            gal_ofmt: str = None,
            void_ofmt: str = None):
        self.config = config
        self.halo_fmt = halo_fmt
        self.cosmo_par_fname = str(cosmo_par_fname)
        self.fore_survey_labels_dict = fore_survey_labels_dict
        self.gal_ofmt = gal_ofmt
        self.void_ofmt = void_ofmt
        self.scale_factor = 1.0 / (1.0 + config.redshift)

        if set(fore_survey_labels_dict) != set(fore_nofz_fnames_dict):
            raise ValueError("foreground survey labels and n(z) keys must match")

        fore_masks = self._prepare_fore_masks(fore_mask_fnames_dict)
        fore_nofzs = self._prepare_fore_nofzs(fore_nofz_fnames_dict)
        self.cata_loader = CatalogLoader(config=config)
        self.hod_populator = HODPopulator(config=config)
        self.survey_generator = SurveyGenerator(
            config=config, masks=fore_masks, nofzs=fore_nofzs
        )
        self.void_finder = VoidFinder(config=config)
```

Add the foreground setup methods inside `FastPMRunner`; do not edit the original `CosmoGridRunner` methods:

```python
    def _prepare_fore_masks(self, mask_fnames):
        if "boss_veto" not in mask_fnames:
            raise ValueError("fore_mask_fnames_dict must contain boss_veto")

        survey_names = [name for name in mask_fnames if name != "boss_veto"]
        have_boss = any("boss" in name for name in survey_names)
        have_2dflens = any("2dflens" in name for name in survey_names)
        masks = {}

        if have_boss:
            if not mask_fnames["boss_veto"]:
                raise ValueError("Must provide BOSS veto mask files")
            if any("lowze2_ngc" in name or "lowze3_ngc" in name
                   for name in survey_names):
                if "boss_lowz_ngc" not in survey_names:
                    raise ValueError("Must provide LOWZ NGC geometry")
            if any("lowze2_sgc" in name or "lowze3_sgc" in name
                   for name in survey_names):
                if "boss_lowz_sgc" not in survey_names:
                    raise ValueError("Must provide LOWZ SGC geometry")
            masks["boss_geom"] = {}
            masks["boss_masks"] = [
                pymangle.Mangle(path) for path in mask_fnames["boss_veto"]
            ]

        if have_2dflens:
            masks["2dflens_geom"] = {}

        for survey_name in survey_names:
            if "boss" in survey_name:
                masks["boss_geom"][survey_name] = pymangle.Mangle(
                    mask_fnames[survey_name]
                )
            elif "2dflens" in survey_name:
                masks["2dflens_geom"][survey_name] = loadFitsMaps(
                    mask_fnames[survey_name]
                )
            else:
                raise ValueError(
                    f"Unsupported foreground mask: {survey_name}"
                )
        return masks

    def _prepare_fore_nofzs(self, nofz_fnames):
        nofz_info = {}
        for survey_name, nofz_fname in nofz_fnames.items():
            if "boss" in survey_name:
                nofz = np.loadtxt(nofz_fname, usecols=(1, 2, 3, 5))
            elif "2dflens" in survey_name:
                nofz = np.loadtxt(nofz_fname, usecols=(1, 2, 3, 4))
            else:
                raise ValueError(
                    f"Unsupported foreground n(z): {survey_name}"
                )
            nofz_info = make_nofz_info(
                nofz_info,
                survey_name,
                np.append(nofz[:, 0], nofz[-1, 1]),
                nofz[:, 3],
                nofz[:, 2],
            )

        if "boss_cmass" in nofz_info:
            nofz_info["boss_cmass"]["nz_ref"] *= 0.93
        return nofz_info
```

The empty test configuration `{"boss_veto": []}` plus empty n(z)/labels returns empty resources without error.

- [ ] **Step 4: Implement strict cosmology parsing**

Add:

```python
    def _get_cosmo_instance(self, icosmo: int, otype="ccl"):
        with open(self.cosmo_par_fname, "r") as stream:
            fixed_line = stream.readline().strip()
            varying_line = stream.readline().strip()

        if not fixed_line.startswith("#") or not varying_line.startswith("#"):
            raise ValueError("FastPM cosmology file must start with two headers")

        fixed = {}
        for item in fixed_line[1:].split():
            if "=" not in item:
                raise ValueError("malformed fixed cosmology header")
            name, value = item.split("=", 1)
            fixed[name] = float(value)

        varying_names = varying_line[1:].split()
        required_fixed = {"hubble", "Omegab", "ns"}
        required_varying = {"OmegaM", "S8"}
        if not required_fixed.issubset(fixed):
            raise ValueError("missing fixed FastPM cosmology parameters")
        if not required_varying.issubset(varying_names):
            raise ValueError("missing varying FastPM cosmology parameters")

        rows = np.loadtxt(self.cosmo_par_fname, comments="#", ndmin=2)
        if rows.shape[1] != len(varying_names):
            raise ValueError("cosmology header and data column counts differ")
        if icosmo < 0 or icosmo >= len(rows):
            raise IndexError(f"cosmology label {icosmo} is out of range")

        selected = dict(zip(varying_names, rows[icosmo]))
        OmegaM = float(selected["OmegaM"])
        if OmegaM <= fixed["Omegab"]:
            raise ValueError("OmegaM must be larger than Omegab")

        params = {
            **fixed,
            **{name: float(value) for name, value in selected.items()},
            "Omega_c": OmegaM - fixed["Omegab"],
            "sigma8": float(selected["S8"]) / np.sqrt(OmegaM / 0.3),
        }
        if otype == "dict":
            return params
        if otype == "ccl":
            return ccl.Cosmology(
                h=params["hubble"],
                Omega_b=params["Omegab"],
                Omega_c=params["Omega_c"],
                sigma8=params["sigma8"],
                n_s=params["ns"],
                w0=-1.0,
                wa=0.0,
                m_nu=0.0,
            )
        raise NotImplementedError(f"Output type {otype} not implemented")
```

- [ ] **Step 5: Implement validated path and seed helpers**

Add:

```python
    def _get_halo_fname(self, icosmo: int) -> str:
        halo_fname = self.halo_fmt.format(icosmo, self.scale_factor)
        if not os.path.isfile(halo_fname):
            raise FileNotFoundError(
                f"Parent-processed Rockstar catalog not found: {halo_fname}"
            )
        with open(halo_fname, "r") as stream:
            columns = stream.readline().lstrip("#").split()
        if "PID" not in columns:
            raise ValueError(f"Rockstar catalog has no PID column: {halo_fname}")
        return halo_fname

    def _get_sampling_seed_offset(self, icosmo, irlz):
        return icosmo * self.config.nrlzs_per_cosmo + irlz

    def _get_hod_seed_offset(self, icosmo, irlz, ihod):
        return (
            icosmo
            * self.config.nrlzs_per_cosmo
            * self.config.nhod_per_cosmo
            + irlz * self.config.nhod_per_cosmo
            + ihod
        )
```

- [ ] **Step 6: Run focused and existing tests to verify GREEN**

Run:

```bash
env MPLCONFIGDIR=/tmp/cosmogrid_mock_pipe_mpl \
/Users/suqikuai777/miniforge3/envs/mock_pipe/bin/python -m unittest \
tests.test_fastpm_runner.FastPMRunnerCoreTests -v
```

Expected: all `FastPMRunnerCoreTests` pass.

Then run:

```bash
env MPLCONFIGDIR=/tmp/cosmogrid_mock_pipe_mpl \
/Users/suqikuai777/miniforge3/envs/mock_pipe/bin/python -m unittest discover -v
```

Expected: existing and new tests pass with zero failures/errors.

- [ ] **Step 7: Commit core FastPM support**

```bash
git add runner.py tests/test_fastpm_runner.py
git commit -m "feat: add FastPM cosmology and halo resolution"
```

---

### Task 3: Add Rockstar HOD Loading and HOD Parameter Sampling

**Files:**
- Modify: `runner.py` (`FastPMRunner` only)
- Modify: `tests/test_fastpm_runner.py`

**Interfaces:**
- Consumes: `_get_cosmo_instance(icosmo)`, `_get_halo_fname(icosmo)`, `CatalogLoader.load_rstar_halocat(..., host_only=True)`, and `_get_sampling_seed_offset`.
- Produces: `_load_hod_halocat(icosmo) -> tuple[ccl.Cosmology, UserSuppliedHaloCatalog]` and `sample_hod_params(icosmo, irlz=0)`.

- [ ] **Step 1: Add a complete synthetic Rockstar fixture and failing HOD test**

The test module already imports the full Rockstar header (including final `PID`) and defines the two exact rows in Task 2. Add the fixture writer and behavior test below to `FastPMRunnerCoreTests`. The host has `PID=-1`, the subhalo has `PID=10`, and the host has `Halfmass_Radius=50` kpc/h so the test also proves there is no `rHalf > 0.1` cut.

```python
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
```

- [ ] **Step 2: Add a failing sampling test using a behavior fake**

Define a fake whose result exposes both the halo count and seed offset rather than asserting mock call counts. Add the test method to `FastPMRunnerCoreTests`:

```python
class SamplingHODPopulator:
    def find_hod_params(self, halo_catalog, seed_offset):
        return {
            "halo_count": len(halo_catalog.halo_table),
            "seed_offset": seed_offset,
        }
```

Add this test method to `FastPMRunnerCoreTests`:

```python
    def test_sample_hod_params_uses_realization_seed_and_host_catalog(self):
        self.assertTrue(
            hasattr(self.runner, "sample_hod_params"),
            "FastPMRunner.sample_hod_params must be implemented",
        )
        self.write_rstar_catalog()
        self.runner.hod_populator = SamplingHODPopulator()
        result = self.runner.sample_hod_params(0, irlz=3)
        self.assertEqual(result, {"halo_count": 1, "seed_offset": 3})
```

- [ ] **Step 3: Run the new tests to verify RED**

Run:

```bash
env MPLCONFIGDIR=/tmp/cosmogrid_mock_pipe_mpl \
/Users/suqikuai777/miniforge3/envs/mock_pipe/bin/python -m unittest \
tests.test_fastpm_runner.FastPMRunnerCoreTests.test_load_hod_halocat_uses_only_hosts_without_rhalf_cut \
tests.test_fastpm_runner.FastPMRunnerCoreTests.test_sample_hod_params_uses_realization_seed_and_host_catalog -v
```

Expected: assertion failures naming missing `_load_hod_halocat` and `sample_hod_params`.

- [ ] **Step 4: Implement HOD loading and sampling**

Add to `FastPMRunner`:

```python
    def _load_hod_halocat(self, icosmo):
        cosmo = self._get_cosmo_instance(icosmo, otype="ccl")
        halo_fname = self._get_halo_fname(icosmo)
        halo_catalog = self.cata_loader.load_rstar_halocat(
            halo_fname,
            cosmo=cosmo,
            ofmt="hod",
            clean=False,
            host_only=True,
        )
        return cosmo, halo_catalog

    def sample_hod_params(self, icosmo, irlz=0):
        _, halo_catalog = self._load_hod_halocat(icosmo)
        return self.hod_populator.find_hod_params(
            halo_catalog,
            seed_offset=self._get_sampling_seed_offset(icosmo, irlz),
        )
```

- [ ] **Step 5: Run focused and full tests to verify GREEN**

Run:

```bash
env MPLCONFIGDIR=/tmp/cosmogrid_mock_pipe_mpl \
/Users/suqikuai777/miniforge3/envs/mock_pipe/bin/python -m unittest \
tests.test_fastpm_runner.FastPMRunnerCoreTests.test_load_hod_halocat_uses_only_hosts_without_rhalf_cut \
tests.test_fastpm_runner.FastPMRunnerCoreTests.test_sample_hod_params_uses_realization_seed_and_host_catalog -v
```

Then run:

```bash
env MPLCONFIGDIR=/tmp/cosmogrid_mock_pipe_mpl \
/Users/suqikuai777/miniforge3/envs/mock_pipe/bin/python -m unittest discover -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit Rockstar HOD loading and sampling**

```bash
git add runner.py tests/test_fastpm_runner.py
git commit -m "feat: load FastPM Rockstar HOD catalogs"
```

---

### Task 4: Add the Foreground Galaxy Workflow

**Files:**
- Modify: `runner.py` (`FastPMRunner` only)
- Modify: `tests/test_fastpm_runner.py`

**Interfaces:**
- Consumes: `_load_hod_halocat`, `_get_hod_seed_offset`, `HODPopulator`, `SurveyGenerator`, foreground labels, and optional `gal_ofmt`.
- Produces: `gen_mock_gal(icosmo, irlz, ihod, ihod_param, save=False) -> np.ndarray`.

- [ ] **Step 1: Add failing helper and workflow tests**

Use behavior fakes at the expensive stochastic/survey boundary. Add the three test methods to `FastPMRunnerCoreTests`:

```python
class GalaxyHODPopulator:
    def populate_galaxies(self, halo_catalog, model_params_dict, indx, OmegaM):
        galaxies = np.zeros(
            1,
            dtype=[
                ("x", "f8"), ("y", "f8"), ("z", "f8"),
                ("vx", "f8"), ("vy", "f8"), ("vz", "f8"),
            ],
        )
        galaxies["x"] = indx
        galaxies["vx"] = OmegaM
        return {"dummy": galaxies}

    def get_galaxy_features(self, galaxies, features):
        pos = np.column_stack([galaxies["x"], galaxies["y"], galaxies["z"]])
        vel = np.column_stack([galaxies["vx"], galaxies["vy"], galaxies["vz"]])
        return pos, vel


class GalaxySurveyGenerator:
    def box_to_lightcone(self, cosmo, gal_pos, gal_adj_props):
        result = np.zeros(1, dtype=[("marker", "f8"), ("survey", "i4")])
        result["marker"] = gal_pos[0, 0] + gal_adj_props["gal_vel"][0, 0]
        return result

    def gen_boss_like(self, galcone, survey_name, survey_label, make_nz=True):
        result = galcone.copy()
        result["survey"] = survey_label
        return result
```

Add these test methods to `FastPMRunnerCoreTests`:

```python
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

    def test_gen_mock_gal_requires_output_format_when_saving(self):
        self.assertTrue(
            hasattr(self.runner, "gen_mock_gal"),
            "FastPMRunner.gen_mock_gal must be implemented",
        )
        with self.assertRaisesRegex(ValueError, "gal_ofmt"):
            self.runner.gen_mock_gal(0, 0, 0, np.ones(6), save=True)

    def test_unsupported_foreground_survey_is_rejected(self):
        self.assertTrue(
            hasattr(self.runner, "_pick_gen_mock_func"),
            "FastPMRunner._pick_gen_mock_func must be implemented",
        )
        with self.assertRaisesRegex(ValueError, "unknown"):
            self.runner._pick_gen_mock_func("unknown")
```

The save-validation branch must execute before catalog loading so the second test does not need a catalog fixture.

- [ ] **Step 2: Run the galaxy tests to verify RED**

Run:

```bash
env MPLCONFIGDIR=/tmp/cosmogrid_mock_pipe_mpl \
/Users/suqikuai777/miniforge3/envs/mock_pipe/bin/python -m unittest \
tests.test_fastpm_runner.FastPMRunnerCoreTests.test_gen_mock_gal_returns_survey_catalog_with_fastpm_seed \
tests.test_fastpm_runner.FastPMRunnerCoreTests.test_gen_mock_gal_requires_output_format_when_saving \
tests.test_fastpm_runner.FastPMRunnerCoreTests.test_unsupported_foreground_survey_is_rejected -v
```

Expected: assertion failures naming missing `gen_mock_gal` and `_pick_gen_mock_func`.

- [ ] **Step 3: Implement FastPM galaxy helpers and routing**

Add:

```python
    def _make_hod_param_dict(self, hod_param):
        return dict(zip(self.config.model_params_names, hod_param))

    def _gsample_dict_to_array(self, dict_of_gsamples):
        return dict_of_gsamples[next(iter(dict_of_gsamples))]

    def _pick_gen_mock_func(self, survey_name):
        boss_like = {
            "boss_lowz_ngc", "boss_cmass_ngc",
            "boss_lowz_sgc", "boss_cmass_sgc",
        }
        boss_trim = {
            "boss_lowze2_ngc", "boss_lowze3_ngc",
            "boss_lowze2_sgc", "boss_lowze3_sgc",
        }
        if survey_name in boss_like:
            return self.survey_generator.gen_boss_like
        if survey_name in boss_trim:
            return self.survey_generator.gen_boss_like_trim
        if survey_name == "2dflens_south":
            return self.survey_generator.gen_2dflens_like
        raise ValueError(f"Unsupported foreground survey: {survey_name}")
```

- [ ] **Step 4: Implement `gen_mock_gal`**

Add:

```python
    def gen_mock_gal(
            self, icosmo, irlz, ihod, ihod_param: np.ndarray,
            save=False):
        if save and self.gal_ofmt is None:
            raise ValueError("gal_ofmt is required when save=True")

        cosmo, halo_catalog = self._load_hod_halocat(icosmo)
        OmegaM = cosmo.omega_x(a=1.0, species="matter")
        model_params = self._make_hod_param_dict(ihod_param)
        seed_offset = self._get_hod_seed_offset(icosmo, irlz, ihod)
        samples = self.hod_populator.populate_galaxies(
            halo_catalog,
            model_params,
            indx=seed_offset,
            OmegaM=OmegaM,
        )
        galaxies = self._gsample_dict_to_array(samples)
        gal_pos, gal_vel = self.hod_populator.get_galaxy_features(
            galaxies, features=["pos", "vel"]
        )
        fullsky = self.survey_generator.box_to_lightcone(
            cosmo, gal_pos=gal_pos, gal_adj_props={"gal_vel": gal_vel}
        )

        survey_catalogs = []
        for survey_name, survey_label in self.fore_survey_labels_dict.items():
            generator = self._pick_gen_mock_func(survey_name)
            survey_catalogs.append(
                generator(fullsky, survey_name, survey_label)
            )
        if not survey_catalogs:
            raise ValueError("No foreground surveys configured")
        result = np.concatenate(survey_catalogs)

        if save:
            Table(result).write(self.gal_ofmt.format(icosmo, irlz, ihod))
        return result
```

- [ ] **Step 5: Run focused and full tests to verify GREEN**

Run:

```bash
env MPLCONFIGDIR=/tmp/cosmogrid_mock_pipe_mpl \
/Users/suqikuai777/miniforge3/envs/mock_pipe/bin/python -m unittest \
tests.test_fastpm_runner.FastPMRunnerCoreTests.test_gen_mock_gal_returns_survey_catalog_with_fastpm_seed \
tests.test_fastpm_runner.FastPMRunnerCoreTests.test_gen_mock_gal_requires_output_format_when_saving \
tests.test_fastpm_runner.FastPMRunnerCoreTests.test_unsupported_foreground_survey_is_rejected -v
```

Then run:

```bash
env MPLCONFIGDIR=/tmp/cosmogrid_mock_pipe_mpl \
/Users/suqikuai777/miniforge3/envs/mock_pipe/bin/python -m unittest discover -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit galaxy workflow**

```bash
git add runner.py tests/test_fastpm_runner.py
git commit -m "feat: generate FastPM galaxy catalogs"
```

---

### Task 5: Add the Void Workflow

**Files:**
- Modify: `runner.py` (`FastPMRunner` only)
- Modify: `tests/test_fastpm_runner.py`

**Interfaces:**
- Consumes: `_get_cosmo_instance`, `_pick_gen_mock_func`, `VoidFinder`, foreground labels, and optional `void_ofmt`.
- Produces: `gen_mock_void(icosmo, irlz, ihod, galcone_survey, dive_input, dive_output, save=False) -> np.ndarray`.

- [ ] **Step 1: Add failing void behavior tests**

Define the two boundary fakes:

```python
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
```

Add these test methods to `FastPMRunnerCoreTests`:

```python
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
```

- [ ] **Step 2: Run the void tests to verify RED**

Run:

```bash
env MPLCONFIGDIR=/tmp/cosmogrid_mock_pipe_mpl \
/Users/suqikuai777/miniforge3/envs/mock_pipe/bin/python -m unittest \
tests.test_fastpm_runner.FastPMRunnerCoreTests.test_gen_mock_void_filters_redshift_and_preserves_survey \
tests.test_fastpm_runner.FastPMRunnerCoreTests.test_gen_mock_void_requires_output_format_when_saving -v
```

Expected: assertion failures naming missing `gen_mock_void`.

- [ ] **Step 3: Implement `gen_mock_void`**

Add:

```python
    def gen_mock_void(
            self, icosmo, irlz, ihod, galcone_survey,
            dive_input, dive_output, save=False):
        if save and self.void_ofmt is None:
            raise ValueError("void_ofmt is required when save=True")

        cosmo = self._get_cosmo_instance(icosmo, otype="ccl")
        survey_catalogs = []
        for survey_name, survey_label in self.fore_survey_labels_dict.items():
            selected = galcone_survey[galcone_survey["survey"] == survey_label]
            if len(selected) == 0:
                continue
            voids = self.void_finder.galcone_to_voidcone(
                selected,
                cosmo,
                survey=survey_label,
                dive_input=dive_input,
                dive_output=dive_output,
            )
            redshift_cut = (
                (voids["z"] >= self.config.zmin_lightcone)
                & (voids["z"] <= self.config.zmax_lightcone)
            )
            generator = self._pick_gen_mock_func(survey_name)
            survey_catalogs.append(
                generator(
                    voids[redshift_cut],
                    survey_name,
                    survey_label,
                    make_nz=False,
                )
            )
        if not survey_catalogs:
            raise ValueError("No non-empty foreground surveys in galaxy catalog")
        result = np.concatenate(survey_catalogs)

        if save:
            Table(result).write(self.void_ofmt.format(icosmo, irlz, ihod))
        return result
```

- [ ] **Step 4: Run focused and full tests to verify GREEN**

Run:

```bash
env MPLCONFIGDIR=/tmp/cosmogrid_mock_pipe_mpl \
/Users/suqikuai777/miniforge3/envs/mock_pipe/bin/python -m unittest \
tests.test_fastpm_runner.FastPMRunnerCoreTests.test_gen_mock_void_filters_redshift_and_preserves_survey \
tests.test_fastpm_runner.FastPMRunnerCoreTests.test_gen_mock_void_requires_output_format_when_saving -v
```

Then run:

```bash
env MPLCONFIGDIR=/tmp/cosmogrid_mock_pipe_mpl \
/Users/suqikuai777/miniforge3/envs/mock_pipe/bin/python -m unittest discover -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit void workflow**

```bash
git add runner.py tests/test_fastpm_runner.py
git commit -m "feat: generate FastPM void catalogs"
```

---

### Task 6: Real-Data Integration and Final Verification

**Files:**
- Verify: `runner.py`
- Verify: `handler.py`
- Verify: `utils/io_func.py`
- Verify: `tests/test_fastpm_runner.py`
- Read externally: `cosmo_list.txt` and `out_0_wPID.list`

**Interfaces:**
- Consumes: completed `FastPMRunner` and parent-processed real catalog.
- Produces: verification evidence that cosmo0, `a_0.7692`, and host-only HOD loading work at real scale without changing `CosmoGridRunner`.

- [ ] **Step 1: Run the complete automated suite**

```bash
env MPLCONFIGDIR=/tmp/cosmogrid_mock_pipe_mpl \
/Users/suqikuai777/miniforge3/envs/mock_pipe/bin/python -m unittest discover -v
```

Expected: all tests pass with zero failures and errors.

- [ ] **Step 2: Verify the real cosmology and filename**

Run:

```bash
env MPLCONFIGDIR=/tmp/cosmogrid_mock_pipe_mpl \
/Users/suqikuai777/miniforge3/envs/mock_pipe/bin/python -c '
from handler import PipeConfig
from runner import FastPMRunner

halo_fmt = (
    "/Users/suqikuai777/Dataspace/FastPM/Cosmology/"
    "L1000_N1024_1000cosmo/cosmo{:d}/"
    "a_{:5.4f}/rstar/out_0_wPID.list"
)
runner = FastPMRunner(
    config=PipeConfig(Lbox=1000.0, Npart=1024, redshift=0.3),
    halo_fmt=halo_fmt,
    cosmo_par_fname="/Users/suqikuai777/Dataspace/FastPM/Cosmology/cosmo_list.txt",
    fore_mask_fnames_dict={"boss_veto": []},
    fore_nofz_fnames_dict={},
    fore_survey_labels_dict={},
)
params = runner._get_cosmo_instance(0, otype="dict")
assert params["OmegaM"] == 0.200614
assert runner.scale_factor == 1.0 / 1.3
path = runner._get_halo_fname(0)
assert path.endswith("cosmo0/a_0.7692/rstar/out_0_wPID.list")
print(path)
'
```

Expected: exit 0 and the exact `cosmo0/a_0.7692` path.

- [ ] **Step 3: Verify real host-only HOD conversion**

Run:

```bash
env MPLCONFIGDIR=/tmp/cosmogrid_mock_pipe_mpl \
/Users/suqikuai777/miniforge3/envs/mock_pipe/bin/python -c '
from handler import PipeConfig
from runner import FastPMRunner

runner = FastPMRunner(
    config=PipeConfig(Lbox=1000.0, Npart=1024, redshift=0.3),
    halo_fmt=(
        "/Users/suqikuai777/Dataspace/FastPM/Cosmology/"
        "L1000_N1024_1000cosmo/cosmo{:d}/"
        "a_{:5.4f}/rstar/out_0_wPID.list"
    ),
    cosmo_par_fname="/Users/suqikuai777/Dataspace/FastPM/Cosmology/cosmo_list.txt",
    fore_mask_fnames_dict={"boss_veto": []},
    fore_nofz_fnames_dict={},
    fore_survey_labels_dict={},
)
cosmo, catalog = runner._load_hod_halocat(0)
table = catalog.halo_table
assert len(table) > 0
assert (table["halo_upid"] == -1).all()
assert (table["halo_hostid"] == table["halo_id"]).all()
assert (table["halo_nfw_conc"] > 0).all()
print(f"hosts={len(table)} particle_mass={catalog.particle_mass}")
'
```

Expected: exit 0, positive host count, host-only hierarchy, and positive catalog-derived concentrations.

- [ ] **Step 4: Verify syntax, diff hygiene, and original-runner preservation**

```bash
/Users/suqikuai777/miniforge3/envs/mock_pipe/bin/python -m py_compile \
runner.py handler.py utils/io_func.py tests/test_fastpm_runner.py
git diff --check
git diff --unified=0 cb97c97 -- runner.py
```

Expected: compilation and `git diff --check` exit 0. Inspect the runner diff and confirm every production addition is inside the new `FastPMRunner`; no existing `CosmoGridRunner` line changed.

- [ ] **Step 5: Request independent code review**

Use `superpowers:requesting-code-review` with the design spec, this plan, the current base SHA, and the implementation HEAD. Fix every Critical and Important issue, rerunning the complete suite and real-data checks after changes.

- [ ] **Step 6: Final verification checkpoint**

Use `superpowers:verification-before-completion`. Re-run Steps 1, 3, and 4 fresh, read each exit code and full output, and only then report completion.
