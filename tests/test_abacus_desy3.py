import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import healpy as hp
import numpy as np
from astropy.io import fits

from runner import AbacusRunner
from utils.mkback_utils import (
    gen_angle_positions_from_healpix,
    gen_redshifts_from_nofz,
    make_nofz,
)


class AbacusDesY3MaskTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)

    def write_nested_science_mask(self):
        nested_mask = np.zeros(hp.nside2npix(2), dtype=np.uint8)
        nested_mask[[3, 17, 31]] = 1
        column = fits.Column(
            name="SCIENCE",
            format=f"{len(nested_mask)}B",
            array=nested_mask.reshape(1, -1),
        )
        table = fits.BinTableHDU.from_columns([column])
        table.header["PIXTYPE"] = "HEALPIX"
        table.header["ORDERING"] = "NESTED"
        table.header["COORDSYS"] = "C"
        table.header["NSIDE"] = 2
        path = self.root / "des_y3_mask.fits"
        fits.HDUList([fits.PrimaryHDU(), table]).writeto(path)
        return path, nested_mask

    def test_des_y3_science_mask_is_loaded_in_ring_order(self):
        # This catches reading the legacy VALUE field or treating NESTED
        # pixel indices as RING indices.
        path, nested_mask = self.write_nested_science_mask()
        runner = object.__new__(AbacusRunner)

        masks = runner._prepare_back_masks({"DES-Y3": path})

        expected = hp.reorder(nested_mask, n2r=True)
        np.testing.assert_array_equal(masks["DES-Y3"], expected)

    def test_random_source_count_uses_healpix_footprint_area(self):
        # This catches deriving the catalog size from an RA/Dec bounding box.
        mask = np.zeros(hp.nside2npix(2), dtype=np.uint8)
        mask[[0, 1]] = 1
        footprint_arcmin2 = (
            2 * hp.nside2pixarea(2) * (180.0 / np.pi) ** 2 * 3600.0
        )
        density = 7.0 / footprint_arcmin2

        ra, dec = gen_angle_positions_from_healpix(density, mask)

        self.assertEqual(len(ra), 7)
        self.assertEqual(len(dec), 7)
        pixels = hp.ang2pix(2, ra, dec, lonlat=True)
        self.assertTrue(np.all(mask[pixels] == 1))

    def test_random_positions_remain_inside_after_float32_storage(self):
        # This catches accepting a float64 position that crosses a HEALPix
        # boundary when the catalog stores its coordinates as float32.
        nside = 256
        mask = (np.arange(hp.nside2npix(nside)) % 2).astype(np.uint8)
        footprint_area = np.count_nonzero(mask) * hp.nside2pixarea(
            nside, degrees=True
        )
        density = 200_000 / (footprint_area * 3600.0)
        random_state = np.random.get_state()
        self.addCleanup(np.random.set_state, random_state)
        np.random.seed(0)

        ra, dec = gen_angle_positions_from_healpix(density, mask)

        stored_pixels = hp.ang2pix(
            nside,
            ra.astype(np.float32),
            dec.astype(np.float32),
            lonlat=True,
        )
        self.assertTrue(np.all(mask[stored_pixels] == 1))

    def test_random_positions_wrap_float32_ra_at_360(self):
        mask = np.ones(hp.nside2npix(1), dtype=np.uint8)
        footprint_area = hp.nside2pixarea(1, degrees=True) * len(mask)
        density = 1.0 / (footprint_area * 3600.0)
        almost_one = np.nextafter(1.0, 0.0)
        random_values = [
            np.full(1_000, almost_one),
            np.zeros(1_000),
        ]

        with patch(
            "utils.mkback_utils.np.random.uniform",
            side_effect=random_values,
        ):
            ra, _ = gen_angle_positions_from_healpix(density, mask)

        self.assertEqual(len(ra), 1)
        self.assertEqual(float(ra[0]), 0.0)

    def test_des_y3_runner_loads_all_four_tomographic_bins(self):
        # This catches accidentally configuring only the tomo4 file named in
        # the original request, or treating the three-column metadata line as
        # a redshift-density row.
        from abacus_runs.run_mock_shape_desy3 import build_des_y3_runner

        mask_path, _ = self.write_nested_science_mask()
        nofz_dir = self.root / "desy3_nofzs"
        nofz_dir.mkdir()
        for tomo in range(1, 5):
            path = nofz_dir / (
                "pzwei_sources_abacushuge_desy3_"
                f"tom{tomo}_dz0pt05.dat"
            )
            path.write_text(
                "0.0 2.0  4\n"
                "0.25 0.0\n"
                "0.75 1.0\n"
                "1.25 1.0\n"
                "1.75 0.0\n"
            )

        runner = build_des_y3_runner(
            shear_map_fmt=str(self.root / "gamma{:d}_z{:.2f}.fits"),
            mask_path=mask_path,
            nofz_dir=nofz_dir,
            output_path=self.root / "shape.fits",
            redshift_src_list=[0.5, 1.0],
        )

        self.assertEqual(
            runner.back_ngals_dict,
            {
                "tomo1": 1.476,
                "tomo2": 1.479,
                "tomo3": 1.484,
                "tomo4": 1.461,
            },
        )
        self.assertEqual(
            runner.tomo_labels_dict,
            {"tomo1": 1, "tomo2": 2, "tomo3": 3, "tomo4": 4},
        )
        self.assertEqual(
            set(runner.shear_assigner.nofzs),
            {"tomo1", "tomo2", "tomo3", "tomo4"},
        )
        self.assertAlmostEqual(runner.config.sigma_e, 0.261)

    def test_density_nofz_produces_the_requested_catalog_size(self):
        # This catches interpreting a normalized probability density whose
        # values sum to 1/dz as direct per-bin probabilities.
        nofz = make_nofz(
            np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55]),
            np.array([0.0, 5.0, 10.0, 5.0, 0.0, 0.0]),
        )

        z_true, z_phot = gen_redshifts_from_nofz(101, nofz)

        self.assertAlmostEqual(float(nofz["nz"].sum()), 1.0)
        self.assertEqual(len(z_true), 101)
        self.assertEqual(len(z_phot), 101)


if __name__ == "__main__":
    unittest.main()
