import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyccl as ccl

from handler import CatalogLoader, PipeConfig
from utils.io_func import pkd_to_hod_type


RSTAR_HEADER = (
    "#ID DescID Mvir Vmax Vrms Rvir Rs Np X Y Z VX VY VZ "
    "JX JY JZ Spin rs_klypin Mvir_all M200b M200c M500c M2500c "
    "Xoff Voff spin_bullock b_to_a c_to_a A[x] A[y] A[z] "
    "b_to_a(500c) c_to_a(500c) A[x](500c) A[y](500c) A[z](500c) "
    "T/|U| M_pe_Behroozi M_pe_Diemer Halfmass_Radius PID\n"
)


class RockstarCatalogLoaderTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.catalog_path = Path(self.tempdir.name) / "out_0.list"
        rows = (
            "10 -1 2.0e12 200 100 200 40 50 1 2 3 10 20 30 "
            "0 0 0 0.1 40 2.0e12 0 0 0 0 0 0 0 1 1 0 0 0 "
            "1 1 0 0 0 0.5 0 0 50 -1\n"
            "11 -1 5.0e11 120 60 100 25 20 4 5 6 40 50 60 "
            "0 0 0 0.2 25 5.0e11 0 0 0 0 0 0 0 1 1 0 0 0 "
            "1 1 0 0 0 0.6 0 0 200 10\n"
        )
        self.catalog_path.write_text(RSTAR_HEADER + "#a = 0.769231\n" + rows)
        self.config = PipeConfig(Lbox=1000.0, Npart=1024, redshift=0.3)
        self.loader = CatalogLoader(self.config)
        self.cosmo = ccl.Cosmology(
            Omega_c=0.2607,
            Omega_b=0.0490,
            h=0.6766,
            sigma8=0.8102,
            n_s=0.9665,
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def test_load_rstar_halocat_defaults_to_host_halos(self):
        self.assertTrue(hasattr(self.loader, "load_rstar_halocat"))

        result = self.loader.load_rstar_halocat(
            self.catalog_path,
            self.cosmo,
            ofmt="rstar",
        )

        np.testing.assert_array_equal(result["ID"], [10])
        np.testing.assert_array_equal(result["PID"], [-1])

    def test_load_rstar_halocat_does_not_apply_rhalf_cleaning(self):
        self.assertTrue(hasattr(self.loader, "load_rstar_halocat"))

        result = self.loader.load_rstar_halocat(
            self.catalog_path,
            self.cosmo,
            ofmt="rstar",
            clean=True,
        )

        np.testing.assert_allclose(result["rHalf"], [0.05])

    def test_load_rstar_halocat_matches_pkd_hod_output_contract(self):
        self.assertTrue(hasattr(self.loader, "load_rstar_halocat"))

        rstar_catalog = self.loader.load_rstar_halocat(
            self.catalog_path,
            self.cosmo,
            ofmt="hod",
        )
        expected_particle_mass = 8.00503777322644e10
        pkd_catalog = pkd_to_hod_type(
            {
                "pos": np.array([[1.0, 2.0, 3.0]]),
                "vel": np.array([[10.0, 20.0, 30.0]]),
                "mass": np.array([2.0e12]),
                "rHalf": np.array([0.05]),
            },
            cosmo=self.cosmo,
            pmass=expected_particle_mass,
            boxsize=1000.0,
            redshift=0.3,
        )

        self.assertIs(type(rstar_catalog), type(pkd_catalog))
        self.assertEqual(
            set(rstar_catalog.halo_table.colnames),
            set(pkd_catalog.halo_table.colnames),
        )
        self.assertAlmostEqual(
            rstar_catalog.particle_mass,
            expected_particle_mass,
        )
        self.assertEqual(rstar_catalog.redshift, 0.3)
        np.testing.assert_allclose(rstar_catalog.Lbox, [1000.0] * 3)
        self.assertEqual(len(rstar_catalog.halo_table), 1)
        np.testing.assert_array_equal(rstar_catalog.halo_table["halo_id"], [10])
        np.testing.assert_array_equal(
            rstar_catalog.halo_table["halo_upid"],
            [-1],
        )
        np.testing.assert_array_equal(
            rstar_catalog.halo_table["halo_hostid"],
            [10],
        )
        np.testing.assert_allclose(
            rstar_catalog.halo_table["halo_nfw_conc"],
            [5.0],
        )


if __name__ == "__main__":
    unittest.main()
