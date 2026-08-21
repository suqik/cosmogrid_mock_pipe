import tempfile
import unittest
from pathlib import Path

import numpy as np

from utils import io_func


RSTAR_HEADER = (
    "#ID DescID Mvir Vmax Vrms Rvir Rs Np X Y Z VX VY VZ "
    "JX JY JZ Spin rs_klypin Mvir_all M200b M200c M500c M2500c "
    "Xoff Voff spin_bullock b_to_a c_to_a A[x] A[y] A[z] "
    "b_to_a(500c) c_to_a(500c) A[x](500c) A[y](500c) A[z](500c) "
    "T/|U| M_pe_Behroozi M_pe_Diemer Halfmass_Radius PID\n"
)


class RockstarIOTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.catalog_path = Path(self.tempdir.name) / "out_0.list"
        rows = (
            "10 -1 2.0e12 200 100 200 40 50 1 2 3 10 20 30 "
            "0 0 0 0.1 40 2.0e12 0 0 0 0 0 0 0 1 1 0 0 0 "
            "1 1 0 0 0 0.5 0 0 100 -1\n"
            "11 -1 5.0e11 120 60 100 25 20 4 5 6 40 50 60 "
            "0 0 0 0.2 25 5.0e11 0 0 0 0 0 0 0 1 1 0 0 0 "
            "1 1 0 0 0 0.6 0 0 50 10\n"
        )
        self.catalog_path.write_text(RSTAR_HEADER + "#a = 0.769231\n" + rows)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_get_rstar_halo_attrs_maps_hod_fields_and_units(self):
        self.assertTrue(hasattr(io_func, "get_rstar_halo_attrs"))
        attrs = [
            "pos",
            "vel",
            "mass",
            "rvir",
            "rHalf",
            "concentration",
            "ID",
            "PID",
        ]

        result = io_func.get_rstar_halo_attrs(
            self.catalog_path, attrs=attrs, host_only=False
        )

        np.testing.assert_allclose(result["pos"], [[1, 2, 3], [4, 5, 6]])
        np.testing.assert_allclose(result["vel"], [[10, 20, 30], [40, 50, 60]])
        np.testing.assert_allclose(result["mass"], [2.0e12, 5.0e11])
        np.testing.assert_allclose(result["rvir"], [0.2, 0.1])
        np.testing.assert_allclose(result["rHalf"], [0.1, 0.05])
        np.testing.assert_allclose(result["concentration"], [5.0, 4.0])
        np.testing.assert_array_equal(result["ID"], [10, 11])
        np.testing.assert_array_equal(result["PID"], [-1, 10])
        self.assertTrue(np.issubdtype(result["ID"].dtype, np.integer))
        self.assertTrue(np.issubdtype(result["PID"].dtype, np.integer))

    def test_get_rstar_halo_attrs_host_only_filters_every_attribute(self):
        self.assertTrue(hasattr(io_func, "get_rstar_halo_attrs"))
        result = io_func.get_rstar_halo_attrs(
            self.catalog_path,
            host_only=True,
        )

        np.testing.assert_allclose(result["pos"], [[1, 2, 3]])
        np.testing.assert_allclose(result["vel"], [[10, 20, 30]])
        np.testing.assert_allclose(result["mass"], [2.0e12])
        np.testing.assert_allclose(result["rvir"], [0.2])
        np.testing.assert_allclose(result["rHalf"], [0.1])
        np.testing.assert_allclose(result["concentration"], [5.0])
        np.testing.assert_array_equal(result["ID"], [10])
        np.testing.assert_array_equal(result["PID"], [-1])

    def test_rstar_to_hod_type_preserves_catalog_properties_and_hierarchy(self):
        self.assertTrue(hasattr(io_func, "get_rstar_halo_attrs"))
        self.assertTrue(hasattr(io_func, "rstar_to_hod_type"))
        infos = io_func.get_rstar_halo_attrs(self.catalog_path, host_only=False)

        halo_catalog = io_func.rstar_to_hod_type(
            infos,
            pmass=8.0e10,
            boxsize=1000.0,
            redshift=0.3,
        )

        table = halo_catalog.halo_table
        np.testing.assert_array_equal(table["halo_id"], [10, 11])
        np.testing.assert_array_equal(table["halo_upid"], [-1, 10])
        np.testing.assert_array_equal(table["halo_hostid"], [10, 10])
        np.testing.assert_allclose(table["halo_x"], [1, 4])
        np.testing.assert_allclose(table["halo_y"], [2, 5])
        np.testing.assert_allclose(table["halo_z"], [3, 6])
        np.testing.assert_allclose(table["halo_vx"], [10, 40])
        np.testing.assert_allclose(table["halo_vy"], [20, 50])
        np.testing.assert_allclose(table["halo_vz"], [30, 60])
        np.testing.assert_allclose(table["halo_rvir"], [0.2, 0.1])
        np.testing.assert_allclose(table["halo_mvir"], [2.0e12, 5.0e11])
        np.testing.assert_allclose(table["halo_nfw_conc"], [5.0, 4.0])
        np.testing.assert_allclose(table["halo_rhalf"], [0.1, 0.05])


if __name__ == "__main__":
    unittest.main()
