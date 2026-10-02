import importlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.cosmology import units as cu
from astropy.table import Table
from dsigma.physics import critical_surface_density
from dsigma.stacking import excess_surface_density


class AbacusVoidLensingTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('abacus_runs.run_vl'),
                             'The Abacus void-lensing driver has not been implemented')
        self.vl = importlib.import_module('abacus_runs.run_vl')

    def test_void_loader_preserves_every_radius_and_real_redshift(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / 'voids.txt'
            np.savetxt(filename, [[180, 0, 0.4, 0.1], [181, 1, 0.5, 17.5], [182, 2, 0.6, 90]])
            catalog = self.vl.load_voids(filename)
        self.assertEqual(len(catalog), 3)
        np.testing.assert_allclose(catalog.Rv, [0.1, 17.5, 90])
        np.testing.assert_allclose(catalog.z, [0.4, 0.5, 0.6])
        np.testing.assert_array_equal(catalog.w, [1, 1, 1])

    def test_tomo_selection_uses_the_lower_bin_boundary(self):
        self.assertEqual(self.vl.eligible_tomos(0.6), [4, 5])
        self.assertEqual(self.vl.eligible_tomos(0.8), [5])

    def test_source_selection_uses_true_redshift_and_requested_noise(self):
        data = Table(dict(ra=[180., 181., 182.], dec=[0., 0., 0.],
                          z=[0.1, 1.9, 0.2], z_true=[1., 1.1, 0.9],
                          tomo=[4, 5, 3], w=[2., 3., 1.],
                          g1_pure=[0.02, 0.03, 0.04], g2_pure=[0.01, 0.02, 0.03],
                          g1=[0.2, 0.3, 0.4], g2=[0.1, 0.2, 0.3]))
        pure = self.vl.source_table(data, 4, noisy=False)
        noisy = self.vl.source_table(data, 4, noisy=True)
        self.assertEqual(len(pure), 1)
        np.testing.assert_allclose(pure['z'], [1.])
        np.testing.assert_allclose(pure['z_l_max'], [1.])
        np.testing.assert_allclose(pure['e_1'], [-0.02])
        np.testing.assert_allclose(pure['e_2'], [0.01])
        np.testing.assert_allclose(pure['w'], [2.])
        np.testing.assert_allclose(noisy['e_1'], [-0.2])

    def make_pair_fixture(self, gamma=0.02):
        cosmo = self.vl.get_cosmology()
        chi_mpch = cosmo.comoving_distance(0.5).to_value(u.Mpc / cu.littleh, cu.with_H0(cosmo.H0))
        sources = Table(dict(ra=180 + np.rad2deg(np.array([1.2, 2.2]) / chi_mpch),
                             dec=[0., 0.], z=[1., 1.], z_l_max=[1., 1.],
                             w=[1., 1.], e_1=[-gamma, -gamma], e_2=[0., 0.]))
        lenses = Table(dict(ra=[180.], dec=[0.], z=[0.5], w_sys=[1.]))
        return cosmo, lenses, sources

    def test_pair_bins_keep_mpch_units_and_tangential_sign(self):
        cosmo, lenses, sources = self.make_pair_fixture()
        edges = self.vl.radial_edges(10., 0.1, 0.3, 2, 'mpc_h')
        result = self.vl.precompute_pairs(lenses, sources, edges, cosmo, jobs=1)
        np.testing.assert_array_equal(result['sum 1'], [[1, 1]])
        observed = excess_surface_density(result).to_value(u.Msun / u.pc**2, cu.with_H0(cosmo.H0))
        sigma_crit = critical_surface_density(0.5, 1., cosmology=cosmo, comoving=True)
        expected = 0.02 * sigma_crit.to_value(u.Msun / u.pc**2, cu.with_H0(cosmo.H0))
        np.testing.assert_allclose(observed, [expected, expected], rtol=1e-4)

    def test_zero_shear_returns_zero_signal(self):
        cosmo, lenses, sources = self.make_pair_fixture(gamma=0.)
        result = self.vl.precompute_pairs(lenses, sources,
                                          self.vl.radial_edges(10., 0.1, 0.3, 2, 'mpc_h'), cosmo, jobs=1)
        np.testing.assert_allclose(excess_surface_density(result).value, [0., 0.], atol=1e-15)

    def test_geometric_pruning_only_removes_zero_pair_lenses(self):
        cosmo, first, sources = self.make_pair_fixture()
        lenses = Table(dict(ra=[180., 180.03, 10.], dec=[0., 0., 50.],
                            z=[0.5, 0.5, 0.5], w_sys=[1., 1., 1.]))
        edges = self.vl.radial_edges(10., 0.1, 0.3, 2, 'mpc_h')
        mask = self.vl.possible_pair_mask(lenses, sources, edges[-1], cosmo)
        np.testing.assert_array_equal(mask, [True, True, False])
        full = self.vl.precompute_pairs(lenses.copy(), sources, edges, cosmo, jobs=1)
        pruned = self.vl.precompute_pairs(lenses[mask], sources, edges, cosmo, jobs=1)
        np.testing.assert_array_equal(full['sum 1'].sum(axis=0), pruned['sum 1'].sum(axis=0))
        np.testing.assert_allclose(excess_surface_density(full).value,
                                   excess_surface_density(pruned).value, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
