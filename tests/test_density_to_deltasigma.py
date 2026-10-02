import importlib
import tempfile
import unittest
from pathlib import Path

import numpy as np


class DensityToDeltaSigmaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = importlib.import_module('abacus_runs.density_to_deltasigma')

    def test_load_profiles_preserves_counts_and_builds_requested_rbins(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'profiles.txt'
            np.savetxt(path, [[12, -0.5, -0.2, 0.0],
                              [15, -0.4, -0.1, 0.1]])
            counts, radii, profiles = self.module.load_profiles(path, spacing=0.025)

        np.testing.assert_array_equal(counts, [12, 15])
        np.testing.assert_allclose(radii, [0.025, 0.050, 0.075])
        np.testing.assert_allclose(profiles,
                                   [[-0.5, -0.2, 0.0], [-0.4, -0.1, 0.1]])

    def test_uniform_density_contrast_sphere_matches_analytic_solution(self):
        radii = np.arange(1, 121) * 0.025
        projected_radii = np.array([0.5, 1.0, 2.0])
        observed = self.module.delta_sigma_kernel(
            radii, np.ones_like(radii), projected_radii
        )

        sphere_radius = 3.0
        root = np.sqrt(sphere_radius**2 - projected_radii**2)
        sigma = 2.0 * root
        mean_sigma = 4.0 / (3.0 * projected_radii**2) * (
            sphere_radius**3 - root**3
        )
        expected = mean_sigma - sigma
        np.testing.assert_allclose(observed, expected, rtol=2e-7, atol=2e-9)

    def test_zero_density_contrast_gives_zero_delta_sigma(self):
        radii = np.arange(1, 121) * 0.025
        observed = self.module.delta_sigma_kernel(
            radii, np.zeros_like(radii), radii
        )
        np.testing.assert_allclose(observed, 0.0, atol=1e-14)

    def test_physical_normalization_uses_comoving_mean_matter_density(self):
        # Hand calculation: rho_m,0 * Rv =
        # 2.77536627e11 * Om0 * h^2 * (Rv/h) / 1e12 Msun/pc^2.
        observed = self.module.surface_density_scale(
            rv_mpch=10.0, omega_m=0.3, h=0.7
        )
        self.assertAlmostEqual(observed, 0.5828269167, places=10)


if __name__ == '__main__':
    unittest.main()
