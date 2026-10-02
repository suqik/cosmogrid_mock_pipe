import importlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.table import Table


class PlotDensityDeltaSigmaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = importlib.import_module('abacus_runs.plot_density_deltasigma')

    def test_weighted_mean_uses_void_counts(self):
        values = np.array([[1.0, 2.0], [5.0, 10.0]])
        observed = self.module.void_weighted_mean(values, np.array([1, 3]))
        np.testing.assert_allclose(observed, [4.0, 8.0])

    def write_pure_measurements(self, directory):
        directory.mkdir()
        for tomo, signal in [(4, [-0.3, -0.2, -0.1]),
                             (5, [-0.4, -0.3, -0.15])]:
            table = Table(
                dict(rp_over_Rv=[0.1, 0.2, 0.4],
                     ds=np.array(signal) * u.Msun / u.pc**2,
                     ds_err=np.array([0.03, 0.02, 0.01]) * u.Msun / u.pc**2)
            )
            table.write(directory / f'vl_tomo{tomo}_pure.fits')

    def test_load_measurements_reads_only_tomo4_and_tomo5_pure(self):
        with tempfile.TemporaryDirectory() as directory:
            measurement_dir = Path(directory) / 'measurements'
            self.write_pure_measurements(measurement_dir)

            observed = self.module.load_pure_measurements(measurement_dir)

        self.assertEqual(sorted(observed), [4, 5])
        np.testing.assert_allclose(observed[4]['radius'], [0.1, 0.2, 0.4])
        np.testing.assert_allclose(observed[4]['delta_sigma'], [-0.3, -0.2, -0.1])
        np.testing.assert_allclose(observed[5]['delta_sigma'], [-0.4, -0.3, -0.15])

    def test_plot_file_writes_nonempty_png_and_pdf(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / 'profiles.npz'
            output = directory / 'profile_plot'
            measurement_dir = directory / 'measurements'
            self.write_pure_measurements(measurement_dir)
            np.savez_compressed(
                source,
                void_count=np.array([10, 20]),
                r_over_Rv=np.array([0.1, 0.2, 0.4]),
                density_contrast=np.array([[-0.5, -0.2, 0.0],
                                           [-0.4, -0.1, 0.1]]),
                delta_sigma=np.array([[-0.3, -0.2, -0.1],
                                      [-0.2, -0.1, 0.0]]),
                delta_sigma_unit='Msun/pc^2 comoving',
                rv_median_mpch=17.5,
            )

            paths = self.module.plot_file(source, measurement_dir, output)

            self.assertEqual(paths, (Path(f'{output}.png'), Path(f'{output}.pdf')))
            for path in paths:
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 1000)


if __name__ == '__main__':
    unittest.main()
