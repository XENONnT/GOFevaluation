import scipy.stats as sps
import numpy as np
import unittest

from GOFevaluation import binned_poisson_chi2_gof
from GOFevaluation import point_to_point_gof


class Test_binned_poisson_chi2_gof(unittest.TestCase):
    def test_dimensions(self):
        # test nD binned GOF in different dimensions:
        signal_expectation = 100
        xs = np.linspace(0, 1, 100)
        for nD in [1, 2, 3, 5]:
            data = np.vstack([xs for i in range(nD)]).T
            bins = [np.linspace(0, 1, 3) for i in range(nD)]
            mus = np.ones([2 for i in range(nD)])
            mus /= np.sum(mus)

            nbins = 2**nD
            ns_flat = np.zeros(nbins)
            ns_flat[0:2] = 50
            mus_flat = np.ones(nbins)
            mus_flat /= np.sum(mus_flat)
            gof_flat = 2 * np.sum(
                sps.poisson(ns_flat).logpmf(ns_flat) -
                sps.poisson(mus_flat * signal_expectation).logpmf(ns_flat))

            gofclass = binned_poisson_chi2_gof(
                data, mus, bins, signal_expectation)
            gof = gofclass.calculate_gof()

            self.assertLess(abs(gof_flat - gof), 1e-8)


class Test_point_to_point_gof(unittest.TestCase):
    def test_distances(self):
        # test if number of distance values is correct
        xs = np.linspace(0, 1, 100)
        xs_ref = np.linspace(0, 1, 200)
        for nD in [1, 2, 3, 5]:
            data = np.vstack([xs for i in range(nD)]).T
            reference = np.vstack([xs_ref for i in range(nD)]).T
            gofclass = point_to_point_gof(data, reference)
            gofclass.get_distances()

            self.assertEqual(len(gofclass.d_data_data), gofclass.nevents_data *
                             (gofclass.nevents_data-1) / 2)
            self.assertEqual(len(gofclass.d_ref_ref), gofclass.nevents_ref *
                             (gofclass.nevents_ref-1) / 2)
            self.assertEqual(len(gofclass.d_data_ref), gofclass.nevents_ref *
                             gofclass.nevents_data)

    def test_symmetry(self):
        # the pointwise energy test is symmetrical in reference and science sample:
        xs_a = sps.uniform().rvs(50)[:, None]
        xs_b = sps.uniform().rvs(50)[:, None]
        gofclass_ab = point_to_point_gof(xs_a, xs_b)
        gofclass_ab.d_min = 0.01  # set explicitly to avoid asymmetry in setting d_min
        gofclass_ab.get_distances()
        gof_ab = gofclass_ab.calculate_gof()
        gofclass_ba = point_to_point_gof(xs_b, xs_a)
        gofclass_ba.d_min = 0.01  # set explicitly to avoid asymmetry in setting d_min
        gofclass_ba.get_distances()
        gof_ba = gofclass_ba.calculate_gof()

        # it seems precision is a bit low in this case
        self.assertAlmostEqual(gof_ab, gof_ba, places=6)

    def test_value(self):
        # simple values:
        xs_a = np.array([0])[:, None]
        xs_b = np.array([1, 2])[:, None]

        e_data_ref = np.log(2)/2
        gofclass_ab = point_to_point_gof(xs_a, xs_b)
        gofclass_ab.d_min = 0.01  # set explicitly to avoid asymmetry in setting d_min
        gofclass_ab.get_distances()
        gof_ab = gofclass_ab.calculate_gof()
        # it seems precision is a bit low in this case
        self.assertAlmostEqual(gof_ab, e_data_ref, places=6)


if __name__ == "__main__":
    unittest.main()
