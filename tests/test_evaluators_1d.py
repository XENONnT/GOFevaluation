import scipy.stats as sps
import numpy as np
from scipy.interpolate import interp1d
import unittest
import warnings

from GOFevaluation.evaluators_1d import KSTestGOF
from GOFevaluation.evaluators_1d import KSTestTwoSampleGOF
from GOFevaluation.evaluators_1d import ADTestTwoSampleGOF
from GOFevaluation.evaluators_nd import PointToPointGOF
from GOFevaluation.evaluators_nd import BinnedPoissonChi2GOF
from GOFevaluation.evaluators_nd import BinnedChi2GOF


class TestKSTestGOF(unittest.TestCase):
    def test_value(self):
        """Compare result of method to manually calculated gof."""

        # Generate Test Data
        n_samples = 100
        # pseudo random data with fixed seed for reproducibility
        data = sps.norm.rvs(size=n_samples, random_state=300)

        bin_edges = np.linspace(-4, 4, 101)
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
        bin_widths = bin_edges[1:] - bin_edges[:-1]

        normed_gauss_pdf = sps.norm.pdf(bin_centers) * bin_widths
        interp_cdf = interp1d(bin_centers, np.cumsum(normed_gauss_pdf), kind="cubic")

        # Calculate GOF 'by hand'
        ecdf = np.arange(n_samples + 1, dtype=float) / n_samples
        dn = np.abs(interp_cdf(np.sort(data)) - ecdf[:-1])

        # Calculate GOF
        gofclass = KSTestGOF(data_sample=data, pdf=normed_gauss_pdf, bin_edges=bin_edges)
        gof = gofclass.get_gof()

        self.assertEqual(max(dn), gof)


class TestKSTestTwoSampleGOF(unittest.TestCase):
    def get_ecdf(self, values, data_sample):
        """Calculate ecdf by hand."""
        n_data = len(data_sample)
        cdf = []
        for value in values:
            cdf.append(len(data_sample[data_sample <= value]))
        cdf = np.array(cdf) / n_data
        return cdf

    def test_value(self):
        """Compare result of method to manually calculated gof."""

        # Generate Test Data (simple case of n_sample=n_reference)
        n_samples = 100
        n_reference = 300
        # pseudo random data with fixed seed for reproducibility
        data = sps.norm.rvs(size=n_samples, random_state=300)
        reference = sps.norm.rvs(size=n_reference, random_state=500)

        # Calculate GOF 'by hand'
        x = np.linspace(-3, 3, 10000)
        ecdf_data = self.get_ecdf(x, data)
        ecdf_reference = self.get_ecdf(x, reference)
        dn = np.abs(ecdf_data - ecdf_reference)

        # Calculate GOF
        gofclass = KSTestTwoSampleGOF(data_sample=data, reference_sample=reference)
        gof = gofclass.get_gof()

        self.assertAlmostEqual(max(dn), gof, places=6)

    def test_symmetry(self):
        xs_a = sps.norm().rvs(50)
        xs_b = sps.norm().rvs(50)

        gofclass_ab = KSTestTwoSampleGOF(xs_a, xs_b)
        gof_ab = gofclass_ab.get_gof()
        gofclass_ba = KSTestTwoSampleGOF(xs_b, xs_a)
        gof_ba = gofclass_ba.get_gof()

        self.assertEqual(gof_ab, gof_ba)


class TestADTestTwoSampleGOF(unittest.TestCase):
    def test_symmetry(self):
        xs_a = sps.norm().rvs(50)
        xs_b = sps.norm().rvs(50)

        gofclass_ab = ADTestTwoSampleGOF(xs_a, xs_b)
        gof_ab = gofclass_ab.get_gof()
        gofclass_ba = ADTestTwoSampleGOF(xs_b, xs_a)
        gof_ba = gofclass_ba.get_gof()

        self.assertEqual(gof_ab, gof_ba)


class TestPvalues(unittest.TestCase):
    def test_two_sample_value(self):
        """Test if p-value for two identical samples is 1."""
        # Fixed Standard Normal distributed data
        data = np.array(
            [
                -0.80719796,
                0.39138662,
                0.12886947,
                -0.4383365,
                0.88404481,
                0.98167819,
                1.22302837,
                0.1138414,
                0.45974904,
                0.48926863,
            ]
        )

        gof_objects = [
            ADTestTwoSampleGOF(data, data),
            KSTestTwoSampleGOF(data, data),
            PointToPointGOF(data, data),
        ]
        d_mins = [None, None, 0.00001]

        # Ignore warning here since this is what we want to test
        warnings.filterwarnings("ignore", message="p-value is 1.*")
        n_perm = 300
        for gof_object, d_min in zip(gof_objects, d_mins):
            if d_min is not None:
                p_value = gof_object.get_pvalue(n_perm=n_perm, d_min=d_min)
            else:
                p_value = gof_object.get_pvalue(n_perm=n_perm)
            self.assertTrue(p_value > 0.97)


class TestBinnedPoissonChi2GOF(unittest.TestCase):
    def test_bin_equiprobable(self):
        """Test if from_binned and bin_equiprobable init give same result."""
        n_data = 10
        n_expected = 12
        n_partitions = 5
        data_sample = np.linspace(0, 1, n_data)
        reference_sample = np.linspace(0, 1, int(10 * n_data))

        binned_data = np.full(n_partitions, n_data / n_partitions)
        binned_reference = np.full(n_partitions, n_expected / n_partitions)
        gofclass_from_binned = BinnedPoissonChi2GOF.from_binned(binned_data, binned_reference)
        gof_from_binned = gofclass_from_binned.get_gof()

        gofclass_bin_equiprobable = BinnedPoissonChi2GOF.bin_equiprobable(
            data_sample, reference_sample, nevents_expected=n_expected, n_partitions=n_partitions
        )
        gof_bin_equiprobable = gofclass_bin_equiprobable.get_gof()
        self.assertAlmostEqual(gof_bin_equiprobable, gof_from_binned, 10)


class TestBinnedChi2GOF(unittest.TestCase):
    def test_bin_equiprobable(self):
        """Test if from_binned and bin_equiprobable init give same result."""
        n_data = 10
        n_expected = 12
        n_partitions = 5
        data_sample = np.linspace(0, 1, n_data)
        reference_sample = np.linspace(0, 1, int(10 * n_data))

        binned_data = np.full(n_partitions, n_data / n_partitions)
        binned_reference = np.full(n_partitions, n_expected / n_partitions)
        gofclass_from_binned = BinnedChi2GOF.from_binned(binned_data, binned_reference)
        gof_from_binned = gofclass_from_binned.get_gof()

        gofclass_bin_equiprobable = BinnedChi2GOF.bin_equiprobable(
            data_sample, reference_sample, nevents_expected=n_expected, n_partitions=n_partitions
        )
        gof_bin_equiprobable = gofclass_bin_equiprobable.get_gof()
        self.assertAlmostEqual(gof_bin_equiprobable, gof_from_binned, 10)


if __name__ == "__main__":
    unittest.main()
