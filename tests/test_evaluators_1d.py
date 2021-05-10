import scipy.stats as sps
import numpy as np
from scipy.interpolate import interp1d
import unittest

from GOFevaluation import kstest_gof
# from GOFevaluation import kstest_two_sample_gof


class Test_kstest_gof(unittest.TestCase):
    def test_value(self):
        """compare result of method to manually calculated gof"""

        # Generate Test Data
        n_samples = 100
        # pseudo random data with fixed seed for reproducibility
        data = sps.norm.rvs(size=n_samples, random_state=300)

        bin_edges = np.linspace(-4, 4, 101)
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
        bin_widths = bin_edges[1:] - bin_edges[:-1]

        normed_gauss_pdf = sps.norm.pdf(bin_centers) * bin_widths
        interp_cdf = interp1d(bin_centers, np.cumsum(normed_gauss_pdf),
                              kind='cubic')

        # Calculate GoF 'by hand'
        ecdf = np.arange(n_samples+1, dtype=float)/n_samples
        dn = np.abs(interp_cdf(np.sort(data)) - ecdf[:-1])

        # Calculate GoF
        gofclass = kstest_gof(data=data,
                              pdf=normed_gauss_pdf,
                              bin_edges=bin_edges)
        gof = gofclass.calculate_gof()

        self.assertEqual(max(dn), gof)


if __name__ == "__main__":
    unittest.main()
