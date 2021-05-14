import scipy.stats as sps
import numpy as np
import unittest

from GOFevaluation import binned_poisson_chi2_gof
from GOFevaluation import point_to_point_gof
from GOFevaluation import binned_chi2_gof


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
            gof = gofclass.get_gof()

            self.assertLess(abs(gof_flat - gof), 1e-8)

    def test_from_binned(self):
        """Test if regular init and frtom_binned init give same result"""
        for nD in range(1, 5+1):
            # generate uniformly distributed data points and bibn data
            n_events_per_bin = 5
            n_bins_per_dim = int(32**(1/nD))
            n_events = int(n_bins_per_dim**nD * n_events_per_bin)

            data_points = sps.uniform().rvs(size=[n_events, nD])
            bin_edges = np.linspace(0, 1, n_bins_per_dim+1)
            bin_edges = np.array([bin_edges for i in range(nD)])
            binned_data, _ = np.histogramdd(data_points, bins=bin_edges)

            # generate binned pdf
            normed_pdf = np.ones(binned_data.shape)
            normed_pdf /= np.sum(normed_pdf)
            expected_events = normed_pdf * np.sum(binned_data)

            # calculate gof with both inits
            gofclass = binned_poisson_chi2_gof.from_binned(
                data=binned_data, expectations=expected_events)
            gof_from_binned = gofclass.get_gof()

            gofclass = binned_poisson_chi2_gof(data=data_points,
                                               pdf=normed_pdf,
                                               bin_edges=bin_edges,
                                               nevents_expected=n_events)
            gof = gofclass.get_gof()

            self.assertEqual(gof, gof_from_binned)


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
        # the pointwise energy test is symmetrical in reference and
        # science sample:
        xs_a = sps.uniform().rvs(50)[:, None]
        xs_b = sps.uniform().rvs(50)[:, None]
        gofclass_ab = point_to_point_gof(xs_a, xs_b)
        # set d_min explicitly to avoid asymmetry in setting d_min
        gof_ab = gofclass_ab.get_gof(d_min=0.01)
        gofclass_ba = point_to_point_gof(xs_b, xs_a)
        # set d_min explicitly to avoid asymmetry in setting d_min
        gof_ba = gofclass_ba.get_gof(d_min=0.01)

        # it seems precision is a bit low in this case
        self.assertAlmostEqual(gof_ab, gof_ba, places=6)

    def test_value(self):
        # simple values:
        xs_a = np.array([0])[:, None]
        xs_b = np.array([1, 2])[:, None]

        e_data_ref = np.log(2)/2
        gofclass_ab = point_to_point_gof(xs_a, xs_b)
        # set d_min explicitly to avoid asymmetry in setting d_min
        gof_ab = gofclass_ab.get_gof(d_min=0.01)
        # it seems precision is a bit low in this case
        self.assertAlmostEqual(gof_ab, e_data_ref, places=6)


class Test_binned_chi2_gof(unittest.TestCase):
    def test_chi2_distribution(self):
        """check, if binned data follows chi2-distribution with
        ndof = n_bins - 1 as one would expect. Test for 1-5 dimensions."""

        n_testvalues = 100
        model = sps.uniform()
        for nD in range(1, 5+1):
            # have same number of events per bin and total number
            # of bins for all tests
            n_events_per_bin = 20
            n_bins_per_dim = int(32**(1/nD))
            n_events = int(n_bins_per_dim**nD * n_events_per_bin)

            bin_edges = np.linspace(0, 1, n_bins_per_dim+1)
            bin_edges = np.array([bin_edges for i in range(nD)])

            chi2_vals = []
            for i in range(n_testvalues):
                # generate uniformly distributed rvs with fixed random
                # states for reproducibility
                data_points = model.rvs(
                    size=[n_events, nD], random_state=300+i)
                binned_data, _ = np.histogramdd(data_points, bins=bin_edges)

                normed_pdf = np.ones(binned_data.shape)
                normed_pdf /= np.sum(normed_pdf)
                expected_events = normed_pdf * np.sum(binned_data)

                gofclass = binned_chi2_gof.from_binned(
                    data=binned_data, expectations=expected_events)
                chi2_val = gofclass.get_gof()
                chi2_vals.append(chi2_val)

            ndof = n_bins_per_dim**nD - 1

            # compare histogram of chi2s to expected chi2(ndof) distribution:
            n_chi2_bins = 20
            n, bin_edges = np.histogram(chi2_vals, bins=n_chi2_bins,
                                        range=(np.quantile(chi2_vals, .01),
                                               np.quantile(chi2_vals, .99)))
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            bin_widths = bin_edges[1:] - bin_edges[:-1]
            chi2_pdf = sps.chi2.pdf(bin_centers, df=ndof) * \
                bin_widths * n_testvalues

            # calculate 'reduced chi2' to estimate agreement of chi2 values
            # and chi2 pdf
            test_chi2 = np.sum((chi2_pdf-n)**2 / chi2_pdf)/n_chi2_bins
            self.assertTrue((test_chi2 > 1/3) & (test_chi2 < 3))

    def test_from_binned(self):
        """Test if regular init and frtom_binned init give same result"""
        for nD in range(1, 5+1):
            # generate uniformly distributed data points and bibn data
            n_events_per_bin = 5
            n_bins_per_dim = int(32**(1/nD))
            n_events = int(n_bins_per_dim**nD * n_events_per_bin)

            data_points = sps.uniform().rvs(size=[n_events, nD])
            bin_edges = np.linspace(0, 1, n_bins_per_dim+1)
            bin_edges = np.array([bin_edges for i in range(nD)])
            binned_data, _ = np.histogramdd(data_points, bins=bin_edges)

            # generate binned pdf
            normed_pdf = np.ones(binned_data.shape)
            normed_pdf /= np.sum(normed_pdf)
            expected_events = normed_pdf * np.sum(binned_data)

            # calculate gof with both inits
            gofclass = binned_chi2_gof.from_binned(
                data=binned_data, expectations=expected_events)
            gof_from_binned = gofclass.get_gof()

            gofclass = binned_chi2_gof(data=data_points,
                                       pdf=normed_pdf,
                                       bin_edges=bin_edges,
                                       nevents_expected=n_events)
            gof = gofclass.get_gof()

            self.assertEqual(gof, gof_from_binned)


if __name__ == "__main__":
    unittest.main()
