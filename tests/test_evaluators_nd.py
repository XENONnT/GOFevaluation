import scipy.stats as sps
import numpy as np
import unittest
import warnings

from GOFevaluation import BinnedPoissonChi2GOF
from GOFevaluation import PointToPointGOF
from GOFevaluation import BinnedChi2GOF


class TestBinnedPoissonChi2GOF(unittest.TestCase):
    def test_dimensions(self):
        # test nD binned GOF in different dimensions:
        for nD in range(2, 5+1):
            # generate uniformly distributed data points and bin data
            n_events_per_bin = 30
            n_bins_per_dim = int(32**(1/nD))
            n_events = int(n_bins_per_dim**nD * n_events_per_bin)

            data_points = sps.uniform().rvs(size=[n_events, nD])
            bin_edges = np.linspace(0, 1, n_bins_per_dim+1)
            bin_edges = np.array([bin_edges for i in range(nD)])
            binned_data, _ = np.histogramdd(data_points, bins=bin_edges)
            binned_data_flat = binned_data.reshape(-1)

            # generate binned pdf
            normed_pdf = np.ones(binned_data.shape)
            normed_pdf /= np.sum(normed_pdf)
            binned_reference = normed_pdf * np.sum(binned_data)
            binned_reference_flat = binned_reference.reshape(-1)

            # calculate gof for nD and flat:
            gofclass = BinnedPoissonChi2GOF.from_binned(binned_data,
                                                        binned_reference)
            gof = gofclass.get_gof()

            gofclass_flat = BinnedPoissonChi2GOF.from_binned(
                binned_data_flat, binned_reference_flat)
            gof_flat = gofclass_flat.get_gof()

            self.assertEqual(gof_flat, gof)

    def test_from_binned(self):
        """Test if regular init and from_binned init give same result"""
        for nD in range(1, 5+1):
            # generate uniformly distributed data points and bin data
            n_events_per_bin = 30
            n_bins_per_dim = int(32**(1/nD))
            n_events = int(n_bins_per_dim**nD * n_events_per_bin)

            data_points = sps.uniform().rvs(size=[n_events, nD])
            bin_edges = np.linspace(0, 1, n_bins_per_dim+1)
            bin_edges = np.array([bin_edges for i in range(nD)])
            binned_data, _ = np.histogramdd(data_points, bins=bin_edges)

            # generate binned pdf
            normed_pdf = np.ones(binned_data.shape)
            normed_pdf /= np.sum(normed_pdf)
            binned_reference = normed_pdf * np.sum(binned_data)

            # calculate gof with both inits
            gofclass_from_classmethod = BinnedPoissonChi2GOF.from_binned(
                binned_data=binned_data, binned_reference=binned_reference)
            gof_from_binned = gofclass_from_classmethod.get_gof()

            gofclass_from_init = BinnedPoissonChi2GOF(
                data_sample=data_points,
                pdf=normed_pdf,
                bin_edges=bin_edges,
                nevents_expected=n_events)
            gof = gofclass_from_init.get_gof()

            self.assertEqual(gof, gof_from_binned)
            # ensure that no matter what you use for creating the object keys
            # are the same
            self.assertEqual(sorted(gofclass_from_classmethod.__dict__.keys()),
                             sorted(gofclass_from_init.__dict__.keys()))


class TestPointToPointGOF(unittest.TestCase):
    def test_distances(self):
        # test if number of distance values is correct
        xs = np.linspace(0, 1, 100)
        xs_ref = np.linspace(0, 1, 200)
        for nD in [1, 2, 3, 5]:
            data = np.vstack([xs for i in range(nD)]).T
            reference = np.vstack([xs_ref for i in range(nD)]).T

            nevents_data = len(data)
            nevents_ref = len(reference)

            gofclass = PointToPointGOF(data, reference)
            d_data_data, d_ref_ref, d_data_ref = gofclass.get_distances(
                data, reference)

            self.assertEqual(len(d_data_data), nevents_data *
                             (nevents_data-1) / 2)
            self.assertEqual(len(d_ref_ref), nevents_ref *
                             (nevents_ref-1) / 2)
            self.assertEqual(len(d_data_ref), nevents_ref *
                             nevents_data)

    def test_symmetry(self):
        # the pointwise energy test is symmetrical in reference and
        # science sample:
        xs_a = sps.uniform().rvs(50)[:, None]
        xs_b = sps.uniform().rvs(50)[:, None]
        gofclass_ab = PointToPointGOF(xs_a, xs_b)
        # set d_min explicitly to avoid asymmetry in setting d_min
        gof_ab = gofclass_ab.get_gof(d_min=0.01)
        gofclass_ba = PointToPointGOF(xs_b, xs_a)
        # set d_min explicitly to avoid asymmetry in setting d_min
        gof_ba = gofclass_ba.get_gof(d_min=0.01)

        # it seems precision is a bit low in this case
        self.assertAlmostEqual(gof_ab, gof_ba, places=6)

    def test_value(self):
        # simple values:
        xs_a = np.array([0])[:, None]
        xs_b = np.array([1, 2])[:, None]

        e_data_ref = np.log(2)/2
        gofclass_ab = PointToPointGOF(xs_a, xs_b)
        # set d_min explicitly to avoid asymmetry in setting d_min
        gof_ab = gofclass_ab.get_gof(d_min=0.01)
        # it seems precision is a bit low in this case
        self.assertAlmostEqual(gof_ab, e_data_ref, places=6)


class TestBinnedChi2GOF(unittest.TestCase):

    def test_dimensions(self):
        # test nD binned GOF in different dimensions:
        for nD in range(2, 5+1):
            # generate uniformly distributed data points and bin data
            n_events_per_bin = 30
            n_bins_per_dim = int(32**(1/nD))
            n_events = int(n_bins_per_dim**nD * n_events_per_bin)

            data_points = sps.uniform().rvs(size=[n_events, nD])
            bin_edges = np.linspace(0, 1, n_bins_per_dim+1)
            bin_edges = np.array([bin_edges for i in range(nD)])
            binned_data, _ = np.histogramdd(data_points, bins=bin_edges)
            binned_data_flat = binned_data.reshape(-1)

            # generate binned pdf
            normed_pdf = np.ones(binned_data.shape)
            normed_pdf /= np.sum(normed_pdf)
            binned_reference = normed_pdf * np.sum(binned_data)
            binned_reference_flat = binned_reference.reshape(-1)

            # calculate gof for nD and flat:
            gofclass = BinnedChi2GOF.from_binned(binned_data,
                                                 binned_reference)
            gof = gofclass.get_gof()

            gofclass_flat = BinnedChi2GOF.from_binned(
                binned_data_flat, binned_reference_flat)
            gof_flat = gofclass_flat.get_gof()

            self.assertEqual(gof_flat, gof)

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
                binned_reference = normed_pdf * np.sum(binned_data)

                gofclass = BinnedChi2GOF.from_binned(
                    binned_data=binned_data, binned_reference=binned_reference)
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
        """Test if regular init and from_binned init give same result"""
        for nD in range(1, 5+1):
            # generate uniformly distributed data points and bibn data
            n_events_per_bin = 15
            n_bins_per_dim = int(32**(1/nD))
            n_events = int(n_bins_per_dim**nD * n_events_per_bin)

            data_points = sps.uniform().rvs(size=[n_events, nD])
            bin_edges = np.linspace(0, 1, n_bins_per_dim+1)
            bin_edges = np.array([bin_edges for i in range(nD)])
            binned_data, _ = np.histogramdd(data_points, bins=bin_edges)

            # generate binned pdf
            normed_pdf = np.ones(binned_data.shape)
            normed_pdf /= np.sum(normed_pdf)
            binned_reference = normed_pdf * np.sum(binned_data)

            # calculate gof with both inits
            gofclass_from_classmethod = BinnedChi2GOF.from_binned(
                binned_data=binned_data, binned_reference=binned_reference)
            gof_from_binned = gofclass_from_classmethod.get_gof()

            gofclass_from_init = BinnedChi2GOF(data_sample=data_points,
                                               pdf=normed_pdf,
                                               bin_edges=bin_edges,
                                               nevents_expected=n_events)
            gof = gofclass_from_init.get_gof()

            self.assertEqual(gof, gof_from_binned)

            # ensure that no matter what you use for creating the object keys
            # are the same
            self.assertEqual(sorted(gofclass_from_classmethod.__dict__.keys()),
                             sorted(gofclass_from_init.__dict__.keys()))


class TestPvalue(unittest.TestCase):
    def test_dimension_two_sample(self):
        """Test if p-value for two identical samples is 1
        for higher dimensional samples."""
        d_min = .00001
        for nD in [2, 3, 4]:
            # Fixed Standard Normal distributed data
            data = np.array([-0.80719796,  0.39138662,  0.12886947, -0.4383365,
                             0.88404481, 0.98167819,  1.22302837,  0.1138414,
                             0.45974904,  0.48926863])
            data = np.vstack([data for i in range(nD)]).T
            gof_object = PointToPointGOF(data, data)

            # Ignore warning here since this is what we want to test
            warnings.filterwarnings("ignore", message="p-value is set to 1.*")
            n_perm = 300
            p_value = gof_object.get_pvalue(n_perm=n_perm,
                                            d_min=d_min)

            self.assertEqual(p_value, 1)

    def test_value(self):
        """Test for 1D if binned_data = binned_reference gives p-value
        of one."""
        n_bins = 3
        n_events_per_bin = 30

        data = np.ones(n_bins) * n_events_per_bin

        gof_objects = [BinnedChi2GOF.from_binned(data, data),
                       BinnedPoissonChi2GOF.from_binned(data, data)]

        # Ignore warning here since this is what we want to test
        warnings.filterwarnings("ignore", message="p-value is set to 1.*")
        n_mc = 400
        for gof_object in gof_objects:
            p_value = gof_object.get_pvalue(n_mc=n_mc)
            self.assertEqual(p_value, 1)


if __name__ == "__main__":
    unittest.main()
