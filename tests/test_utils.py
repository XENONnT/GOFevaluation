# import scipy.stats as sps
import numpy as np
import unittest
import warnings

from GOFevaluation.utils import equiprobable_histogram
from GOFevaluation.utils import _get_finite_bin_edges
from GOFevaluation.utils import _get_count_density
from GOFevaluation.utils import check_for_ties
from GOFevaluation.utils import check_dimensionality_for_eqpb


class TestEqpb(unittest.TestCase):
    def test_eqpb_1d(self):
        """Test if 1d eqpb in fact gives an equiprobable binning that conserves the
        number of data points and has the correct number of bin edges."""
        n_data = 10
        n_partitions = 5
        data_sample = np.linspace(0, 1, n_data)
        reference_sample = np.linspace(0, 1, int(10 * n_data))
        n, bin_edges = equiprobable_histogram(
            data_sample=data_sample, reference_sample=reference_sample, n_partitions=n_partitions
        )
        self.assertEqual(np.sum(n), n_data)
        for expect in n:
            self.assertEqual(expect, n_data / n_partitions)
        self.assertEqual(len(bin_edges) - 1, n_partitions)
        self.assertEqual(len(n), n_partitions)

    def test_eqpb_2d_lin(self):
        """Test if 2d eqpb in fact gives an equiprobable binning that conserves the
        number of data points and has the correct number of bin edges.

        Weights are in uniform histogram.

        """
        n_data = 12
        n_partitions = [2, 3]
        data_sample = np.linspace(0, 1, n_data)
        reference_sample = np.linspace(0, 1, int(10 * n_data))
        reference_sample_weights_l = [None, np.ones(10 * n_data)]
        data_sample = np.vstack([data_sample for i in range(2)]).T
        reference_sample = np.vstack([reference_sample for i in range(2)]).T

        for reference_sample_weights in reference_sample_weights_l:
            self.eqpb_2d(
                data_sample, reference_sample, n_partitions, reference_sample_weights, n_data
            )

    def test_eqpb_2d_log(self):
        """Test if 2d eqpb in fact gives an equiprobable binning that conserves the
        number of data points and has the correct number of bin edges.

        Weights are in log-like histogram.

        """
        n_data = 12
        n_partitions = [2, 3]
        data_sample = np.logspace(0, 1, n_data)
        reference_sample = np.linspace(1, 10, int(10 * n_data))
        weights = np.log(np.linspace(1, 10, int(10 * n_data)))
        reference_sample_weights = np.hstack([0, np.diff(weights)])
        reference_sample_weights_l = [reference_sample_weights]
        data_sample = np.vstack([data_sample for i in range(2)]).T
        reference_sample = np.vstack([reference_sample for i in range(2)]).T

        for reference_sample_weights in reference_sample_weights_l:
            self.eqpb_2d(
                data_sample, reference_sample, n_partitions, reference_sample_weights, n_data
            )

    def eqpb_2d(
        self, data_sample, reference_sample, n_partitions, reference_sample_weights, n_data
    ):
        """Test 2d equiprobable binning.

        Test this for both orderings.

        """
        for order in [[0, 1], [1, 0]]:
            n, bin_edges = equiprobable_histogram(
                data_sample=data_sample,
                reference_sample=reference_sample,
                n_partitions=n_partitions,
                order=order,
                reference_sample_weights=reference_sample_weights,
            )
            self.assertEqual(np.sum(n), n_data)
            for expect in n.reshape(-1):
                self.assertEqual(expect, n_data / np.prod(n_partitions))
            self.assertEqual(np.shape(bin_edges[0])[0] - 1, n_partitions[order[0]])
            self.assertEqual(np.shape(bin_edges[1])[0], n_partitions[order[0]])
            self.assertEqual(np.shape(bin_edges[1])[1] - 1, n_partitions[order[1]])
            self.assertEqual(list(np.shape(n)), [n_partitions[order[0]], n_partitions[order[1]]])

    def test_eqpb_2d_none(self):
        """Test if equiprobable binning with weights=None and weights=np.ones give the
        same result."""
        n_data = 12
        n_partitions = [2, 3]
        data_sample = np.linspace(0, 1, n_data)
        reference_sample = np.linspace(0, 1, int(10 * n_data))
        data_sample = np.vstack([data_sample for i in range(2)]).T
        reference_sample = np.vstack([reference_sample for i in range(2)]).T
        order = [0, 1]

        _, bin_edges_none = equiprobable_histogram(
            data_sample=data_sample,
            reference_sample=reference_sample,
            n_partitions=n_partitions,
            order=order,
            reference_sample_weights=None,
        )

        _, bin_edges_ones = equiprobable_histogram(
            data_sample=data_sample,
            reference_sample=reference_sample,
            n_partitions=n_partitions,
            order=order,
            reference_sample_weights=np.ones(10 * n_data),
        )

        self.assertEqual(len(bin_edges_none[0]), len(bin_edges_ones[0]))
        for i in range(bin_edges_none[0].shape[0]):
            self.assertAlmostEqual(bin_edges_none[0][i], bin_edges_ones[0][i], places=12)
        for i in range(bin_edges_none[1].shape[0]):
            for j in range(bin_edges_none[1].shape[1]):
                self.assertAlmostEqual(bin_edges_none[1][i][j], bin_edges_ones[1][i][j], places=12)

    def test__get_finite_bin_edges(self):
        """Test if get_finite_bin_edges in fact gives the bin edges that effectively
        contain all of the data without being infinite."""
        n_data = 12
        n_partitions = [2, 3]
        data_sample = np.linspace(0, 1, n_data)
        data_sample = np.vstack([data_sample for i in range(2)]).T
        reference_sample = np.linspace(0, 1, int(10 * n_data))
        reference_sample = np.vstack([reference_sample for i in range(2)]).T
        edges = []
        for order in [[0, 1], [1, 0]]:
            n, bin_edges = equiprobable_histogram(
                data_sample=data_sample,
                reference_sample=reference_sample,
                n_partitions=n_partitions,
                order=order,
            )
            edges = _get_finite_bin_edges(bin_edges, data_sample, order)
            self.assertEqual(edges[0][-1], max(data_sample[:, 1]))
            self.assertEqual(edges[0][0], min(data_sample[:, 1]))
            for i in range(1, len(edges[0]) - 1):
                self.assertEqual(edges[0][i], bin_edges[0][i])

            for i in range(0, len(edges[1])):
                self.assertEqual(edges[1][i][-1], max(data_sample[:, 0]))
                self.assertEqual(edges[1][i][0], min(data_sample[:, 0]))
                for j in range(1, len(edges[1][i]) - 1):
                    self.assertEqual(edges[1][i][j], bin_edges[1][i][j])

    def test__get_count_density(self):
        """Test if _get_count_density can correctly count the density of data points in
        2D  for both orderings."""
        # Define data that is equally spaced on a 6x6 grid.
        # Binning 36 data points with a 3x2 partitions you get a
        # count density of 1 in each bin.
        sqrt_n_data = 6
        n_partitions = [2, 3]

        x = np.linspace(0, sqrt_n_data, sqrt_n_data)
        xx, yy = np.meshgrid(x, x)
        data_sample = np.vstack([xx.reshape(-1), yy.reshape(-1)]).T

        x = np.linspace(0, sqrt_n_data, int(sqrt_n_data * 10))
        xx, yy = np.meshgrid(x, x)
        reference_sample = np.vstack([xx.reshape(-1), yy.reshape(-1)]).T

        edges = []
        for order in [[0, 1], [1, 0]]:
            n, bin_edges = equiprobable_histogram(
                data_sample=data_sample,
                reference_sample=reference_sample,
                n_partitions=n_partitions,
                order=order,
            )
            edges = _get_finite_bin_edges(bin_edges, data_sample, order)
            count_density = _get_count_density(n.copy(), edges[0], edges[1], data_sample)
            self.assertEqual(np.shape(n), np.shape(count_density))
            for i in range(0, len(edges[0]) - 1):
                for j in range(0, len(edges[1][i]) - 1):
                    if order == [0, 1]:
                        self.assertAlmostEqual(1, count_density[i][j], places=12)
                    elif order == [1, 0]:
                        self.assertAlmostEqual(1, count_density[i][j], places=12)

    def test_check_for_ties_1d(self):
        a = np.linspace(0, 1, 10)

        sample = np.concatenate((a, a))
        warning_raised = False
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            try:
                check_for_ties(sample, dim=1)
            except Warning:
                warning_raised = True
        self.assertTrue(warning_raised)

        sample = a
        warning_raised = False
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            try:
                check_for_ties(sample, dim=1)
            except Warning:
                warning_raised = True
        self.assertFalse(warning_raised)

    def test_check_for_ties_2d(self):
        a = np.vstack((np.linspace(0, 1, 10), np.linspace(0, 1, 10))).T

        sample = np.concatenate((a, a))
        warning_raised = False
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            try:
                check_for_ties(sample, dim=2)
            except Warning:
                warning_raised = True
        self.assertTrue(warning_raised)

        sample = a
        warning_raised = False
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            try:
                check_for_ties(sample, dim=2)
            except Warning:
                warning_raised = True
        self.assertFalse(warning_raised)

    def test_check_dimensionality_for_eqpb_1d(self):
        data_sample = np.linspace(0, 1, 10)
        reference_sample = np.linspace(0, 1, 20)
        n_partitions = 4
        order = None

        error_raised = False
        try:
            check_dimensionality_for_eqpb(
                data_sample=data_sample,
                reference_sample=reference_sample,
                n_partitions=n_partitions,
                order=order,
            )
        except AssertionError:
            error_raised = True
        self.assertFalse(error_raised)

        error_raised = False
        reference_sample = np.vstack((np.linspace(0, 1, 20), np.linspace(0, 1, 20))).T
        try:
            check_dimensionality_for_eqpb(
                data_sample=data_sample,
                reference_sample=reference_sample,
                n_partitions=n_partitions,
                order=order,
            )
        except AssertionError:
            error_raised = True
        self.assertTrue(error_raised)

    def test_check_dimensionality_for_eqpb_2d(self):
        data_sample = np.vstack((np.linspace(0, 1, 10), np.linspace(0, 1, 10))).T
        reference_sample = np.vstack((np.linspace(0, 1, 20), np.linspace(0, 1, 20))).T
        n_partitions = [2, 2]
        order = None

        error_raised = False
        try:
            check_dimensionality_for_eqpb(
                data_sample=data_sample,
                reference_sample=reference_sample,
                n_partitions=n_partitions,
                order=order,
            )
        except AssertionError:
            error_raised = True
        self.assertFalse(error_raised)

        error_raised = False
        reference_sample = np.vstack(
            (np.linspace(0, 1, 20), np.linspace(0, 1, 20), np.linspace(0, 1, 20))
        ).T
        try:
            check_dimensionality_for_eqpb(
                data_sample=data_sample,
                reference_sample=reference_sample,
                n_partitions=n_partitions,
                order=order,
            )
        except AssertionError:
            error_raised = True
        self.assertTrue(error_raised)
