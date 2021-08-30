# import scipy.stats as sps
import numpy as np
import unittest

from GOFevaluation import equiprobable_histogram


class TestEqpb(unittest.TestCase):
    def test_eqpb_1d(self):
        '''Test if 1d eqpb in fact gives an equiprobable binning
        that conserves the number of data points and has the correct
        number of bin edges.'''
        n_data = 10
        n_partitions = 5
        data_sample = np.linspace(0, 1, n_data)
        reference_sample = np.linspace(0, 1, int(10 * n_data))
        n, bin_edges = equiprobable_histogram(data_sample=data_sample,
                                              reference_sample=reference_sample,
                                              n_partitions=n_partitions)
        self.assertEqual(np.sum(n), n_data)
        for expect in n:
            self.assertEqual(expect, n_data / n_partitions)
        self.assertEqual(len(bin_edges) - 1, n_partitions)
        self.assertEqual(len(n), n_partitions)

    def test_eqpb_2d(self):
        '''Test if 2d eqpb in fact gives an equiprobable binning
        that conserves the number of data points and has the correct
        number of bin edges. Test this for both orderings.'''
        n_data = 12
        n_partitions = [2, 3]
        data_sample = np.linspace(0, 1, n_data)
        data_sample = np.vstack([data_sample for i in range(2)]).T
        reference_sample = np.linspace(0, 1, int(10 * n_data))
        reference_sample = np.vstack([reference_sample for i in range(2)]).T

        for order in [[0, 1], [1, 0]]:
            n, bin_edges = equiprobable_histogram(data_sample=data_sample,
                                                  reference_sample=reference_sample,
                                                  n_partitions=n_partitions,
                                                  order=order)
            self.assertEqual(np.sum(n), n_data)
            for expect in n.reshape(-1):
                self.assertEqual(expect, n_data / np.product(n_partitions))
            self.assertEqual(
                np.shape(bin_edges[0])[0] - 1, n_partitions[order[0]])
            self.assertEqual(np.shape(bin_edges[1])[0], n_partitions[order[0]])
            self.assertEqual(
                np.shape(bin_edges[1])[1] - 1, n_partitions[order[1]])
            self.assertEqual(list(np.shape(n)),
                             [n_partitions[order[0]], n_partitions[order[1]]])
