# import scipy.stats as sps
import numpy as np
import unittest

from GOFevaluation import equiprobable_histogram
from GOFevaluation import apply_irregular_binning
from GOFevaluation import _get_finite_bin_edges
from GOFevaluation import _get_count_density


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

    def test__get_finite_bin_edges(self):
        '''Test if get_finite_bin_edges in fact gives the bin edges 
        that effectively contain all of the data without being
        infinite'''
        n_data = 12
        n_partitions = [2, 3]
        data_sample = np.linspace(0, 1, n_data)
        data_sample = np.vstack([data_sample for i in range(2)]).T
        reference_sample = np.linspace(0, 1, int(10 * n_data))
        reference_sample = np.vstack([reference_sample for i in range(2)]).T
        edges = []
        for order in [[0, 1], [1, 0]]:
            n, bin_edges = equiprobable_histogram(data_sample=data_sample,
                                                  reference_sample=reference_sample,
                                                  n_partitions=n_partitions,
                                                  order=order)
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
        '''Test if _get_count_density can correctly count the density
        of data points in 2D  for both orderings'''
        n_data = 12
        n_partitions = [2, 3]
        data_sample = np.linspace(0, 1, n_data)
        data_sample = np.vstack([data_sample for i in range(2)]).T
        reference_sample = np.linspace(0, 1, int(10 * n_data))
        reference_sample = np.vstack([reference_sample for i in range(2)]).T
        edges = []
        for order in [[0, 1], [1, 0]]:
            n, bin_edges = equiprobable_histogram(data_sample=data_sample,
                                                  reference_sample=reference_sample,
                                                  n_partitions=n_partitions,
                                                  order=order)
            edges = _get_finite_bin_edges(bin_edges, data_sample, order)
            count_density = _get_count_density(n.copy(), edges[0], edges[1], data_sample)
            self.assertEqual(np.shape(n), np.shape(count_density))
            testarray0=np.array([[24.203389830508474, 24.203389830508474, 5.974895397489539], [5.97489539748954, 24.203389830508463, 24.203389830508478]])
            testarray1=np.array([[36.61538461538461,7.175879396984923],[11.999999999999998, 11.999999999999998],[7.175879396984927,36.615384615384606]])

            for i in range(0, len(edges[0]) - 1):
                for j in range(0, len(edges[1][i]) - 1):
                    if(order==[0,1]):
                        self.assertEqual(testarray0[i][j],count_density[i][j])
                    elif(order==[1,0]):
                        self.assertEqual(testarray1[i][j],count_density[i][j])
