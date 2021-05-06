import scipy.stats as sps
import numpy as np
from collections import OrderedDict
from scipy.interpolate import interp1d
from sklearn.neighbors import DistanceMetric
from GOFevaluation import test_statistics
from GOFevaluation import test_statistics_sample


class nd_test_statistics(test_statistics):
    """
        Override binning to work in arbitrary dimensions
    """

    def bin_data(self, bin_edges):
        # function to bin nD data:
        self.binned_data, _ = np.histogramdd(self.data, bins=bin_edges)


class binned_poisson_chi2_gof(nd_test_statistics):
    """
        computes the binned poisson modified Chi2 from Baker+Cousins
        In the limit of large bin counts (10+) this is Chi2 distributed.
        In the general case you may have to toyMC the distribution yourself.
        Input:
         - nevents_expected: array with expected # of events (not rates) for each bin
         - data: array of equal shape as nevents_expected
             containing observed events in each bin
        Output:
         - the sum of the binned poisson Chi2.
        reference: https://doi.org/10.1016/0167-5087(84)90016-4
        While the absolute likelihood is a poor GOF measure
        (see http://www.physics.ucla.edu/~cousins/stats/cousins_saturated.pdf)
    """

    def __minimalinit__(self, data, expectations):
        # initialise with already binned data + expectations (typical case):
        nd_test_statistics.__init__(self=self,
                                    data=data,
                                    pdf=expectations / np.sum(expectations),
                                    nevents_expected=np.sum(expectations))
        self._name = self.__class__.__name__
        self.binned_data = data

    def __init__(self, data, pdf, bin_edges, nevents_expected):
        # initialise with the common call signature
        nd_test_statistics.__init__(self=self,
                                    data=data,
                                    pdf=pdf,
                                    nevents_expected=nevents_expected)
        self._name = self.__class__.__name__

        self.bin_edges = bin_edges
        self.bin_data(bin_edges=bin_edges)

    def calculate_gof(self):
        ret = sps.poisson(self.binned_data).logpmf(self.binned_data)
        ret -= sps.poisson(self.pdf * self.nevents_expected).logpmf(
            self.binned_data)
        return 2 * np.sum(ret)


class binned_chi2_gof(nd_test_statistics):
    # TODO Implement!
    pass


class point_to_point_gof(test_statistics_sample):
    """computes point-to-point gof as described in 
    https://arxiv.org/abs/hep-ex/0203010.
    Input:
    - nevents_data n-dim data samples with shape (nevents_data, n)
    - nevents_reference n-dim referencesamples with shape (nevents_ref, n)
    Output:
    Test Statistic based on 'Statisticsl Energy' """

    def __init__(self, data, reference_sample):
        test_statistics_sample.__init__(self=self,
                                        data=data,
                                        reference_sample=reference_sample)
        self._name = self.__class__.__name__
        self.nevents_data = np.shape(self.data)[0]
        self.nevents_ref = np.shape(self.reference_sample)[0]

    def get_distances(self):
        """get distances of data-data, reference-reference 
        and data-reference"""
        # TODO: What about scaling of different units/orders of magnitude?
        # Calculate distance matrices for tthe two samples
        dist = DistanceMetric.get_metric('euclidean')

        d_data_data = np.triu(dist.pairwise(self.data))
        d_data_data.reshape(-1)
        self.d_data_data = d_data_data[d_data_data > 0]

        d_ref_ref = np.triu(dist.pairwise(self.reference_sample))
        d_ref_ref.reshape(-1)
        self.d_ref_ref = d_ref_ref[d_ref_ref > 0]

        self.d_data_ref = dist.pairwise(
            self.data, self.reference_sample).reshape(-1)

    def calculate_gof(self):
        self.get_distances()
        ret_data_data = 1/self.nevents_data**2 * \
            np.sum(-np.log(self.d_data_data))
        ret_ref_ref = 1/self.nevents_ref**2 * np.sum(-np.log(self.d_ref_ref))
        ret_data_ref = -1/self.nevents_ref/self.nevents_data * \
            np.sum(-np.log(self.d_data_ref))
        ret = ret_data_data + ret_ref_ref + ret_data_ref
        return ret
