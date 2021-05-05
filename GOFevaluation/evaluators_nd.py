import scipy.stats as sps
import numpy as np
from collections import OrderedDict
from scipy.interpolate import interp1d
from GOFevaluation import test_statistics


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
        #initialise with the common call signature
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
