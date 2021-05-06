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
        if len(self.data.shape)==1:
            self.binned_data, _ = np.histogram(self.data, bins=bin_edges)
        else:
            self.binned_data, _ = np.histogramdd(self.data, bins=bin_edges)


class binned_poisson_chi2_gof(nd_test_statistics):
    """
        computes the binned poisson modified Chi2 from Baker+Cousins
        In the limit of large bin counts (10+) this is Chi2 distributed.
        In the general case you may have to toyMC the distribution yourself.
        Input:
         - data: array of equal shape as nevents_expected
             containing observed events in each bin
         - pdf: normalized histogram of binned expectations
         - bin_edges: bin-edges of the pdf
         - nevents_expected: expectation, can be mean of expectation pdf
        Output:
         - the sum of the binned poisson Chi2.
        reference: https://doi.org/10.1016/0167-5087(84)90016-4
        While the absolute likelihood is a poor GOF measure
        (see http://www.physics.ucla.edu/~cousins/stats/cousins_saturated.pdf)
    """
    @classmethod
    def from_binned(cls, data, expectations):
        """Initialize with already binned data + expectations

        In this case the bin-edges don't matter, so we bypass the usual init
        """
        self = cls(None, None, None, None)
        nd_test_statistics.__init__(self=self,
                                    data=data,
                                    pdf=expectations / np.sum(expectations),
                                    nevents_expected=np.sum(expectations))
        self._name = self.__class__.__name__
        self.binned_data = data
        return self

    def __init__(self, data, pdf, bin_edges, nevents_expected):
        """Initialize with unbinned data and a normalized pdf
        """
        if data is None:
            # bypass init, using binned data
            return
        # initialise with the common call signature
        nd_test_statistics.__init__(self=self,
                                    data=data,
                                    pdf=pdf,
                                    nevents_expected=nevents_expected)
        self._name = self.__class__.__name__

        self.bin_edges = bin_edges
        self.bin_data(bin_edges=bin_edges)
        return

    @classmethod
    def calculate_binned_gof(cls, binned_data, binned_expectations):
        """Get Chi2 GoF from binned data & expectations
        """
        ret = sps.poisson(binned_data).logpmf(binned_data)
        ret -= sps.poisson(binned_expectations).logpmf(binned_data)
        return 2 * np.sum(ret)

    def calculate_gof(self):
        """Get Chi2 GoF using current class attributes
        """
        gof = binned_poisson_chi2_gof.calculate_binned_gof(
            self.binned_data,
            self.pdf * self.nevents_expected
        )
        return gof

    def sample_gofs(self, n_mc=1000):
        """Sample n_mc random Chi2 GoF's

        Simulates random data from the PDF and calculates its GoF n_mc times
        """
        fake_gofs = np.zeros(n_mc)
        for i in range(n_mc):
            samples = sps.poisson(self.pdf * self.nevents_expected).rvs()
            fake_gofs[i] = binned_poisson_chi2_gof.calculate_binned_gof(
                samples,
                self.pdf * self.nevents_expected, 
            )
        return fake_gofs 

    def get_pvalue(self, n_mc=1000):
        """Get the p-value of the data under the null hypothesis

        Gets the distribution of the GoF statistic, and compares it to the
        GoF of the data given the expectations.
        """
        gof = self.calculate_gof()
        fake_gofs = self.sample_gofs(n_mc=n_mc)
        hist, bin_edges = np.histogram(fake_gofs, bins=1000)
        cumulative_density = 1.0 - np.cumsum(hist)/np.sum(hist)
        bin_centers = np.mean(np.vstack([bin_edges[0:-1], bin_edges[1:]]), axis=0)
        try:
            pvalue = cumulative_density[np.digitize(gof, bin_edges)-1]
        except IndexError:
            raise ValueError('Not enough MC\'s run -- GoF is outside toy distribution!')
        return pvalue
