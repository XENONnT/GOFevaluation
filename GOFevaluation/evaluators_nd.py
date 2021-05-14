import scipy.stats as sps
import numpy as np
from sklearn.neighbors import DistanceMetric
from GOFevaluation import test_statistics
from GOFevaluation import test_statistics_sample


class binned_poisson_chi2_gof(test_statistics):
    """
        Computes the binned poisson modified Chi2 from Baker+Cousins
        In the limit of large bin counts (10+) this is Chi2 distributed.
        In the general case you may have to toyMC the distribution yourself.

        Input (unbinned data):
        - data: array of unbinned data
        - pdf: normalized histogram of binned expectations
        - bin_edges: bin-edges of the pdf
        - nevents_expected: expectation, can be mean of expectation pdf

        Input (binned data):
        initialise with binned_poisson_chi2_gof.from_binned(...)
        - data: array of binned data
        - expectations: array of binned expectations

        Output:
        - the sum of the binned poisson Chi2.

        Reference: https://doi.org/10.1016/0167-5087(84)90016-4
        While the absolute likelihood is a poor GOF measure
        (see http://www.physics.ucla.edu/~cousins/stats/cousins_saturated.pdf)
    """
    @classmethod
    def from_binned(cls, data, expectations):
        """Initialize with already binned data + expectations

        In this case the bin-edges don't matter, so we bypass the usual init
        """
        assert (data.shape == expectations.shape), \
            "Shape of binned data does not match shape of expectations!"
        self = cls(None, None, None, None)
        test_statistics.__init__(self=self,
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
        test_statistics.__init__(self=self,
                                 data=data,
                                 pdf=pdf,
                                 nevents_expected=nevents_expected)
        self._name = self.__class__.__name__

        self.bin_edges = bin_edges
        self.bin_data(bin_edges=bin_edges)
        return

    @classmethod
    def calculate_binned_gof(cls, binned_data, binned_expectations):
        """Get binned poisson chi2 GoF from binned data & expectations
        """
        ret = sps.poisson(binned_data).logpmf(binned_data)
        ret -= sps.poisson(binned_expectations).logpmf(binned_data)
        return 2 * np.sum(ret)

    def calculate_gof(self):
        """
            Get binned poisson chi2 GoF using current class attributes
        """
        gof = binned_poisson_chi2_gof.calculate_binned_gof(
            self.binned_data,
            self.pdf * self.nevents_expected
        )
        self.gof = gof
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
        if not hasattr(self, 'gof'):
            _ = self.calculate_gof()
        fake_gofs = self.sample_gofs(n_mc=n_mc)
        hist, bin_edges = np.histogram(fake_gofs, bins=1000)
        cumulative_density = 1.0 - np.cumsum(hist) / np.sum(hist)
        try:
            pvalue = cumulative_density[np.digitize(self.gof, bin_edges) - 1]
        except IndexError:
            raise ValueError(
                'Not enough MC\'s run -- GoF is outside toy distribution!')
        return pvalue


class binned_chi2_gof(test_statistics):
    """Compoutes the binned chi2 GoF based on Pearson's chi2.

    Input (unbinned data):
    - data: array of unbinned data
    - pdf: normalized histogram of binned expectations
    - bin_edges: bin-edges of the pdf
    - nevents_expected: expectation, can be mean of expectation pdf

    Input (binned data):
    initialise with binned_chi2_gof.from_binned(...)
        - data: array of binned data
        - expectations: array of binned expectations

    Output:
    - Pearson's Chi2

    Reference: https://www.itl.nist.gov/div898/handbook/eda/section3/eda35f.htm
    """
    @classmethod
    def from_binned(cls, data, expectations):
        """Initialize with already binned data + expectations

        In this case the bin-edges don't matter, so we bypass the usual init
        """
        assert (data.shape == expectations.shape), \
            "Shape of binned data does not match shape of expectations!"
        self = cls(None, None, None, None)
        test_statistics.__init__(self=self,
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
        test_statistics.__init__(self=self,
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
        gof = sps.chisquare(binned_data,
                            binned_expectations, axis=None)[0]
        return gof

    def calculate_gof(self):
        """
            Get Chi2 GoF using current class attributes
        """
        gof = binned_chi2_gof.calculate_binned_gof(
            self.binned_data, self.pdf * self.nevents_expected)
        return gof


class point_to_point_gof(test_statistics_sample):
    """computes point-to-point gof as described in
    https://arxiv.org/abs/hep-ex/0203010.

    Input:
    - nevents_data n-dim data samples with shape (nevents_data, n)
    - nevents_reference n-dim referencesamples with shape (nevents_ref, n)

    Output:
    Test Statistic based on 'Statisticsl Energy'

    Samples should be pre-processed to have similar scale in each analysis
    dimension."""

    def __init__(self, data, reference_sample):
        test_statistics_sample.__init__(
            self=self, data=data, reference_sample=reference_sample
        )
        self._name = self.__class__.__name__
        self.nevents_data = np.shape(self.data)[0]
        self.nevents_ref = np.shape(self.reference_sample)[0]

    def get_distances(self):
        """get distances of data-data, reference-reference
        and data-reference"""

        dist = DistanceMetric.get_metric("euclidean")

        d_data_data = np.triu(dist.pairwise(self.data))
        d_data_data.reshape(-1)
        self.d_data_data = d_data_data[d_data_data > 0]

        d_ref_ref = np.triu(dist.pairwise(self.reference_sample))
        d_ref_ref.reshape(-1)
        self.d_ref_ref = d_ref_ref[d_ref_ref > 0]

        self.d_data_ref = dist.pairwise(
            self.data, self.reference_sample).reshape(-1)

    def get_d_min(self):
        """find d_min as the average distance of reference_simulation
        points in the region of highest density"""
        # For now a very simple approach is chosen as the paper states that
        # the precise value of this is not critical. One might want to
        # look into a more proficient way in the future.
        self.d_min = np.quantile(self.d_ref_ref, 0.001)

    def weighting_function(self, d):
        """Weigh distances d according to log profile. Pole at d = 0
        is omitted by introducing d_min that replaces the distance for
        d < d_min
        """
        if not hasattr(self, "d_min"):
            self.get_d_min()
        d[d <= self.d_min] = self.d_min

        return -np.log(d)

    def calculate_gof(self, *args, **kwargs):

        self.get_distances()
        ret_data_data = (1 / self.nevents_data ** 2 *
                         np.sum(self.weighting_function(self.d_data_data)))
        ret_ref_ref = (1 / self.nevents_ref ** 2 *
                       np.sum(self.weighting_function(self.d_ref_ref)))
        ret_data_ref = (-1 / self.nevents_ref / self.nevents_data *
                        np.sum(self.weighting_function(self.d_data_ref)))
        ret = ret_data_data + ret_ref_ref + ret_data_ref
        return ret

# %%
