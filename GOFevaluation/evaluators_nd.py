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
    def calculate_gof(cls, binned_data, binned_expectations):
        """Get binned poisson chi2 GoF from binned data & expectations
        """
        ret = sps.poisson(binned_data).logpmf(binned_data)
        ret -= sps.poisson(binned_expectations).logpmf(binned_data)
        return 2 * np.sum(ret)

    def get_gof(self):
        """
            Get binned poisson chi2 GoF using current class attributes
        """
        gof = binned_poisson_chi2_gof.calculate_gof(
            self.binned_data,
            self.pdf * self.nevents_expected
        )
        self.gof = gof
        return gof


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
    def calculate_gof(cls, binned_data, binned_expectations):
        """Get Chi2 GoF from binned data & expectations
        """
        gof = sps.chisquare(binned_data,
                            binned_expectations, axis=None)[0]
        return gof

    def get_gof(self):
        """
            Get Chi2 GoF using current class attributes
        """
        gof = binned_chi2_gof.calculate_gof(
            self.binned_data, self.pdf * self.nevents_expected)
        self.gof = gof
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

    @classmethod
    def get_distances(cls, data, reference_sample):
        """get distances of data-data, reference-reference
        and data-reference"""

        dist = DistanceMetric.get_metric("euclidean")

        d_data_data = np.triu(dist.pairwise(data))
        d_data_data.reshape(-1)
        d_data_data = d_data_data[d_data_data > 0]

        d_ref_ref = np.triu(dist.pairwise(reference_sample))
        d_ref_ref.reshape(-1)
        d_ref_ref = d_ref_ref[d_ref_ref > 0]

        d_data_ref = dist.pairwise(data, reference_sample).reshape(-1)

        return d_data_data, d_ref_ref, d_data_ref

    @classmethod
    def get_d_min(cls, d_ref_ref):
        """find d_min as the average distance of reference_simulation
        points in the region of highest density"""
        # For now a very simple approach is chosen as the paper states that
        # the precise value of this is not critical. One might want to
        # look into a more proficient way in the future.
        d_min = np.quantile(d_ref_ref, 0.001)
        return d_min

    @classmethod
    def weighting_function(cls, d, d_min):
        """Weigh distances d according to log profile. Pole at d = 0
        is omitted by introducing d_min that replaces the distance for
        d < d_min
        """
        d[d <= d_min] = d_min

        return -np.log(d)

    @classmethod
    def calculate_gof(cls, data, reference_sample, d_min=None):
        """Calculate point-to-point GoF.
        If d_min=None, d_min is calculated according to a typical distance
        of the reference sample."""

        nevents_data = np.shape(data)[0]
        nevents_ref = np.shape(reference_sample)[0]

        d_data_data, d_ref_ref, d_data_ref = cls.get_distances(
            data, reference_sample)
        if d_min is None:
            d_min = cls.get_d_min(d_ref_ref)

        ret_data_data = (1 / nevents_data ** 2 *
                         np.sum(cls.weighting_function(d_data_data, d_min)))
        ret_ref_ref = (1 / nevents_ref ** 2 *
                       np.sum(cls.weighting_function(d_ref_ref, d_min)))
        ret_data_ref = (-1 / nevents_ref / nevents_data *
                        np.sum(cls.weighting_function(d_data_ref, d_min)))
        gof = ret_data_data + ret_ref_ref + ret_data_ref
        return gof

    def get_gof(self, d_min=None):
        # self.get_distances()
        gof = point_to_point_gof.calculate_gof(self.data,
                                               self.reference_sample,
                                               d_min)
        self.gof = gof
        return gof
