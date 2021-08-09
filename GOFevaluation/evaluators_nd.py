import scipy.stats as sps
import numpy as np
from sklearn.neighbors import DistanceMetric
import warnings

from GOFevaluation import EvaluatorBaseBinned
from GOFevaluation import EvaluatorBaseSample


class BinnedPoissonChi2GOF(EvaluatorBaseBinned):
    """Goodness of Fit based on the binned poisson modified Chi2

        Computes the binned poisson modified Chi2 from Baker+Cousins
        In the limit of large bin counts (10+) this is Chi2 distributed.
        In the general case you may have to toyMC the distribution yourself.

        - **unbinned data**
            :param data_sample: sample of unbinned data
            :type data_sample: array_like, n-Dimensional
            :param pdf: binned pdf to be tested
            :type pdf: array_like, n-Dimensional
            :param bin_edges: binning of the pdf
            :type bin_edges: array_like, n-Dimensional
            :param nevents_expected: expectation, can be mean of expected pdf
            :type nevents_expected: float

        - **binned data**
            initialise with .from_binned(...)
            :param binned_data: array of binned data
            :type binned_data: array_like, n-Dimensional
            :param binned_reference: array of binned reference
            :type binned_reference: array_like, n-Dimensional

    .. note::
        Reference:
        https://doi.org/10.1016/0167-5087(84)90016-4
        While the absolute likelihood is a poor GOF measure
        (see http://www.physics.ucla.edu/~cousins/stats/cousins_saturated.pdf)
    """

    def __init__(self, data_sample, pdf, bin_edges, nevents_expected):
        """Initialize with unbinned data and a normalized pdf
        """
        super().__init__(data_sample, pdf, bin_edges, nevents_expected)

    @staticmethod
    def calculate_gof(binned_data, binned_reference):
        """Get binned poisson chi2 GoF from binned data & reference
        """
        critical_bin_count = 10
        if (binned_data < critical_bin_count).any():
            warnings.warn(f'Binned data contains bin count(s) below '
                          f'{critical_bin_count}. GoF not well defined!',
                          stacklevel=2)
        if (binned_reference < critical_bin_count).any():
            warnings.warn(f'Binned reference contains bin count(s) below '
                          f'{critical_bin_count}. GoF not well defined!',
                          stacklevel=2)
        ret = sps.poisson(binned_data).logpmf(binned_data)
        ret -= sps.poisson(binned_reference).logpmf(binned_data)
        return 2 * np.sum(ret)

    def get_gof(self):
        """gof is calculated using current class attributes

        :return: Goodness of Fit
        :rtype: float
        """
        gof = self.calculate_gof(self.binned_data, self.binned_reference)
        self.gof = gof
        return gof

    def get_pvalue(self, n_mc=1000, return_fake_gofs=False):
        return super().get_pvalue(n_mc, return_fake_gofs=return_fake_gofs)


class BinnedChi2GOF(EvaluatorBaseBinned):
    """Compoutes the binned chi2 GoF based on Pearson's chi2.

        - **unbinned data**
            :param data_sample: sample of unbinned data
            :type data_sample: array_like, n-Dimensional
            :param pdf: binned pdf to be tested
            :type pdf: array_like, n-Dimensional
            :param bin_edges: binning of the pdf
            :type bin_edges: array_like, n-Dimensional
            :param nevents_expected: expectation, can be mean of expected pdf
            :type nevents_expected: float

        - **binned data**
            initialise with .from_binned(...)
            :param binned_data: array of binned data
            :type binned_data: array_like, n-Dimensional
            :param binned_reference: array of binned reference
            :type binned_reference: array_like, n-Dimensional

    .. note::
        Reference:
        https://www.itl.nist.gov/div898/handbook/eda/section3/eda35f.htm
    """

    def __init__(self, data_sample, pdf, bin_edges, nevents_expected):
        """Initialize with unbinned data and a normalized pdf
        """
        super().__init__(data_sample, pdf, bin_edges, nevents_expected)

    @staticmethod
    def calculate_gof(binned_data, binned_reference):
        """Get Chi2 GoF from binned data & expectations
        """
        critical_bin_count = 5
        if (binned_data < critical_bin_count).any():
            warnings.warn(f'Binned data contains bin count(s) below '
                          f'{critical_bin_count}. GoF not well defined!',
                          stacklevel=2)
        if (binned_reference < critical_bin_count).any():
            warnings.warn(f'Binned reference contains bin count(s) below '
                          f'{critical_bin_count}. GoF not well defined!',
                          stacklevel=2)
        gof = sps.chisquare(binned_data,
                            binned_reference, axis=None)[0]
        return gof

    def get_gof(self):
        """gof is calculated using current class attributes

        :return: Goodness of Fit
        :rtype: float
        """
        gof = self.calculate_gof(self.binned_data, self.binned_reference)
        self.gof = gof
        return gof

    def get_pvalue(self, n_mc=1000, return_fake_gofs=False):
        return super().get_pvalue(n_mc, return_fake_gofs=return_fake_gofs)


class PointToPointGOF(EvaluatorBaseSample):
    """Goodness of Fit based on point-to-point method

    :param data_sample: sample of unbinned data
    :type data_sample: array_like, n-Dimensional
    :param reference_sample: sample of unbinned reference
    :type reference_sample: array_like, n-Dimensional

    .. note::
        * Samples should be pre-processed to have similar scale in each
          analysis dimension.
        * Reference:
          https://arxiv.org/abs/hep-ex/0203010
    """

    def __init__(self, data_sample, reference_sample):
        super().__init__(data_sample=data_sample,
                         reference_sample=reference_sample)

    @staticmethod
    def get_distances(data_sample, reference_sample):
        """get distances of data-data, reference-reference
        and data-reference

        :param data_sample: sample of unbinned data
        :type data_sample: array_like, n-Dimensional
        :param reference_sample: sample of unbinned reference
        :type reference_sample: array_like, n-Dimensional
        :retrun: distance of (data-data, reference-reference, data-reference)
        :rtype: triple of arrays
        """
        # For 1D input, arrays need to be transformed in
        # order for the distance measure method to work
        if data_sample.ndim == 1:
            data_sample = np.vstack(data_sample)
        if reference_sample.ndim == 1:
            reference_sample = np.vstack(reference_sample)

        dist = DistanceMetric.get_metric("euclidean")

        d_data_data = np.triu(dist.pairwise(data_sample))
        d_data_data.reshape(-1)
        d_data_data = d_data_data[d_data_data > 0]

        # d_ref_ref = np.triu(dist.pairwise(reference_sample))
        # d_ref_ref.reshape(-1)
        # d_ref_ref = d_ref_ref[d_ref_ref > 0]

        d_data_ref = dist.pairwise(data_sample, reference_sample).reshape(-1)

        return d_data_data, d_data_ref

    @staticmethod
    def get_d_min(d_ref_ref):
        """find d_min as the average distance of reference_sample
        points in the region of highest density"""
        # For now a very simple approach is chosen as the paper states that
        # the precise value of this is not critical. One might want to
        # look into a more proficient way in the future.
        d_min = np.quantile(d_ref_ref, 0.001)
        return d_min

    @staticmethod
    def weighting_function(d, d_min, w_func='log'):
        """Weigh distances d according to log profile. Pole at d = 0
        is omitted by introducing d_min that replaces the distance for
        d < d_min
        """
        d[d <= d_min] = d_min
        if w_func == 'log':
            ret = -np.log(d)
        elif w_func == 'x2':
            ret = d**2
        elif w_func == 'x':
            ret = d
        elif w_func == '1/x':
            ret = 1 / d
        else:
            raise KeyError(f'w_func {w_func} is not defined.')
        return ret

    @classmethod
    def calculate_gof(cls, data_sample, reference_sample, d_min=None):
        """Internal function to calculate gof for :func:`get_gof`
        and :func:`get_pvalue`"""

        nevents_data = np.shape(data_sample)[0]
        nevents_ref = np.shape(reference_sample)[0]

        d_data_data, d_data_ref = cls.get_distances(
            data_sample, reference_sample)
        if d_min is None:
            d_min = cls.get_d_min(d_data_ref)

        ret_data_data = (1 / nevents_data ** 2 *
                         np.sum(cls.weighting_function(d_data_data, d_min)))
        # ret_ref_ref = (1 / nevents_ref ** 2 *
        #                np.sum(cls.weighting_function(d_ref_ref, d_min)))
        ret_data_ref = (-1 / nevents_ref / nevents_data *
                        np.sum(cls.weighting_function(d_data_ref, d_min)))
        gof = ret_data_data + ret_data_ref  # ret_data_data + ret_ref_ref + ret_data_ref
        return gof

    def get_gof(self, d_min=None):
        """gof is calculated using current class attributes

        :param d_min: Replaces the distance for distance d < d_min.
            If None, d_min is estimated by :func:`get_dmin`,
            defaults to None
        :type d_min: float, optional
        :return: Goodness of Fit
        :rtype: float

        .. note::
            d_min should be a typical distance of the reference_sample in
            the region of highest density
        """

        gof = self.calculate_gof(
            self.data_sample, self.reference_sample, d_min)
        self.gof = gof
        return gof

    def get_pvalue(self, n_perm=1000, d_min=None, return_fake_gofs=False):
        """p-value is calculated

        Computes the p-value by means of re-sampling data sample
        and reference sample. For each re-sampling, the gof is calculated.
        The p-value can then be obtained from the distribution of these
        fake-gofs.

        :param n_perm: Number of fake-gofs calculated, defaults to 1000
        :type n_perm: int, optional
        :param d_min: Replaces the distance for distance d < d_min.
            If None, d_min is estimated by :func:`get_dmin`,
            defaults to None
        :type d_min: float, optional
        :return: p-value
        :rtype: float
        """
        return super().get_pvalue(n_perm, d_min,
                                  return_fake_gofs=return_fake_gofs)
