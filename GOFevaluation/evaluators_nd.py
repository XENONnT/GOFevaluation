import scipy.stats as sps
import numpy as np
# from sklearn.neighbors import DistanceMetric
from sklearn.metrics import pairwise_distances
from itertools import product
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
        # print('-' * 20)
        # print(binned_data, binned_reference)
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

    def get_pvalue(self, n_mc=1000):
        return super().get_pvalue(n_mc)


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

    def get_pvalue(self, n_mc=1000):
        return super().get_pvalue(n_mc)


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
        self.distance_matrix = None

    @staticmethod
    def get_tuple_arrays(indices1, indices2=None):
        if indices2 is None:
            index_tuples = list(product(indices1, repeat=2))
            tuple_array1 = np.array([i[0] for i in index_tuples
                                     if i[0] < i[1]])
            tuple_array2 = np.array([i[1] for i in index_tuples
                                     if i[0] < i[1]])
        else:
            index_tuples = list(product(indices1, indices2))
            tuple_array1 = np.array([i[0] for i in index_tuples])
            tuple_array2 = np.array([i[1] for i in index_tuples])
        return tuple_array1, tuple_array2

    @classmethod
    def group_distances(cls, distance_matrix, data_indices, reference_indices):
        d_data_data = np.triu(distance_matrix[data_indices].T[data_indices], k=1)
        d_data_data.reshape(-1)
        d_data_data = d_data_data[d_data_data > 0]
        
        d_ref_ref = np.triu(distance_matrix[reference_indices].T[reference_indices], k=1)
        d_ref_ref.reshape(-1)
        d_ref_ref = d_ref_ref[d_ref_ref > 0]
        d_data_ref = distance_matrix[data_indices].T[reference_indices].reshape(-1)
        
        return d_data_data, d_ref_ref, d_data_ref

    def get_distances(self, data_sample=None, reference_sample=None,
                      data_indices=None, reference_indices=None):
        """get distances of data-data, reference-reference
        and data-reference

        :param data_sample: sample of unbinned data
        :type data_sample: array_like, n-Dimensional
        :param reference_sample: sample of unbinned reference
        :type reference_sample: array_like, n-Dimensional
        :retrun: distance of (data-data, reference-reference, data-reference)
        :rtype: triple of arrays
        """
        if self.distance_matrix is None:
            # print('---1---')
            assert (data_sample is not None and reference_sample is not None), (
                'Samples have to be passed if the distance matrix is \
                    not yet calculated!')
            distance_matrix = self.get_distance_matrix(data_sample,
                                                       reference_sample)
        else:
            # print('---2---')
            distance_matrix = self.distance_matrix
        
        if data_indices is None and reference_indices is None:
            data_indices = np.arange(0, len(data_sample))
            reference_indices = np.arange(len(data_sample),
                                          len(data_sample) + len(reference_sample))

        return self.group_distances(distance_matrix, data_indices,
                                    reference_indices)

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
    def weighting_function(d, d_min):
        """Weigh distances d according to log profile. Pole at d = 0
        is omitted by introducing d_min that replaces the distance for
        d < d_min
        """
        d[d <= d_min] = d_min
        return -np.log(d)

    # @classmethod
    def calculate_gof(self, data_sample=None, reference_sample=None,
                      d_min=None, data_indices=None, reference_indices=None):
        """Internal function to calculate gof for :func:`get_gof`
        and :func:`get_pvalue`"""

        nevents_data = np.shape(data_sample)[0]
        nevents_ref = np.shape(reference_sample)[0]

        d_data_data, d_ref_ref, d_data_ref = self.get_distances(
            data_sample, reference_sample, data_indices, reference_indices)
        if d_min is None:
            d_min = self.get_d_min(d_ref_ref)

        ret_data_data = (1 / nevents_data ** 2 *
                         np.sum(self.weighting_function(d_data_data, d_min)))
        ret_ref_ref = (1 / nevents_ref ** 2 *
                       np.sum(self.weighting_function(d_ref_ref, d_min)))
        ret_data_ref = (-1 / nevents_ref / nevents_data *
                        np.sum(self.weighting_function(d_data_ref, d_min)))
        gof = ret_data_data + ret_ref_ref + ret_data_ref
        return gof
    
    @staticmethod
    def get_distance_matrix(data_sample, reference_sample):
        samples = np.concatenate([data_sample, reference_sample])
        return pairwise_distances(samples)

    def permutation_gofs(self, n_perm=1000, d_min=None):
        """Generate fake GoFs by re-sampling data and reference sample
        This overrides evaluator_base.EvaluatorBaseSample.permutation_gofs()

        :param n_perm: Number of fake-gofs calculated, defaults to 1000
        :type n_perm: int, optional
        :param d_min: Only for PointToPointGOF, defaults to None
        :type d_min: float, optional
        :return: Array of fake GoFs
        :rtype: array_like
        """
        # print('I MADE IT HERE!')
        n_data = len(self.data_sample)
        n_reference = len(self.reference_sample)
        indices = np.arange(n_data + n_reference)
        fake_gofs = np.zeros(n_perm)
        for i in range(n_perm):
            rng = np.random.default_rng()
            rng.shuffle(indices, axis=0)

            data_indices = indices[:n_data]
            reference_indices = indices[n_data:]
            # print(data_indices[0])
            fake_gofs[i] = self.calculate_gof(data_sample=self.data_sample,
                                              reference_sample=self.reference_sample,
                                              d_min=d_min,
                                              data_indices=data_indices,
                                              reference_indices=reference_indices)
        return fake_gofs

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

    def get_pvalue(self, n_perm=1000, d_min=None):
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
        self.distance_matrix = self.get_distance_matrix(self.data_sample,
                                                        self.reference_sample)
        # print('USING DISTANCE MATRIX')
        return super().get_pvalue(n_perm, d_min)
