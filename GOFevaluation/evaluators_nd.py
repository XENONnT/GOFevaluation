import scipy.stats as sps
import numpy as np
from sklearn.metrics import pairwise_distances
from itertools import product
import warnings
# import time

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
        self.weighted_distance_matrix = None
        self.d_min = None

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
        # print(f'4.2.3.6.1:    {time.perf_counter():.4f} s')
        d_data_data = np.triu(
            distance_matrix[np.ix_(data_indices, data_indices)], k=1)
        d_data_data.reshape(-1)
        # d_data_data = d_data_data[d_data_data > 0]

        # print(f'4.2.3.6.2:    {time.perf_counter():.4f} s')
        d_ref_ref = distance_matrix[np.ix_(
            reference_indices, reference_indices)]
        # print(f'4.2.3.6.3:    {time.perf_counter():.4f} s')
        d_ref_ref = np.triu(d_ref_ref, k=1)
        # print(f'4.2.3.6.4:    {time.perf_counter():.4f} s')
        d_ref_ref.reshape(-1)
        # print(f'4.2.3.6.5:    {time.perf_counter():.4f} s')
        # d_ref_ref = d_ref_ref[d_ref_ref > 0]

        # print(f'4.2.3.6.6:    {time.perf_counter():.4f} s')
        d_data_ref = distance_matrix[np.ix_(
            data_indices, reference_indices)].reshape(-1)

        return d_data_data, d_ref_ref, d_data_ref

    def get_weighted_distances(self, data_sample=None, reference_sample=None,
                               data_indices=None, reference_indices=None,
                               store_matrix=False):
        """get weighted distances of data-data, reference-reference
        and data-reference

        :param data_sample: sample of unbinned data
        :type data_sample: array_like, n-Dimensional
        :param reference_sample: sample of unbinned reference
        :type reference_sample: array_like, n-Dimensional
        :retrun: distance of (data-data, reference-reference, data-reference)
        :rtype: triple of arrays
        """
        # print(f'4.2.3.4:    {time.perf_counter():.4f} s')
        if self.weighted_distance_matrix is None:
            assert (data_sample is not None and reference_sample is not None), (
                'Samples have to be passed if the distance matrix is \
                    not yet calculated!')
            distance_matrix = self.get_distance_matrix(data_sample,
                                                       reference_sample)
            weighted_distance_matrix = self.weighting_function(
                distance_matrix, self.d_min)
            if store_matrix:
                self.weighted_distance_matrix = weighted_distance_matrix
                # print('SAVE MATRIX')
        else:
            weighted_distance_matrix = self.weighted_distance_matrix

        # print(f'4.2.3.5:    {time.perf_counter():.4f} s')
        if data_indices is None and reference_indices is None:
            data_indices = np.arange(0, len(data_sample))
            reference_indices = np.arange(len(data_sample),
                                          len(data_sample) + len(reference_sample))
        # print(f'4.2.3.6:    {time.perf_counter():.4f} s')
        ret = self.group_distances(weighted_distance_matrix, data_indices,
                                   reference_indices)
        return ret

    @staticmethod
    def get_d_min(d_ref_ref):
        """find d_min as the average distance of reference_sample
        points in the region of highest density"""
        # For now a very simple approach is chosen as the paper states that
        # the precise value of this is not critical. One might want to
        # look into a more proficient way in the future.
        # print('CALCULATE DMIN')
        d_min = np.quantile(d_ref_ref, 0.001)
        return d_min

    @staticmethod
    def weighting_function(d, d_min):
        """Weigh distances d according to log profile. Pole at d = 0
        is omitted by introducing d_min that replaces the distance for
        d < d_min
        """
        # print(f'4.2.3.1:    {time.perf_counter():.4f} s')
        d[d <= d_min] = d_min
        # print(f'4.2.3.2:    {time.perf_counter():.4f} s')
        ret = -np.log(d)
        # print(f'4.2.3.3:    {time.perf_counter():.4f} s')
        return ret

    # @classmethod
    def calculate_gof(self, data_sample=None, reference_sample=None,
                      data_indices=None, reference_indices=None,
                      store_matrix=False):
        """Internal function to calculate gof for :func:`get_gof`
        and :func:`get_pvalue`"""
        # print(f'4.2.1:    {time.perf_counter():.4f} s')
        nevents_data = np.shape(data_sample)[0]
        nevents_ref = np.shape(reference_sample)[0]

        # d_data_data, d_ref_ref, d_data_ref = self.get_distances(
        #     data_sample, reference_sample, data_indices, reference_indices)
        # print(f'4.2.2:    {time.perf_counter():.4f} s')
        if self.d_min is None:
            d_ref_ref = np.triu(pairwise_distances(reference_sample), k=1)
            d_ref_ref.reshape(-1)
            d_ref_ref = d_ref_ref[d_ref_ref > 0]
            self.d_min = self.get_d_min(d_ref_ref)
        # print(f'4.2.3:    {time.perf_counter():.4f} s')
        wd_data_data, wd_ref_ref, wd_data_ref = self.get_weighted_distances(
            data_sample, reference_sample, data_indices, reference_indices,
            store_matrix=store_matrix)
        # print(f'4.2.4:    {time.perf_counter():.4f} s')
        ret_data_data = (1 / nevents_data ** 2 * np.sum(wd_data_data))
        ret_ref_ref = (1 / nevents_ref ** 2 * np.sum(wd_ref_ref))
        ret_data_ref = (-1 / nevents_ref / nevents_data * np.sum(wd_data_ref))
        gof = ret_data_data + ret_ref_ref + ret_data_ref
        # print(f'4.2.5:    {time.perf_counter():.4f} s')
        return gof

    @staticmethod
    def get_distance_matrix(data_sample, reference_sample):
        # print('Calculating Distance Matrix')
        samples = np.concatenate([data_sample, reference_sample])
        # use all cores for processing
        return pairwise_distances(samples, n_jobs=-1)

    def _calculate_fake_gof(self, indices, n_data):
        rng = np.random.default_rng()
        rng.shuffle(indices, axis=0)
        data_indices = indices[:n_data]
        reference_indices = indices[n_data:]
        # print(f'4.2({i}):    {time.perf_counter():.4f} s')
        fake_gof = self.calculate_gof(data_sample=self.data_sample,
                                      reference_sample=self.reference_sample,
                                      data_indices=data_indices,
                                      reference_indices=reference_indices)
        return fake_gof

    def permutation_gofs(self, n_perm=1000):
        """Generate fake GoFs by re-sampling data and reference sample
        This overrides evaluator_base.EvaluatorBaseSample.permutation_gofs()

        :param n_perm: Number of fake-gofs calculated, defaults to 1000
        :type n_perm: int, optional
        :return: Array of fake GoFs
        :rtype: array_like
        """
        # print(f'4.1:    {time.perf_counter():.4f} s')
        n_data = len(self.data_sample)
        n_reference = len(self.reference_sample)
        indices = np.arange(n_data + n_reference)
        fake_gofs = np.zeros(n_perm)
        for i in range(n_perm):
            fake_gofs[i] = self._calculate_fake_gof(indices, n_data)
            # print(f'4.3({i}):    {time.perf_counter():.4f} s')
        return fake_gofs

    def get_gof(self, d_min=None, store_matrix=False):
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
        if self.gof is None:
            self.d_min = d_min
            gof = self.calculate_gof(
                self.data_sample, self.reference_sample, store_matrix=store_matrix)
            self.gof = gof
        else:
            gof = self.gof
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
        if self.d_min is not None:
            self.d_min = d_min
        _ = self.get_gof(self.d_min, store_matrix=True)
        # print(f'1:    {time.perf_counter():.4f} s')
        # distance_matrix = self.get_distance_matrix(self.data_sample,
        #                                            self.reference_sample)
        # self.weighted_distance_matrix = self.weighting_function(distance_matrix, self.d_min)
        # print(f'2:    {time.perf_counter():.4f} s')
        return super().get_pvalue(n_perm, d_min=d_min)
