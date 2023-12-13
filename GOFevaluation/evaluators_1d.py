import scipy.stats as sps
import numpy as np
import warnings
from scipy.interpolate import interp1d

from GOFevaluation.evaluator_base import EvaluatorBasePdf
from GOFevaluation.evaluator_base import EvaluatorBaseSample
from GOFevaluation.evaluator_base import EvaluatorBaseMCUnbinned


class ADTestTwoSampleGOF(EvaluatorBaseSample):
    """Goodness of Fit based on the two-sample Anderson-Darling Test

    The test is described in https://www.doi.org/10.1214/aoms/1177706788
    and https://www.doi.org/10.2307/2288805. test if two samples
    come from the same pdf. Similar to :class:`KSTestTwoSampleGOF` but more
    weight is given on tail differences due to a different weighting function.


    :param data_sample: sample of unbinned data
    :type data_sample: array_like, 1-Dimensional
    :param reference_sample: sample of unbinned reference
    :type reference_sample: array_like, 1-Dimensional
    """

    def __init__(self, data_sample, reference_sample):
        super().__init__(data_sample=data_sample,
                         reference_sample=reference_sample)

    @staticmethod
    def calculate_gof(data_sample, reference_sample):
        """Internal function to calculate gof for :func:`get_gof`
        and :func:`get_pvalue`"""
        # mute specific warnings from sps p-value calculation
        # as this value is not used here anyways:
        warnings.filterwarnings(
            "ignore", message="p-value floored: true value smaller than 0.001")
        warnings.filterwarnings(
            "ignore", message="p-value capped: true value larger than 0.25")
        gof = sps.anderson_ksamp([data_sample, reference_sample])[0]
        return gof

    def get_gof(self):
        """gof is calculated using current class attributes

        This method uses `sps.anderson_ksamp`.
        :return: Goodness of Fit
        :rtype: float
        """
        gof = self.calculate_gof(self.data_sample, self.reference_sample)
        self.gof = gof
        return gof

    def get_pvalue(self, n_perm=1000):
        pvalue = super().get_pvalue(n_perm)
        return pvalue


class KSTestGOF(EvaluatorBasePdf):
    """Goodness of Fit based on the Kolmogorov-Smirnov Test

    Test if data_sample comes from given pdf.

    :param data_sample: sample of unbinned data
    :type data_sample: array_like, 1-Dimensional
    :param pdf: binned pdf to be tested
    :type pdf: array_like, 1-Dimensional
    :param bin_edges: binning of the pdf
    :type bin_edges: array_like, 1-Dimensional
    """

    def __init__(self, data_sample, pdf, bin_edges):
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        assert ((min(data_sample) >= min(bin_centers))
                & (max(data_sample) <= max(bin_centers))), (
            "Data point(s) outside of pdf bins. Can't compute GoF.")

        super().__init__(data_sample=data_sample, pdf=pdf)
        self.bin_centers = bin_centers

    @staticmethod
    def calculate_gof(data_sample, cdf):
        """Internal function to calculate gof for :func:`get_gof`
        and :func:`get_pvalue`"""
        gof = sps.kstest(data_sample, cdf=cdf)[0]
        return gof

    def get_gof(self):
        """gof is calculated using current class attributes

        The CDF is interpolated from pdf and binning. The gof is then
        calculated by `sps.kstest`

        :return: Goodness of Fit
        :rtype: float
        """
        interp_cdf = interp1d(self.bin_centers,
                              np.cumsum(self.pdf),
                              kind='cubic')
        gof = self.calculate_gof(self.data_sample, interp_cdf)
        self.gof = gof
        return gof

    def get_pvalue(self):
        pvalue = super().get_pvalue()
        return pvalue


class KSTestTwoSampleGOF(EvaluatorBaseSample):
    """Goodness of Fit based on the Kolmogorov-Smirnov Test for two samples

    Test if two samples come from the same pdf.

    :param data_sample: sample of unbinned data
    :type data_sample: array_like, 1-Dimensional
    :param reference_sample: sample of unbinned reference
    :type reference_sample: array_like, 1-Dimensional
    """

    def __init__(self, data_sample, reference_sample):
        super().__init__(data_sample=data_sample,
                         reference_sample=reference_sample)

    @staticmethod
    def calculate_gof(data_sample, reference_sample):
        """Internal function to calculate gof for :func:`get_gof`
        and :func:`get_pvalue`"""
        gof = sps.ks_2samp(data_sample, reference_sample, mode='asymp')[0]
        return gof

    def get_gof(self):
        """gof is calculated using current class attributes

        This method uses `scipy.stats.kstest`.
        :return: Goodness of Fit
        :rtype: float
        """
        gof = self.calculate_gof(self.data_sample, self.reference_sample)
        self.gof = gof
        return gof

    def get_pvalue(self, n_perm=1000):
        pvalue = super().get_pvalue(n_perm)
        return pvalue


class FractionInSlice(EvaluatorBaseMCUnbinned):
    """
    A function that finds how likely it is that a uniformly sampled 2pi circle contain
    an equal or greater portion in a slice set in the generator
    angles in radians
    """
    def __init__(self, data, opening_angle=np.pi, fixed_length=True):
        mu = len(data)
        assert 0 < mu  # otherwise, pointless
        if fixed_length:
            get_uniform_thetas = lambda: sps.uniform(-np.pi, 2*np.pi).rvs(mu)
        else:
            get_uniform_thetas = lambda: sps.uniform(-np.pi, 2*np.pi).rvs(sps.poisson(mu).rvs())
        distance = self.get_best_partition
        super().__init__(data, get_uniform_thetas, distance)

    @staticmethod
    def dtheta(t0, t1):
        """Compute smallest angle between two directions"""
        return np.abs((t0 - t1 + np.pi) % (2 * np.pi) - np.pi)

    @staticmethod
    def get_best_partition(data_t, opening_angle=np.pi, test_angles=None, return_best_angle=False):
        """
        Find the angle and number of events that is the most events you can fit into opening_angle
        If test_angles = None, the direction of all data-points will be tried, but if there are many 1000s of points,
        it can be more performant and good enough to just pass np.linspace(0, 2*np.pi, 100) instead
        """
        if len(data_t) == 0:
            if return_best_angle:
                return 1., 0.
            else:
                return 1.

        if test_angles is None:
            test_angles = data_t

        ns = np.zeros(len(data_t))
        for i, t in enumerate(test_angles):
            ns[i] = np.sum(FractionInSlice.dtheta(data_t, t) < 0.5 * opening_angle)
        topt = test_angles[np.argmax(ns)]
        opt = np.max(ns)
        if return_best_angle:
            return opt / len(data_t), topt
        else:
            return opt / len(data_t)
