import scipy.stats as sps
import numpy as np
import warnings
from scipy.interpolate import interp1d

from GOFevaluation.evaluator_base import EvaluatorBasePdf
from GOFevaluation.evaluator_base import EvaluatorBaseSample


class ADTestTwoSampleGOF(EvaluatorBaseSample):
    """Goodness of Fit based on the two-sample Anderson-Darling Test.

    The test is described in https://www.doi.org/10.1214/aoms/1177706788
    and https://www.doi.org/10.2307/2288805. It test if two samples
    come from the same pdf. Similar to :class:`KSTestTwoSampleGOF` but more
    weight is given on tail differences due to a different weighting function.


    :param data_sample: sample of unbinned data
    :type data_sample: array_like, 1-Dimensional
    :param reference_sample: sample of unbinned reference
    :type reference_sample: array_like, 1-Dimensional

    """

    def __init__(self, data_sample, reference_sample):
        super().__init__(data_sample=data_sample, reference_sample=reference_sample)

    @staticmethod
    def calculate_gof(data_sample, reference_sample):
        """Internal function to calculate gof for :func:`get_gof` and
        :func:`get_pvalue`"""
        # mute specific warnings from sps p-value calculation
        # as this value is not used here anyways:
        warnings.filterwarnings("ignore", message="p-value floored: true value smaller than 0.001")
        warnings.filterwarnings("ignore", message="p-value capped: true value larger than 0.25")
        gof = sps.anderson_ksamp([data_sample, reference_sample])[0]
        return gof

    def get_gof(self):
        """GOF is calculated using current class attributes.

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
    """Goodness of Fit based on the Kolmogorov-Smirnov Test.

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
        assert (min(data_sample) >= min(bin_centers)) & (
            max(data_sample) <= max(bin_centers)
        ), "Data point(s) outside of pdf bins. Can't compute GOF."

        super().__init__(data_sample=data_sample, pdf=pdf)
        self.bin_centers = bin_centers

    @staticmethod
    def calculate_gof(data_sample, cdf):
        """Internal function to calculate gof for :func:`get_gof` and
        :func:`get_pvalue`"""
        gof = sps.kstest(data_sample, cdf=cdf)[0]
        return gof

    def get_gof(self):
        """GOF is calculated using current class attributes.

        The CDF is interpolated from pdf and binning. The gof is then
        calculated by `sps.kstest`

        :return: Goodness of Fit
        :rtype: float

        """
        interp_cdf = interp1d(self.bin_centers, np.cumsum(self.pdf), kind="cubic")
        gof = self.calculate_gof(self.data_sample, interp_cdf)
        self.gof = gof
        return gof

    def get_pvalue(self):
        pvalue = super().get_pvalue()
        return pvalue


class KSTestTwoSampleGOF(EvaluatorBaseSample):
    """Goodness of Fit based on the Kolmogorov-Smirnov Test for two samples.

    Test if two samples come from the same pdf.

    :param data_sample: sample of unbinned data
    :type data_sample: array_like, 1-Dimensional
    :param reference_sample: sample of unbinned reference
    :type reference_sample: array_like, 1-Dimensional

    """

    def __init__(self, data_sample, reference_sample):
        super().__init__(data_sample=data_sample, reference_sample=reference_sample)

    @staticmethod
    def calculate_gof(data_sample, reference_sample):
        """Internal function to calculate gof for :func:`get_gof` and
        :func:`get_pvalue`"""
        gof = sps.ks_2samp(data_sample, reference_sample, mode="asymp")[0]
        return gof

    def get_gof(self):
        """GOF is calculated using current class attributes.

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
