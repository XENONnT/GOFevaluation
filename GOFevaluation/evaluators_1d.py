import scipy.stats as sps
import numpy as np
import warnings
from scipy.interpolate import interp1d
from GOFevaluation import test_statistics_pdf
from GOFevaluation import test_statistics_sample


class adtest_two_sample_gof(test_statistics_sample):
    """Goodness of Fit based on the two-sample Anderson-Darling test
    as described in https://www.doi.org/10.1214/aoms/1177706788
    and https://www.doi.org/10.2307/2288805.
    Test if two samples come from the same pdf.

    Similar to kstest_two_sample_gof but more weight is given on tail
    differences due to a different weighting function.


    Input:
    - data_sample: sample of unbinned data_sample
    - reference_sample: sample of unbinned reference

    Output:
    - gof: gof statistic calculated with scipy.stats.anderson_ksamp"""

    def __init__(self, data_sample, reference_sample):
        super().__init__(data_sample=data_sample,
                         reference_sample=reference_sample)

    @staticmethod
    def calculate_gof(data_sample, reference_sample):
        # mute specific warnings from sps p-value calculation
        # as this value is not used here anyways:
        warnings.filterwarnings(
            "ignore", message="p-value floored: true value smaller than 0.001")
        warnings.filterwarnings(
            "ignore", message="p-value capped: true value larger than 0.25")
        gof = sps.anderson_ksamp([data_sample, reference_sample])[0]
        return gof

    def get_gof(self):
        gof = self.calculate_gof(self.data_sample, self.reference_sample)
        self.gof = gof
        return gof

    def get_pvalue(self, n_perm=1000):
        pvalue = super().get_pvalue(n_perm)
        return pvalue


class kstest_gof(test_statistics_pdf):
    """Goodness of Fit based on the Kolmogorov-Smirnov Test.
    Test if data_sample comes from given pdf.

    Input:
    - data_sample: sample of unbinned data
    - pdf: binned pdf to be tested
    - bin_edges: binning of the pdf

    Output:
    - gof: supremum of the absolute value of the difference of CDF and ECDF
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
        gof = sps.kstest(data_sample, cdf=cdf)[0]
        return gof

    def get_gof(self):
        """
        Interpolate CDF from binned pdf and calculate supremum of the
        absolute value of the difference of CDF and ECDF via scipy.stats.kstest
        """
        interp_cdf = interp1d(self.bin_centers,
                              np.cumsum(self.pdf),
                              kind='cubic')
        gof = self.calculate_gof(self.data_sample, interp_cdf)
        self.gof = gof
        return gof

    def get_pvalue(self):
        pvalue = super().get_pvalue()
        self.pvalue = pvalue
        return pvalue


class kstest_two_sample_gof(test_statistics_sample):
    """Goodness of Fit based on the Kolmogorov-Smirnov Test for two samples.
    Test if two samples come from the same pdf.

    Input:
    - data: sample of unbinned data
    - reference_sample: sample of unbinned reference

    Output:
    - gof: supremum of the absolute value of the difference of both ECDF
    """

    def __init__(self, data_sample, reference_sample):
        super().__init__(data_sample=data_sample,
                         reference_sample=reference_sample)

    @staticmethod
    def calculate_gof(data_sample, reference_sample):
        gof = sps.ks_2samp(data_sample, reference_sample)[0]
        return gof

    def get_gof(self):
        """
        calculate supremum of the absolute value of the difference
        of both ECDF via scipy.stats.kstest
        """
        gof = self.calculate_gof(self.data_sample, self.reference_sample)
        self.gof = gof
        return gof

    def get_pvalue(self, n_perm=1000):
        pvalue = super().get_pvalue(n_perm)
        return pvalue
