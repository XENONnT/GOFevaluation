import scipy.stats as sps
import numpy as np
import warnings
from collections import OrderedDict
from scipy.interpolate import interp1d
from GOFevaluation import test_statistics
from GOFevaluation import test_statistics_sample
from GOFevaluation import binned_chi2_gof


class adtest_two_sample_gof(test_statistics_sample):
    """Goodness of Fit based on the two-sample Anderson-Darling test
    as described in https://www.doi.org/10.1214/aoms/1177706788
    and https://www.doi.org/10.2307/2288805.
    Test if two samples come from the same pdf.

    Similar to kstest_two_sample_gof but more weight is given on tail
    differences due to a different weighting function.


    Input:
    - data: sample of unbinned data
    - reference_sample: sample of unbinned reference

    Output:
    - gof: gof statistic calculated with scipy.stats.anderson_ksamp"""

    def __init__(self, data, reference_sample):
        test_statistics_sample.__init__(
            self=self, data=data, reference_sample=reference_sample)

        self._name = self.__class__.__name__

    @staticmethod
    def calculate_gof(data, reference_sample):
        # mute specific warnings from sps p-value calculation
        # as this value is not used here anyways:
        warnings.filterwarnings(
            "ignore", message="p-value floored: true value smaller than 0.001")
        warnings.filterwarnings(
            "ignore", message="p-value capped: true value larger than 0.25")
        gof = sps.anderson_ksamp([data, reference_sample])[0]
        return gof

    def get_gof(self):
        gof = self.calculate_gof(self.data, self.reference_sample)
        self.gof = gof
        return gof


class kstest_gof(test_statistics):
    """Goodness of Fit based on the Kolmogorov-Smirnov Test.
    Test if data sample comes from given pdf.

    Input:
    - data: sample of unbinned data
    - pdf: binned pdf to be tested
    - bin_edges: binning of the pdf

    Output:
    - gof: supremum of the absolute value of the difference of CDF and ECDF
    """

    def __init__(self, data, pdf, bin_edges):
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        assert ((min(data) >= min(bin_centers))
                & (max(data) <= max(bin_centers))), (
            "Data point(s) outside of pdf bins. Can't compute GoF.")

        test_statistics.__init__(self,
                                 data=data,
                                 pdf=pdf,
                                 nevents_expected=None)
        self._name = self.__class__.__name__
        self.bin_edges = bin_edges
        self.bin_centers = bin_centers

    @staticmethod
    def calculate_gof(data, cdf):
        gof = sps.kstest(data, cdf=cdf)[0]
        return gof

    def get_gof(self):
        """
        Interpolate CDF from binned pdf and calculate supremum of the
        absolute value of the difference of CDF and ECDF via scipy.stats.kstest
        """
        interp_cdf = interp1d(self.bin_centers,
                              np.cumsum(self.pdf),
                              kind='cubic')
        gof = self.calculate_gof(self.data, interp_cdf)
        self.gof = gof
        return gof

    def get_pvalue(self, n_mc=1000):
        # This method is not implemented yet. We are working on adding it in
        # the near future.
        raise NotImplementedError("p-value computation not yet implemented!")


class kstest_two_sample_gof(test_statistics_sample):
    """Goodness of Fit based on the Kolmogorov-Smirnov Test for two samples.
    Test if two samples come from the same pdf.

    Input:
    - data: sample of unbinned data
    - reference_sample: sample of unbinned reference

    Output:
    - gof: supremum of the absolute value of the difference of both ECDF
    """

    def __init__(self, data, reference_sample):
        test_statistics_sample.__init__(
            self=self, data=data, reference_sample=reference_sample
        )
        self._name = self.__class__.__name__

    @staticmethod
    def calculate_gof(data, reference_sample):
        gof = sps.ks_2samp(data, reference_sample)[0]
        return gof

    def get_gof(self):
        """
        calculate supremum of the absolute value of the difference
        of both ECDF via scipy.stats.kstest
        """
        gof = self.calculate_gof(self.data, self.reference_sample)
        self.gof = gof
        return gof


class evaluators_1d(object):
    """Evaluation class for goodnes of fit measures in Xenon"""

    def __init__(self, data, pdf, nevents_expected, bin_edges):
        self.pdf = pdf
        self.data = data

        self.l_measures_to_calculate = [
            binned_chi2_gof(data=data,
                            pdf=pdf,
                            nevents_expected=nevents_expected,
                            bin_edges=bin_edges),
            kstest_gof(data=data,
                       pdf=pdf,
                       bin_edges=bin_edges)
        ]

    def calculate_gof_values(self):
        d_results = OrderedDict()
        for measure in self.l_measures_to_calculate:
            res = measure.get_result_as_dict()
            d_results.update(dict(res))
        return d_results
