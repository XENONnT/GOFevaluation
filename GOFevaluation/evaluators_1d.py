import scipy.stats as sps
import numpy as np
from collections import OrderedDict
from scipy.interpolate import interp1d
from GOFevaluation import test_statistics
from GOFevaluation import test_statistics_sample


class binned_poisson_gof(test_statistics):
    """Docstring for binned_poisson_gof. """

    def __init__(self, data, pdf, nevents_expected, bin_edges):
        test_statistics.__init__(self=self,
                                 data=data,
                                 pdf=pdf,
                                 nevents_expected=nevents_expected)
        self._name = self.__class__.__name__

        self.bin_edges = bin_edges
        self.bin_data(bin_edges=bin_edges)

    def calculate_gof(self):
        """
        TODO: Docstring needs to go here
        """
        value = sps.poisson(self.expected_events).logpmf(
            self.binned_data).sum()
        return value


class chi2_gof(test_statistics):
    """Docstring for chi2_gof. """

    def __init__(self, data, pdf, bin_edges, nevents_expected):
        test_statistics.__init__(self=self,
                                 data=data,
                                 pdf=pdf,
                                 nevents_expected=nevents_expected)
        self._name = self.__class__.__name__
        self.bin_edges = bin_edges
        self.bin_data(bin_edges=bin_edges)

    def calculate_gof(self):
        value = sps.chisquare(self.binned_data,
                              self.expected_events)[0]
        return value


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

    def calculate_gof(self):
        value = sps.anderson_ksamp([self.data, self.reference_sample])[0]
        return value


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

    def calculate_gof(self):
        """
        Interpolate CDF from binned pdf and calculate supremum of the
        absolute value of the difference of CDF and ECDF via scipy.stats.kstest
        """
        interp_cdf = interp1d(self.bin_centers,
                              np.cumsum(self.pdf),
                              kind='cubic')
        value = sps.kstest(self.data, cdf=interp_cdf)[0]
        return value


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

    def calculate_gof(self):
        """
        calculate supremum of the absolute value of the difference
        of both ECDF via scipy.stats.kstest
        """
        value = sps.ks_2samp(self.data, self.reference_sample)[0]
        return value


class evaluators_1d(object):
    """Evaluation class for goodnes of fit measures in Xenon"""

    def __init__(self, data, pdf, nevents_expected, bin_edges):
        self.pdf = pdf
        self.data = data

        self.l_measures_to_calculate = [
            chi2_gof(data=data,
                     pdf=pdf,
                     nevents_expected=nevents_expected,
                     bin_edges=bin_edges,
                     empty_bin_value=0.1),
            chi2_gof(data=data,
                     pdf=pdf,
                     nevents_expected=nevents_expected,
                     bin_edges=bin_edges,
                     empty_bin_value=0.01),
            chi2_gof(data=data,
                     pdf=pdf,
                     nevents_expected=nevents_expected,
                     bin_edges=bin_edges,
                     empty_bin_value=0.001),
            binned_poisson_gof(data=data,
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
