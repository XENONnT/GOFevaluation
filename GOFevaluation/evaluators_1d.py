import scipy.stats as sps
import numpy as np
from collections import OrderedDict
from GOFevaluation import test_statistics


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
        value = sps.poisson(self.expected_events).logpmf(self.binned_data).sum()
        return value


class chi2_gof(test_statistics):
    """Docstring for chi2_gof. """
    def __init__(self, data, pdf, bin_edges, nevents_expected,
                 empty_bin_value):
        test_statistics.__init__(self=self,
                                 data=data,
                                 pdf=pdf,
                                 nevents_expected=nevents_expected)
        self._name = self.__class__.__name__ + "_" + str(empty_bin_value)
        self.empty_bin_value = empty_bin_value

        self.bin_edges = bin_edges
        self.bin_data(bin_edges=bin_edges)

    def calculate_gof(self):
        no_empty_bins_data_events = np.where(self.binned_data == 0,
                                             self.empty_bin_value,
                                             self.binned_data)
        value = sps.chisquare(self.expected_events, no_empty_bins_data_events)[0]
        return value


class anderson_test_gof(test_statistics):
    """Docstring for anderson_test_gof. """
    def __init__(self, data, pdf, nevents_expected, bin_edges):
        """TODO: to be defined. """
        test_statistics.__init__(self=self,
                                 data=data,
                                 pdf=pdf,
                                 nevents_expected=nevents_expected)

        self._name = "anderson_test_gof"

        self.bin_edges = bin_edges
        self.bin_data(bin_edges=bin_edges)

    def calculate_gof(self):
        value = sps.anderson_ksamp([self.expected_events, self.binned_data])[0]
        return value

# TODO needs to be tested
class kstest_gof(test_statistics):
    """Docstring for kstest_gof. """
    def __init__(self, data, interpolated_cdf):
        """TODO: to be defined. """
        test_statistics.__init__(self, data=data, pdf=interpolated_cdf)
        self._name = self.__class__.__name__

    def calculate_gof(self):
        """
        tested with logc2 as data
        """
        value = sps.kstest(self.data, cdf=np.cumsum(self.pdf))[0]
        return value


class evaluators_1d(object):
    """Evaluation class for goodnes of fit measures in Xenon"""
    def __init__(self, pdf, data, nevents_expected, bin_edges):
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
            anderson_test_gof(data=data,
                              pdf=pdf,
                              nevents_expected=nevents_expected,
                              bin_edges=bin_edges)
        ]

    def calculate_gof_values(self):
        d_results = OrderedDict()
        for measure in self.l_measures_to_calculate:
            res = measure.get_result_as_dict()
            d_results.update(dict(res))
        return d_results
