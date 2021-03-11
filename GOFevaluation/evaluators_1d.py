import scipy.stats as sps
import numpy as np
from collections import OrderedDict


class evaluators_1d(object):
    """Evaluation class for goodnes of fit measures in Xenon"""
    def __init__(self, expected_events, data_points, bin_edges):
        self.expected_events = expected_events
        self.data_points = data_points

        self.bin_edges = bin_edges
        self.binned_data = None
        self.bin_data(bin_edges=self.bin_edges)

    def bin_data(self, bin_edges):
        self.binned_data, _ = np.histogram(self.data_points, bins=bin_edges)

    def _calculate_gof_values(self):
        """
        expected_events: binned expectation per bin (1d)
        data_events: toy-data or real data per bin (1d)
        """
        gof_binned_poisson = sps.poisson(self.expected_events).logpmf(
            self.binned_data).sum()

        # chisquare
        no_empty_bins_data_events = np.where(self.binned_data == 0, 0.1,
                                             self.binned_data)
        gof_binned_chi_square0_1 = sps.chisquare(self.expected_events,
                                                 no_empty_bins_data_events)

        no_empty_bins_data_events = np.where(self.binned_data == 0, 0.01,
                                             self.binned_data)
        gof_binned_chi_square0_01 = sps.chisquare(self.expected_events,
                                                  no_empty_bins_data_events)

        no_empty_bins_data_events = np.where(self.binned_data == 0, 0.001,
                                             self.binned_data)
        gof_binned_chi_square0_001 = sps.chisquare(self.expected_events,
                                                   no_empty_bins_data_events)

        # anderson-test
        gof_anderson_test = sps.anderson_ksamp([self.expected_events, self.binned_data])

        # TODO: Fix me
        #  # ks-test
        #  pdf_best_fit = pdf_reduced.get_pdf(**rfit)
        #  projected_pdf = pdf_best_fit.project(axis=axis)
        #  interp_cdf = interp1d(projected_pdf.bin_centers,
        #                        projected_pdf.cumulative_density,
        #                        kind='cubic')
        #  gof_ks_test = sps.kstest(toydata['logcs2'], cdf=interp_cdf)

        results_dict = OrderedDict([
            ('GOF_binned_poisson', gof_binned_poisson),
            ('GOF_binned_chisquare0_1', gof_binned_chi_square0_1[0]),
            ('GOF_binned_chisquare0_01', gof_binned_chi_square0_01[0]),
            ('GOF_binned_chisquare0_001', gof_binned_chi_square0_001[0]),
            ('GOF_anderson_test', gof_anderson_test[0]),
            #  ('GOF_ks_test', gof_ks_test[0]), # special treatment, has TODO-above
        ])

        return results_dict
