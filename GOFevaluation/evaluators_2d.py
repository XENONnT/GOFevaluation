#  import scipy.stats as sps
import numpy as np
from collections import OrderedDict
from collections import namedtuple


class evaluators_2d(object):
    """Evaluation class for goodnes of fit measures in Xenon"""

    def __init__(self, expected_events, x_data_points, y_data_points,
                 x_bin_edges, y_bin_edges):
        self.expected_events = expected_events
        self.x_data_points = x_data_points
        self.y_data_points = y_data_points

        self.x_bin_edges = x_bin_edges
        self.y_bin_edges = y_bin_edges

        bin_edges = namedtuple("bin_edges", ["x", "y"])
        self.bin_edges = bin_edges(x=x_bin_edges, y=y_bin_edges)

        self.binned_data = None

        self.bin_data(bin_edges=self.bin_edges)

    def bin_data(self, bin_edges):
        """
        bin_edges: named tuple with "x" and "y" with the bin_edges for both
        dimensions
        """
        self.binned_data, _, _ = np.histogram2d(x=self.x_data_points,
                                                y=self.y_data_points,
                                                bins=(bin_edges.x,
                                                      bin_edges.y))

    def _calculate_gof_values(self):
        # TODO: Fixme
        # Two-dimensional binned events & tests:
        #  # best-fit model is: best_fit_hist_generated_mm
        #  hist_data = best_fit_hist_generated_mm.similar_blank_hist()
        #  hist_data.add(unbinned_ll._data)
        #
        #  gof_binned_chi_square_2d = (hist_data.histogram -
        #                              best_fit_hist_generated_mm.histogram
        #                              )**2 / best_fit_hist_generated_mm.histogram
        #  gof_binned_chi_square_2d[
        #      best_fit_hist_generated_mm.histogram <= 0.] = 0.
        #  gof_binned_chi_square_2d = np.sum(gof_binned_chi_square_2d)
        #
        #  gof_binned_poisson_2d = np.sum(
        #      sps.poisson(best_fit_hist_generated_mm.histogram.flatten()).logpmf(
        #          hist_data.histogram.flatten()))
        #
        #  gof_binned_chi_square_2d_rebin2 = (
        #      hist_data.rebin(2, 2).histogram - best_fit_hist_generated_mm.rebin(
        #          2, 2).histogram)**2 / best_fit_hist_generated_mm.rebin(
        #              2, 2).histogram
        #  gof_binned_chi_square_2d_rebin2[
        #      best_fit_hist_generated_mm.rebin(2, 2).histogram <= 0.] = 0.
        #  gof_binned_chi_square_2d_rebin2 = np.sum(
        #      gof_binned_chi_square_2d_rebin2)
        #  gof_binned_chi_square_2d_rebin3 = (
        #      hist_data.rebin(3, 3).histogram - best_fit_hist_generated_mm.rebin(
        #          3, 3).histogram)**2 / best_fit_hist_generated_mm.rebin(
        #              3, 3).histogram
        #  gof_binned_chi_square_2d_rebin3[
        #      best_fit_hist_generated_mm.rebin(3, 3).histogram <= 0.] = 0.
        #  gof_binned_chi_square_2d_rebin3 = np.sum(
        #      gof_binned_chi_square_2d_rebin3)
        results_dict = OrderedDict([
            #  ("GOF_binned_chisquare_2d", gof_binned_chi_square_2d),
            #  ("GOF_binned_chisquare_2d_rebin2",
            #   gof_binned_chi_square_2d_rebin2),
            #  ("GOF_binned_chisquare_2d_rebin3",
            #   gof_binned_chi_square_2d_rebin3),
            #  ("GOF_binned_poisson_2d", gof_binned_poisson_2d),
        ])

        return results_dict
