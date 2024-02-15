import scipy.stats as sps
import numpy as np
import unittest

from GOFevaluation.utils import equiprobable_histogram
from GOFevaluation.utils import plot_equiprobable_histogram


class TestPlotEqualProbable(unittest.TestCase):
    def test_plot_arguments(self):
        n_data = 10
        n_reference = 100
        data_sample = sps.uniform(-3, 6).rvs(size=[n_data, 2])
        x = sps.norm().rvs(size=n_reference)
        y = sps.uniform(-3, 6).rvs(size=n_reference)
        reference_sample = np.stack([x, y]).T

        n_partitions = [5, 3]
        nevents_expected = sps.poisson(mu=n_data).rvs()

        order = [0, 1]
        _, bin_edges = equiprobable_histogram(
            data_sample, reference_sample, n_partitions, order=order, plot=False
        )

        kwargs = {}
        plot_equiprobable_histogram(
            data_sample=data_sample,
            bin_edges=bin_edges,
            order=order,
            reference_sample=reference_sample,
            nevents_expected=nevents_expected,
            plot_mode="sigma_deviation",
            **kwargs
        )

        plot_xlim = [-3, 3]
        plot_ylim = [-3, 3]
        plot_equiprobable_histogram(
            data_sample=data_sample,
            bin_edges=bin_edges,
            order=order,
            reference_sample=reference_sample,
            nevents_expected=nevents_expected,
            plot_xlim=plot_xlim,
            plot_ylim=plot_ylim,
            plot_mode="sigma_deviation",
            **kwargs
        )

        kwargs = {"vmin": -1, "vmax": 2}
        plot_equiprobable_histogram(
            data_sample=data_sample,
            bin_edges=bin_edges,
            order=order,
            reference_sample=reference_sample,
            nevents_expected=nevents_expected,
            plot_xlim=plot_xlim,
            plot_ylim=plot_ylim,
            plot_mode="sigma_deviation",
            **kwargs
        )

        kwargs = {"vmin": -3, "vmax": 3}
        plot_equiprobable_histogram(
            data_sample=data_sample,
            bin_edges=bin_edges,
            order=order,
            reference_sample=reference_sample,
            nevents_expected=nevents_expected,
            plot_xlim=plot_xlim,
            plot_ylim=plot_ylim,
            plot_mode="sigma_deviation",
            **kwargs
        )

        order = [1, 0]
        _, bin_edges = equiprobable_histogram(
            data_sample, reference_sample, n_partitions, order=order, plot=False
        )

        kwargs = {"vmin": -1, "vmax": 1}
        plot_equiprobable_histogram(
            data_sample=data_sample,
            bin_edges=bin_edges,
            order=order,
            nevents_expected=nevents_expected,
            plot_xlim=plot_xlim,
            plot_ylim=plot_ylim,
            plot_mode="num_counts",
            **kwargs
        )

        plot_equiprobable_histogram(
            data_sample=data_sample,
            bin_edges=bin_edges,
            order=order,
            nevents_expected=nevents_expected,
            plot_mode="count_density",
            **kwargs
        )

        try:
            error_raised = True
            plot_equiprobable_histogram(
                data_sample=data_sample,
                bin_edges=bin_edges,
                order=order,
                nevents_expected=nevents_expected,
                plot_xlim=plot_xlim,
                plot_ylim=plot_ylim,
                plot_mode="count_density",
                **kwargs
            )
            error_raised = False
        except Exception:
            print("Error correctly raised when count_density" " mode with x or y limit specified")
        else:
            if not error_raised:
                raise RuntimeError(
                    "Should raise error when count_density" " mode with x or y limit specified"
                )


if __name__ == "__main__":
    unittest.main()
