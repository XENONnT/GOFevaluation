import scipy.stats as sps
import numpy as np
import unittest
from collections import OrderedDict

from GOFevaluation import adtest_two_sample_gof
# from GOFevaluation import kstest_gof
from GOFevaluation import kstest_two_sample_gof
from GOFevaluation import binned_poisson_chi2_gof
from GOFevaluation import binned_chi2_gof
from GOFevaluation import point_to_point_gof
from GOFevaluation import evaluate_gof


class Test_evaluate_gof(unittest.TestCase):

    def test_gof(self):
        """Check if gof values of wrapper object is the same as
        for individual calculation"""

        # Generate data and reference (as sample and binned) to use
        # to calculate all GoFs at once
        model = sps.uniform
        nevents_expected = 100
        data_sample = model.rvs(size=nevents_expected)
        reference_sample = model.rvs(size=nevents_expected*3)
        bin_edges = np.linspace(0, 1, 11)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        binned_data, _ = np.histogram(data_sample, bins=bin_edges)
        pdf = model.pdf(bin_centers)
        pdf /= np.sum(pdf)
        binned_reference = pdf * nevents_expected

        d_min = 0.01  # define this manually

        # Calculate GoF with wrapper:
        gof_measure_dict = evaluate_gof.gof_measure_dict
        gof_list = gof_measure_dict.keys()

        gof_object = evaluate_gof(gof_list=gof_list,
                                  data_sample=data_sample,
                                  reference_sample=reference_sample,
                                  binned_data=binned_data,
                                  binned_reference=binned_reference,
                                  pdf=pdf,
                                  nevents_expected=nevents_expected,
                                  bin_edges=bin_edges
                                  )

        gofs_wrapper = gof_object.get_gofs(d_min=d_min)

        # Calculate GoFs individually: (skip kstest_gof for now)
        gofs_individual = OrderedDict()

        gof_measure_dict_individual = {
            'adtest_two_sample_gof': adtest_two_sample_gof(
                data_sample=data_sample,
                reference_sample=reference_sample),
            # 'kstest_gof': kstest_gof(
            #     data_sample=data_sample,
            #     pdf=pdf,
            #     bin_edges=bin_edges),
            'kstest_two_sample_gof': kstest_two_sample_gof(
                data_sample=data_sample,
                reference_sample=reference_sample),
            'binned_poisson_chi2_gof': binned_poisson_chi2_gof(
                data_sample=data_sample,
                pdf=pdf,
                bin_edges=bin_edges,
                nevents_expected=nevents_expected),
            'binned_poisson_chi2_gof_from_binned':
                binned_poisson_chi2_gof.from_binned(
                binned_data=binned_data,
                binned_reference=binned_reference),
            'binned_chi2_gof': binned_chi2_gof(
                data_sample=data_sample,
                pdf=pdf,
                bin_edges=bin_edges,
                nevents_expected=nevents_expected),
            'binned_chi2_gof_from_binned':
                binned_chi2_gof.from_binned(
                binned_data=binned_data,
                binned_reference=binned_reference),
            'point_to_point_gof': point_to_point_gof(
                data_sample=data_sample,
                reference_sample=reference_sample)
        }

        for key in gof_measure_dict_individual:
            if key == 'point_to_point_gof':
                gof = gof_measure_dict_individual[key].get_gof(d_min=d_min)
            else:
                gof = gof_measure_dict_individual[key].get_gof()
            gofs_individual[key] = gof

        # Compare the results. Iterating through gofs_individual
        # rather than requireing equality of the dictionaries
        # allows for stability of the test when measures are added
        # to evaluate_gof.gof_measure_dict

        for key in gof_measure_dict_individual:
            self.assertEqual(gofs_wrapper[key], gofs_individual[key])


if __name__ == "__main__":
    unittest.main()
