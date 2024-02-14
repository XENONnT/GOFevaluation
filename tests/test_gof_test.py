import scipy.stats as sps
import numpy as np
import unittest
from collections import OrderedDict

from GOFevaluation.evaluators_1d import ADTestTwoSampleGOF
from GOFevaluation.evaluators_1d import KSTestTwoSampleGOF
from GOFevaluation.evaluators_nd import BinnedPoissonChi2GOF
from GOFevaluation.evaluators_nd import BinnedChi2GOF
from GOFevaluation.evaluators_nd import PointToPointGOF
from GOFevaluation.gof_test import GOFTest


class TestGOFTest(unittest.TestCase):

    def test_gof(self):
        """Check if gof values of wrapper object is the same as for individual
        calculation."""

        # Generate data and reference (as sample and binned) to use
        # to calculate all GOFs at once
        model = sps.uniform
        nevents_expected = 300
        data_sample = model.rvs(size=nevents_expected)
        reference_sample = model.rvs(size=nevents_expected * 3)
        bin_edges = np.linspace(0, 1, 11)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        binned_data, _ = np.histogram(data_sample, bins=bin_edges)
        pdf = model.pdf(bin_centers)
        pdf /= np.sum(pdf)
        binned_reference = pdf * nevents_expected

        d_min = 0.01  # define this manually

        n_partitions = 10

        # Calculate GOF with wrapper:
        gof_list = GOFTest.allowed_gof_str

        gof_object = GOFTest(
            gof_list=gof_list,
            data_sample=data_sample,
            reference_sample=reference_sample,
            binned_data=binned_data,
            binned_reference=binned_reference,
            pdf=pdf,
            nevents_expected=nevents_expected,
            bin_edges=bin_edges,
            n_partitions=n_partitions,
        )

        gofs_wrapper = gof_object.get_gofs(d_min=d_min)

        # Calculate GOFs individually: (skip kstest_gof for now)
        gofs_individual = OrderedDict()

        gof_measure_dict_individual = {
            "ADTestTwoSampleGOF": ADTestTwoSampleGOF(
                data_sample=data_sample, reference_sample=reference_sample
            ),
            # 'kstest_gof': kstest_gof(
            #     data_sample=data_sample,
            #     pdf=pdf,
            #     bin_edges=bin_edges),
            "KSTestTwoSampleGOF": KSTestTwoSampleGOF(
                data_sample=data_sample, reference_sample=reference_sample
            ),
            "BinnedPoissonChi2GOF": BinnedPoissonChi2GOF(
                data_sample=data_sample,
                pdf=pdf,
                bin_edges=bin_edges,
                nevents_expected=nevents_expected,
            ),
            "BinnedPoissonChi2GOF.from_binned": BinnedPoissonChi2GOF.from_binned(
                binned_data=binned_data, binned_reference=binned_reference
            ),
            "BinnedPoissonChi2GOF.bin_equiprobable": BinnedPoissonChi2GOF.bin_equiprobable(
                data_sample=data_sample,
                reference_sample=reference_sample,
                nevents_expected=nevents_expected,
                n_partitions=n_partitions,
            ),
            "BinnedChi2GOF": BinnedChi2GOF(
                data_sample=data_sample,
                pdf=pdf,
                bin_edges=bin_edges,
                nevents_expected=nevents_expected,
            ),
            "BinnedChi2GOF.from_binned": BinnedChi2GOF.from_binned(
                binned_data=binned_data, binned_reference=binned_reference
            ),
            "BinnedChi2GOF.bin_equiprobable": BinnedChi2GOF.bin_equiprobable(
                data_sample=data_sample,
                reference_sample=reference_sample,
                nevents_expected=nevents_expected,
                n_partitions=n_partitions,
            ),
            "PointToPointGOF": PointToPointGOF(
                data_sample=data_sample, reference_sample=reference_sample
            ),
        }

        for key in gof_measure_dict_individual:
            if key == "PointToPointGOF":
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
