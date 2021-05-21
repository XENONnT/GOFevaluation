from GOFevaluation import adtest_two_sample_gof
from GOFevaluation import kstest_gof
from GOFevaluation import kstest_two_sample_gof
from GOFevaluation import binned_poisson_chi2_gof
from GOFevaluation import binned_chi2_gof
from GOFevaluation import point_to_point_gof


class evaluate_gof(object):
    """This wrapper class is meant to streamline the creation of commonly used
    function calls of the package"""

    gof_measure_dict = {
        'adtest_two_sample_gof': adtest_two_sample_gof,
        'kstest_gof': kstest_gof,
        'kstest_two_sample_gof': kstest_two_sample_gof,
        'binned_poisson_chi2_gof': binned_poisson_chi2_gof,
        'binned_chi2_gof': binned_chi2_gof,
        'point_to_point_gof': point_to_point_gof
    }

    def __init__(self, gof_list, **kwargs):
        self.gof_objects = dict()
        for key in gof_list:
            self.gof_objects[key] = self.gof_measure_dict[key](**kwargs)

    @classmethod
    def from_binned(cls):
        pass

    def get_gofs(self):
        pass

    def get_pvalues(self):
        pass
