from collections import OrderedDict
from GOFevaluation import adtest_two_sample_gof
from GOFevaluation import kstest_gof
from GOFevaluation import kstest_two_sample_gof
from GOFevaluation import binned_poisson_chi2_gof
from GOFevaluation import binned_chi2_gof
from GOFevaluation import point_to_point_gof


class evaluate_gof(object):
    """This wrapper class is meant to streamline the creation of commonly used
    function calls of the package

    Input:
    - gof_list: list of strings referring to the gof measures from
    evaluators_nd and evaluators_1d

    - kwargs: All parameters of evaluators_nd and evaluators_1d are viable
    kwargs
    """

    gof_measure_dict = {
        'adtest_two_sample_gof': adtest_two_sample_gof,
        'kstest_gof': kstest_gof,
        'kstest_two_sample_gof': kstest_two_sample_gof,
        'binned_poisson_chi2_gof': binned_poisson_chi2_gof,
        'binned_poisson_chi2_gof_from_binned':
            binned_poisson_chi2_gof.from_binned,
        'binned_chi2_gof': binned_chi2_gof,
        'binned_chi2_gof_from_binned': binned_chi2_gof.from_binned,
        'point_to_point_gof': point_to_point_gof
    }

    def __init__(self, gof_list, **kwargs):
        self.gof_objects = OrderedDict()
        self.gof_list = gof_list
        self.gofs = None
        self.pvalues = None
        for key in self.gof_list:
            try:
                self.gof_objects[key] = self.gof_measure_dict[key](**kwargs)
            except KeyError:
                print(
                    f'Key "{key}" is not defined in {self.__class__.__name__}')

    def __repr__(self):
        return f'{self.__class__.__module__}, {self.__dict__}'

    def __str__(self):
        args = ['GoF measures: '+", ".join(self.gof_list)]
        if self.gofs:
            gofs_str = ", ".join([str(g) for g in self.gofs.values()])
            args.append('gofs = ' + gofs_str)
        if self.pvalues:
            pvalues_str = ", ".join([str(p) for p in self.pvalues.values()])
            args.append('p-values = ' + pvalues_str)
        args_str = "\n".join(args)
        return f'{self.__class__.__module__}\n{args_str}'

    def get_gofs(self, **kwargs):
        self.gofs = OrderedDict()
        for key in self.gof_objects:
            gof = self.gof_objects[key].get_gof(**kwargs)
            self.gofs[key] = gof
        return self.gofs

    def get_pvalues(self, **kwargs):
        self.pvalues = OrderedDict()
        for key in self.gof_objects:
            pvalue = self.gof_objects[key].get_pvalue(**kwargs)
            self.pvalues[key] = pvalue
        return self.pvalues
