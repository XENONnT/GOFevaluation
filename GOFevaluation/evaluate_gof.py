from collections import OrderedDict
from inspect import signature
import warnings
from GOFevaluation import adtest_two_sample_gof
# from GOFevaluation import kstest_gof
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

    Possible entries in gof_list:
    'adtest_two_sample_gof'
    'kstest_two_sample_gof'
    'binned_poisson_chi2_gof':
    'binned_poisson_chi2_gof_from_binned'
    'binned_chi2_gof'
    'binned_chi2_gof_from_binned'
    'point_to_point_gof'

    A user warning is issued if unused keyword arguments are passed.
    """

    gof_measure_dict = {
        'adtest_two_sample_gof': adtest_two_sample_gof,
        # 'kstest_gof': kstest_gof,
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
        kwargs_used = []  # check if all kwargs were used
        for key in self.gof_list:
            try:
                func = self.gof_measure_dict[key]
            except KeyError:
                raise KeyError(
                    f'Key "{key}" is not defined in {self.__class__.__name__}')

            specific_kwargs = self._get_specific_kwargs(func, kwargs)
            self.gof_objects[key] = func(**specific_kwargs)
            kwargs_used += specific_kwargs.keys()
        self._check_kwargs_used(kwargs_used, kwargs)

    @staticmethod
    def _get_specific_kwargs(func, kwargs):
        """Extract only the kwargs that are required for the function
            to ommit passing not used parameters:"""
        specific_kwargs = OrderedDict()
        params = signature(func).parameters.keys()
        for key in kwargs:
            if key in params:
                specific_kwargs[key] = kwargs[key]
        # Check if all required arguments are passed
        _ = signature(func).bind(**specific_kwargs)

        return specific_kwargs

    @staticmethod
    def _check_kwargs_used(kwargs_used, kwargs):
        for kwarg in kwargs:
            if kwarg not in kwargs_used:
                warnings.warn(f'Keyword argument {kwarg} was not used!')

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
        kwargs_used = []  # check if all kwargs were used
        self.gofs = OrderedDict()
        for key in self.gof_objects:
            func = self.gof_objects[key].get_gof
            specific_kwargs = self._get_specific_kwargs(func, kwargs)
            gof = func(**specific_kwargs)
            self.gofs[key] = gof
            kwargs_used += specific_kwargs.keys()
        self._check_kwargs_used(kwargs_used, kwargs)
        return self.gofs

    def get_pvalues(self, **kwargs):
        kwargs_used = []  # check if all kwargs were used
        self.pvalues = OrderedDict()
        for key in self.gof_objects:
            func = self.gof_objects[key].get_pvalue
            specific_kwargs = self._get_specific_kwargs(func, kwargs)
            pvalue = func(**specific_kwargs)
            self.pvalues[key] = pvalue
            kwargs_used += specific_kwargs.keys()
        self._check_kwargs_used(kwargs_used, kwargs)
        return self.pvalues
