from collections import OrderedDict
from inspect import signature
import warnings

from GOFevaluation import ADTestTwoSampleGOF
from GOFevaluation import KSTestTwoSampleGOF
from GOFevaluation import BinnedPoissonChi2GOF
from GOFevaluation import BinnedChi2GOF
from GOFevaluation import PointToPointGOF


class GOFTest(object):
    """This wrapper class is meant to streamline the creation of commonly used
    function calls of the package

    :param gof_list: list of strings referring to the gof measures from
        evaluators_nd and evaluators_1d
    :param gof_list: list
    :param kwargs: All parameters of evaluators_nd and evaluators_1d are viable
        kwargs
    :raises ValueError: If gof_list contains not allowed strings (see below)

    .. note::
        * Possible entries in gof_list:
            * 'ADTestTwoSampleGOF',
            * 'KSTestTwoSampleGOF',
            * 'BinnedPoissonChi2GOF',
            * 'BinnedPoissonChi2GOF.from_binned',
            * 'BinnedPoissonChi2GOF.bin_equiprobable',
            * 'BinnedChi2GOF',
            * 'BinnedChi2GOF.from_binned',
            * 'BinnedChi2GOF.bin_equiprobable',
            * 'PointToPointGOF'
        * A user warning is issued if unused keyword arguments are passed.
    """

    allowed_gof_str = [
        'ADTestTwoSampleGOF',
        'KSTestTwoSampleGOF',
        'BinnedPoissonChi2GOF',
        'BinnedPoissonChi2GOF.from_binned',
        'BinnedPoissonChi2GOF.bin_equiprobable',
        'BinnedChi2GOF',
        'BinnedChi2GOF.from_binned',
        'BinnedChi2GOF.bin_equiprobable',
        'PointToPointGOF'
    ]

    def __init__(self, gof_list, **kwargs):
        self.gof_objects = OrderedDict()
        self.gof_list = gof_list
        self.gofs = OrderedDict()
        self.pvalues = OrderedDict()
        kwargs_used = []  # check if all kwargs were used
        for gof_str in self.gof_list:
            if gof_str in self.allowed_gof_str:
                func = eval(gof_str)
            else:
                allowed_str = ", ".join(
                    ['"' + str(p) + '"' for p in self.allowed_gof_str])
                raise ValueError(f'"{gof_str}" is not an allowed value '
                                 f'for gof_list. Possible values are '
                                 f'{allowed_str}')

            specific_kwargs = self._get_specific_kwargs(func, kwargs)
            self.gof_objects[gof_str] = func(**specific_kwargs)
            kwargs_used += specific_kwargs.keys()
        self._check_kwargs_used(kwargs_used, kwargs)

    def __repr__(self):
        return f'{self.__class__.__module__}:\n{self.__dict__}'

    def __str__(self):
        args = ['GoF measures: ' + ", ".join(self.gof_list)]
        if self.gofs:
            gofs_str = ", ".join([str(g) for g in self.gofs.values()])
            args.append('gofs = ' + gofs_str)
        if self.pvalues:
            pvalues_str = ", ".join([str(p) for p in self.pvalues.values()])
            args.append('p-values = ' + pvalues_str)
        args_str = "\n".join(args)
        return f'{self.__class__.__module__}\n{args_str}'

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
        """Check if all kwargs were used, issue a warning if not."""
        for kwarg in kwargs:
            if (kwarg not in kwargs_used) and (kwarg != 'gof_list'):
                warnings.warn(f'Keyword argument {kwarg} was not used!')

    def get_gofs(self, **kwargs):
        """Calculate GoF values for individual GoF measures.

        :param kwargs: All parameters from a get_gof method are viable kwargs.
            gof_list: A list of gof_measures for which GoF should be
            calculated. If none is given, the GoF value is calculated for all
            GoF measures defined at initialisation.
        """
        kwargs_used = []  # check if all kwargs were used
        for key in kwargs.get('gof_list', self.gof_objects):
            func = self.gof_objects[key].get_gof
            specific_kwargs = self._get_specific_kwargs(func, kwargs)
            gof = func(**specific_kwargs)
            self.gofs[key] = gof
            kwargs_used += specific_kwargs.keys()
        self._check_kwargs_used(kwargs_used, kwargs)
        return self.gofs

    def get_pvalues(self, **kwargs):
        """Calculate p-values for individual GoF measures.

        :param kwargs: All parameters from a get_pvalue method are viable kwargs.
            gof_list: A list of gof_measures for which p-value should be
            calculated. If none is given, the p-value is calculated for all
            GoF measures defined at initialisation.
        """
        kwargs_used = []  # check if all kwargs were used
        for key in kwargs.get('gof_list', self.gof_objects):
            func = self.gof_objects[key].get_pvalue
            specific_kwargs = self._get_specific_kwargs(func, kwargs)
            pvalue = func(**specific_kwargs)
            self.pvalues[key] = pvalue
            kwargs_used += specific_kwargs.keys()
        self._check_kwargs_used(kwargs_used, kwargs)
        return self.pvalues
