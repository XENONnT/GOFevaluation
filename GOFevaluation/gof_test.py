from collections import OrderedDict
from inspect import signature
import warnings

from GOFevaluation.evaluators_1d import ADTestTwoSampleGOF  # noqa: F401
from GOFevaluation.evaluators_1d import KSTestTwoSampleGOF  # noqa: F401
from GOFevaluation.evaluators_nd import BinnedPoissonChi2GOF  # noqa: F401
from GOFevaluation.evaluators_nd import BinnedChi2GOF  # noqa: F401
from GOFevaluation.evaluators_nd import PointToPointGOF  # noqa: F401


class GOFTest(object):
    """This wrapper class is meant to streamline the creation of commonly used function
    calls of the package.

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
        "ADTestTwoSampleGOF",
        "KSTestTwoSampleGOF",
        "BinnedPoissonChi2GOF",
        "BinnedPoissonChi2GOF.from_binned",
        "BinnedPoissonChi2GOF.bin_equiprobable",
        "BinnedChi2GOF",
        "BinnedChi2GOF.from_binned",
        "BinnedChi2GOF.bin_equiprobable",
        "PointToPointGOF",
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
                self.gofs[gof_str] = None
                self.pvalues[gof_str] = None
            else:
                allowed_str = ", ".join(['"' + str(p) + '"' for p in self.allowed_gof_str])
                raise ValueError(
                    f'"{gof_str}" is not an allowed value '
                    f"for gof_list. Possible values are "
                    f"{allowed_str}"
                )

            specific_kwargs = self._get_specific_kwargs(func, kwargs)
            self.gof_objects[gof_str] = func(**specific_kwargs)
            kwargs_used += specific_kwargs.keys()
        self._check_kwargs_used(kwargs_used, kwargs)

    def __repr__(self):
        return f"{self.__class__.__module__}:\n{self.__dict__}"

    def __str__(self):
        args = ["GOF measures: " + ", ".join(self.gof_list)] + ["\n"]
        for key, gof in self.gofs.items():
            pval = self.pvalues[key]
            results_str = "\033[1m" + key + "\033[0m" + "\n"
            results_str += f"gof = {gof}\n"
            results_str += f"p-value = {pval}\n"
            args.append(results_str)
        args_str = "\n".join(args)
        return f"{self.__class__.__module__}\n{args_str}"

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
            if (kwarg not in kwargs_used) and (kwarg != "gof_list"):
                warnings.warn(f"Keyword argument {kwarg} was not used!")

    def get_gofs(self, **kwargs):
        """Calculate GOF values for individual GOF measures.

        :param kwargs: All parameters from a get_gof method are viable kwargs.
            gof_list: A list of gof_measures for which GOF should be
            calculated. If none is given, the GOF value is calculated for all
            GOF measures defined at initialisation.

        """
        kwargs_used = []  # check if all kwargs were used
        for key in kwargs.get("gof_list", self.gof_objects):
            func = self.gof_objects[key].get_gof
            specific_kwargs = self._get_specific_kwargs(func, kwargs)
            gof = func(**specific_kwargs)
            self.gofs[key] = gof
            kwargs_used += specific_kwargs.keys()
        self._check_kwargs_used(kwargs_used, kwargs)
        return self.gofs

    def get_pvalues(self, **kwargs):
        """Calculate the approximate p-values for individual GOF measures.

        :param kwargs: All parameters from a get_pvalue method are viable kwargs.
            gof_list: A list of gof_measures for which p-value should be
            calculated. If none is given, the p-value is calculated for all
            GOF measures defined at initialisation.

        """
        kwargs_used = []  # check if all kwargs were used
        for key in kwargs.get("gof_list", self.gof_objects):
            func = self.gof_objects[key].get_pvalue
            specific_kwargs = self._get_specific_kwargs(func, kwargs)
            pvalue = func(**specific_kwargs)
            self.pvalues[key] = pvalue
            self.gofs[key] = self.gof_objects[key].gof
            kwargs_used += specific_kwargs.keys()
        self._check_kwargs_used(kwargs_used, kwargs)
        return self.pvalues
