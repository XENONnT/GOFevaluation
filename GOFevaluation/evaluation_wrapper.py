# from GOFevaluation import evaluators_1d
from GOFevaluation import evaluators_nd
from GOFevaluation import test_statistics


class evaluation_wrapper(test_statistics):
    """This wrapper class is meant to streamline the creation of commonly used
    function calls of the package"""

    def __init__(self, data, pdf, bin_edges, nevents_expected):
        self.gofs = dict()
        self.gofs["binned_chi2_gof"] = evaluators_nd.binned_chi2_gof(
            data=data,
            pdf=pdf,
            bin_edges=bin_edges,
            nevents_expected=nevents_expected)
        self.gofs[
            "binned_poisson_chi2_gof"] = evaluators_nd.binned_poisson_chi2_gof(
                data=data,
                pdf=pdf,
                bin_edges=bin_edges,
                nevents_expected=nevents_expected)

    @classmethod
    def from_binned(cls, data, expectations):
        """Initialize with already binned data + expectations

        In this case the bin-edges don't matter, so we bypass the usual init
        """
        self = cls(None, None, None, None)
        self.gofs = dict()
        self.gofs[
            "binned_chi2_gof"] = evaluators_nd.binned_chi2_gof.from_binned(
                data=data, expectations=expectations)
        self.gofs[
            "binned_poisson_chi2_gof"] = evaluators_nd.binned_poisson_chi2_gof.from_binned(
                data=data, expectations=expectations)
        return self

    def get_pvalues(self):
        return {key: obj.get_pvalue() for key, obj in self.gof_objects.items()}
