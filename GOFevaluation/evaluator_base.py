import numpy as np
import scipy.stats as sps
import warnings
import numbers
from GOFevaluation.utils import equiprobable_histogram
from GOFevaluation.utils import apply_irregular_binning
from GOFevaluation.utils import plot_equiprobable_histogram
from GOFevaluation.utils import check_sample_sanity


class EvaluatorBase(object):
    """Parent class for all evaluator base classes."""

    def __init__(self):
        self._name = self.__class__.__name__
        self.gof = None
        self.pvalue = None

    def __repr__(self):
        return f"{self.__class__.__module__}.{self.__class__.__qualname__} ({self.__dict__.keys()})"

    def __str__(self):
        args = [self._name]
        if self.gof is not None:
            args.append(f"gof = {self.gof}")
        if self.pvalue is not None:
            args.append(f"p-value = {self.pvalue}")
        args_str = "\n".join(args)
        return f"{self.__class__.__module__}\n{args_str}"

    @staticmethod
    def calculate_gof():
        raise NotImplementedError("calculate_gof is not implemented yet!")

    def get_gof(self):
        raise NotImplementedError("get_gof is not implemented yet!")

    def get_pvalue(self):
        raise NotImplementedError("get_pvalue is not implemented yet!")


class EvaluatorBaseBinned(EvaluatorBase):
    """Evaluator base class for binned expectations reference input."""

    def __init__(self, data_sample, pdf, bin_edges, nevents_expected):
        check_sample_sanity(data_sample)
        super().__init__()
        self.pdf = pdf
        assert isinstance(nevents_expected, numbers.Real) and not isinstance(
            nevents_expected, bool
        ), f"nevents_expected must be numeric but is of type {type(nevents_expected)}."
        self.binned_reference = self.pdf * nevents_expected

        if bin_edges is None:
            # In this case data_sample is binned data!
            assert (
                data_sample.shape == pdf.shape
            ), "Shape of binned data does not match shape of the pdf!"

            # Convert to int. Make sure the deviation is purely from
            # dtype conversion, i.e. the values provided are actually
            # bin counts and not float values.
            binned_data_int = np.abs(data_sample.round(0).astype(int))
            assert np.sum(np.abs(data_sample - binned_data_int)) < 1e-10, (
                "Deviation encounterd when converting dtype of binned_data to"
                "int. Make sure binned_data contains natural numbers!"
            )
            self.binned_data = binned_data_int
        else:
            self.bin_data(data_sample=data_sample, bin_edges=bin_edges)
        return

    @classmethod
    def from_binned(cls, binned_data, binned_reference):
        """Initialize with already binned data + expectations."""
        # bin_edges=None will set self.binned_data=binned_data
        # in the init
        return cls(
            data_sample=binned_data,
            pdf=binned_reference / np.sum(binned_reference),
            bin_edges=None,
            nevents_expected=np.sum(binned_reference),
        )

    @classmethod
    def bin_equiprobable(
        cls,
        data_sample,
        reference_sample,
        nevents_expected,
        n_partitions,
        order=None,
        plot=False,
        plot_mode="sigma_deviation",
        reference_sample_weights=None,
        **kwargs,
    ):
        """Initialize with data and reference sample that are binned such that the
        expectation value is the same in each bin.

        kwargs are passed to `plot_equiprobable_histogram` if plot is True.

        """
        check_sample_sanity(data_sample)
        check_sample_sanity(reference_sample)
        if len(reference_sample) < 50 * len(data_sample):
            warnings.warn(
                f"Number of reference samples ({len(reference_sample)}) "
                + "should be much larger than number of data samples "
                + f"({len(data_sample)}) to ensure negligible statistical "
                + "fluctuations for the equiprobable binning.",
                stacklevel=2,
            )

        pdf, bin_edges = equiprobable_histogram(
            data_sample=reference_sample,
            reference_sample=reference_sample,
            n_partitions=n_partitions,
            order=order,
            data_sample_weights=reference_sample_weights,
            reference_sample_weights=reference_sample_weights,
            **kwargs,
        )
        pdf = pdf / np.sum(pdf)

        expected_one_hot = pdf.ravel() * nevents_expected
        if not np.allclose(expected_one_hot, expected_one_hot[0], rtol=1e-4, atol=1e-2):
            warnings.warn(
                "Could not achieve exact equiprobable binning, "
                + "the approximately equiprobable binning is used which "
                + f"has a minimum expectation value of {min(expected_one_hot):.3f}"
                + f"  and a maximum expectation value of {max(expected_one_hot):.3f}.",
                stacklevel=2,
            )

        binned_data = apply_irregular_binning(
            data_sample=data_sample, bin_edges=bin_edges, order=order
        )

        if plot:
            plot_equiprobable_histogram(
                data_sample=data_sample,
                bin_edges=bin_edges,
                order=order,
                reference_sample=reference_sample,
                nevents_expected=nevents_expected,
                reference_sample_weights=reference_sample_weights,
                plot_mode=plot_mode,
                **kwargs,
            )

        # bin_edges=None will set self.binned_data=binned_data
        # in the init
        return cls(
            data_sample=binned_data, pdf=pdf, bin_edges=None, nevents_expected=nevents_expected
        )

    def bin_data(self, data_sample, bin_edges):
        """Function to bin nD data sample."""
        if len(data_sample.shape) == 1:
            self.binned_data, _ = np.histogram(data_sample, bins=bin_edges)
        else:
            self.binned_data, _ = np.histogramdd(data_sample, bins=bin_edges)

        assert (
            self.binned_data.shape == self.pdf.shape
        ), "Shape of binned data doesn not match shape of pdf!"

    def sample_gofs(self, n_mc=1000):
        """Generate fake GOFs for toy data sampled from binned reference.

        :param n_mc: Number of fake-gofs calculated, defaults to 1000
        :type n_mc: int, optional
        :return: Array of fake GOFs
        :rtype: array_like

        """
        fake_gofs = np.zeros(n_mc)
        for i in range(n_mc):
            samples = sps.poisson(self.binned_reference).rvs()
            fake_gofs[i] = self.calculate_gof(samples, self.binned_reference)
        return fake_gofs

    def _get_pvalue(self, n_mc=1000):
        if self.gof is None:
            _ = self.get_gof()
        fake_gofs = self.sample_gofs(n_mc=n_mc)
        percentile = sps.percentileofscore(fake_gofs, self.gof, kind="strict")
        pvalue = 1 - percentile / 100

        if pvalue == 0:
            warnings.warn(
                f"p-value is 0.0. (Observed GOF: "
                f"{self.gof:.2e}, maximum of simulated GOFs: "
                f"{max(fake_gofs):.2e}). For a more "
                f"precise result, increase n_mc!",
                stacklevel=2,
            )
        elif pvalue == 1:
            warnings.warn(
                f"p-value is 1.0. (Observed GOF "
                f"{self.gof:.2e}, minimum of simulated GOFs: "
                f"{min(fake_gofs):.2e}). For a more "
                f"precise result, increase n_mc!",
                stacklevel=2,
            )

        self.pvalue = pvalue
        return pvalue, fake_gofs

    def get_pvalue(self, n_mc=1000):
        """The approximate p-value is calculated.

        Computes the p-value by means of generating toyMCs and calculating their GOF.
        The p-value can then be obtained from the distribution of these fake-gofs.

        Note that this is only an approximate method, since the model is not refitted
        to the toy data. Especially with low statistics and many fit parameters, this
        can introduce a bias towards larger p-values.

        :param n_mc: Number of fake-gofs calculated, defaults to 1000
        :type n_mc: int, optional
        :return: p-value
        :rtype: float

        """
        pvalue, _ = self._get_pvalue(n_mc=n_mc)
        return pvalue

    def get_pvalue_return_fake_gofs(self, n_mc=1000):
        """The approximate p-value is calculated.

        Computes the p-value by means of generating toyMCs and calculating their GOF.
        The p-value can then be obtained from the distribution of these fake-gofs. The
        array of fake-gofs is returned together with the p-value.

        Note that this is only an approximate method, since the model is not refitted
        to the toy data. Especially with low statistics and many fit parameters, this
        can introduce a bias towards larger p-values.

        :param n_mc: Number of fake-gofs calculated, defaults to 1000
        :type n_mc: int, optional
        :return: p-value
        :rtype: float

        """
        pvalue, fake_gofs = self._get_pvalue(n_mc=n_mc)
        return pvalue, fake_gofs


class EvaluatorBasePdf(EvaluatorBase):
    """Evaluator base class for sample data, binned pdf reference input."""

    def __init__(self, data_sample, pdf):
        check_sample_sanity(data_sample)
        super().__init__()
        self.data_sample = data_sample
        self.pdf = pdf

    def get_pvalue(self):
        # This method is not implemented yet. We are working on adding it in
        # the near future.
        raise NotImplementedError("p-value computation not yet implemented!")


class EvaluatorBaseSample(EvaluatorBase):
    """Evaluator base class for sample data and reference input."""

    def __init__(self, data_sample, reference_sample):
        check_sample_sanity(data_sample)
        check_sample_sanity(reference_sample)
        super().__init__()
        self.data_sample = data_sample
        self.reference_sample = reference_sample

    def permutation_gofs(self, n_perm=1000, d_min=None):
        """Generate fake GOFs by re-sampling data and reference sample.

        :param n_perm: Number of fake-gofs calculated, defaults to 1000
        :type n_perm: int, optional
        :param d_min: Only for PointToPointGOF, defaults to None
        :type d_min: float, optional
        :return: Array of fake GOFs
        :rtype: array_like

        """
        n_data = len(self.data_sample)
        mixed_sample = np.concatenate([self.data_sample, self.reference_sample], axis=0)
        fake_gofs = np.zeros(n_perm)
        for i in range(n_perm):
            rng = np.random.default_rng()
            rng.shuffle(mixed_sample, axis=0)

            data_perm = mixed_sample[:n_data]
            reference_perm = mixed_sample[n_data:]
            if d_min is not None:
                fake_gofs[i] = self.calculate_gof(
                    data_sample=data_perm, reference_sample=reference_perm, d_min=d_min
                )
            else:
                fake_gofs[i] = self.calculate_gof(
                    data_sample=data_perm, reference_sample=reference_perm
                )
        return fake_gofs

    def _get_pvalue(self, n_perm=1000, d_min=None):
        if self.gof is None:
            if d_min is not None:
                _ = self.get_gof(d_min=d_min)
            else:
                _ = self.get_gof()
        fake_gofs = self.permutation_gofs(n_perm=n_perm, d_min=d_min)

        percentile = sps.percentileofscore(fake_gofs, self.gof, kind="strict")
        pvalue = 1 - percentile / 100

        if pvalue == 0:
            warnings.warn(
                f"p-value is 0.0. (Observed GOF: "
                f"{self.gof:.2e}, maximum of simulated GOFs: "
                f"{max(fake_gofs):.2e}). For a more "
                f"precise result, increase n_perm!",
                stacklevel=2,
            )
        elif pvalue == 1:
            warnings.warn(
                f"p-value is 1.0. (Observed GOF "
                f"{self.gof:.2e}, minimum of simulated GOFs: "
                f"{min(fake_gofs):.2e}). For a more "
                f"precise result, increase n_perm!",
                stacklevel=2,
            )
        self.pvalue = pvalue

        return pvalue, fake_gofs

    def get_pvalue(self, n_perm=1000, d_min=None):
        """The approximate p-value is calculated.

        Computes the p-value by means of re-sampling data sample and reference sample.
        For each re-sampling, the gof is calculated. The p-value can then be obtained
        from the distribution of these fake-gofs.

        Note that this is only an approximate method, since the model is not refitted
        to the re-sampled data. Especially with low statistics and many fit parameters,
        this can introduce a bias towards larger p-values.

        :param n_perm: Number of fake-gofs calculated, defaults to 1000
        :type n_perm: int, optional
        :return: p-value
        :rtype: float

        """
        pvalue, _ = self._get_pvalue(n_perm=n_perm, d_min=d_min)
        return pvalue

    def get_pvalue_return_fake_gofs(self, n_perm=1000, d_min=None):
        """The approximate p-value is calculated.

        Computes the p-value by means of re-sampling data sample and reference sample.
        For each re-sampling, the gof is calculated. The p-value can then be obtained
        from the distribution of these fake-gofs. The array of fake-gofs is returned
        together with the p-value.

        Note that this is only an approximate method, since the model is not refitted
        to the re-sampled. Especially with low statistics and many fit parameters, this
        can introduce a bias towards larger p-values.

        :param n_perm: Number of fake-gofs calculated, defaults to 1000
        :type n_perm: int, optional
        :return: p-value, fake_gofs
        :rtype: float

        """
        pvalue, fake_gofs = self._get_pvalue(n_perm=n_perm, d_min=d_min)
        return pvalue, fake_gofs
