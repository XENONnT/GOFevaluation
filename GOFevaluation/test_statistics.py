import numpy as np
import scipy.stats as sps


class test_statistics_core(object):
    """Parent class for all test statistic classes"""

    def __init__(self, data):
        self.data = data
        self._name = self.__class__.__name__

    @staticmethod
    def calculate_gof():
        raise NotImplementedError("calculate_gof is not implemented yet!")

    def get_gof(self,  **_):
        raise NotImplementedError("get_gof is not implemented yet!")

    def get_result_as_dict(self):
        value = self.get_gof()
        return {self._name: value}


class test_statistics(test_statistics_core):
    """Test statistics class for binned expectations reference input."""

    def __init__(self, data, pdf, bin_edges, nevents_expected):
        super().__init__(data=data)
        self.pdf = pdf
        self.expected_events = self.pdf * nevents_expected

        if bin_edges is None:
            assert (data.shape == pdf.shape), \
                "Shape of binned data does not match shape of the pdf!"
            self.binned_data = data
        else:
            self.bin_data(bin_edges=bin_edges)
        return

    @classmethod
    def from_binned(cls, data, expectations):
        """Initialize with already binned data + expectations
        """
        # bin_edges=None will set binned_data=data
        return cls(data=data,
                   pdf=expectations / np.sum(expectations),
                   bin_edges=None,
                   nevents_expected=np.sum(expectations))

    def bin_data(self, bin_edges):
        # function to bin nD data:
        if len(self.data.shape) == 1:
            self.binned_data, _ = np.histogram(self.data, bins=bin_edges)
        else:
            self.binned_data, _ = np.histogramdd(self.data, bins=bin_edges)

        assert (self.binned_data.shape == self.pdf.shape), \
            "Shape of binned data doesn not match shape of pdf!"

    def sample_gofs(self, n_mc=1000):
        """Sample n_mc random GoF's

        Simulates random data from the PDF and calculates its GoF n_mc times
        """
        fake_gofs = np.zeros(n_mc)
        for i in range(n_mc):
            samples = sps.poisson(self.expected_events).rvs()
            fake_gofs[i] = self.calculate_gof(
                samples, self.expected_events)
        return fake_gofs

    def get_pvalue(self, n_mc=1000):
        """Get the p-value of the data under the null hypothesis

        Gets the distribution of the GoF statistic, and compares it to the
        GoF of the data given the expectations.
        """
        if not hasattr(self, 'gof'):
            _ = self.get_gof()
        fake_gofs = self.sample_gofs(n_mc=n_mc)
        hist, bin_edges = np.histogram(fake_gofs, bins=1000)
        # add 0 bin to the front and truncate cumulative_density
        # at the end to get pvalue[0] = 1
        hist = np.concatenate([[0], hist])
        cumulative_density = (1.0 - np.cumsum(hist) / np.sum(hist))[:-1]
        index_pvalue = np.digitize(self.gof, bin_edges) - 1

        if index_pvalue == (len(hist) - 1):
            raise ValueError(
                f'Index {index_pvalue} is out of bounds. '
                + 'Not enough MC\'s run!')
        elif index_pvalue == -1:
            raise ValueError(
                f'Index {index_pvalue} is out of bounds. '
                + 'Not enough MC\'s run!')
        else:
            pvalue = cumulative_density[index_pvalue]

        return pvalue


class test_statistics_pdf(test_statistics_core):
    """Test statistics class for sample data, binned pdf reference input."""

    def __init__(self, data, pdf):
        super().__init__(data=data)
        self.pdf = pdf

    def get_pvalue(self):
        # This method is not implemented yet. We are working on adding it in
        # the near future.
        raise NotImplementedError("p-value computation not yet implemented!")


class test_statistics_sample(test_statistics_core):
    """Test statistics class for sample data and reference input."""

    def __init__(self, data, reference_sample):
        super().__init__(data=data)
        self.reference_sample = reference_sample

    def permutation_gofs(self, n_perm=1000, d_min=None):
        """Get n_perm GoF's by randomly permutating data and reference sample
        """
        n_data = len(self.data)
        mixed_sample = np.concatenate([self.data, self.reference_sample],
                                      axis=0)
        fake_gofs = np.zeros(n_perm)
        for i in range(n_perm):
            rng = np.random.default_rng()
            rng.shuffle(mixed_sample, axis=0)

            data_perm = mixed_sample[:n_data]
            reference_perm = mixed_sample[n_data:]
            if d_min is not None:
                fake_gofs[i] = self.calculate_gof(
                    data=data_perm, reference_sample=reference_perm,
                    d_min=d_min)
            else:
                fake_gofs[i] = self.calculate_gof(
                    data=data_perm, reference_sample=reference_perm)
        return fake_gofs

    def get_pvalue(self, n_perm=1000, d_min=None):
        """Get the p-value of the data under the null hypothesis

        Computes the p-value by means of a permutation test of data sample
        and reference sample.
        """
        if not hasattr(self, 'gof'):
            if d_min is not None:
                _ = self.get_gof(d_min=d_min)
            else:
                _ = self.get_gof()
        if d_min is not None:
            fake_gofs = self.permutation_gofs(n_perm=n_perm, d_min=d_min)
        else:
            fake_gofs = self.permutation_gofs(n_perm=n_perm)
        hist, bin_edges = np.histogram(fake_gofs, bins=1000)
        # add 0 bin to the front and truncate cumulative_density
        # at the end to get pvalue[0] = 1
        hist = np.concatenate([[0], hist])
        cumulative_density = (1.0 - np.cumsum(hist) / np.sum(hist))[:-1]

        index_pvalue = np.digitize(self.gof, bin_edges) - 1

        if index_pvalue == (len(hist) - 1):
            raise ValueError(
                f'Index {index_pvalue} is out of bounds. '
                + 'Not enough permutations run!')
        elif index_pvalue == -1:
            raise ValueError(
                f'Index {index_pvalue} is out of bounds. '
                + 'Not enough permutations run!')
        else:
            pvalue = cumulative_density[index_pvalue]

        return pvalue
