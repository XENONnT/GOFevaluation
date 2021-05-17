import numpy as np
import scipy.stats as sps


class test_statistics_core(object):
    def __init__(self, data):
        self.data = data
        self.nevents = len(data)
        self._name = None

    @classmethod
    def calculate_gof(cls):
        raise NotImplementedError("calculate_gof mus be implemented!")

    def get_gof(self):
        raise NotImplementedError("Your goodnes of fit computation goes here!")

    def get_result_as_dict(self):
        assert self._name is not None, (str(self.__class__.__name__)
                                        + ": You need to define self._name "
                                        + "for your goodnes of fit measure!")
        value = self.get_gof()
        return {self._name: value}


class test_statistics(test_statistics_core):
    def __init__(self, data, pdf, nevents_expected):
        test_statistics_core.__init__(self=self,
                                      data=data)
        self.pdf = pdf
        if nevents_expected is not None:
            self.nevents_expected = nevents_expected
            self.expected_events = self.pdf * self.nevents_expected

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
                samples, self.pdf * self.nevents_expected)
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
        try:
            pvalue = cumulative_density[index_pvalue]
        except IndexError:
            raise ValueError(
                f'Index {index_pvalue} is out of bounds. '
                + 'Not enough MC\'s run!')
        return pvalue


class test_statistics_sample(test_statistics_core):
    def __init__(self, data, reference_sample):
        test_statistics_core.__init__(self=self,
                                      data=data)
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
        try:
            pvalue = cumulative_density[index_pvalue]
        except IndexError:
            raise ValueError(
                f'Index {index_pvalue} is out of bounds. '
                + 'Not enough permutations run!')
        return pvalue
