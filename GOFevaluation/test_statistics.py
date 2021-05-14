import numpy as np
import scipy.stats as sps


class test_statistics_core(object):
    def __init__(self, data):
        self.data = data
        self.nevents = len(data)
        self._name = None

    def calculate_gof(self):
        raise NotImplementedError("Your goodnes of fit computation goes here!")

    def get_result_as_dict(self):
        assert self._name is not None, (str(self.__class__.__name__)
                                        + ": You need to define self._name "
                                        + "for your goodnes of fit measure!")
        value = self.calculate_gof()
        return {self._name: value}


class test_statistics(test_statistics_core):
    def __init__(self, data, pdf, nevents_expected):
        test_statistics_core.__init__(self=self,
                                      data=data)
        self.pdf = pdf
        if nevents_expected is not None:
            self.nevents_expected = nevents_expected
            self.expected_events = self.pdf * self.nevents_expected

    @classmethod
    def calculate_binned_gof(cls, binned_data, binned_expectations):
        raise NotImplementedError("calculate_binned_gof mus be implemented!")

    def bin_data(self, bin_edges):
        # function to bin nD data:
        if len(self.data.shape) == 1:
            self.binned_data, _ = np.histogram(self.data, bins=bin_edges)
        else:
            self.binned_data, _ = np.histogramdd(self.data, bins=bin_edges)

        assert (self.binned_data.shape == self.pdf.shape), \
            "Shape of binned data doesn not match shape of pdf!"

    def sample_gofs(self, n_mc=1000):
        """Sample n_mc random Chi2 GoF's

        Simulates random data from the PDF and calculates its GoF n_mc times
        """
        fake_gofs = np.zeros(n_mc)
        for i in range(n_mc):
            samples = sps.poisson(self.pdf * self.nevents_expected).rvs()
            fake_gofs[i] = self.calculate_binned_gof(
                samples, self.pdf * self.nevents_expected)
        return fake_gofs

    def get_pvalue(self, n_mc=1000):
        """Get the p-value of the data under the null hypothesis

        Gets the distribution of the GoF statistic, and compares it to the
        GoF of the data given the expectations.
        """
        if not hasattr(self, 'gof'):
            _ = self.calculate_gof()
        fake_gofs = self.sample_gofs(n_mc=n_mc)
        hist, bin_edges = np.histogram(fake_gofs, bins=1000)
        cumulative_density = 1.0 - np.cumsum(hist) / np.sum(hist)
        try:
            pvalue = cumulative_density[np.digitize(self.gof, bin_edges) - 1]
        except IndexError:
            raise ValueError(
                'Not enough MC\'s run -- GoF is outside toy distribution!')
        return pvalue


class test_statistics_sample(test_statistics_core):
    def __init__(self, data, reference_sample):
        test_statistics_core.__init__(self=self,
                                      data=data)
        self.reference_sample = reference_sample
