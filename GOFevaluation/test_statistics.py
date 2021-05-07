import numpy as np

class test_statistics(object):
    def __init__(self, data, pdf, nevents_expected):
        self.data = data
        self.pdf = pdf
        self.nevents = len(data)
        self.nevents_expected = nevents_expected
        self.expected_events = self.pdf * self.nevents_expected
        self._name = None

    def calculate_gof(self):
        raise NotImplementedError("Your goodnes of fit computation goes here!")

    def bin_data(self, bin_edges):
        # function to bin nD data:
        if len(self.data.shape)==1:
            self.binned_data, _ = np.histogram(self.data, bins=bin_edges)
        else:
            self.binned_data, _ = np.histogramdd(self.data, bins=bin_edges)

    def get_result_as_dict(self):
        assert self._name is not None, str(self.__class__.__name__) + ": You need to define self._name for your goodnes of fit measure!"
        value = self.calculate_gof()
        return {self._name: value}
