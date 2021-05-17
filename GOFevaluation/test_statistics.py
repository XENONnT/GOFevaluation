import numpy as np


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

    def bin_data(self, bin_edges):
        # function to bin nD data:
        if len(self.data.shape) == 1:
            self.binned_data, _ = np.histogram(self.data, bins=bin_edges)
        else:
            self.binned_data, _ = np.histogramdd(self.data, bins=bin_edges)

        assert (self.binned_data.shape == self.pdf.shape), \
            "Shape of binned data doesn not match shape of pdf!"


class test_statistics_sample(test_statistics_core):
    def __init__(self, data, reference_sample):
        test_statistics_core.__init__(self=self,
                                      data=data)
        self.reference_sample = reference_sample
