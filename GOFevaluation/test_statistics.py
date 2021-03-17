import numpy as np


class test_statistics(object):

    def __init__(self, data, expectation):
        self.name = self.__class__.__name__
        self.data = data
        self.expectation = expectation

    def calculate_gof(self):
        raise NotImplementedError

    def bin_data(self, bin_edges):
        self.binned_data, _ = np.histogram(self.data, bins=bin_edges)

    def get_result_as_dict(self):
        value = self.calculate_gof()
        return {self.name: value}
