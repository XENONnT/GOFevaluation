# scratchpad for testing
import numpy as np


test_data_x = [0, 0, 20, 1,  10, 200, 12, 234]
test_data = test_data_x
test_data_y = [0, 0, 11, 12, 234, 21, 10, 200]

bin_edges_x = [-10, 10, 301]
bin_edges_y = [-10, 10, 300]
bin_edges = bin_edges_x

hist, bin_edges = np.histogram(test_data, bins=bin_edges)
print(hist)
print(bin_edges)

X, test_edges_x, test_edges_y = np.histogram2d(x=test_data_x, y=test_data_y,
        bins=[bin_edges_x, bin_edges_y])

print(X)
print(test_edges_x)
print(test_edges_y)
