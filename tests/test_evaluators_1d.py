import GOFevaluation
import scipy.stats as sps
import numpy as np


bin_edges = np.linspace(-4, 4, 101)
bin_centers = bin_edges[:-1] + (bin_edges[1:] - bin_edges[:-1]) / 2
bin_widths = bin_edges[1:] - bin_edges[:-1]
normed_gauss_pdf = sps.norm.pdf(bin_centers) * bin_widths

n_events = 1000

expected_events = n_events * normed_gauss_pdf
data_points = sps.norm.rvs(size=1000)

evaluator = GOFevaluation.evaluators_1d(expected_events=expected_events,
        data_points=data_points,
        bin_edges=bin_edges)
result = evaluator._calculate_gof_values()
print(result)
