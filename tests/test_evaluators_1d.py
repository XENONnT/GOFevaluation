import GOFevaluation
import scipy.stats as sps
import numpy as np


bin_edges = np.linspace(-4, 4, 101)
bin_centers = bin_edges[:-1] + (bin_edges[1:] - bin_edges[:-1]) / 2
bin_widths = bin_edges[1:] - bin_edges[:-1]
normed_gauss_pdf = sps.norm.pdf(bin_centers) * bin_widths
#
n_events = 1000

expected_events = n_events * normed_gauss_pdf
data_points = sps.norm.rvs(size=1000)
#
#  evaluator = GOFevaluation.evaluators_1d(expected_events=expected_events,
#          data_points=data_points,
#          bin_edges=bin_edges)
#  result = evaluator._calculate_gof_values()
#  print(result)

#  evaluator = GOFevaluation.binned_poisson(data=data_points, expectation=expected_events)

from GOFevaluation import binned_poisson_gof
from GOFevaluation import chi2_gof

#  test = binned_poisson_gof(data=data_points, expectation=expected_events, bin_edges=bin_edges)
#  chi2 = chi2_gof(data=data_points, expectation=expected_events, bin_edges=bin_edges, empty_bin_value=0.1)
#
#  print(test.calculate_gof())
#  print(test.get_result_as_dict())
#  print(chi2.calculate_gof())
#  print(chi2.get_result_as_dict())

evaluator = GOFevaluation.evaluators_1d(expected_events=expected_events,
        data_points=data_points,
        bin_edges=bin_edges)
results = evaluator.calculate_gof_values()
print(results)
