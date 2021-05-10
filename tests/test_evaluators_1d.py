# import GOFevaluation
# import scipy.stats as sps
# import numpy as np


# bin_edges = np.linspace(-4, 4, 101)
# bin_centers = bin_edges[:-1] + (bin_edges[1:] - bin_edges[:-1]) / 2
# bin_widths = bin_edges[1:] - bin_edges[:-1]
# normed_gauss_pdf = sps.norm.pdf(bin_centers) * bin_widths

# n_events = 1000

# expected_events = n_events * normed_gauss_pdf
# data_points = sps.norm.rvs(size=1000)


# from GOFevaluation import binned_poisson_gof
# from GOFevaluation import chi2_gof


# evaluator = GOFevaluation.evaluators_1d(data=data_points,
#        pdf=normed_gauss_pdf,
#        nevents_expected=n_events,
#        bin_edges=bin_edges)
# results = evaluator.calculate_gof_values()
# print(results)

# test = GOFevaluation.kstest_gof(data=data_points, pdf=normed_gauss_pdf, nevents_expected=n_events, bin_edges=bin_edges)
# test.calculate_gof()
