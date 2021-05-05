import GOFevaluation
import scipy.stats as sps
import numpy as np

from GOFevaluation import binned_poisson_chi2_gof


def test_dimensions():
    #test nD binned GOF in different dimensions:
    signal_expectation = 100
    xs = np.linspace(0, 1, 100)
    for nD in [1, 2, 3, 5]:
        data = np.vstack([xs for i in range(nD)]).T
        bins = [np.linspace(0, 1, 3) for i in range(nD)]
        mus = np.ones([2 for i in range(nD)])
        mus /= np.sum(mus)

        nbins = 2**nD
        ns_flat = np.zeros(nbins)
        ns_flat[0:2] = 50
        mus_flat = np.ones(nbins)
        mus_flat /= np.sum(mus_flat)
        gof_flat = 2 * np.sum(
            sps.poisson(ns_flat).logpmf(ns_flat) -
            sps.poisson(mus_flat * signal_expectation).logpmf(ns_flat))

        gofclass = binned_poisson_chi2_gof(data, mus, bins, signal_expectation)
        gof = gofclass.calculate_gof()

        assert abs(gof_flat - gof) < 1e-8


if __name__ == "__main__":
    test_dimensions()
    print("nd tests passed")
