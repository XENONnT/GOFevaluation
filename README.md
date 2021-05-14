# GOFevaluation
![Test package](https://github.com/XENONnT/GOFevaluation/actions/workflows/python-package.yml/badge.svg)

## Implemented GoF measures
| GoF measure                   | Class                     |    data input   | reference input | dim |
|-------------------------------|---------------------------|:---------------:|:---------------:|:---:|
| Kolmogorov-Smirnov            | `kstest_gof`              |      sample     |      binned     |  1D |
| Two-Sample Kolmogorov-Smirnov | `kstest_two_sample_gof`   |      sample     |      sample     |  1D |
| Two-Sample Anderson-Darling   | `adtest_two_sample_gof`   |      sample     |      sample     |  1D |
| Poisson Chi2                  | `binned_poisson_chi2_gof` | binned / sample |      binned     |  nD |
| Chi2                          | `binned_chi2_gof`         | binned / sample |      binned     |  nD |
| Point-to-point                | `point_to_point_gof`      |      sample     |      sample     |  nD |
## Installation

1. Choose the environment of your liking. ```source activate strax``` is recommended.
2. Install the package with ```python setup.py develop --user```
