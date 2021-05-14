# GOFevaluation
![Test package](https://github.com/XENONnT/GOFevaluation/actions/workflows/python-package.yml/badge.svg)

## Implemented GoF measures
| GoF measure                   | Class                     |    data input   | reference input | dim |         GoF        |       p-value      |
|-------------------------------|---------------------------|:---------------:|:---------------:|:---:|:------------------:|:------------------:|
| Kolmogorov-Smirnov            | `kstest_gof`              |      sample     |      binned     |  1D | :white_check_mark: |                    |
| Two-Sample Kolmogorov-Smirnov | `kstest_two_sample_gof`   |      sample     |      sample     |  1D | :white_check_mark: | :white_check_mark: |
| Two-Sample Anderson-Darling   | `adtest_two_sample_gof`   |      sample     |      sample     |  1D | :white_check_mark: | :white_check_mark: |
| Poisson Chi2                  | `binned_poisson_chi2_gof` | binned / sample |      binned     |  nD | :white_check_mark: | :white_check_mark: |
| Chi2                          | `binned_chi2_gof`         | binned / sample |      binned     |  nD | :white_check_mark: | :white_check_mark: |
| Point-to-point                | `point_to_point_gof`      |      sample     |      sample     |  nD | :white_check_mark: | :white_check_mark: |
## Installation

1. Choose the environment of your liking. ```source activate strax``` is recommended.
2. Install the package with ```python setup.py develop --user```
