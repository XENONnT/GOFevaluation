# GOFevaluation
![Test package](https://github.com/XENONnT/GOFevaluation/actions/workflows/python-package.yml/badge.svg)

## Implemented GoF measures
| GoF measure        | Class                                |   In: binned data  |  In: unbinned data |   In: binned pdf   | In: sample from pdf |       In: 1D       |       In: nD       |      Out: GoF      |    Out: p-value    |
|--------------------|--------------------------------------|:------------------:|:------------------:|:------------------:|:-------------------:|:------------------:|:------------------:|:------------------:|:------------------:|
| Kolmogorov-Smirnov | `kstest_gof` `kstest_two_sample_gof` | :white_check_mark: | :white_check_mark: | :white_check_mark: |  :white_check_mark: | :white_check_mark: |         :x:        | :white_check_mark: |                    |
| Anderson-Darling   | `adtest_two_sample_gof`              |         :x:        | :white_check_mark: |         :x:        |  :white_check_mark: | :white_check_mark: |         :x:        | :white_check_mark: |                    |
| Poisson Chi2       | `binned_poisson_chi2_gof`            | :white_check_mark: | :white_check_mark: | :white_check_mark: |                     | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| Chi2               | `binned_chi2_gof`                    | :white_check_mark: | :white_check_mark: | :white_check_mark: |                     | :white_check_mark: | :white_check_mark: | :white_check_mark: |                    |
| Point-to-point     | `point_to_point_gof`                 |         :x:        | :white_check_mark: |         :x:        |  :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |                    |               |                 |
## Installation

1. Choose the environment of your liking. ```source activate strax``` is recommended.
2. Install the package with ```python setup.py develop --user```
