# GOFevaluation
Evaluate the Goodness-of-Fit (GOF) for binned or unbinned data.
![Test package](https://github.com/XENONnT/GOFevaluation/actions/workflows/python-package.yml/badge.svg)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/XENONnT/GOFevaluation/HEAD)
[![PyPI version shields.io](https://img.shields.io/pypi/v/GOFevaluation.svg)](https://pypi.python.org/pypi/GOFevaluation/)
[![CodeFactor](https://www.codefactor.io/repository/github/xenonnt/gofevaluation/badge)](https://www.codefactor.io/repository/github/xenonnt/gofevaluation)
[![Coverage Status](https://coveralls.io/repos/github/XENONnT/GOFevaluation/badge.svg?branch=master)](https://coveralls.io/github/XENONnT/GOFevaluation?branch=master)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/XENONnT/GOFevaluation/master.svg)](https://results.pre-commit.ci/latest/github/XENONnT/GOFevaluation/master)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5626909.svg)](https://doi.org/10.5281/zenodo.5626909)

This GOF suite comprises the possibility to calculate different 1D / nD, binned / two-sample (unbinned) GOF measures and the corresponding approximate p-value. A list of implemented measures is given below.


## Implemented GOF measures
| GOF measure                   | Class                     |    data input   | reference input | dim |
|-------------------------------|---------------------------|:---------------:|:---------------:|:---:|
| Kolmogorov-Smirnov            | `KSTestGOF`               |      sample     |      binned     |  1D |
| Two-Sample Kolmogorov-Smirnov | `KSTestTwoSampleGOF`      |      sample     |      sample     |  1D |
| Two-Sample Anderson-Darling   | `ADTestTwoSampleGOF`      |      sample     |      sample     |  1D |
| Poisson Chi2                  | `BinnedPoissonChi2GOF`    | binned / sample |      binned     |  nD |
| Chi2                          | `BinnedChi2GOF`           | binned / sample |      binned     |  nD |
| Point-to-point                | `PointToPointGOF`         |      sample     |      sample     |  nD |


## Installation and Set-Up

### Regular installation:
```
pip install GOFevaluation
```

### Developer setup:
Clone the repository:

```
git clone https://github.com/XENONnT/GOFevaluation
cd GOFevaluation
```
Install the requirements in your environment:
```
pip install -r requirements.txt
```

Then install the package:
```
python setup.py install --user
```
You are now good to go!

## Usage
The best way to start with the `GOFevaluation` package is to have a look at the tutorial notebook. If you click on the [mybinder](https://mybinder.org/v2/gh/XENONnT/GOFevaluation/HEAD) badge, you can execute the interactive notebook and give it a try yourself without the need of a local installation.
### Individual GOF Measures
Depending on your data and reference input you can initialise a `gof_object` in one of the following ways:
```python
import GOFevaluation as ge

# Data Sample + Binned PDF
gof_object = ge.BinnedPoissonChi2GOF(data_sample, pdf, bin_edges, nevents_expected)

# Binned Data + Binned PDF
gof_object = ge.BinnedPoissonChi2GOF.from_binned(binned_data, binned_reference)

# Data Sample + Reference Sample
gof_object = ge.PointToPointGOF(data_sample, reference_sample)
```

With any `gof_object` you can calculate the GOF and the corresponding p-value as follows:
```python
gof = gof_object.get_gof()
p_value = gof_object.get_pvalue()
```

### Multiple GOF Measures at once
You can compute GOF and p-values for multiple measures at once with the `GOFTest` class.

**Example:**
```python
import GOFevaluation as ge
import scipy.stats as sps

# random_state makes sure the gof values are reproducible.
# For the p-values, a slight variation is expected due to
# the random re-sampling method that is used.
data_sample = sps.uniform.rvs(size=100, random_state=200)
reference_sample = sps.uniform.rvs(size=300, random_state=201)

# Initialise all two-sample GOF measures:
gof_object = ge.GOFTest(data_sample=data_sample,
                        reference_sample=reference_sample,
                        gof_list=['ADTestTwoSampleGOF',
                                  'KSTestTwoSampleGOF',
                                  'PointToPointGOF'])
# Calculate GOFs and p-values:
d_min = 0.01
gof_object.get_gofs(d_min=d_min)
# OUTPUT:
# OrderedDict([('ADTestTwoSampleGOF', 1.6301454042304904),
#              ('KSTestTwoSampleGOF', 0.14),
#              ('PointToPointGOF', -0.7324060759792504)])

gof_object.get_pvalues(d_min=d_min)
# OUTPUT:
# OrderedDict([('ADTestTwoSampleGOF', 0.08699999999999997),
#              ('KSTestTwoSampleGOF', 0.10699999999999998),
#              ('PointToPointGOF', 0.31200000000000006)])

# Re-calculate p-value only for one measure:
gof_object.get_pvalues(d_min=.001, gof_list=['PointToPointGOF'])
# OUTPUT:
# OrderedDict([('ADTestTwoSampleGOF', 0.08699999999999997),
#              ('KSTestTwoSampleGOF', 0.10699999999999998),
#              ('PointToPointGOF', 0.128)])

print(gof_object)
# OUTPUT:
# GOFevaluation.gof_test
# GOF measures: ADTestTwoSampleGOF, KSTestTwoSampleGOF, PointToPointGOF


# ADTestTwoSampleGOF
# gof = 1.6301454042304904
# p-value = 0.08499999999999996

# KSTestTwoSampleGOF
# gof = 0.13999999999999996
# p-value = 0.09799999999999998

# PointToPointGOF
# gof = -0.7324060759792504
# p-value = 0.128
```




## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.
