# GOFevaluation
Evaluate the Goodness-of-Fit (GoF) for binned or unbinned data.  
![Test package](https://github.com/XENONnT/GOFevaluation/actions/workflows/python-package.yml/badge.svg)

This GoF suite comprises the possibility to calculate different 1D / nD, binned / two-sample (unbinned) GoF measures and the corresponding p-value. A list of implemented measures is given below. 

 
## Implemented GoF measures
| GoF measure                   | Class                     |    data input   | reference input | dim |
|-------------------------------|---------------------------|:---------------:|:---------------:|:---:|
| Kolmogorov-Smirnov            | `KSTestGOF`               |      sample     |      binned     |  1D |
| Two-Sample Kolmogorov-Smirnov | `KSTestTwoSampleGOF`      |      sample     |      sample     |  1D |
| Two-Sample Anderson-Darling   | `ADTestTwoSampleGOF`      |      sample     |      sample     |  1D |
| Poisson Chi2                  | `BinnedPoissonChi2GOF`    | binned / sample |      binned     |  nD |
| Chi2                          | `BinnedChi2GOF`           | binned / sample |      binned     |  nD |
| Point-to-point                | `PointToPointGOF`         |      sample     |      sample     |  nD |


## Installation and Set-Up

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
### Individual GoF Measures
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

With any `gof_object` you can calculate the GoF and the corresponding p-value as follows:
```python
gof = gof_object.get_gof()
p_value = gof_object.get_pvalue()
```

### Multiple GoF Measures at once
You can compute GoF and p-values for multiple measures at once with the `GOFTest` class. 

**Example:**
```python
import GOFevaluation as ge
import scipy.stats as sps

# random_state makes sure the gof values are reproducible.
# For the p-values, a slight variation is expected due to
# the random re-sampling method that is used.
data_sample = sps.uniform.rvs(size=100, random_state=200)
reference_sample = sps.uniform.rvs(size=300, random_state=201)

# Initialise all two-sample GoF measures:
gof_object = ge.GOFTest(data_sample=data_sample, 
                        reference_sample=reference_sample,
                        gof_list=['ADTestTwoSampleGOF', 
                                  'KSTestTwoSampleGOF', 
                                  'PointToPointGOF'])
# Calculate GoFs and p-values:
d_min = 0.01
gof_object.get_gofs(d_min=d_min)
# OUTPUT:
# OrderedDict([('ADTestTwoSampleGOF', 1.6301454042304904),
#              ('KSTestTwoSampleGOF', 0.14),
#              ('PointToPointGOF', 0.00048491049630050576)])

gof_object.get_pvalues(d_min=d_min)
# OUTPUT:
# OrderedDict([('ADTestTwoSampleGOF', 0.08699999999999997),
#              ('KSTestTwoSampleGOF', 0.10699999999999998),
#              ('PointToPointGOF', 0.14300000000000002)])

# Re-calculate p-value only for one measure:
gof_object.get_pvalues(d_min=.3, gof_list=['PointToPointGOF'])
# OUTPUT:
# OrderedDict([('ADTestTwoSampleGOF', 0.08699999999999997),
#              ('KSTestTwoSampleGOF', 0.10699999999999998),
#              ('PointToPointGOF', 0.03400000000000003)])

print(gof_object)
# OUTPUT:
# GOFevaluation.gof_test
# GoF measures: ADTestTwoSampleGOF, KSTestTwoSampleGOF, PointToPointGOF
# gofs = 1.6301454042304904, 0.14, 0.00048491049630050576
# p-values = 0.08699999999999997, 0.10699999999999998, 0.03400000000000003
```




## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.
