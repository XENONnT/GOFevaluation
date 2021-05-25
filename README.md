# GOFevaluation
Evaluate the Goodness-of-Fit (GoF) for binned or unbinned data.  
![Test package](https://github.com/XENONnT/GOFevaluation/actions/workflows/python-package.yml/badge.svg)

This GoF suite comprises the possibility to calculate different 1D / nD, binned / two-sample (unbinned) GoF measures and the corresponding p-value. A list of implemented measures is given below. 

 
## Implemented GoF measures
| GoF measure                   | Class                     |    data input   | reference input | dim |
|-------------------------------|---------------------------|:---------------:|:---------------:|:---:|
| Kolmogorov-Smirnov            | `kstest_gof`              |      sample     |      binned     |  1D |
| Two-Sample Kolmogorov-Smirnov | `kstest_two_sample_gof`   |      sample     |      sample     |  1D |
| Two-Sample Anderson-Darling   | `adtest_two_sample_gof`   |      sample     |      sample     |  1D |
| Poisson Chi2                  | `binned_poisson_chi2_gof` | binned / sample |      binned     |  nD |
| Chi2                          | `binned_chi2_gof`         | binned / sample |      binned     |  nD |
| Point-to-point                | `point_to_point_gof`      |      sample     |      sample     |  nD |


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
python setup.py --user
```
You are now good to go!

## Usage
### Individual GoF Measures
Depending on your data and reference input you can initialise a `gof_object` in one of the following ways:
```python
import GOFevaluation as ge

# Data Sample + Binned PDF
gof_object = ge.binned_poisson_chi2_gof(data, pdf, bin_edges, nevents_expected)

# Binned Data + Binned PDF
gof_object = ge.binned_poisson_chi2_gof.from_binned(data, expectations)

# Data Sample + Reference Sample
gof_object = ge.point_to_point_gof(data, reference_sample)
```

With any `gof_object` you can calculate the GoF and the corresponding p-value as follows:
```python
gof = gof_object.get_gof()
p_value = gof_object.get_pvalue()
```

### Multiple GoF Measures at once
You can compute GoF and p-values for multiple measures at once with the `evaluate_gof` class. 

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
gof_object = ge.evaluate_gof(data_sample=data_sample, 
                             reference_sample=reference_sample,
                             gof_list=['adtest_two_sample_gof', 
                                       'kstest_two_sample_gof', 
                                       'point_to_point_gof'])
# Calculate GoFs and p-values:
d_min = 0.01
gof_object.get_gofs(d_min=d_min)
# OUTPUT:
# OrderedDict([('adtest_two_sample_gof', 1.6301454042304904),
#              ('kstest_two_sample_gof', 0.14),
#              ('point_to_point_gof', 0.00048491049630050576)])

gof_object.get_pvalues(d_min=d_min)
# OUTPUT:
# OrderedDict([('adtest_two_sample_gof', 0.061000000000000054),
#             ('kstest_two_sample_gof', 0.10699999999999998),
#             ('point_to_point_gof', 0.119)])

print(gof_object)
# OUTPUT:
# GOFevaluation.evaluate_gof
# GoF measures: adtest_two_sample_gof, kstest_two_sample_gof, point_to_point_gof
# gofs = 1.6301454042304904, 0.14, 0.00048491049630050576
# p-values = 0.06999999999999995, 0.09899999999999998, 0.125
```




## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.
