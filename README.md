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
python setup.py develop --user
```
You are now good to go!

## Usage
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

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
