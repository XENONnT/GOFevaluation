v0.1.2
==================
* Add colorbar switch, set 2D histogram x&y limit by @dachengx in #39
* Some plotting bug fixes by @hoetzsch in #41
* Homemade equiprobable_binning, still based on ECDF by @dachengx in #43
* a few patches by @hammannr in #38
* Exercise notebook by @hammannr in #44

v0.1.1
===================
* Add an example notebook that can be used as a guide when using the package for the first time (#29)
* Improve and extend plotting of equiprobable binnings. This adds the option of plotting the binwise count density (#35)

v0.1.0
===================
* Multiple GOF tests (binned and unbinned) can be performed (#1, #5, #10, #12, #13)
* The p-value is calculated based on toy sampling from the reference or a permutation test (#2, #14)
* A wrapper class makes it convenient to perform multiple GOF tests in parallel (#19)
* An equiprobable binning algorithm is implemented. The binning can be applied upon initialisation of the GOF object and a few visualization tools are provided. (#25, #26)
* CI workflow implemented (#7)
