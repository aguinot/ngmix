ngmix
=====

Gaussian mixture models and other code for working with for 2d images,
implemented in python.   The code is made fast using the numba package.  Note
the old c-extension based code is still available in the tag v0.9.5

For some examples, please see [the wiki](https://github.com/esheldon/ngmix/wiki).

dependencies
------------

* numpy
* numba

optional dependencies
---------------------
* scipy: optional needed for image fitting using the Levenberg-Marquardt fitter
* galsim: optional for doing metacalibration operations.
* skikit-learn:  for sampling multivariate PDFs
* statsmodels: optional for importance sampling (multivariate student
    T distribution)
* emcee: optional for doing MCMC fitting: http://dan.iel.fm/emcee/current/ Affine invariant MCMC sampler.

installation
------------

With conda
```bash
conda install ngmix
```
For the above to work you may need to add the conda-forge channel
```bash
conda config --add channels conda-forge
```

From source
```bash
python setup.py install
```
If installing from source, you will need to also install numba.
By far the easiest way is using conda

```bash
conda install numba
```
