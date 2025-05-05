# Python wrapper for BM4D denoising - from Tampere with love, again

Python wrapper for BM4D for stationary correlated noise (including white noise).

BM4D is an algorithm for attenuation of additive spatially correlated
stationary (aka colored) Gaussian noise for volumetric data.
This package provides a wrapper for the BM4D binaries for Python for the 
denoising of volumetric and volumetric multichannel data. For denoising of 
images/2-D multichannel data, see also the [bm3d](https://pypi.org/project/bm3d)  package.

These newer binaries (v4+) are designed only for dealing with additive Gaussian
white or correlated noise. Special features like embedded handling of Rice noise
and adaptive groupwise variance estimation are supported by the legacy binaries
v3.2 at https://webpages.tuni.fi/foi/GCF-BM3D/ .

This implementation is based on
- Y. Mäkinen, L. Azzari, A. Foi, 2020, "Collaborative Filtering of Correlated Noise: Exact Transform-Domain Variance
  for Improved Shrinkage and Patch Matching", in IEEE Transactions on Image Processing, vol. 29, pp. 8339-8354.
- M. Maggioni, V. Katkovnik, K. Egiazarian, A. Foi, 2013, "Nonlocal Transform-Domain Filter for Volumetric Data Denoising
  and Reconstruction", in IEEE Transactions on Image Processing, vol. 22, pp. 119-133.
- Y. Mäkinen, S. Marchesini, A. Foi, 2022, "Ring Artifact and Poisson Noise Attenuation via Volumetric Multiscale
  Nonlocal Collaborative Filtering of Spatially Correlated Noise", in Journal of Synchrotron Radiation, vol. 29, pp. 829-842.

The package contains the BM4D binaries compiled for:
- Windows (Win11, MinGW-64)
- Linux (Manjaro 23.1, 64-bit)
- Mac OSX (El Capitan, 64-bit): no longer maintained and will be removed from future releases.
- macOS (Sonoma, 64-bit ARM)

The package is available for non-commercial use only. For details, see LICENSE.

Basic usage:
```python
y_hat = bm4d(z, sigma)  # white noise: include noise std
y_hat = bm4d(z, psd)  # correlated noise: include noise PSD (size of z)
```

For usage examples, see the examples folder of the full source (bm4d-***.tar.gz) from https://pypi.org/project/bm4d/#files .


Contact: Ymir Mäkinen <ymir.makinen@tuni.fi> 
