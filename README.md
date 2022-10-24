# hpar-reader
Tools to read and extract TUWien harmonic parameters.


## Description
Package containing tools for reading and manipulating TUWien Harmonic Parameters, specifically for generating Day of
Year Sentinel-1 SIG0 backscatter estimate.


## Installation

Dependencies:
-  [yeoda](https://github.com/TUW-GEO/yeoda)

ATTENTION: Packages like gdal, cartopy, or geopandas need more OS support and have more dependencies than other packages
and can therefore not be installed solely via pip. Thus, for a fresh setup, an existing environment with the conda
dependencies listed in conda_env.yml is expected. To create such an environment, you can run:

```
conda create -n "hpar-reader" -c conda-forge python=3.8 mamba
conda activate hpar-reader
mamba install -c conda-forge gdal geopandas cartopy yeoda
```

## Usage
after installation a commandline interface accessing using `hpar_reader` should be available.

commandline inputs include:

* `-path` - file path to the HPAR file directory
* `-sg` - suggrid of the HPAR to be read, using 2 character continent code e.g. 'EU'
* `-t` - [Equi7grid]() tile name to be generated.
* `-o` - Sentinel-1 relative orbit e.g. `A175`
*  `-date` - date to be estimated in `YYYYMMDD` format.
