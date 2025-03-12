# pysampled

[![src](https://img.shields.io/badge/src-github-blue)](https://github.com/praneethnamburi/pysampled)
[![PyPI - Version](https://img.shields.io/pypi/v/pysampled.svg?logo=pypi&label=PyPI&logoColor=gold)](https://pypi.org/project/pysampled/)
[![Documentation Status](https://readthedocs.org/projects/pysampled/badge/?version=latest)](https://pysampled.readthedocs.io)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/praneethnamburi/pysampled/main/LICENSE)

*A toolkit for working with uniformly sampled time series data.*

`pysampled` streamlines the exploration of time series data and the development of signal processing pipelines. It enables researchers and engineers to analyze time series data—including audio signals and physiological data—efficiently and intuitively. With its user-friendly interface and well-documented examples, the package makes signal processing accessible for both basic manipulations and analyses like filtering, resampling, trend extraction, and spectral analysis.

## Installation

**1. Installing from PyPI (Recommended)**

```sh
pip install pysampled && download-airpls
```

You can optionally use `pip install pysampled[minimal]` to skip installing scikit-learn and matplotlib.

> *Note:* The `download-airpls` command is defined in `pyproject.toml` and ensures that the required `airPLS.py` file is properly downloaded. More information on airPLS [here](https://github.com/zmzhang/airPLS/tree/master).



**2. Installing from the GitHub Repository**

```sh
pip install git+https://github.com/praneethnamburi/pysampled.git && download-airpls
```

Alternatively, you can clone the repository locally and set up your environment using the `requirements.yml` file. If you do this, download `airPLS.py` manually from [here](https://github.com/zmzhang/airPLS/tree/master) and add it to the `pysampled` folder inside the cloned repository.

```sh
git clone https://github.com/praneethnamburi/pysampled.git
cd pysampled
conda env create -n pysampled -f requirements.yml
```


## Quickstart

```python
import pysampled as sampled

# Generate a 10 Hz signal sampled at 100 Hz. Sum of three sine waves (1, 3, and 5 Hz).
sig = sampled.generate_signal("three_sine_waves")[:5.0] 

# Only keep first 5 seconds of the signal
sig = sig[:5.0]

# visualize the signal, before and after applying a bandpass filter between 2 and 4 Hz
sampled.plot([sig, sig.bandpass(2, 4)])
```


## License

This project is licensed under the MIT License. See the `LICENSE` file for details.


## Contact

[Praneeth Namburi](https://praneethnamburi.com)

Project Link: [https://github.com/praneethnamburi/pysampled](https://github.com/praneethnamburi/pysampled)


## Acknowledgments

This tool was developed as part of the ImmersionToolbox initiative at the [MIT.nano Immersion Lab](https://immersion.mit.edu). Thanks to NCSOFT for supporting this initiative.
