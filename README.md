# pysampled

[![PyPI - Version](https://img.shields.io/pypi/v/pysampled.svg?logo=pypi&label=PyPI&logoColor=gold)](https://pypi.org/project/pysampled/)
![Supported Python Versions](https://img.shields.io/static/v1?label=python&message=>=3.7&color=green)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/praneethnamburi/pysampled/main/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/pysampled/badge/?version=latest)](https://pysampled.readthedocs.io)
[![PyPI Downloads](https://img.shields.io/pypi/dw/pysampled.svg?label=PyPI%20downloads)](
https://pypi.org/project/pysampled/)
[![Downloads](https://pepy.tech/badge/pysampled)](https://pepy.tech/project/pysampled)

*Tools for working with uniformly sampled (time series) data.*

## Installation

You can install `pysampled` via PyPI, Conda-Forge, or directly from the GitHub repository. Follow the instructions below based on your preferred method.

---

**1. Installing from PyPI (Recommended)**

```sh
pip install pysampled && download-airpls
```

You can optionally use `pip install pysampled[minimal]` to skip installing scikit-learn and matplotlib.

> *Note:* The `download-airpls` command is defined in `pyproject.toml` and ensures that the required `airPLS.py` file is properly downloaded. More information on airPLS [here](https://github.com/zmzhang/airPLS/tree/master).



**2. Installing from the GitHub Repository (For Development Versions)**
For the latest (possibly unstable) version directly from GitHub:

```sh
pip install git+https://github.com/yourusername/pysampled.git && download-airpls
```

---


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
