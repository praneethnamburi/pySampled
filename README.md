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

You can install `pySampled` via PyPI, Conda-Forge, or directly from the GitHub repository. Follow the instructions below based on your preferred method.

---

### **1. Installing from PyPI (Recommended for Most Users)**

For a streamlined one-liner command:
```sh
pip install pySampled && download-airpls
```

> **Note:** The `download-airpls` command is defined in `pyproject.toml` and ensures that the required `airPLS.py` file is properly downloaded. More information on airPLS [here](https://github.com/zmzhang/airPLS/tree/master)

---

### **2. Installing from Conda-Forge**
Once `pySampled` is available on `conda-forge`, you can install it with:

```sh
conda install -c conda-forge pySampled
```

After installation, run:
```sh
download-airpls
```

> **Note:** Since `airPLS.py` is not part of `conda-forge`, this step ensures it is correctly fetched.

---

### **3. Installing from the GitHub Repository (For Development Versions)**
For the latest (possibly unstable) version directly from GitHub:

```bash
pip install git+https://github.com/yourusername/pySampled.git
```

Then run:
```bash
download-airpls
```

---


### Quickstart

### Tutorial

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.


## Contact

[Praneeth Namburi](https://praneethnamburi.com)

Project Link: [https://github.com/praneethnamburi/pysampled](https://github.com/praneethnamburi/pysampled)


## Acknowledgments

This tool was developed as part of the ImmersionToolbox initiative at the [MIT.nano Immersion Lab](https://immersion.mit.edu). Thanks to NCSOFT for supporting this initiative.
