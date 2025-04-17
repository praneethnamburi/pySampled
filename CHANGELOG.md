# Change Log
All notable changes to this project will be documented in this file.

## [1.1.1]

### Changed
Bugfixes in the `pysampled.Data.apply` method for issues related to the indexing functionality introduced in 1.1.0. Also corrected `ValueError` in `pysampled.Data.__getitem__` to `KeyError`.

## [1.1.0]

### Added
Added "indexing" functionality to `Data` class. For example, if an instance of data represented 6 signals with 1000 samples each coming from two 3-axis accelerometers acc1, and acc2 each with coordinates x, y, and z, then we can simply index subsections of this signal using `s["acc1"]` or `s["x"]` or `s["acc2"]["x"]`. 


## [1.0.2] - 2025-03-12
 
First major release.
