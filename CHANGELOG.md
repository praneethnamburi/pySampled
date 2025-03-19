# Change Log
All notable changes to this project will be documented in this file.

## [1.1.0]

### Added
Added "indexing" functionality to `Data` class. For example, if an instance of data represented 6 signals with 1000 samples each coming from two 3-axis accelerometers acc1, and acc2 each with coordinates x, y, and z, then we can simply index subsections of this signal using `s["acc1"]` or `s["x"]` or `s["acc2"]["x"]`. 


## [1.0.2] - 2025-03-12
 
First major release.
