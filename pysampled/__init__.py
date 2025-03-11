"""
:py:mod:`pysampled` provides tools for working with uniformly sampled (time series) data.

:class:`pysampled.Data` is the most important class in this module. It allows for easy signal splicing, and includes wrappers for basic signal processing techniques. The :class:`pysampled.Data` class encapsulates signal values (data) with the sampling rate and provides wrappers for performing basic signal processing. It uses the :class:`pysampled.Time` class to ease the burden of managing time and converting between time (in seconds) and sample numbers.

Classes:
    Data: Provides various signal processing methods for sampled data.
    
    Time: Encapsulates sampling rate, sample number, and time for sampled data.
    Interval: Represents an interval with start and end times, includes iterator protocol.
    
    Siglets: A collection of signal pieces for event-triggered analyses.
    
    RunningWin: Manages running windows for data processing.
    DataList: A list of :class:`pysampled.Data` objects with filtering capabilities based on metadata.
    Event: An interval with labels for event handling.
    Events: A list of Event objects with label-based selection.

Functions:
    uniform_resample: Uniformly resamples a signal at a given sampling rate.
    
    interpnan: Interpolates NaNs in a 1D signal.
    onoff_samples: Finds onset and offset samples of a 1D boolean signal.
    frac_power: Calculates the fraction of power in a specific frequency band.

Examples:
    CAUTION: In this module, when referring to time, integers are interpreted as sample numbers, and floats are interpreted as time in seconds.

    .. code-block:: python

        sig = Data(np.random.random((10, 3)), sr=2, t0=5.) # 5 seconds
        x3 = sig[5.:5.05]()
        x3.interval().end
        x3[:1]()                                           # retrieve the first sample (NOT until 1 s)
        x3[0:5.5](), x3[5.0:5.5]()

        sig.apply_running_win(lambda x: np.sqrt(np.mean(x**2)), win_size=0.25, win_inc=0.1)
"""

from .__version__ import __version__
from .core import (
    Data,
    Time,
    Interval,
    Siglets,
    RunningWin,
    DataList,
    Event,
    Events,
    uniform_resample,
)
