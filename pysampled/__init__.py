"""
Tools for working with uniformly sampled (time series) data.

This module provides classes and functions to handle and process sampled data, referring to uniformly sampled time series data. 

`sampled.Data is the most important class in this module. It allows for easy signal splicing, and includes wrappers for basic signal processing techniques. The `sampled.Data` class encapsulates signal values (data) with the sampling rate and provides wrappers for performing basic signal processing. It uses the `sampled.Time` class to ease the burden of managing time and converting between time (in seconds) and sample numbers.

Classes:
    Data - Provides various signal processing methods for sampled data.
    
    Time - Encapsulates sampling rate, sample number, and time for sampled data.
    Interval - Represents an interval with start and end times, includes iterator protocol.
    
    Siglets - A collection of signal pieces for event-triggered analyses.
    
    # support classes for Data
    RunningWin - Manages running windows for data processing.
    
    # classes to extend the functionality of Data and Interval classes
    DataList - A list of `sampled.Data` objects with filtering capabilities based on metadata.
    Event - An interval with labels for event handling.
    Events - A list of Event objects with label-based selection.

Functions:
    # support functions for the `sampled.Data` class
    interpnan - Interpolates NaNs in a 1D signal.
    onoff_samples - Finds onset and offset samples of a 1D boolean signal.
    uniform_resample - Uniformly resamples a signal at a given sampling rate.
    frac_power - Calculates the fraction of power in a specific frequency band.

Examples:
    CAUTION: In this module, when referring to time, integers are interpreted as sample numbers, and floats are interpreted as time in seconds.

    sig = Data(np.random.random((10, 3)), sr=2, t0=5.) # 5 seconds
    x3 = sig[5.:5.05]()
    x3.interval().end
    x3[:1]()                                           # retrieve the first sample (NOT until 1 s)
    x3[0:5.5](), x3[5.0:5.5]()

    sig.apply_running_win(lambda x: np.sqrt(np.mean(x**2)), win_size=0.25, win_inc=0.1)
"""

import numpy as np
from scipy.fft import fft, fftfreq
from scipy.integrate import simpson
from scipy.interpolate import interp1d
from scipy.signal import (butter, filtfilt, firwin, hilbert, iirnotch,
                          resample, welch)


class Time:
    """Manage timestamps when working with sampled data. 
    
    CAUTION: INTEGER IMPLIES SAMPLE NUMBER, FLOAT IMPLIES TIME IN SECONDS.
    Use this to encapsulate sampling rate (sr), sample number (sample), and time (s).
    When the sampling rate is changed, the sample number is updated, but the time is held constant.
    When the time is changed, sample number is updated.
    When the sample number is changed, the time is updated
    When working in Premiere Pro, use 29.97 fps drop-frame timecode to show the actual time in video.
    You should see semicolons instead of colons
        inp 
            (str)   hh;mm;ss;frame#
            (float) assumes provided input is time in seconds!
            (int)   assumes the provided input is the sample number
            (tuple) assumes (timestamp/time/sample, sampling rate)
        sr 
            sampling rate, in Hz. casted into a float.

    Examples:
        t = Time('00;09;53;29', 30)
        t = Time(9.32, 180)
        t = Time(12531, 180)
        t = Time((9.32, sr=180))
        t = Time((9.32, 180), 30) # DO NOT DO THIS, sampling rate will be 180
        t.time
        t.sample
    """
    def __init__(self, inp, sr=30.):
        # set the sampling rate
        if isinstance(inp, tuple):
            assert len(inp) == 2
            self._sr = float(inp[1])
            inp = inp[0] # input is now either a string, float, or int!
        else:
            self._sr = float(sr)

        # set the sample number before setting the time
        assert isinstance(inp, (str, float, int))
        if isinstance(inp, str):
            inp = [int(x) for x in inp.split(';')]
            self._sample = round((inp[0]*60*60 + inp[1]*60 + inp[2])*self.sr + inp[3])
        if isinstance(inp, float): # time to sample
            self._sample = round(inp*self.sr)
        if isinstance(inp, int):
            self._sample = inp
        
        # set the time based on the sample number
        self._time = float(self._sample)/self._sr

    @property
    def sr(self):
        return self._sr

    @sr.setter
    def sr(self, sr_val):
        """When changing the sampling rate, time is kept the same, and the sample number is NOT"""
        sr_val = float(sr_val)
        self._sr = sr_val
        self._sample = int(self._time*self._sr)
    
    def change_sr(self, new_sr):
        self.sr = new_sr
        return self

    @property
    def sample(self):
        return self._sample
    
    @sample.setter
    def sample(self, sample_val):
        self._sample = int(sample_val)
        self._time  = float(self._sample)/self._sr
    
    @property
    def time(self):
        """Return time in seconds"""
        return self._time

    @time.setter
    def time(self, s_val):
        """If time is changed, then the sample number should be reset as well"""
        self._sample = int(float(s_val)*self._sr)
        self._time = float(self._sample)/self._sr

    def __add__(self, other):
        x = self._arithmetic(other)
        return Time(x[2].__add__(x[0], x[1]), self.sr)

    def __sub__(self, other):
        x = self._arithmetic(other)
        return Time(x[2].__sub__(x[0], x[1]), self.sr)

    def _arithmetic(self, other):
        if isinstance(other, self.__class__):
            assert other.sr == self.sr
            return (self.sample, other.sample, int)
        elif isinstance(other, int):
            # integer implies sample, float implies time
            return (self.sample, other, int)
        elif isinstance(other, float):
            return (self.time, other, float)
        else:
            raise TypeError(other, "Unexpected input type! Input either a float for time, integer for sample, or time object")

    def to_interval(self, iter_rate=None):
        """Return an interval object with start and end times being the same"""
        return Interval(self, self, self.sr, iter_rate)
    
    def __repr__(self):
        return "time={:.3f} s, sample={}, sr={} Hz ".format(self.time, self.sample, self.sr) + super().__repr__()


class Interval:
    """
    Interval object with start and stop times. Implements the iterator protocol.
    INCLUDES BOTH START AND END SAMPLES
    Pictoral understanding:
    start           -> |                                           | <-
    frames          -> |   |   |   |   |   |   |   |   |   |   |   | <- [self.sr, len(self)=12, self.t_data, self.t]
    animation times -> |        |        |        |        |         <- [self.iter_rate, self._index, self.t_iter]
    Frame sampling is used to pick the nearest frame corresponding to the animation times
    Example:
        intvl = Interval(('00;09;51;03', 30), ('00;09;54;11', 30), sr=180, iter_rate=env.Key().fps)
        intvl.iter_rate = 24 # say 24 fps for animation
        for nearest_sample, time, index in intvl:
            print((nearest_sample, time, index))
    """
    def __init__(self, start, end, sr=30., iter_rate=None):
        # if isinstance(start, (int, float)) and sr is not None:
        self.start = self._process_inp(start, sr)
        self.end = self._process_inp(end, sr)

        assert self.start.sr == self.end.sr # interval is defined for a specific sampled dataset
        
        self._index = 0
        if iter_rate is None:
            self.iter_rate = self.sr # this will be the animation fps when animating data at a different rate
        else:
            self.iter_rate = float(iter_rate)

    @staticmethod
    def _process_inp(inp, sr):
        if isinstance(inp, Time):
            return inp # sr is ignored, superseded by input's sampling rate
        return Time(inp, sr) # string, float, int or tuple. sr is ignored if tuple.

    @property
    def sr(self):
        return self.start.sr
    
    @sr.setter
    def sr(self, sr_val):
        sr_val = float(sr_val)
        self.start.sr = sr_val
        self.end.sr = sr_val
        
    def change_sr(self, new_sr):
        self.sr = new_sr
        return self

    @property
    def dur_time(self):
        """Duration in seconds"""
        return self.end.time - self.start.time
    
    @property
    def dur_sample(self):
        """Duration in number of samples"""
        return self.end.sample - self.start.sample + 1 # includes both start and end samples
    
    def __len__(self):
        return self.dur_sample

    # iterator protocol - you can do: for sample, time, index in interval
    def __iter__(self):
        """Iterate from start sample to end sample"""
        return self
    
    def __next__(self):
        index_interval = 1./self.iter_rate
        if self._index <= int(self.dur_time*self.iter_rate)+1:
            time = self.start.time + self._index*index_interval
            nearest_sample = self.start.sample + int(self._index*index_interval*self.sr)
            result = (nearest_sample, time, self._index)
        else:
            self._index = 0
            raise StopIteration
        self._index += 1
        return result
    
    # time vectors
    @property
    def t_iter(self):
        """Time Vector for the interval at iteration frame rate"""
        return self._t(self.iter_rate)

    @property
    def t_data(self):
        """Time vector at the data sampling rate"""
        return self._t(self.sr)

    @property
    def t(self):
        """Time Vector relative to t_zero"""
        return self.t_data
        
    def _t(self, rate):
        _t = [self.start.time]
        while (_t[-1] + 1./rate) <= self.end.time:
            _t.append(_t[-1] + 1./rate)
        return _t

    def __add__(self, other):
        """Used to shift an interval, use union to find a union"""
        return Interval(self.start+other, self.end+other, sr=self.sr, iter_rate=self.iter_rate)

    def __sub__(self, other):
        return Interval(self.start-other, self.end-other, sr=self.sr, iter_rate=self.iter_rate)

    def add(self, other):
        """Add to object, rather than returning a new object"""
        self.start = self.start + other
        self.end = self.end + other

    def sub(self, other):
        self.start = self.start - other
        self.end = self.end - other

    def union(self, other):
        """ 
        Merge intervals to make an interval from minimum start time to
        maximum end time. Other can be an interval, or a tuple of intervals.

        iter_rate and sr are inherited from the original
        event. Therefore, e1.union(e2) doesn't have to be the same as
        e2.union(e1)
        """
        assert self.sr == other.sr
        this_start = (self.start, other.start)[np.argmin((self.start.time, other.start.time))]
        this_end = (self.end, other.end)[np.argmax((self.end.time, other.end.time))]
        return Interval(this_start, this_end, sr=self.sr, iter_rate=self.iter_rate)

    def intersection(self, other):
        assert self.sr == other.sr
        if (other.start.time > self.end.time) | (self.start.time > other.end.time):
            return ()
        this_start = (self.start, other.start)[np.argmax((self.start.time, other.start.time))]
        this_end = (self.end, other.end)[np.argmin((self.end.time, other.end.time))]
        return  Interval(this_start, this_end, sr=self.sr, iter_rate=self.iter_rate)

class Data: # Signal processing
    def __init__(self, sig, sr, axis=None, history=None, t0=0., meta=None):
        """
        axis (int) time axis
        t0 (float) time at start sample
        meta is metadata that you can store in sampled data that is propagated by the clone method
        NOTE: When inheriting from this class, if the parameters of the
        __init__ method change, then make sure to rewrite the _clone method
        """
        self._sig = np.asarray(sig) # assumes sig is uniformly resampled
        assert self._sig.ndim in (1, 2)
        if not hasattr(self, 'sr'): # in case of multiple inheritance - see ot.Marker
            self.sr = sr
        if axis is None:
            self.axis = np.argmax(np.shape(self._sig))
        else:
            self.axis = axis
        if history is None:
            self._history = [('initialized', None)]
        else:
            assert isinstance(history, list)
            self._history = history
        self._t0 = t0
        self.meta = meta
    
    def __call__(self, col=None):
        """Return either a specific column or the entire set 2D signal"""
        if col is None:
            return self._sig

        if isinstance(col, str): # supply an empty string to take advantage of easy plotting
            return self.t, self._sig

        assert isinstance(col, int) and col < len(self)
        slc = [slice(None)]*self._sig.ndim
        slc[self.get_signal_axis()] = col
        return self._sig[tuple(slc)] # not converting slc to tuple threw a FutureWarning

    def _clone(self, proc_sig, his_append=None, **kwargs):
        if his_append is None:
            his = self._history # only useful when cloning without manipulating the data, e.g. returning a subset of columns
        else:
            his = self._history + [his_append]
        
        if hasattr(self, 'meta'):
            meta = self.meta
        else:
            meta = None
        axis = kwargs.pop('axis', self.axis)
        t0 = kwargs.pop('t0', self._t0)
        return self.__class__(proc_sig, self.sr, axis, his, t0, meta=meta)

    def analytic(self):
        proc_sig = hilbert(self._sig, axis=self.axis)
        return self._clone(proc_sig, ('analytic', None))

    def envelope(self, type='upper', lowpass=True):
        # analytic envelope, optionally low-passed
        assert type in ('upper', 'lower')
        if type == 'upper':
            proc_sig = np.abs(hilbert(self._sig, axis=self.axis))
        else:
            proc_sig = -np.abs(hilbert(-self._sig, axis=self.axis))

        if lowpass:
            if lowpass is True: # set cutoff frequency to lower end of bandpass filter
                assert 'bandpass' in [h[0] for h in self._history]
                lowpass = [h[1]['low'] for h in self._history if h[0] == 'bandpass'][0]
            assert isinstance(lowpass, (int, float)) # cutoff frequency
            return self._clone(proc_sig, ('envelope_'+type, None)).lowpass(lowpass)
        return self._clone(proc_sig, ('envelope_'+type, None))
    
    def phase(self):
        proc_sig = np.unwrap(np.angle(hilbert(self._sig, axis=self.axis)))
        return self._clone(proc_sig, ('instantaneous_phase', None))
    
    def instantaneous_frequency(self):
        proc_sig = np.diff(self.phase()._sig) / (2.0*np.pi) * self.sr
        return self._clone(proc_sig, ('instantaneous_frequency', None))

    def bandpass(self, low, high, order=None):
        if order is None:
            order = int(self.sr/2) + 1
        filt_pts = firwin(order, (low, high), fs=self.sr, pass_zero='bandpass')
        proc_sig = filtfilt(filt_pts, 1, self._sig, axis=self.axis)
        return self._clone(proc_sig, ('bandpass', {'filter':'firwin', 'low':low, 'high':high, 'order':order}))

    def _butterfilt(self, cutoff, order, btype):
        assert btype in ('low', 'high')
        if order is None:
            order = 6
        b, a = butter(order, cutoff/(0.5*self.sr), btype=btype, analog=False)

        nan_manip = False
        nan_bool = np.isnan(self._sig)
        if nan_bool.any():
            nan_manip = True
            self = self.interpnan() # interpolate missing values before applying an IIR filter

        proc_sig = filtfilt(b, a, self._sig, axis=self.axis)
        if nan_manip:
            proc_sig[nan_bool] = np.NaN # put back the NaNs in the same place

        return self._clone(proc_sig, (btype+'pass', {'filter':'butter', 'cutoff':cutoff, 'order':order, 'NaN manipulation': nan_manip}))

    def notch(self, cutoff, q=30):
        b, a = iirnotch(cutoff, q, self.sr)
        proc_sig = filtfilt(b, a, self._sig, axis=self.axis)

        return self._clone(proc_sig, ('notch', {'filter': 'iirnotch', 'cutoff': cutoff, 'q': q}))

    def lowpass(self, cutoff, order=None):
        return self._butterfilt(cutoff, order, 'low')
    
    def highpass(self, cutoff, order=None):
        return self._butterfilt(cutoff, order, 'high')

    def smooth(self, window_len=10, window='hanning'):
        if self._sig.ndim != 1:
            raise ValueError("smooth only accepts 1 dimension arrays.")

        if self._sig.size < window_len:
            raise ValueError("Input vector needs to be bigger than window size.")

        if window_len < 3:
            return self

        if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError("""Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'""")

        sig = np.r_[2 * self._sig[0] - self._sig[window_len:0:-1], self._sig, 2 * self._sig[-1] - self._sig[-2:-window_len - 2:-1]]

        if window == 'flat':  # moving average
            win = np.ones(window_len, 'd')
        else:
            win = eval('np.' + window + '(window_len)')

        sig_conv = np.convolve(win / win.sum(), sig, mode='same')

        return sig_conv[window_len: -window_len]
    
    def get_trend_airPLS(self, *args, **kwargs):
        from airPLS import airPLS
        trend = np.apply_along_axis(airPLS, self.axis, self._sig, *args, **kwargs)
        return self._clone(trend, ('get_trend_airPLS', {'args':args, **kwargs}))
        
    def detrend_airPLS(self, *args, **kwargs):
        trend = self.get_trend_airPLS(*args, **kwargs)
        proc_sig = self._sig - trend()
        return self._clone(proc_sig, ('detrend_airPLS', {'args':args, **kwargs}))


    def medfilt(self, order=11):
        """
        Median filter the signal
        
        order is the number of samples in the kernel if it is an int, and treated as time if it is a float
        """
        sw = np.lib.stride_tricks.sliding_window_view # this should be much faster than using running window
        if isinstance(order, float):
            order = int(order*self.sr)
        assert isinstance(order, int)
        order = (order // 2)*2 + 1 # ensure order is odd for simpler handling of time
        proc_sig_middle = np.median(sw(self._sig, order, axis=self.axis), axis=-1)
        pre_fill = np.take(self._sig, np.r_[:order//2], axis=self.axis)
        post_fill = np.take(self._sig, np.r_[-order//2+1:0], axis=self.axis)
        proc_sig = np.concatenate((pre_fill, proc_sig_middle, post_fill)) # ends of the signal will not be filtered
        return self._clone(proc_sig, ('median_filter', {'order': order, 'kernel_size_s': order/self.sr}))
    
    def interpnan(self, maxgap=None, **kwargs):
        """
        Only interpolate values within the mask
        kwargs will be passed to scipy.interpolate.interp1d
        """
        proc_sig = np.apply_along_axis(interpnan, self.axis, self._sig, maxgap, **kwargs)
        return self._clone(proc_sig, ('instantaneous_phase', None))

    def shift_baseline(self, offset=None): 
        # you can use numpy broadcasting to shift each signal if multi-dimensional
        if offset is None:
            offset = np.nanmean(self._sig, self.axis)
        return self._clone(self._sig - offset, ('shift_baseline', offset))
    
    def shift_left(self, time:float=None):
        ret = self._clone(self._sig, ('shift_left', time))
        if time is None: # shift to zero
            time = self._t0
        ret._t0 = self._t0 - time
        return ret
    
    def get_total_left_shift(self) -> float:
        """Return the total amount of time by which the signal was shifted to the left."""
        l_shift = [x[1] for x in self._history if x[0] == 'shift_left']
        return float(sum(l_shift))

    def reset_left_shift(self):
        return self.shift_left(-self.get_total_left_shift())

    def scale(self, scale_factor):
        return self._clone(self._sig/scale_factor, ('scale', scale_factor))
        
    def __len__(self):
        return np.shape(self._sig)[self.axis]

    @property
    def t(self):
        n_samples = len(self)
        return np.linspace(self._t0, self._t0 + (n_samples-1)/self.sr, n_samples)
    
    @property
    def dur(self):
        return (len(self)-1)/self.sr
    
    def t_start(self):
        return self._t0
    
    def t_end(self):
        return self._t0 + (len(self)-1)/self.sr
    
    def interval(self):
        return Interval(self.t_start(), self.t_end(), sr=self.sr)

    def _slice_to_interval(self, key: slice) -> Interval:
        assert key.step is None # otherwise, the sampling rate is going to change, and could cause aliasing without proper filtering
        # IF INTEGERS, assume indices, IF FLOAT, assume time
        if isinstance(key.start, str): # for things like data['t_start':'t_end']
            assert hasattr(self, 'meta') and key.start in self.meta
            key = slice(self.meta[key.start], key.stop, None)
        if isinstance(key.stop, str):
            assert hasattr(self, 'meta') and key.stop in self.meta
            key = slice(key.start, self.meta[key.stop], None)
        if isinstance(key.start, float) or isinstance(key.stop, float):
            intvl_start = key.start
            if key.start is None:
                intvl_start = self.t_start()
            intvl_end = key.stop
            if key.stop is None:
                intvl_end = self.t_end()
        else: # if samples, do python indexing and don't include the end?
            assert isinstance(key.start, (int, type(None))) and isinstance(key.stop, (int, type(None)))
            if key.start is None:
                intvl_start = self.t_start()
            else:
                intvl_start = self.t[sorted((0, key.start, len(self)-1))[1]] # clip to limits
            if key.stop is None:
                intvl_end = self.t_end()
            else:
                intvl_end = self.t[sorted((0, key.stop-1, len(self)-1))[1]]
        return Interval(float(intvl_start), float(intvl_end), sr=self.sr)

    def _interval_to_index(self, key: Interval):
        assert key.sr == self.sr
        offset = round(self._t0*self.sr)
        rng_start = sorted((0, key.start.sample-offset, len(self)-1))[1]
        rng_end = sorted((0, key.end.sample-offset+1, len(self)))[1] # +1 because interval object includes both ends!
        return rng_start, rng_end

    def take_by_interval(self, key: Interval):
        his = self._history + [('slice', key)]
        rng_start, rng_end = self._interval_to_index(key)
        proc_sig = self._sig.take(indices=range(rng_start, rng_end), axis=self.axis)
        if hasattr(self, 'meta'):
            meta = self.meta
        else:
            meta = None
        return self.__class__(proc_sig, self.sr, self.axis, his, self.t[rng_start], meta)

    def __getitem__(self, key):
        """
        Use this function to slice the signal in time.
        Use __call__ to retrieve one column of data, or all columns.

        Example usage:
            x3 = sampled.Data(np.random.random((10, 3)), sr=2, t0=5.)
            
            Indexing with list, tuple, int, or float will return numpy arrays:
                x3[[5.05, 5.45]]                    # returns linearly interpolated values
                x3[5.05]                            # returns linearly interpolated value
                x3[2.]                              # this should error out because it is outside the range
                x3[2], x3[-1], x3[len(x3)-1]        # this is effectively like array-indexing, last two should be the same

            Indexing with interval or slice returns sampled.Data:
                x3[5.:5.05]()                       # should return only one value
                x3[5.:5.05].interval().end          # should return 5
                x3[:1]()                            # retrieve by position if it is an integer - Equivalent to x3[0], but for signals with axis=1, x3[:1] will preserve dimensionality of retrieved signal
                x3[0.:1.]()                         # this will return an empty signal, and the interval() on that won't make sense
                x3[:5.5]()                          # should return first two values 
                x3[0:5.5](), x3[5.0:5.5]()          # should be the same as above, also examine x3[0:5.5].interval().start -> this should be 5.0
        """
        if isinstance(key, (list, tuple, float, int)): # return signal (interpolated if needbe) values at those times
            if isinstance(key, int):
                key = self.t[key]
            return interp1d(self.t, self._sig, axis=self.axis)(key)

        if isinstance(key, str):
            if hasattr(self, 'meta') and key in self.meta:
                return self.meta[key]
            
        assert isinstance(key, (Interval, slice))
        if isinstance(key, slice):
            key = self._slice_to_interval(key)
        return self.take_by_interval(key)

    def make_running_win(self, win_size=0.25, win_inc=0.1):
        win_size_samples = (round(win_size*self.sr)//2)*2 + 1 # ensure odd number of samples
        win_inc_samples = round(win_inc*self.sr)
        n_samples = len(self)
        return RunningWin(n_samples, win_size_samples, win_inc_samples)

    def apply_running_win(self, func, win_size=0.25, win_inc=0.1):
        """
        Process the signal using a running window by applying func to each window.
        Returns:
            Sampled data 
        Example:
            Extract RMS envelope
            self.apply_running_win(lambda x: np.sqrt(np.mean(x**2)), win_size, win_inc)
        """
        rw = self.make_running_win(win_size, win_inc)
        ret_sig = np.array([func(self._sig[r_win], self.axis) for r_win in rw()])
        ret_sr = self.sr/round(win_inc*self.sr)
        return Data(ret_sig, ret_sr, axis=self.axis, t0=self.t[rw.center_idx[0]])
    
    def __le__(self, other): return self._comparison('__le__', other)
    def __ge__(self, other): return self._comparison('__ge__', other)
    def __lt__(self, other): return self._comparison('__lt__', other)
    def __gt__(self, other): return self._comparison('__gt__', other)
    def __eq__(self, other): return self._comparison('__eq__', other)
    def __ne__(self, other): return self._comparison('__ne__', other)

    def _comparison(self, dunder, other):
        cmp_dunder_dict = {'__le__':'<=', '__ge__':'>=', '__lt__':'<', '__gt__':'>', '__eq__':'==', '__ne__':'!='}
        assert dunder in cmp_dunder_dict
        assert isinstance(other, (int, float))
        return self._clone(getattr(self._sig, dunder)(other), (cmp_dunder_dict[dunder], other))
    
    def onoff_times(self):
        """Onset and offset times of a thresholded 1D sampled.Data object"""
        onset_samples, offset_samples = onoff_samples(self._sig)
        return [self.t[x] for x in onset_samples], [self.t[x] for x in offset_samples]
    
    def find_crossings(self, th=0., th_time=None):
        """Find the times at which the signal crosses a given threshold th.
        th_time - Ignore crossings that are less than th_time apart. Caution - uses median filter, check carefully.
        """
        if th_time is None:
            neg_to_pos, pos_to_neg = (self > th).onoff_times()
        else:
            neg_to_pos, pos_to_neg = ((self > th).medfilt(order=round(self.sr*th_time*2)) > 0.5).onoff_times()
        return neg_to_pos, pos_to_neg
        
    def get_signal_axis(self):
        if self().ndim == 1:
            return None # there is no signal axis for a 1d signal
        return (self.axis+1)%self().ndim
    
    def n_signals(self):
        if self().ndim == 1:
            return 1
        return self().shape[self.get_signal_axis()]

    def split_to_1d(self):
        if self().ndim == 1:
            return [self]
        return [self._clone(self(col), his_append=('split', col), axis=0) for col in range(self.n_signals())]
    
    def transpose(self):
        if self().ndim == 1:
            return self # nothing done
        return self._clone(self._sig.T, axis=self.get_signal_axis())
    
    def fft(self, win_size=None, win_inc=None, zero_mean=False):
        T = 1/self.sr
        if win_size is None and win_inc is None:
            N = len(self)
            f = fftfreq(N, T)[:N//2]
            sig = self._clone(self._sig)
            if zero_mean:
                sig = sig.shift_baseline()
            if np.ndim(sig) == 1:
                amp = 2.0/N * np.abs(fft(sig)[0:N//2])
            else:
                amp = np.array([2.0/N * np.abs(fft(s())[0:N//2]) for s in sig.split_to_1d()]).T
            return f, amp
        
        # do a sliding window fft
        if win_inc is None:
            win_inc = win_size # no overlap
            
        rw = self.make_running_win(win_size, win_inc)
        if np.ndim(self._sig) == 1:
            amp_all = []
            for this_rw in rw():
                sig = self[this_rw]
                if zero_mean:
                    sig = sig.shift_baseline()
                N = len(sig)
                this_amp = 2.0/N * np.abs(fft(sig())[0:N//2])
                amp_all.append(this_amp)
            f = fftfreq(N, T)[:N//2]
            amp = np.mean(amp_all, axis=0)
            return f, amp
        if np.ndim(self._sig) == 2:
            amp_all = []
            for sig in self.split_to_1d():
                f, amp = sig.fft(win_size, win_inc, zero_mean)
                amp_all.append(amp)
            return f, np.array(amp_all).T
    
    def fft_as_sampled(self, *args, **kwargs):
        f, amp = self.fft(*args, **kwargs)
        df = (f[-1] - f[0])/(len(f)-1)
        return Data(amp, sr=1/df, t0=f[0]) # think of it as sr number of samples per Hz (instead of samples per second)
    
    def psd(self, win_size=5.0, win_inc=None, **kwargs): 
        """compute the power spectral density using the welch method"""
        kwargs_default = dict(nperseg=round(self.sr*win_size), scaling='density')
        kwargs = kwargs_default | kwargs
        if win_inc is not None:
            kwargs['noverlap'] = kwargs['nperseg'] - round(self.sr*win_inc)
        else:
            kwargs['noverlap'] = None
        if self().ndim == 1:
            f, Pxx = welch(self._sig, self.sr, **(kwargs_default | kwargs))
            return f, Pxx
        Pxx = []
        for s in self.split_to_1d():
            f, this_Pxx = welch(s._sig, s.sr, **kwargs)
            Pxx.append(this_Pxx)
        Pxx = np.vstack(Pxx).T
        return f, Pxx

    def psd_as_sampled(self, *args, **kwargs):
        f, Pxx = self.psd(*args, **kwargs)
        df = (f[-1] - f[0])/(len(f)-1)
        return Data(Pxx, sr=1/df, t0=f[0])

    def diff(self, order=1):
        if self._sig.ndim == 2:
            if self.axis == 1:
                pp_value = (self._sig[:, 1] - self._sig[:, 0])[:, None]
                fn = np.hstack
            else: # self.axis == 0
                pp_value = self._sig[1] - self._sig[0]
                fn = np.vstack
        else: # self._sig.ndim == 1
            pp_value = self._sig[1] - self._sig[0]
            fn = np.hstack

        # returning a marker type even though this is technically not true
        return self._clone(fn((pp_value, np.diff(self._sig, axis=self.axis, n=order)))*self.sr, ('diff', None))
    
    def magnitude(self):
        if self._sig.ndim == 1:
            return self
        assert self._sig.ndim == 2 # magnitude does not make sense for a 1D signal (in that case, use np.linalg.norm directly)
        return Data(np.linalg.norm(self._sig, axis=(self.axis+1)%2), self.sr, history=self._history+[('magnitude', 'None')], t0=self._t0, meta=self.meta)

    def apply(self, func, *args, **kwargs):
        """apply a function func along the time axis"""
        try:
            kwargs['axis'] = self.axis
            proc_sig = func(self._sig, *args, **kwargs)
        except TypeError:
            kwargs.pop('axis')
            proc_sig = func(self._sig, *args, **kwargs)
        return self._clone(proc_sig, ('apply', {'func': str(func), 'args': args, 'kwargs': kwargs}))
    
    def apply_along_signals(self, func, *args, **kwargs):
        """apply a function func along the signal axis"""
        try:
            kwargs['axis'] = self.get_signal_axis()
            proc_sig = func(self._sig, *args, **kwargs)
        except TypeError:
            kwargs.pop('axis')
            proc_sig = func(self._sig, *args, **kwargs)
        return self._clone(proc_sig, ('apply_along_signals', {'func': str(func), 'args': args, 'kwargs': kwargs}))
    
    def apply_to_each_signal(self, func, *args, **kwargs):
        """Apply a function to each signal (if self is a collection of signals) separately, and put it back together"""
        assert self().ndim == 2
        proc_sig = np.vstack([func(s._sig.copy(), *args, **kwargs) for s in self.split_to_1d()])
        if self.axis == 0:
            proc_sig = proc_sig.T
        return self._clone(proc_sig, ('apply_to_each_signal', {'func': str(func), 'args': args, 'kwargs': kwargs}))
    
    def regress(self, ref_sig):
        """Regress a reference signal out of the current signal"""
        from sklearn.linear_model import LinearRegression
        assert ref_sig().ndim == self().ndim == 1 # currently only defined for 1D signals
        assert ref_sig.sr == self.sr
        assert len(ref_sig) == len(self)
        reg = LinearRegression().fit(ref_sig().reshape(-1, 1), self())
        prediction = reg.coef_[0]*ref_sig() + reg.intercept_
        return self._clone(self() - prediction, ('Regressed with reference', ref_sig()))
    
    def resample(self, new_sr, *args, **kwargs):
        """args and kwargs will be passed to scipy.signal.resample"""
        proc_sig, proc_t = resample(self._sig, round(len(self)*new_sr/self.sr), t=self.t, axis=self.axis, *args, **kwargs)
        if hasattr(self, 'meta'):
            meta = self.meta
        else:
            meta = None
        return self.__class__(proc_sig, sr=new_sr, axis=self.axis, history=self._history+[('resample', new_sr)], t0=proc_t[0], meta=meta)
    
    def smooth(self, win_size=0.5):
        """Moving average smoothing while preserving the number of samples in the signal"""
        stride = round(win_size*self.sr)
        proc_sig = np.lib.stride_tricks.sliding_window_view(self._sig, stride, axis=self.axis).mean(axis=-1)
        t_start_offset = (stride-1)/(2*self.sr)
        return self.__class__(proc_sig, sr=self.sr, axis=self.axis, history=self._history+[('moving average with stride', stride)], t0=self._t0+t_start_offset, meta=self.meta)
    
    def xlim(self):
        return self.t_start(), self.t_end()
    
    def ylim(self):
        return np.nanmin(self._sig), np.nanmax(self._sig)

    def logdj(self, interpnan_maxgap=None):
        """
        CAUTION: makes sense ONLY if self is a velocity signal
        Computes the log dimensionless jerk of marker velocity.
        interpnan_maxgap - maximum gap (in number of samples) to interpolate.
            - None (default) interpolates all gaps. Supply 0 to not interpolate.
        """
        vel = self.interpnan(maxgap=interpnan_maxgap)

        dt = 1/self.sr
        scale = np.power(self.dur, 3) / np.power(np.max(vel._sig), 2)

        # jerk = vel.apply_to_each_signal(np.diff, 2).apply(lambda x: x/dt**2) # there is a small difference between the values when using diff and gradient.
        jerk = vel.apply_to_each_signal(np.gradient, dt).apply_to_each_signal(np.gradient, dt)
        return -np.log(scale * simpson(np.power(jerk.magnitude()(), 2), dx=dt))

    def logdj2(self, interpnan_maxgap=None):
        """
        CAUTION: makes sense ONLY if self is a speed signal
        Computes the log dimensionless jerk of marker velocity. Variation with speed instead of velocity
        interpnan_maxgap - maximum gap (in number of samples) to interpolate.
            - None (default) interpolates all gaps. Supply 0 to not interpolate.
        """
        speed = self.interpnan(maxgap=interpnan_maxgap)

        dt = 1/self.sr
        scale = np.power(self.dur, 3) / np.power(np.max(speed._sig), 2)

        jerk = speed.apply(np.gradient, dt).apply(np.gradient, dt)
        return -np.log(scale * simpson(np.power(jerk(), 2), dx=dt))

    def sparc(self, fc=10.0, amp_th=0.05, interpnan_maxgap=None, shift_baseline=False, mean_normalize=True):
        """
        CAUTION: makes sense ONLY if self is a speed signal Computes the SPARC
        smoothness metric. 
            interpnan_maxgap - maximum gap (in number of samples) to interpolate.
                - None (default) interpolates all gaps. Supply 0 to not interpolate.
            shift_baseline - Subtract the mean. Defaults to False mean_normalize -
                Divide the signal by the mean. Requred to make smoothness metric
                insensitive to signal amplitude. Defaults to True. 
        """
        speed = self.interpnan(maxgap=interpnan_maxgap)
        if shift_baseline:
            speed = speed.shift_baseline()
        if mean_normalize:
            speed = speed.apply(lambda x: x/np.nanmean(x))

        freq, Mfreq = speed.fft()

        freq_sel = freq[freq <= fc]
        Mfreq_sel = Mfreq[freq <= fc]

        inx = ((Mfreq_sel >= amp_th) * 1).nonzero()[0]
        fc_inx = range(inx[0], inx[-1] + 1)
        freq_sel = freq_sel[fc_inx]
        Mfreq_sel = Mfreq_sel[fc_inx]

        # Calculate arc length
        Mf_sel_diff = np.gradient(Mfreq_sel) / np.mean(np.diff(freq_sel))
        fc = freq_sel[-1]
        integrand = np.sqrt((1 / fc) ** 2 + Mf_sel_diff ** 2)
        sparc = -simpson(integrand, freq_sel)
        return sparc

    def set_nan(self, interval_list):
        """Set parts of a signal to np.nan. 
        E.g. interval_list = [(90.5, 91.2), (93, 93.5)]
        """
        def set_nan(np_arr: np.ndarray, idx_list):
            np_arr[idx_list] = np.nan
            return np_arr

        sel = np.zeros(len(self), dtype=bool)
        for start_time, end_time in interval_list:
            intvl = Interval(float(start_time), float(end_time), sr=self.sr)
            start_index, end_index_inc = self._interval_to_index(intvl)
            sel[start_index:end_index_inc] = True
            
        return self.apply_to_each_signal(set_nan, idx_list=sel)
    
    def remove_and_interpolate(self, interval_list, maxgap=None, **kwargs):
        """Remove parts of a signal, and interpolate between those points."""
        if not interval_list:
            return self
        return self.set_nan(interval_list).interpnan(maxgap=maxgap, **kwargs)

class DataList(list):
    def __call__(self, **kwargs):
        ret = self
        for key, val in kwargs.items():
            if key.endswith('_lim') and (key.removesuffix('_lim')) in self[0].meta:
                assert len(val) == 2
                ret = [s for s in ret if val[0] <= s.meta[key] <= val[1]]
            elif key.endswith('_any') and (key.removesuffix('_lim')) in self[0].meta:
                ret = [s for s in ret if s.meta[key] in val]
            elif key in self[0].meta:
                ret = [s for s in ret if s.meta[key] == val]
            else:
                continue # key was not in meta
        return self.__class__(ret)

class Event(Interval):
    def __init__(self, start, end=None, **kwargs):
        """
        Interval with labels.

        kwargs:
        labels (list of strings) - hastags defining the event
        """
        if end is None: # typecast interval into an event
            assert isinstance(start, Interval)
            end = start.end
            start = start.start
        self.labels = kwargs.pop('labels', [])
        super().__init__(start, end, **kwargs)
    
    def add_labels(self, *new_labels):
        self.labels += list(new_labels)
    
    def remove_labels(self, *labels_to_remove):
        self.labels = [label for label in self.labels if label not in labels_to_remove]


class Events(list):
    """List of event objects that can be selected by labels using the 'get' method."""
    def append(self, key):
        assert isinstance(key, (Event, Interval))
        super().append(Event(key))
    
    def get(self, label):
        return Events([e for e in self if label in e.labels])


class RunningWin:
    def __init__(self, n_samples, win_size, win_inc=1, step=None, offset=0):
        """
        n_samples, win_size, and win_inc are integers (not enforced, but expected!)
        offset (int) offsets all running windows by offset number of samples.
            This is useful when the object you're slicing has an inherent offset that you need to consider.
            For example, consider creating running windows on a sliced optitrack marker
            Think of offset as start_sample
        Attributes of interest:
            run_win (array of slice objects)
            center_idx (indices of center samples)
        """
        self.n_samples = int(n_samples)
        self.win_size = int(win_size)
        self.win_inc = int(win_inc)
        self.n_win = int(np.floor((n_samples-win_size)/win_inc) + 1)
        self.start_index = int(offset)

        run_win = []
        center_idx = []
        for win_count in range(0, self.n_win):
            win_start = (win_count * win_inc) + offset
            win_end = win_start + win_size
            center_idx.append(win_start + win_size//2)
            run_win.append(slice(win_start, win_end, step))
        
        self._run_win = run_win
        self.center_idx = center_idx
        
    def __call__(self, data=None):
        if data is None: # return slice objects
            return self._run_win
        # if data is supplied, apply slice objects to the data
        assert len(data) == self.n_samples
        return [data[x] for x in self._run_win]
    
    def __len__(self):
        return self.n_win


class Siglets:
    """A collection of pieces of signals to do event-triggered analyses"""
    AX_TIME, AX_TRIALS = 0, 1

    def __init__(self, sig:Data, events:Events, window=None, cache=None):
        self.parent = sig
        if window is not None: # use window when all events are of the same length
            if isinstance(window, Interval):
                assert window.sr == sig.sr
            else:
                assert len(window) == 2
                window = Interval(window[0], window[1], sr=sig.sr)
            assert isinstance(events, (list, tuple))
            events = Events([Event(window + ev_time) for ev_time in events])
        self.window = window
        self.events = events
        assert self.is_uniform()
    
    def _parse_ax(self, axis):
        if isinstance(axis, int):
            return axis
        assert isinstance(axis, str)
        if axis in ('t', 'time'):
            return self.AX_TIME
        return self.AX_TRIALS # axis is anything, but ideally in ('ev', 'events', 'sig', 'signals', 'data', 'trials')

    @property
    def sr(self):
        return self.parent.sr
    
    @property
    def t(self):
        """Return the time vector of the event window"""
        return self.window.t
    
    @property
    def n(self):
        """Return the number of siglets"""
        return len(self.events)
    
    def __len__(self):
        """Number of time points"""
        return len(self.window)

    def __call__(self, func=None, axis='events', *args, **kwargs):
        siglet_list = [self.parent[ev]() for ev in self.events]
        if func is None:
            return np.asarray(siglet_list).T
        return self.apply(func, axis=self._parse_ax(axis), *args, **kwargs)

    def apply_along_events(self, func, *args, **kwargs) -> np.ndarray:
        return func(self(), axis=self.AX_TRIALS, *args, **kwargs)
    
    def apply_along_time(self, func, *args, **kwargs) -> np.ndarray:
        return func(self(), axis=self.AX_TIME, *args, **kwargs)
    
    def apply(self, func, axis='events', *args, **kwargs) -> np.ndarray: # by default, applies to each siglet
        return func(self(), axis=self._parse_ax(axis), *args, **kwargs)
    
    def mean(self, axis='events') -> np.ndarray:
        return self(np.mean, axis=axis)
    
    def sem(self, axis='events') -> np.ndarray:
        return self(np.std, axis=axis)/np.sqrt(self.n)
    
    def is_uniform(self):
        return (len(set([ev.dur_sample for ev in self.events])) == 1) # if all events are of the same size


def interpnan(sig, maxgap=None, min_data_frac=0.2, **kwargs):
    """
    Interpolate NaNs in a 1D signal
        sig - 1D numpy array
        maxgap - 
            - (NoneType) all NaN values will be interpolated
            - (int) stretches of NaN values smaller than or equal to maxgap will be interpolated
            - (boolean array) will be used as a mask where interpolation will only happen where maxgap is True
        kwargs - 
            these get passed to scipy.interpolate.interp1d function
            commonly used: kind='cubic'
    """
    assert np.ndim(sig) == 1
    assert 0. <= min_data_frac <= 1.
    if 'fill_value' not in kwargs:
        kwargs['fill_value'] = 'extrapolate'
        
    def nan_helper(y):
        return np.isnan(y), lambda z: z.nonzero()[0]
    proc_sig = sig.copy()
    nans, x = nan_helper(proc_sig)
    if np.mean(~nans) < min_data_frac:
        return sig # interpolate only if there are enough data points

    if maxgap is None:
        mask = np.ones_like(nans)
    elif isinstance(maxgap, int):
        nans = np.isnan(sig)
        mask = np.zeros_like(nans)
        onset_samples, offset_samples = onoff_samples(nans)
        for on_s, off_s in zip(onset_samples, offset_samples):
            assert on_s < off_s
            if off_s - on_s <= maxgap: # interpolate this
                mask[on_s:off_s] = True
    else:
        mask = maxgap
    assert len(mask) == len(sig)
    proc_sig[nans & mask]= interp1d(x(~nans), proc_sig[~nans], **kwargs)(x(nans & mask)) # np.interp(x(nans & mask), x(~nans), proc_sig[~nans])
    return proc_sig

def onoff_samples(tfsig):
    """
    Find onset and offset samples of a 1D boolean signal (e.g. Thresholded TTL pulse)
    Currently works only on 1D signals!
    tfsig is shorthand for true/false signal
    """
    assert tfsig.dtype == bool
    assert np.sum(np.asarray(np.shape(tfsig)) > 1) == 1
    x = np.squeeze(tfsig).astype(int)
    onset_samples = list(np.where(np.diff(x) == 1)[0] + 1)
    offset_samples = list(np.where(np.diff(x) == -1)[0] + 1)
    if tfsig[0]: # is True
        onset_samples = [0] + onset_samples
    if tfsig[-1]:
        offset_samples = offset_samples + [len(tfsig)-1]
    return onset_samples, offset_samples

def uniform_resample(time, sig, sr, t_min=None, t_max=None):
    """
    Uniformly resample a signal at a given sampling rate sr.
    Ideally the sampling rate is determined by the smallest spacing of
    time points.
    Inputs:
        time (list, 1d numpy array) is a non-decreasing array
        sig (list, 1d numpy array)
        sr (float) sampling rate in Hz
        t_min (float) start time for the output array
        t_max (float) end time for the output array
    Returns:
        pn.sampled.Data
    """
    assert len(time) == len(sig)
    time = np.array(time)
    sig = np.array(sig)

    if t_min is None: t_min = time[0]
    if t_max is None: t_max = time[-1]

    n_samples = int((t_max - t_min)*sr) + 1
    t_max = t_min + (n_samples-1)/sr

    t_proc = np.linspace(t_min, t_max, n_samples)
    if np.ndim(sig) == 1:
        sig_proc = np.interp(t_proc, time, sig)
        return Data(sig_proc, sr, t0=t_min)
    sig_proc = np.zeros((len(t_proc), sig.shape[-1]))
    for col_count in range(sig.shape[-1]):
        sig_proc[:, col_count] = np.interp(t_proc, time, sig[:, col_count])
    return Data(sig_proc, sr, t0=t_min)


def frac_power(sig:Data, freq_lim:tuple, win_size:float=5., win_inc:float=2.5, freq_dx:float=0.05, highpass_cutoff:float=0.2) -> Data:
    """
    Calculate the fraction of power in a specific frequency band (similar to synchronymetric in musicrunning project).
    """
    assert len(freq_lim) == 2
    curr_t = sig.t_start()
    ret = []
    while curr_t + win_size < sig.t_end():
        try:
            sig_piece = sig[curr_t:curr_t+win_size]
            if highpass_cutoff > 0:
                f, amp = sig_piece.shift_baseline().highpass(highpass_cutoff).fft()
            else:
                f, amp = sig_piece.shift_baseline().fft()
            area_of_interest = np.trapz(interp1d(f, amp)(np.r_[freq_lim[0]:freq_lim[1]:freq_dx]), dx=freq_dx)
            total_area = np.trapz(amp, f)
            ret.append(area_of_interest / total_area)
            curr_t = curr_t + win_inc
        except ValueError:
            ret.append(np.nan)
            curr_t = curr_t + win_inc
    
    return Data(ret, 1/win_inc, t0=sig.t_start()+win_size/2)
