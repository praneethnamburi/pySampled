import numpy as np
import scipy.fft
import scipy.integrate
import scipy.interpolate
import scipy.signal

from typing import Union, List, Tuple, Callable, Optional, Any

# Optional dependencies - imported inside the methods
## import sklearn.linear_model (for sampled.Data.regress)
## from airPLS import airPLS (for sampled.Data.get_trend_airPLS, sampled.Data.detrend_airPLS)


class Time:
    """
    Time when working with sampled data (including video). INTEGER IMPLIES SAMPLE NUMBER, FLOAT IMPLIES TIME IN SECONDS.
    Use this to encapsulate sampling rate (sr), sample number (sample), and time (s).
    When the sampling rate is changed, the sample number is updated, but the time is held constant.
    When the time is changed, sample number is updated.
    When the sample number is changed, the time is updated
    When working in Premiere Pro, use 29.97 fps drop-frame timecode to show the actual time in video.
    You should see semicolons instead of colons

    Args:
        inp (Union[str, float, int, Tuple[Union[str, float, int], float]]): Input time.
            (str)   hh;mm;ss;frame#
            (float) assumes provided input is time in seconds!
            (int)   assumes the provided input is the sample number
            (tuple) assumes (timestamp/time/sample, sampling rate)
        sr (float): Sampling rate, in Hz. casted into a float.

    Attributes:
        time (float): Time in seconds.
        sample (int): Sample number.
        sr (float): Sampling rate

    Examples:
        .. code-block:: python

            t = Time('00;09;53;29', 30)
            t = Time(9.32, 180)
            t = Time(12531, 180)
            t = Time(9.32, sr=180)
            t = Time(('00;09;53;29', 30), 72) # edge case - for dealing with timestamps from premiere pro
            t.time
            t.sample
    """

    def __init__(
        self,
        inp: Union[str, float, int, Tuple[Union[str, float, int], float]],
        sr: float,
    ):
        if isinstance(inp, list):
            inp = tuple(
                inp
            )  # cast it in case a 2-element list was supplied instead of a tuple
        inp_orig = inp  # for the edge case where time is specified as a string with the last component as the frame number

        # set the sampling rate
        if isinstance(inp, tuple):
            assert len(inp) == 2
            self._sr = float(inp[1])
            inp = inp[0]  # input is now either a string, float, or int!
        else:
            self._sr = float(sr)

        # set the sample number before setting the time
        assert isinstance(inp, (str, float, int))
        if isinstance(inp, str):
            inp = [int(x) for x in inp.split(";")]
            self._sample = round(
                (inp[0] * 60 * 60 + inp[1] * 60 + inp[2]) * self.sr + inp[3]
            )
        if isinstance(inp, float):  # time to sample
            self._sample = round(inp * self.sr)
        if isinstance(inp, int):
            self._sample = inp

        # set the time based on the sample number
        self._time = float(self._sample) / self._sr

        if sr != self.sr:  # t = Time((9.32, 180), 30)
            assert isinstance(inp_orig, tuple)
            self.sr = sr

    @property
    def sr(self) -> float:
        return self._sr

    @sr.setter
    def sr(self, sr_val: float) -> None:
        """When changing the sampling rate, keep `time` the same. This means, the `sample` number will change."""
        sr_val = float(sr_val)
        self._sr = sr_val
        self._sample = int(self._time * self._sr)

    def change_sr(self, new_sr: float) -> "Time":
        self.sr = new_sr
        return self

    @property
    def sample(self) -> int:
        return self._sample

    @sample.setter
    def sample(self, sample_val: int) -> None:
        self._sample = int(sample_val)
        self._time = float(self._sample) / self._sr

    @property
    def time(self) -> float:
        """Return time in seconds"""
        return self._time

    @time.setter
    def time(self, s_val: float) -> None:
        """If time is changed, then the sample number should be reset as well"""
        self._sample = int(float(s_val) * self._sr)
        self._time = float(self._sample) / self._sr

    def __add__(self, other: Union["Time", int, float]) -> "Time":
        x = self._arithmetic(other)
        return Time(x[2].__add__(x[0], x[1]), self.sr)

    def __sub__(self, other: Union["Time", int, float]) -> "Time":
        x = self._arithmetic(other)
        return Time(x[2].__sub__(x[0], x[1]), self.sr)

    def _arithmetic(
        self, other: Union["Time", int, float]
    ) -> Tuple[Union[int, float], Union[int, float], type]:
        if isinstance(other, self.__class__):
            assert other.sr == self.sr
            return (self.sample, other.sample, int)
        elif isinstance(other, int):
            # integer implies sample, float implies time
            return (self.sample, other, int)
        elif isinstance(other, float):
            return (self.time, other, float)
        else:
            raise TypeError(
                other,
                "Unexpected input type! Input either a float for time, integer for sample, or time object",
            )

    def to_interval(self, iter_rate: Optional[float] = None) -> "Interval":
        """Return an interval object with start and end times being the same"""
        return Interval(self, self, self.sr, iter_rate)

    def __repr__(self) -> str:
        return (
            "time={:.3f} s, sample={}, sr={} Hz ".format(
                self.time, self.sample, self.sr
            )
            + super().__repr__()
        )


class Interval:
    r"""
    Interval object with start and stop times. Implements the iterator protocol. INCLUDES BOTH START AND END SAMPLES.

    Pictorial understanding::

        start           -> |                                           | <-
        frames          -> |   |   |   |   |   |   |   |   |   |   |   | <- [self.sr, len(self)=12, self.t_data, self.t]
        animation times -> |        |        |        |        |         <- [self.iter_rate, self._index, self.t_iter]

    Frame sampling is used to pick the nearest frame corresponding to the animation times.

    Args:
        start (Union[Time, str, float, int, Tuple[Union[str, float, int], float]]): Start time.
        end (Union[Time, str, float, int, Tuple[Union[str, float, int], float]]): End time.
        sr (float): Sampling rate, in Hz.
        iter_rate (Optional[float]): Iteration rate.

    Example:
        .. code-block:: python

            intvl = Interval(('00;09;51;03', 30), ('00;09;54;11', 30), sr=180, iter_rate=env.Key().fps)
            intvl.iter_rate = 24  # say 24 fps for animation
            for nearest_sample, time, index in intvl:
                print((nearest_sample, time, index))
    """

    def __init__(
        self,
        start: Union[Time, str, float, int, Tuple[Union[str, float, int], float]],
        end: Union[Time, str, float, int, Tuple[Union[str, float, int], float]],
        sr: float = 30.0,
        iter_rate: Optional[float] = None,
    ):
        # if isinstance(start, (int, float)) and sr is not None:
        self.start = self._process_inp(start, sr)
        self.end = self._process_inp(end, sr)

        assert (
            self.start.sr == self.end.sr
        )  # interval is defined for a specific sampled dataset

        self._index = 0
        if iter_rate is None:
            self.iter_rate = (
                self.sr
            )  # this will be the animation fps when animating data at a different rate
        else:
            self.iter_rate = float(iter_rate)

    @staticmethod
    def _process_inp(
        inp: Union[Time, str, float, int, Tuple[Union[str, float, int], float]],
        sr: float,
    ) -> Time:
        if isinstance(inp, Time):
            return inp  # sr is ignored, superseded by input's sampling rate
        return Time(inp, sr)  # string, float, int or tuple. sr is ignored if tuple.

    @property
    def sr(self) -> float:
        return self.start.sr

    @sr.setter
    def sr(self, sr_val: float) -> None:
        sr_val = float(sr_val)
        self.start.sr = sr_val
        self.end.sr = sr_val

    def change_sr(self, new_sr: float) -> "Interval":
        self.sr = new_sr
        return self

    @property
    def dur_time(self) -> float:
        """Duration in seconds"""
        return self.end.time - self.start.time

    @property
    def dur_sample(self) -> int:
        """Duration in number of samples"""
        return (
            self.end.sample - self.start.sample + 1
        )  # includes both start and end samples

    def __len__(self) -> int:
        return self.dur_sample

    # iterator protocol - you can do: for sample, time, index in interval
    def __iter__(self) -> "Interval":
        """Iterate from start sample to end sample"""
        return self

    def __next__(self) -> Tuple[int, float, int]:
        index_interval = 1.0 / self.iter_rate
        if self._index <= int(self.dur_time * self.iter_rate) + 1:
            time = self.start.time + self._index * index_interval
            nearest_sample = self.start.sample + int(
                self._index * index_interval * self.sr
            )
            result = (nearest_sample, time, self._index)
        else:
            self._index = 0
            raise StopIteration
        self._index += 1
        return result

    # time vectors
    @property
    def t_iter(self) -> List[float]:
        """Time Vector for the interval at iteration frame rate"""
        return self._t(self.iter_rate)

    @property
    def t_data(self) -> List[float]:
        """Time vector at the data sampling rate"""
        return self._t(self.sr)

    @property
    def t(self) -> List[float]:
        """Time Vector relative to t_zero"""
        return self.t_data

    def _t(self, rate: float) -> List[float]:
        _t = [self.start.time]
        while (_t[-1] + 1.0 / rate) <= self.end.time:
            _t.append(_t[-1] + 1.0 / rate)
        return _t

    def __add__(self, other: Union[Time, int, float]) -> "Interval":
        """Used to shift an interval, use union to find a union"""
        return Interval(
            self.start + other, self.end + other, sr=self.sr, iter_rate=self.iter_rate
        )

    def __sub__(self, other: Union[Time, int, float]) -> "Interval":
        return Interval(
            self.start - other, self.end - other, sr=self.sr, iter_rate=self.iter_rate
        )

    def add(self, other: Union[Time, int, float]) -> None:
        """Add to object, rather than returning a new object"""
        self.start = self.start + other
        self.end = self.end + other

    def sub(self, other: Union[Time, int, float]) -> None:
        self.start = self.start - other
        self.end = self.end - other

    def union(self, other: "Interval") -> "Interval":
        """
        Merge intervals to make an interval from minimum start time to
        maximum end time. Other can be an interval, or a tuple of intervals.

        iter_rate and sr are inherited from the original
        event. Therefore, e1.union(e2) doesn't have to be the same as
        e2.union(e1)
        """
        assert self.sr == other.sr
        this_start = (self.start, other.start)[
            np.argmin((self.start.time, other.start.time))
        ]
        this_end = (self.end, other.end)[np.argmax((self.end.time, other.end.time))]
        return Interval(this_start, this_end, sr=self.sr, iter_rate=self.iter_rate)

    def intersection(self, other: "Interval") -> Union["Interval", Tuple]:
        assert self.sr == other.sr
        if (other.start.time > self.end.time) | (self.start.time > other.end.time):
            return ()
        this_start = (self.start, other.start)[
            np.argmax((self.start.time, other.start.time))
        ]
        this_end = (self.end, other.end)[np.argmin((self.end.time, other.end.time))]
        return Interval(this_start, this_end, sr=self.sr, iter_rate=self.iter_rate)


class Data:  # Signal processing
    """
    Signal processing class for sampled data. It is the most important class in this module. It allows for easy signal splicing, and includes wrappers for basic signal processing techniques. This class encapsulates signal values (data) with the sampling rate and provides wrappers for performing basic signal processing. It uses the :class:`pysampled.Time` class to ease the burden of managing time and converting between time (in seconds) and sample numbers.

    NOTE: When inheriting from this class, if the parameters of the ` __init__` method change, then make sure to rewrite the `_clone` method.

    Args:
        sig (np.ndarray): Signal data.
        sr (float): Sampling rate.
        axis (Optional[int]): Time axis.
        history (Optional[List[Tuple[str, Optional[Any]]]]): History of operations.
        t0 (float): Time at start sample.
        meta (Optional[dict]): Metadata.

    Example:
        x3 = sampled.Data(np.random.random((10, 3)), sr=2, t0=5.)
    """

    def __init__(
        self,
        sig: np.ndarray,
        sr: float,
        axis: Optional[int] = None,
        history: Optional[List[Tuple[str, Optional[Any]]]] = None,
        t0: float = 0.0,
        meta: Optional[dict] = None,
    ):
        self._sig = np.asarray(sig)  # assumes sig is uniformly resampled
        assert self._sig.ndim in (1, 2)
        if not hasattr(self, "sr"):  # in case of multiple inheritance - see ot.Marker
            self.sr = sr
        if axis is None:
            self.axis = np.argmax(np.shape(self._sig))
        else:
            self.axis = axis
        if history is None:
            self._history = [("initialized", None)]
        else:
            assert isinstance(history, list)
            self._history = history
        self._t0 = t0
        self.meta = meta

    def __call__(self, col: Optional[Union[int, str]] = None) -> np.ndarray:
        """Return either a specific column or the entire set 2D signal"""
        if col is None:
            return self._sig

        if isinstance(
            col, str
        ):  # supply an empty string to take advantage of easy plotting
            return self.t, self._sig

        assert isinstance(col, int) and col < len(self)
        slc = [slice(None)] * self._sig.ndim
        slc[self.get_signal_axis()] = col
        return self._sig[
            tuple(slc)
        ]  # not converting slc to tuple threw a FutureWarning

    def _clone(
        self,
        proc_sig: np.ndarray,
        his_append: Optional[Tuple[str, Optional[Any]]] = None,
        **kwargs
    ) -> "Data":
        if his_append is None:
            his = (
                self._history
            )  # only useful when cloning without manipulating the data, e.g. returning a subset of columns
        else:
            his = self._history + [his_append]

        if hasattr(self, "meta"):
            meta = self.meta
        else:
            meta = None
        axis = kwargs.pop("axis", self.axis)
        t0 = kwargs.pop("t0", self._t0)
        return self.__class__(proc_sig, self.sr, axis, his, t0, meta=meta)

    def copy(self) -> "Data":
        return self._clone(self._sig.copy())

    def analytic(self) -> "Data":
        """Extract the analytic signal. The analytic signal is a complex-valued signal whose real part is the original signal and whose imaginary part is the Hilbert transform of the original signal. It is useful for calculating the instantaneous attributes of a signal, such as its amplitude envelope, instantaneous phase, and instantaneous frequency (diff of instantaneous phase).

        This method is not intended to be called directly. Instead, use the `envelope`, `phase`, and `instantaneous_frequency` methods to extract the envelope, instantaneous phase, and instantaneous frequency of the signal."""
        proc_sig = scipy.signal.hilbert(self._sig, axis=self.axis)
        return self._clone(proc_sig, ("analytic", None))

    def envelope(
        self, type: str = "upper", lowpass: Union[bool, float] = True
    ) -> "Data":
        """Analytic envelope of the signal. It is optionally lowpass filtered.
        Note that the signal already needs to be bandpass filtered before applying the envelope.
        If not, either a value should be specified for lowpass, or lowpass should be set to False.

        Args:
            type (str, optional): Upper or lowe envelope. Defaults to "upper".
            lowpass (Union[bool, float], optional): Optionally lowpass filter the envelope, with the cutoff frequency defaulting to the lower end of the bandpass filtered signal. Defaults to True.

        Returns:
            Data: Envelope of the signal
        """
        assert type in ("upper", "lower")
        if type == "upper":
            proc_sig = np.abs(scipy.signal.hilbert(self._sig, axis=self.axis))
        else:
            proc_sig = -np.abs(scipy.signal.hilbert(-self._sig, axis=self.axis))

        if lowpass:
            if lowpass is True:  # set cutoff frequency to lower end of bandpass filter
                assert "bandpass" in [h[0] for h in self._history]
                lowpass = [h[1]["low"] for h in self._history if h[0] == "bandpass"][0]
            assert isinstance(lowpass, (int, float))  # cutoff frequency
            return self._clone(proc_sig, ("envelope_" + type, None)).lowpass(lowpass)
        return self._clone(proc_sig, ("envelope_" + type, None))

    def phase(self) -> "Data":
        """Extract the instantaneous phase is the phase of the analytic signal. It is the angle of the complex number formed by the real and imaginary parts of the analytic signal. It is useful for calculating phase-locking (synchronization) between signals."""
        proc_sig = np.unwrap(np.angle(scipy.signal.hilbert(self._sig, axis=self.axis)))
        return self._clone(proc_sig, ("instantaneous_phase", None))

    def instantaneous_frequency(self) -> "Data":
        """Extract the instantaneous frequency of the signal."""
        proc_sig = np.diff(self.phase()._sig) / (2.0 * np.pi) * self.sr
        return self._clone(proc_sig, ("instantaneous_frequency", None))

    def bandpass(self, low: float, high: float, order: Optional[int] = None) -> "Data":
        """Design and apply an FIR (finite impulse response) bandpass filter to the signal. To apply an IIR (infinite impulse response) filter instead, use the `lowpass` and `highpass` methods in tandem.

        Args:
            low (float): Lower cutoff frequency.
            high (float): Upper cutoff frequency.
            order (Optional[int], optional): Order of the filter. Defaults to half a second of the signal.

        Returns:
            Data: Bandpass filtered signal.
        """
        if order is None:
            order = int(self.sr / 2) + 1
        filt_pts = scipy.signal.firwin(
            order, (low, high), fs=self.sr, pass_zero="bandpass"
        )
        proc_sig = scipy.signal.filtfilt(filt_pts, 1, self._sig, axis=self.axis)
        return self._clone(
            proc_sig,
            (
                "bandpass",
                {"filter": "firwin", "low": low, "high": high, "order": order},
            ),
        )

    def _butterfilt(self, cutoff: float, order: Optional[int], btype: str) -> "Data":
        """Design and apply an IIR (infinite impulse response) filter to the signal. Instead of using this method directly, use the `lowpass` and `highpass` methods to apply a Butterworth filter to the signal. Note that missing values are interpolated before applying the filter to avoid common annoyances with missing data values.

        Args:
            cutoff (float): cutoff frequency
            order (Optional[int]): Order of the filter, defaults to 6.
            btype (str): Type of filter, either "low" or "high".

        Returns:
            Data: Filtered signal.
        """
        assert btype in ("low", "high")
        if order is None:
            order = 6
        b, a = scipy.signal.butter(
            order, cutoff / (0.5 * self.sr), btype=btype, analog=False
        )

        nan_manip = False
        nan_bool = np.isnan(self._sig)
        if nan_bool.any():
            nan_manip = True
            self = (
                self.interpnan()
            )  # interpolate missing values before applying an IIR filter

        proc_sig = scipy.signal.filtfilt(b, a, self._sig, axis=self.axis)
        if nan_manip:
            proc_sig[nan_bool] = np.NaN  # put back the NaNs in the same place

        return self._clone(
            proc_sig,
            (
                btype + "pass",
                {
                    "filter": "butter",
                    "cutoff": cutoff,
                    "order": order,
                    "NaN manipulation": nan_manip,
                },
            ),
        )

    def notch(self, cutoff: float, q: float = 30) -> "Data":
        """Apply a notch filter to the signal. A notch filter is a band-stop filter that removes a specific frequency from the signal. It is mostly used to remove power line interference. The cutoff frequency is the frequency to be removed, and the Q factor is the ratio of the center frequency to the bandwidth.

        Args:
            cutoff (float): Frequency to be removed
            q (float, optional): Ratio of the center frequency to the bandwidth. Defaults to 30, setting the bandwidth to 2 Hz for a 60 Hz  power line frequency in North America.

        Returns:
            Data: Notch-filtered signal.
        """
        b, a = scipy.signal.iirnotch(cutoff, q, self.sr)
        proc_sig = scipy.signal.filtfilt(b, a, self._sig, axis=self.axis)

        return self._clone(
            proc_sig, ("notch", {"filter": "iirnotch", "cutoff": cutoff, "q": q})
        )

    def lowpass(self, cutoff: float, order: Optional[int] = None) -> "Data":
        """Apply a lowpass butterworth IIR filter to the signal. Signal below the cutoff frequency is retained, and the order determines the steepness of the roll-off of the filter.

        Args:
            cutoff (float): Cutoff frequency.
            order (Optional[int], optional): Filter order. Defaults to 6.

        Returns:
            Data: Lowpass filtered data.
        """
        return self._butterfilt(cutoff, order, "low")

    def highpass(self, cutoff: float, order: Optional[int] = None) -> "Data":
        """Apply a highpass butterworth IIR filter to the signal. Signal above the cutoff frequency is retained, and the order determines the steepness of the roll-off of the filter.

        Args:
            cutoff (float): Cutoff frequency.
            order (Optional[int], optional): Filter order. Defaults to 6.

        Returns:
            Data: Highpass filtered data.
        """
        return self._butterfilt(cutoff, order, "high")

    def get_trend_airPLS(self, *args, **kwargs) -> "Data":
        try:
            from .airPLS import airPLS
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "airPLS is not installed. Please follow installation instructions in README.md."
            )

        trend = np.apply_along_axis(airPLS, self.axis, self._sig, *args, **kwargs)
        return self._clone(trend, ("get_trend_airPLS", {"args": args, **kwargs}))

    def detrend_airPLS(self, *args, **kwargs) -> "Data":
        trend = self.get_trend_airPLS(*args, **kwargs)
        proc_sig = self._sig - trend()
        return self._clone(proc_sig, ("detrend_airPLS", {"args": args, **kwargs}))

    def medfilt(self, order: Union[int, float] = 11) -> "Data":
        """
        Median filter the signal

        order is the number of samples in the kernel if it is an int, and treated as time if it is a float
        """
        sw = (
            np.lib.stride_tricks.sliding_window_view
        )  # this should be much faster than using running window
        if isinstance(order, float):
            order = int(order * self.sr)
        assert isinstance(order, int)
        order = (order // 2) * 2 + 1  # ensure order is odd for simpler handling of time
        proc_sig_middle = np.median(sw(self._sig, order, axis=self.axis), axis=-1)
        pre_fill = np.take(self._sig, np.r_[: order // 2], axis=self.axis)
        post_fill = np.take(self._sig, np.r_[-order // 2 + 1 : 0], axis=self.axis)
        proc_sig = np.concatenate(
            (pre_fill, proc_sig_middle, post_fill)
        )  # ends of the signal will not be filtered
        return self._clone(
            proc_sig,
            ("median_filter", {"order": order, "kernel_size_s": order / self.sr}),
        )

    def interpnan(
        self, maxgap: Optional[Union[int, float, np.ndarray]] = None, **kwargs
    ) -> "Data":
        """Interpolate NaN values. This method is useful for interpolating missing values in the signal. It uses the `scipy.interpolate.interp1d` function to interpolate the missing values. kwargs will be passed to scipy.interpolate.interp1d

        Args:
            maxgap (Optional[Union[int, float, np.ndarray]]): Various ways to specify where to perform interpolation. Defaults to None.
                (NoneType) all NaN values will be interpolated.
                (int) stretches of NaN values smaller than or equal to maxgap, in samples, will be interpolated.
                (float) stretches of NaN values smaller than or equal to maxgap, in seconds, will be interpolated.
                (boolean array) will be used as a mask where interpolation will only happen where maxgap is True.

        Returns:
            Data: _description_
        """
        if isinstance(maxgap, float):
            maxgap = np.round(maxgap * self.sr)  # seconds to samples

        proc_sig = np.apply_along_axis(
            interpnan, self.axis, self._sig, maxgap, **kwargs
        )
        return self._clone(proc_sig, ("Interpolate NaN values", None))

    def shift_baseline(self, offset: Optional[float] = None) -> "Data":
        # you can use numpy broadcasting to shift each signal if multi-dimensional
        if offset is None:
            offset = np.nanmean(self._sig, self.axis)
        return self._clone(self._sig - offset, ("shift_baseline", offset))

    def shift_left(self, time: Optional[float] = None) -> "Data":
        ret = self._clone(self._sig, ("shift_left", time))
        if time is None:  # shift to zero
            time = self._t0
        ret._t0 = self._t0 - time
        return ret

    def get_total_left_shift(self) -> float:
        """Return the total amount of time by which the signal was shifted to the left."""
        l_shift = [x[1] for x in self._history if x[0] == "shift_left"]
        return float(sum(l_shift))

    def reset_left_shift(self) -> "Data":
        return self.shift_left(-self.get_total_left_shift())

    def scale(self, scale_factor: float) -> "Data":
        return self._clone(self._sig / scale_factor, ("scale", scale_factor))

    def __len__(self) -> int:
        return np.shape(self._sig)[self.axis]

    @property
    def t(self) -> np.ndarray:
        n_samples = len(self)
        return np.linspace(self._t0, self._t0 + (n_samples - 1) / self.sr, n_samples)

    @property
    def dur(self) -> float:
        return (len(self) - 1) / self.sr

    def t_start(self) -> float:
        return self._t0

    def t_end(self) -> float:
        return self._t0 + (len(self) - 1) / self.sr

    def interval(self) -> Interval:
        return Interval(self.t_start(), self.t_end(), sr=self.sr)

    def _slice_to_interval(self, key: slice) -> Interval:
        assert (
            key.step is None
        )  # otherwise, the sampling rate is going to change, and could cause aliasing without proper filtering
        # IF INTEGERS, assume indices, IF FLOAT, assume time
        if isinstance(key.start, str):  # for things like data['t_start':'t_end']
            assert hasattr(self, "meta") and key.start in self.meta
            key = slice(self.meta[key.start], key.stop, None)
        if isinstance(key.stop, str):
            assert hasattr(self, "meta") and key.stop in self.meta
            key = slice(key.start, self.meta[key.stop], None)
        if isinstance(key.start, float) or isinstance(key.stop, float):
            intvl_start = key.start
            if key.start is None:
                intvl_start = self.t_start()
            intvl_end = key.stop
            if key.stop is None:
                intvl_end = self.t_end()
        else:  # if samples, do python indexing and don't include the end?
            assert isinstance(key.start, (int, type(None))) and isinstance(
                key.stop, (int, type(None))
            )
            if key.start is None:
                intvl_start = self.t_start()
            else:
                intvl_start = self.t[
                    sorted((0, key.start, len(self) - 1))[1]
                ]  # clip to limits
            if key.stop is None:
                intvl_end = self.t_end()
            else:
                intvl_end = self.t[sorted((0, key.stop - 1, len(self) - 1))[1]]
        return Interval(float(intvl_start), float(intvl_end), sr=self.sr)

    def _interval_to_index(self, key: Interval) -> Tuple[int, int]:
        assert key.sr == self.sr
        offset = round(self._t0 * self.sr)
        rng_start = sorted((0, key.start.sample - offset, len(self) - 1))[1]
        rng_end = sorted((0, key.end.sample - offset + 1, len(self)))[
            1
        ]  # +1 because interval object includes both ends!
        return rng_start, rng_end

    def take_by_interval(self, key: Interval) -> "Data":
        his = self._history + [("slice", key)]
        rng_start, rng_end = self._interval_to_index(key)
        proc_sig = self._sig.take(indices=range(rng_start, rng_end), axis=self.axis)
        if hasattr(self, "meta"):
            meta = self.meta
        else:
            meta = None
        return self.__class__(
            proc_sig, self.sr, self.axis, his, self.t[rng_start], meta
        )

    def __getitem__(
        self, key: Union[int, float, slice, Interval, str]
    ) -> Union[np.ndarray, "Data", Any]:
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
        if isinstance(
            key, (list, tuple, float, int)
        ):  # return signal (interpolated if needbe) values at those times
            if isinstance(key, int):
                key = self.t[key]
            return scipy.interpolate.interp1d(self.t, self._sig, axis=self.axis)(key)

        if isinstance(key, str):
            if hasattr(self, "meta") and key in self.meta:
                return self.meta[key]

        assert isinstance(key, (Interval, slice))
        if isinstance(key, slice):
            key = self._slice_to_interval(key)
        return self.take_by_interval(key)

    def make_running_win(
        self, win_size: float = 0.25, win_inc: float = 0.1
    ) -> "RunningWin":
        win_size_samples = (
            round(win_size * self.sr) // 2
        ) * 2 + 1  # ensure odd number of samples
        win_inc_samples = round(win_inc * self.sr)
        n_samples = len(self)
        return RunningWin(n_samples, win_size_samples, win_inc_samples)

    def apply_running_win(
        self,
        func: Callable[[np.ndarray, int], Any],
        win_size: float = 0.25,
        win_inc: float = 0.1,
    ) -> "Data":
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
        ret_sr = self.sr / round(win_inc * self.sr)
        return Data(ret_sig, ret_sr, axis=self.axis, t0=self.t[rw.center_idx[0]])

    def __le__(self, other: Union[int, float]) -> "Data":
        return self._comparison("__le__", other)

    def __ge__(self, other: Union[int, float]) -> "Data":
        return self._comparison("__ge__", other)

    def __lt__(self, other: Union[int, float]) -> "Data":
        return self._comparison("__lt__", other)

    def __gt__(self, other: Union[int, float]) -> "Data":
        return self._comparison("__gt__", other)

    def __eq__(self, other: Union[int, float]) -> "Data":
        return self._comparison("__eq__", other)

    def __ne__(self, other: Union[int, float]) -> "Data":
        return self._comparison("__ne__", other)

    def _comparison(self, dunder: str, other: Union[int, float]) -> "Data":
        cmp_dunder_dict = {
            "__le__": "<=",
            "__ge__": ">=",
            "__lt__": "<",
            "__gt__": ">",
            "__eq__": "==",
            "__ne__": "!=",
        }
        assert dunder in cmp_dunder_dict
        assert isinstance(other, (int, float))
        return self._clone(
            getattr(self._sig, dunder)(other), (cmp_dunder_dict[dunder], other)
        )

    def onoff_times(self) -> Tuple[List[float], List[float]]:
        """Onset and offset times of a thresholded 1D sampled.Data object"""
        onset_samples, offset_samples = onoff_samples(self._sig)
        return [self.t[x] for x in onset_samples], [self.t[x] for x in offset_samples]

    def find_crossings(
        self, th: float = 0.0, th_time: Optional[float] = None
    ) -> Tuple[List[float], List[float]]:
        """Find the times at which the signal crosses a given threshold th.
        th_time - Ignore crossings that are less than th_time apart. Caution - uses median filter, check carefully.
        """
        if th_time is None:
            neg_to_pos, pos_to_neg = (self > th).onoff_times()
        else:
            neg_to_pos, pos_to_neg = (
                (self > th).medfilt(order=round(self.sr * th_time * 2)) > 0.5
            ).onoff_times()
        return neg_to_pos, pos_to_neg

    def get_signal_axis(self) -> Optional[int]:
        if self().ndim == 1:
            return None  # there is no signal axis for a 1d signal
        return (self.axis + 1) % self().ndim

    def n_signals(self) -> int:
        if self().ndim == 1:
            return 1
        return self().shape[self.get_signal_axis()]

    def split_to_1d(self) -> List["Data"]:
        if self().ndim == 1:
            return [self]
        return [
            self._clone(self(col), his_append=("split", col), axis=0)
            for col in range(self.n_signals())
        ]

    def transpose(self) -> "Data":
        if self().ndim == 1:
            return self  # nothing done
        return self._clone(self._sig.T, axis=self.get_signal_axis())

    def fft(
        self,
        win_size: Optional[float] = None,
        win_inc: Optional[float] = None,
        zero_mean: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        T = 1 / self.sr
        if win_size is None and win_inc is None:
            N = len(self)
            f = scipy.fft.fftfreq(N, T)[: N // 2]
            sig = self._clone(self._sig)
            if zero_mean:
                sig = sig.shift_baseline()
            if np.ndim(sig) == 1:
                amp = 2.0 / N * np.abs(scipy.fft.fft(sig)[0 : N // 2])
            else:
                amp = np.array(
                    [
                        2.0 / N * np.abs(scipy.fft.fft(s())[0 : N // 2])
                        for s in sig.split_to_1d()
                    ]
                ).T
            return f, amp

        # do a sliding window fft
        if win_inc is None:
            win_inc = win_size  # no overlap

        rw = self.make_running_win(win_size, win_inc)
        if np.ndim(self._sig) == 1:
            amp_all = []
            for this_rw in rw():
                sig = self[this_rw]
                if zero_mean:
                    sig = sig.shift_baseline()
                N = len(sig)
                this_amp = 2.0 / N * np.abs(scipy.fft.fft(sig())[0 : N // 2])
                amp_all.append(this_amp)
            f = scipy.fft.fftfreq(N, T)[: N // 2]
            amp = np.mean(amp_all, axis=0)
            return f, amp
        if np.ndim(self._sig) == 2:
            amp_all = []
            for sig in self.split_to_1d():
                f, amp = sig.fft(win_size, win_inc, zero_mean)
                amp_all.append(amp)
            return f, np.array(amp_all).T

    def fft_as_sampled(self, *args, **kwargs) -> "Data":
        f, amp = self.fft(*args, **kwargs)
        df = (f[-1] - f[0]) / (len(f) - 1)
        return Data(
            amp, sr=1 / df, t0=f[0]
        )  # think of it as sr number of samples per Hz (instead of samples per second)

    def psd(
        self, win_size: float = 5.0, win_inc: Optional[float] = None, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """compute the power spectral density using the welch method"""
        kwargs_default = dict(nperseg=round(self.sr * win_size), scaling="density")
        kwargs = {**kwargs_default, **kwargs}
        if win_inc is not None:
            kwargs["noverlap"] = kwargs["nperseg"] - round(self.sr * win_inc)
        else:
            kwargs["noverlap"] = None
        if self().ndim == 1:
            f, Pxx = scipy.signal.welch(self._sig, self.sr, **kwargs)
            return f, Pxx
        Pxx = []
        for s in self.split_to_1d():
            f, this_Pxx = scipy.signal.welch(s._sig, s.sr, **kwargs)
            Pxx.append(this_Pxx)
        Pxx = np.vstack(Pxx).T
        return f, Pxx

    def psd_as_sampled(self, *args, **kwargs) -> "Data":
        f, Pxx = self.psd(*args, **kwargs)
        df = (f[-1] - f[0]) / (len(f) - 1)
        return Data(Pxx, sr=1 / df, t0=f[0])

    def frac_power(
        self,
        freq_lim: Tuple[float, float],
        win_size: float = 5.0,
        win_inc: float = 2.5,
        freq_dx: float = 0.05,
        highpass_cutoff: float = 0.2,
    ) -> "Data":
        """
        Calculate the fraction of power in a specific frequency band.

        Args:
            freq_lim (Tuple[float, float]): Frequency limits.
            win_size (float): Window size.
            win_inc (float): Window increment.
            freq_dx (float): Frequency resolution.
            highpass_cutoff (float): Highpass cutoff frequency.

        Returns:
            Data: Fraction of power in the specified frequency band.
        """
        assert len(freq_lim) == 2
        curr_t = self.t_start()
        ret = []
        while curr_t + win_size < self.t_end():
            try:
                sig_piece = self[float(curr_t) : float(curr_t + win_size)]
                if highpass_cutoff > 0:
                    f, amp = sig_piece.shift_baseline().highpass(highpass_cutoff).fft()
                else:
                    f, amp = sig_piece.shift_baseline().fft()
                area_of_interest = np.trapz(
                    scipy.interpolate.interp1d(f, amp)(
                        np.r_[freq_lim[0] : freq_lim[1] : freq_dx]
                    ),
                    dx=freq_dx,
                )
                total_area = np.trapz(amp, f)
                ret.append(area_of_interest / total_area)
                curr_t = curr_t + win_inc
            except ValueError:
                ret.append(np.nan)
                curr_t = curr_t + win_inc

        return Data(ret, 1 / win_inc, t0=self.t_start() + win_size / 2)

    def diff(self) -> "Data":
        """Differentiate the signal. Unlike np.diff, the number of samples is preserved, and the units will be in per second, as opposed to per sample. In other words, the np.diff output is multiplied by the sampling rate of the signal."""
        if self._sig.ndim == 2:
            if self.axis == 1:
                pp_value = (self._sig[:, 1] - self._sig[:, 0])[:, None]
                fn = np.hstack
            else:  # self.axis == 0
                pp_value = self._sig[1] - self._sig[0]
                fn = np.vstack
        else:  # self._sig.ndim == 1
            pp_value = self._sig[1] - self._sig[0]
            fn = np.hstack

        return self._clone(
            fn((pp_value, np.diff(self._sig, axis=self.axis, n=1))) * self.sr,
            ("diff", None),
        )

    def magnitude(self) -> "Data":
        if self._sig.ndim == 1:
            return self
        assert (
            self._sig.ndim == 2
        )  # magnitude does not make sense for a 1D signal (in that case, use np.linalg.norm directly)
        return Data(
            np.linalg.norm(self._sig, axis=(self.axis + 1) % 2),
            self.sr,
            history=self._history + [("magnitude", "None")],
            t0=self._t0,
            meta=self.meta,
        )

    def apply(self, func: Callable[..., np.ndarray], *args, **kwargs) -> "Data":
        """apply a function func along the time axis"""
        try:
            kwargs["axis"] = self.axis
            proc_sig = func(self._sig, *args, **kwargs)
        except TypeError:
            kwargs.pop("axis")
            proc_sig = func(self._sig, *args, **kwargs)
        return self._clone(
            proc_sig, ("apply", {"func": str(func), "args": args, "kwargs": kwargs})
        )

    def apply_along_signals(
        self, func: Callable[..., np.ndarray], *args, **kwargs
    ) -> "Data":
        """apply a function func along the signal axis"""
        try:
            kwargs["axis"] = self.get_signal_axis()
            proc_sig = func(self._sig, *args, **kwargs)
        except TypeError:
            kwargs.pop("axis")
            proc_sig = func(self._sig, *args, **kwargs)
        return self._clone(
            proc_sig,
            (
                "apply_along_signals",
                {"func": str(func), "args": args, "kwargs": kwargs},
            ),
        )

    def apply_to_each_signal(
        self, func: Callable[..., np.ndarray], *args, **kwargs
    ) -> "Data":
        """Apply a function to each signal (if self is a collection of signals) separately, and put it back together"""
        if self().ndim == 1:
            return self.apply(func, *args, **kwargs)

        assert self().ndim == 2
        proc_sig = np.vstack(
            [func(s._sig.copy(), *args, **kwargs) for s in self.split_to_1d()]
        )
        if self.axis == 0:
            proc_sig = proc_sig.T
        return self._clone(
            proc_sig,
            (
                "apply_to_each_signal",
                {"func": str(func), "args": args, "kwargs": kwargs},
            ),
        )

    def regress(self, ref_sig: "Data") -> "Data":
        """Regress a reference signal out of the current signal"""
        try:
            import sklearn.linear_model
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "sklearn is not installed. Please install it from https://scikit-learn.org/stable/install.html"
            )
        assert (
            ref_sig().ndim == self().ndim == 1
        )  # currently only defined for 1D signals
        assert ref_sig.sr == self.sr
        assert len(ref_sig) == len(self)
        reg = sklearn.linear_model.LinearRegression().fit(
            ref_sig().reshape(-1, 1), self()
        )
        prediction = reg.coef_[0] * ref_sig() + reg.intercept_
        return self._clone(self() - prediction, ("Regressed with reference", ref_sig()))

    def resample(self, new_sr: float, *args, **kwargs) -> "Data":
        """args and kwargs will be passed to scipy.signal.resample"""
        proc_sig, proc_t = scipy.signal.resample(
            self._sig,
            round(len(self) * new_sr / self.sr),
            t=self.t,
            axis=self.axis,
            *args,
            **kwargs
        )
        if hasattr(self, "meta"):
            meta = self.meta
        else:
            meta = None
        return self.__class__(
            proc_sig,
            sr=new_sr,
            axis=self.axis,
            history=self._history + [("resample", new_sr)],
            t0=proc_t[0],
            meta=meta,
        )

    def smooth(self, win_size: float = 0.5, kernel_type="flat") -> "Data":
        """Moving average smoothing with different kernels while preserving the number of samples in the signal. The kernel_type can be one of the following: 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"""
        kernel_len = round(win_size * self.sr)

        proc_sig = np.apply_along_axis(
            _smooth, self.axis, self._sig, kernel_len, kernel_type
        )
        return self._clone(
            proc_sig, ("smooth", {"win_size": win_size, "kernel_type": kernel_type})
        )

    def moving_average(self, win_size: float = 0.5) -> "Data":
        """Moving average smoothing. Same as applying the "flat" window in the `smooth` method. NOTE: There seems tob e a time offset of half a sample in the output signal compared to the smooth method implemented based on the convlolution. This needs further investigation."""
        stride = round(win_size * self.sr)
        proc_sig = np.lib.stride_tricks.sliding_window_view(
            self._sig, stride, axis=self.axis
        ).mean(axis=-1)
        t_start_offset = (stride - 1) / (2 * self.sr)
        return self._clone(
            proc_sig,
            ("moving average with stride", stride),
            t0=self._t0 + t_start_offset,
        )

    def xlim(self) -> Tuple[float, float]:
        return self.t_start(), self.t_end()

    def ylim(self) -> Tuple[float, float]:
        return np.nanmin(self._sig), np.nanmax(self._sig)

    def logdj(self, interpnan_maxgap: Optional[int] = None) -> float:
        """Compute the log dimensionless jerk, a measure of smoothness of the signal. If the value is closer to zero, then the signal is smoother. CAUTION: makes sense ONLY if self is a velocity signal (as in, a vector, as opposed to a scalar speed signal).

        Args:
            interpnan_maxgap (Optional[int], optional): maximum gap (in number of samples) to interpolate. Defaults to None. None (default) interpolates all gaps. Supply 0 to not interpolate.

        Returns:
            float: log dimensionless jerk
        """
        vel = self.interpnan(maxgap=interpnan_maxgap)

        dt = 1 / self.sr
        scale = np.power(self.dur, 3) / np.power(np.max(vel._sig), 2)

        # jerk = vel.apply_to_each_signal(np.diff, 2).apply(lambda x: x/dt**2) # there is a small difference between the values when using diff and gradient.
        jerk = vel.apply_to_each_signal(np.gradient, dt).apply_to_each_signal(
            np.gradient, dt
        )
        return -np.log(
            scale * scipy.integrate.simpson(np.power(jerk.magnitude()(), 2), dx=dt)
        )

    def logdj2(self, interpnan_maxgap: Optional[int] = None) -> float:
        """Compute the log dimensionless jerk from the speed, a measure of smoothness of the signal. If the value is closer to zero, then the signal is smoother. CAUTION: makes sense ONLY if self is a speed signal (as in, a scalar speed, as opposed to a vector velocity signal). Variation with speed instead of velocity.

        Args:
            interpnan_maxgap (Optional[int], optional): maximum gap (in number of samples) to interpolate. Defaults to None. None (default) interpolates all gaps. Supply 0 to not interpolate.

        Returns:
            float: log dimensionless jerk
        """
        speed = self.interpnan(maxgap=interpnan_maxgap)

        dt = 1 / self.sr
        scale = np.power(self.dur, 3) / np.power(np.max(speed._sig), 2)

        jerk = speed.apply(np.gradient, dt).apply(np.gradient, dt)
        return -np.log(scale * scipy.integrate.simpson(np.power(jerk(), 2), dx=dt))

    def sparc(
        self,
        fc: float = 10.0,
        amp_th: float = 0.05,
        interpnan_maxgap: Optional[int] = None,
        shift_baseline: bool = False,
        mean_normalize: bool = True,
    ) -> float:
        """Compute the spectral arc length, another measure of signal smoothness. The results from sparc were unpredictable, and therefore, we recommend using logdj instead. CAUTION: makes sense ONLY if self is a speed signal (as in, a scalar speed, as opposed to a vector velocity signal).

        Args:
            fc (float, optional): Cutoff frequency. Defaults to 10.0 Hz.
            amp_th (float, optional): Amplitude threshold. Defaults to 0.05.
            interpnan_maxgap (Optional[int], optional): maximum gap (in number of samples) to interpolate. None (default) interpolates all gaps. Supply 0 to not interpolate.
            shift_baseline (bool, optional): Set it to True to subtract the mean from the signal before computing sparc. Defaults to False.
            mean_normalize (bool, optional): Divide the signal by the mean. Requred to make smoothness metric
                insensitive to signal amplitude. Defaults to True.

        Returns:
            float: sparc value
        """
        speed = self.interpnan(maxgap=interpnan_maxgap)
        if shift_baseline:
            speed = speed.shift_baseline()
        if mean_normalize:
            speed = speed.apply(lambda x: x / np.nanmean(x))

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
        integrand = np.sqrt((1 / fc) ** 2 + Mf_sel_diff**2)
        sparc = -scipy.integrate.simpson(integrand, freq_sel)
        return sparc

    def set_nan(self, interval_list: List[Tuple[float, float]]) -> "Data":
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

        return self.copy().apply_to_each_signal(set_nan, idx_list=sel)

    def remove_and_interpolate(
        self,
        interval_list: List[Tuple[float, float]],
        maxgap: Optional[int] = None,
        **kwargs
    ) -> "Data":
        """Remove parts of a signal, and interpolate between those points."""
        if not interval_list:
            return self
        return self.set_nan(interval_list).interpnan(maxgap=maxgap, **kwargs)


class DataList(list):
    """
    A list of :class:`pysampled.Data` objects with filtering capabilities based on metadata.
    """

    def __call__(self, **kwargs) -> "DataList":
        ret = self
        for key, val in kwargs.items():
            if key.endswith("_lim") and (key.removesuffix("_lim")) in self[0].meta:
                assert len(val) == 2
                ret = [s for s in ret if val[0] <= s.meta[key] <= val[1]]
            elif key.endswith("_any") and (key.removesuffix("_lim")) in self[0].meta:
                ret = [s for s in ret if s.meta[key] in val]
            elif key in self[0].meta:
                ret = [s for s in ret if s.meta[key] == val]
            else:
                continue  # key was not in meta
        return self.__class__(ret)


class Event(Interval):
    """
    Interval with labels.

    Args:
        start (Union[Interval, Time, str, float, int, Tuple[Union[str, float, int], float]]): Start time.
        end (Optional[Union[Time, str, float, int, Tuple[Union[str, float, int], float]]]): End time.
        labels (list of strings): (supply as a kwarg) Hashtags defining the event.
    """

    def __init__(
        self,
        start: Union[
            Interval, Time, str, float, int, Tuple[Union[str, float, int], float]
        ],
        end: Optional[
            Union[Time, str, float, int, Tuple[Union[str, float, int], float]]
        ] = None,
        **kwargs
    ):
        """
        Interval with labels.

        kwargs:
        labels (list of strings) - hastags defining the event
        """
        if end is None:  # typecast interval into an event
            assert isinstance(start, Interval)
            end = start.end
            start = start.start
        self.labels = kwargs.pop("labels", [])
        super().__init__(start, end, **kwargs)

    def add_labels(self, *new_labels: str) -> None:
        self.labels += list(new_labels)

    def remove_labels(self, *labels_to_remove: str) -> None:
        self.labels = [label for label in self.labels if label not in labels_to_remove]


class Events(list):
    """
    List of event objects that can be selected by labels using the 'get' method.
    """

    def append(self, key: Union[Event, Interval]) -> None:
        assert isinstance(key, (Event, Interval))
        super().append(Event(key))

    def get(self, label: str) -> "Events":
        return Events([e for e in self if label in e.labels])


class RunningWin:
    """
    Manages running windows for data processing.

    Args:
        n_samples (int): Number of samples.
        win_size (int): Window size.
        win_inc (int): Window increment.
        step (Optional[int]): Step size.
        offset (int): Offset for running windows. This is useful when the object you're slicing has an inherent offset that you need to consider.
            For example, consider creating running windows on a sliced optitrack marker Think of offset as start_sample

    Attributes:
        run_win (List[slice]): List of slice objects, one per running window.
        center_idx (List): Indices of center samples.
    """

    def __init__(
        self,
        n_samples: int,
        win_size: int,
        win_inc: int = 1,
        step: Optional[int] = None,
        offset: int = 0,
    ):
        self.n_samples = int(n_samples)
        self.win_size = int(win_size)
        self.win_inc = int(win_inc)
        self.n_win = int(np.floor((n_samples - win_size) / win_inc) + 1)
        self.start_index = int(offset)

        run_win = []
        center_idx = []
        for win_count in range(0, self.n_win):
            win_start = (win_count * win_inc) + offset
            win_end = win_start + win_size
            center_idx.append(win_start + win_size // 2)
            run_win.append(slice(win_start, win_end, step))

        self._run_win = run_win
        self.center_idx = center_idx

    def __call__(
        self, data: Optional[np.ndarray] = None
    ) -> Union[List[slice], List[np.ndarray]]:
        if data is None:  # return slice objects
            return self._run_win
        # if data is supplied, apply slice objects to the data
        assert len(data) == self.n_samples
        return [data[x] for x in self._run_win]

    def __len__(self) -> int:
        return self.n_win


class Siglets:
    """
    A collection of pieces of signals to do event-triggered analyses.

    Args:
        sig (Data): Signal data.
        events (Events): Events.
        window (Optional[Union[Interval, Tuple[float, float]]]): Window for events.
        cache (Optional[Any]): Cache.
    """

    AX_TIME, AX_TRIALS = 0, 1

    def __init__(
        self,
        sig: Data,
        events: Events,
        window: Optional[Union[Interval, Tuple[float, float]]] = None,
        cache: Optional[Any] = None,
    ):
        self.parent = sig
        if window is not None:  # use window when all events are of the same length
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

    def _parse_ax(self, axis: Union[int, str]) -> int:
        if isinstance(axis, int):
            return axis
        assert isinstance(axis, str)
        if axis in ("t", "time"):
            return self.AX_TIME
        return (
            self.AX_TRIALS
        )  # axis is anything, but ideally in ('ev', 'events', 'sig', 'signals', 'data', 'trials')

    @property
    def sr(self) -> float:
        return self.parent.sr

    @property
    def t(self) -> List[float]:
        """Return the time vector of the event window"""
        return self.window.t

    @property
    def n(self) -> int:
        """Return the number of siglets"""
        return len(self.events)

    def __len__(self) -> int:
        """Number of time points"""
        return len(self.window)

    def __call__(
        self,
        func: Optional[Callable[..., np.ndarray]] = None,
        axis: Union[int, str] = "events",
        *args,
        **kwargs
    ) -> np.ndarray:
        siglet_list = [self.parent[ev]() for ev in self.events]
        if func is None:
            return np.asarray(siglet_list).T
        return self.apply(func, axis=self._parse_ax(axis), *args, **kwargs)

    def apply_along_events(
        self, func: Callable[..., np.ndarray], *args, **kwargs
    ) -> np.ndarray:
        return func(self(), axis=self.AX_TRIALS, *args, **kwargs)

    def apply_along_time(
        self, func: Callable[..., np.ndarray], *args, **kwargs
    ) -> np.ndarray:
        return func(self(), axis=self.AX_TIME, *args, **kwargs)

    def apply(
        self,
        func: Callable[..., np.ndarray],
        axis: Union[int, str] = "events",
        *args,
        **kwargs
    ) -> np.ndarray:  # by default, applies to each siglet
        return func(self(), axis=self._parse_ax(axis), *args, **kwargs)

    def mean(self, axis: Union[int, str] = "events") -> np.ndarray:
        return self(np.mean, axis=axis)

    def sem(self, axis: Union[int, str] = "events") -> np.ndarray:
        return self(np.std, axis=axis) / np.sqrt(self.n)

    def is_uniform(self) -> bool:
        return (
            len(set([ev.dur_sample for ev in self.events])) == 1
        )  # if all events are of the same size


def interpnan(
    sig: np.ndarray,
    maxgap: Optional[Union[int, np.ndarray]] = None,
    min_data_frac: float = 0.2,
    **kwargs
) -> np.ndarray:
    """
    Interpolate NaNs in a 1D signal.

    Args:
        sig (np.ndarray): 1D numpy array.
        maxgap (Optional[Union[int, np.ndarray]]): Maximum gap to interpolate.
            (NoneType) all NaN values will be interpolated.
            (int) stretches of NaN values smaller than or equal to maxgap will be interpolated.
            (boolean array) will be used as a mask where interpolation will only happen where maxgap is True.
        min_data_frac (float): Minimum data fraction.
        **kwargs: Additional arguments for scipy.interpolate.interp1d.
            commonly used: kind='cubic'

    Returns:
        np.ndarray: Interpolated signal.
    """
    assert np.ndim(sig) == 1
    assert 0.0 <= min_data_frac <= 1.0
    if "fill_value" not in kwargs:
        kwargs["fill_value"] = "extrapolate"

    def nan_helper(y):
        return np.isnan(y), lambda z: z.nonzero()[0]

    proc_sig = sig.copy()
    nans, x = nan_helper(proc_sig)
    if np.mean(~nans) < min_data_frac:
        return sig  # interpolate only if there are enough data points

    if maxgap is None:
        mask = np.ones_like(nans)
    elif isinstance(maxgap, int):
        nans = np.isnan(sig)
        mask = np.zeros_like(nans)
        onset_samples, offset_samples = onoff_samples(nans)
        for on_s, off_s in zip(onset_samples, offset_samples):
            assert on_s < off_s
            if off_s - on_s <= maxgap:  # interpolate this
                mask[on_s:off_s] = True
    else:
        mask = maxgap
    assert len(mask) == len(sig)
    proc_sig[nans & mask] = scipy.interpolate.interp1d(
        x(~nans), proc_sig[~nans], **kwargs
    )(
        x(nans & mask)
    )  # np.interp(x(nans & mask), x(~nans), proc_sig[~nans])
    return proc_sig


def onoff_samples(tfsig: np.ndarray) -> Tuple[List[int], List[int]]:
    """
    Find onset and offset samples of a 1D boolean signal (e.g. Thresholded TTL pulse).

    Args:
        tfsig (np.ndarray): 1D boolean signal.

    Returns:
        Tuple[List[int], List[int]]: Onset and offset samples.
    """
    assert tfsig.dtype == bool
    assert np.sum(np.asarray(np.shape(tfsig)) > 1) == 1
    x = np.squeeze(tfsig).astype(int)
    onset_samples = list(np.where(np.diff(x) == 1)[0] + 1)
    offset_samples = list(np.where(np.diff(x) == -1)[0] + 1)
    if tfsig[0]:  # is True
        onset_samples = [0] + onset_samples
    if tfsig[-1]:
        offset_samples = offset_samples + [len(tfsig) - 1]
    return onset_samples, offset_samples


def uniform_resample(
    time: np.ndarray,
    sig: np.ndarray,
    sr: float,
    t_min: Optional[float] = None,
    t_max: Optional[float] = None,
) -> Data:
    """
    Uniformly resample a signal at a given sampling rate sr.

    Args:
        time (np.ndarray): Non-decreasing array of time points.
        sig (np.ndarray): Signal data.
        sr (float): Sampling rate in Hz.
        t_min (Optional[float]): Start time for the output array.
        t_max (Optional[float]): End time for the output array.

    Returns:
        Data: Uniformly resampled data.
    """
    assert len(time) == len(sig)
    time = np.array(time)
    sig = np.array(sig)

    if t_min is None:
        t_min = time[0]
    if t_max is None:
        t_max = time[-1]

    n_samples = int((t_max - t_min) * sr) + 1
    t_max = t_min + (n_samples - 1) / sr

    t_proc = np.linspace(t_min, t_max, n_samples)
    if np.ndim(sig) == 1:
        sig_proc = np.interp(t_proc, time, sig)
        return Data(sig_proc, sr, t0=t_min)
    sig_proc = np.zeros((len(t_proc), sig.shape[-1]))
    for col_count in range(sig.shape[-1]):
        sig_proc[:, col_count] = np.interp(t_proc, time, sig[:, col_count])
    return Data(sig_proc, sr, t0=t_min)


def _smooth(
    sig: np.ndarray, kernel_len: int = 10, kernel_type: str = "hanning"
) -> np.ndarray:
    """Smooth a signal using convolution with a kernel. The kernel can be a flat window, a Hanning window, a Hamming window, a Bartlett window, or a Blackman window. Note that this method only works for 1D signals.

    Args:
        window_len (int, optional): Length of the kernel in number of samples. Defaults to 10.
        window (str, optional): Kernel type. Defaults to "hanning".

    Returns:
        Data: Smoothed signal.
    """
    assert np.ndim(sig) == 1
    assert 3 < kernel_len < sig.size
    assert kernel_type in ("flat", "hanning", "hamming", "bartlett", "blackman")

    sig = np.r_[
        2 * sig[0] - sig[kernel_len:0:-1],
        sig,
        2 * sig[-1] - sig[-2 : -kernel_len - 2 : -1],
    ]

    if kernel_type == "flat":  # moving average
        win = np.ones(kernel_len, "d")
    else:
        win = getattr(np, kernel_type)(kernel_len)

    sig_conv = np.convolve(win / win.sum(), sig, mode="same")

    return sig_conv[kernel_len:-kernel_len]


def generate_signal(
    signal_type: str = "white_noise", sr: float = 100, duration: float = 10
) -> Data:
    """
    Generate a signal of a specific type. Intended for testing and demonstration purposes.

    Args:
        signal_type (str): Signal type. Either "white_noise", "sine_wave", "three_sine_waves", "ekg", or "accelerometer".
            It can also be "1d" or "2d" which return white_noise and accelerometer signals respectively.
        sr (float): Sampling rate.
        duration (float): Duration.

    Returns:
        Data: Signal data.
    """
    assert signal_type in (
        "white_noise",
        "sine_wave",
        "three_sine_waves",
        "ekg",
        "accelerometer",
    )
    signal_type = {"1d": "white_noise", "2d": "accelerometer"}.get(
        signal_type, signal_type
    )

    def _generate_signal(signal_type: str, sr: float, duration: float) -> np.ndarray:
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        if signal_type == "white_noise":
            local_rng = np.random.RandomState(721)
            return local_rng.normal(0, 1, t.shape)
        elif signal_type == "sine_wave":
            return np.sin(2 * np.pi * 1 * t)
        elif signal_type == "three_sine_waves":
            return (
                np.sin(2 * np.pi * 1 * t)
                + 0.5 * np.sin(2 * np.pi * 3 * t + np.pi / 4)
                + 0.25 * np.sin(2 * np.pi * 5 * t + np.pi / 2)
            )
        elif signal_type == "ekg":
            return scipy.signal.chirp(t, f0=0.5, f1=2.5, t1=duration, method="linear")
        elif signal_type == "accelerometer":
            return np.vstack(
                [
                    np.sin(2 * np.pi * 1 * t),
                    np.sin(2 * np.pi * 2 * t + np.pi / 4),
                    np.sin(2 * np.pi * 3 * t + np.pi / 2),
                ]
            ).T
        else:
            raise ValueError("Unknown signal type")

    return Data(_generate_signal(signal_type, sr, duration), sr=sr)
