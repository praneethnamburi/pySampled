import pytest
import numpy as np
from pysampled import Data, Siglets, generate_signal


@pytest.fixture
def white_noise():
    return generate_signal("white_noise", 100, 10)


@pytest.fixture
def sine_wave():
    return generate_signal("sine_wave", 100, 10)


@pytest.fixture
def three_sine_waves():
    return generate_signal("three_sine_waves", 100, 10)


@pytest.fixture
def ekg():
    return generate_signal("ekg", 100, 10)


@pytest.fixture
def accelerometer():
    return generate_signal("accelerometer", 100, 10)


def test_init(white_noise):
    assert white_noise.sr == 100
    assert white_noise._sig.shape == (1000,)


def test_call(sine_wave):
    assert sine_wave().shape == (1000,)
    assert np.allclose(
        sine_wave()[:10],
        np.sin(2 * np.pi * 1 * np.linspace(0, 0.1, 10, endpoint=False)),
    )


def test_clone(three_sine_waves):
    cloned = three_sine_waves._clone(three_sine_waves._sig * 2)
    assert np.allclose(cloned._sig, three_sine_waves._sig * 2)


def test_analytic(white_noise, accelerometer):
    analytic_signal = white_noise.analytic()
    assert np.allclose(np.real(analytic_signal._sig), white_noise._sig)
    analytic_signal = accelerometer.analytic()
    assert np.allclose(np.real(analytic_signal._sig), accelerometer._sig)


def test_envelope(three_sine_waves):
    # envelope only makes sense
    envelope_signal = three_sine_waves.envelope(lowpass=2)
    assert envelope_signal._sig.shape == three_sine_waves._sig.shape


def test_phase(white_noise):
    phase_signal = white_noise.phase()
    assert phase_signal._sig.shape == white_noise._sig.shape


def test_instantaneous_frequency(white_noise):
    inst_freq = white_noise.instantaneous_frequency()
    assert inst_freq._sig.shape == (999,)


def test_bandpass(three_sine_waves):
    bandpassed = three_sine_waves.bandpass(0.5, 2.0)
    assert bandpassed._sig.shape == three_sine_waves._sig.shape


def test_notch(three_sine_waves):
    notched = three_sine_waves.notch(1.0)
    assert notched._sig.shape == three_sine_waves._sig.shape


def test_lowpass(three_sine_waves):
    lowpassed = three_sine_waves.lowpass(2.0)
    assert lowpassed._sig.shape == three_sine_waves._sig.shape


def test_highpass(three_sine_waves):
    highpassed = three_sine_waves.highpass(1.0)
    assert highpassed._sig.shape == three_sine_waves._sig.shape


def test_smooth(white_noise):
    smoothed = white_noise.smooth(window_len=10)
    assert smoothed.shape == white_noise._sig.shape


def test_get_trend_airPLS(ekg):
    trend = ekg.get_trend_airPLS()
    assert trend._sig.shape == ekg._sig.shape


def test_detrend_airPLS(ekg):
    detrended = ekg.detrend_airPLS()
    assert detrended._sig.shape == ekg._sig.shape


def test_medfilt(three_sine_waves):
    medfiltered = three_sine_waves.medfilt(order=11)
    assert medfiltered._sig.shape == three_sine_waves._sig.shape


def test_interpnan(three_sine_waves):
    interpolated = three_sine_waves.interpnan()
    assert interpolated._sig.shape == three_sine_waves._sig.shape


def test_shift_baseline(white_noise):
    shifted = white_noise.shift_baseline()
    assert shifted._sig.shape == white_noise._sig.shape


def test_shift_left(white_noise):
    shifted = white_noise.shift_left(1.0)
    assert shifted._t0 == white_noise._t0 - 1.0


def test_scale(white_noise):
    scaled = white_noise.scale(2.0)
    assert np.allclose(scaled._sig, white_noise._sig / 2.0)


def test_len(white_noise):
    assert len(white_noise) == 1000


def test_t(white_noise):
    assert white_noise.t.shape == (1000,)


def test_dur(white_noise):
    assert white_noise.dur == 9.99


def test_take_by_interval(white_noise):
    interval = white_noise.interval()
    taken = white_noise.take_by_interval(interval)
    assert taken._sig.shape == white_noise._sig.shape


def test_getitem(white_noise):
    assert white_noise[0].shape == ()
    assert white_noise[0.0:1.0]._sig.shape == (101,)
    assert white_noise[:1.5]._sig.shape == (151,)


def test_apply_running_win(white_noise):
    win_inc = 0.1
    applied = white_noise.apply_running_win(np.mean, win_size=0.5, win_inc=win_inc)
    assert applied.sr == white_noise.sr * win_inc
    assert applied._sig.shape[0] == 95


def test_comparison(white_noise):
    assert (white_noise <= 0)._sig.shape == white_noise._sig.shape


def test_onoff_times(white_noise):
    on_times, off_times = (white_noise < 0).onoff_times()
    assert isinstance(on_times, list)
    assert isinstance(off_times, list)


def test_find_crossings(white_noise):
    pos_crossings, neg_crossings = white_noise.find_crossings()
    assert isinstance(pos_crossings, list)
    assert isinstance(neg_crossings, list)


def test_split_to_1d(accelerometer):
    split = accelerometer.split_to_1d()
    assert len(split) == 3
    assert all(s._sig.shape == (1000,) for s in split)


def test_transpose(accelerometer):
    transposed = accelerometer.transpose()
    assert transposed._sig.shape == (3, 1000)
    assert accelerometer.axis == 0
    assert transposed.axis == 1


def test_fft(white_noise):
    f, amp = white_noise.fft()
    assert f.shape == amp.shape


def test_fft_as_sampled(white_noise):
    fft_sampled = white_noise.fft_as_sampled()
    assert fft_sampled._sig.shape[0] == 500


def test_psd(white_noise, accelerometer):
    f, Pxx = white_noise.psd()
    assert f.shape == Pxx.shape
    f, Pxx = accelerometer.psd()
    assert Pxx.shape == (251, 3)


def test_psd_as_sampled(white_noise):
    psd_sampled = white_noise.psd_as_sampled()
    assert psd_sampled._sig.shape[0] == 251


def test_diff(white_noise):
    diffed = white_noise.diff()
    assert diffed._sig.shape == (1000,)


def test_magnitude(accelerometer):
    magnitude = accelerometer.magnitude()
    assert magnitude._sig.shape == (1000,)


def test_apply(white_noise):
    applied = white_noise.apply(lambda x: x**2)
    assert applied._sig.shape == (1000,)


def test_apply_along_signals(accelerometer):
    applied = accelerometer.apply_along_signals(np.mean)
    assert applied._sig.shape == (1000,)


def test_apply_to_each_signal(accelerometer):
    applied = accelerometer.apply_to_each_signal(
        np.mean
    )  # Even though this works, this is not the intention of this method
    assert applied._sig.shape == (1, 3)


def test_regress(three_sine_waves, sine_wave):
    regressed = three_sine_waves.regress(sine_wave)
    assert regressed._sig.shape == sine_wave._sig.shape


def test_resample(white_noise):
    resampled = white_noise.resample(50)
    assert resampled._sig.shape == (500,)


def test_smooth(white_noise):
    smoothed = white_noise.smooth(win_size=0.5)
    assert smoothed._sig.shape == (1000,)


def test_xlim(white_noise):
    xlim = white_noise.xlim()
    assert isinstance(xlim, tuple)


def test_ylim(white_noise):
    ylim = white_noise.ylim()
    assert isinstance(ylim, tuple)


def test_logdj(accelerometer, white_noise):
    logdj = (
        white_noise.logdj()
    )  # this should be a velocity signal, but testing for a point moving in 1D
    assert isinstance(logdj, float)
    logdj = accelerometer.logdj()  # technically, this should be a velocity signal
    assert isinstance(logdj, float)


def test_logdj2(white_noise):
    logdj2 = white_noise.logdj2()
    assert isinstance(logdj2, float)


def test_sparc(white_noise):
    sparc = white_noise.sparc()
    assert isinstance(sparc, float)


def test_set_nan(white_noise):
    set_nan = white_noise.set_nan([(0.5, 1.0)])
    assert np.isnan(set_nan._sig[50:101]).all()


def test_remove_and_interpolate(white_noise):
    removed = white_noise.remove_and_interpolate([(0.5, 1.0)])
    assert not np.isnan(removed._sig).any()


# def test_smooth_vs_moving_average():
#     s = Data(np.hstack((np.arange(20), np.arange(20)[::-1])), sr=12)

#     pysampled.plot([s, s.smooth(0.3), s.moving_average(0.3)]) # even kernel
#     pysampled.plot([s, s.smooth(0.4), s.moving_average(0.4)]) # odd kernel

def test_siglets(three_sine_waves):
    sl = Siglets(three_sine_waves, (-1, 2, 4.7, 6.2, 7.1, 9.9), window=(-1., 2.))
    assert sl.n == 4
    assert sl().shape == (301, 4)
    sl = Siglets(three_sine_waves, (-1, 2, 4.7, 6.2, 7.1, 9.9), window=(-10, 20))
    assert sl().shape == (31, 4)

if __name__ == "__main__":
    pytest.main()
