import numpy as np
import pytest

import pytest
from pysampled import Time, Interval


def test_time_initialization():
    t1 = Time("00;09;53;29", 30)
    assert t1.sample == 17819
    assert t1.time == pytest.approx(593.9667, rel=1e-3)
    assert t1.sr == 30.0

    t2 = Time(9.32, 180)
    assert t2.sample == 1678
    assert t2.time == pytest.approx(9.3222, rel=1e-3)
    assert t2.sr == 180.0

    t3 = Time(12531, 180)
    assert t3.sample == 12531
    assert t3.time == pytest.approx(69.6167, rel=1e-3)
    assert t3.sr == 180.0

    t4 = Time((9.32, 180), 180)
    assert t4.sample == 1678
    assert t4.time == pytest.approx(9.3222, rel=1e-3)
    assert t4.sr == 180.0

    t5 = Time(("00;09;51;03", 30), 180)
    assert t5.sr == 180.0
    assert t5.sample == 106398
    assert t5.time == pytest.approx(591.1, rel=1e-3)


def test_time_sr_change():
    t = Time(9.32, 180)
    t.sr = 90
    assert t.sr == 90.0
    assert t.sample == 839
    assert t.time == pytest.approx(9.3222, rel=1e-3)


def test_time_arithmetic():
    t1 = Time(9.32, 180)
    t2 = Time(5.0, 180)
    t3 = t1 + t2
    assert t3.sample == 2578
    assert t3.time == pytest.approx(14.322, rel=1e-3)

    t4 = t1 - t2
    assert t4.sample == 778
    assert t4.time == pytest.approx(4.322, rel=1e-3)


def test_interval_initialization():
    intvl = Interval(("00;09;51;03", 30), ("00;09;54;11", 30), sr=180, iter_rate=24)
    assert intvl.start.sample == 106398
    assert intvl.end.sample == 106986
    assert intvl.sr == 180.0
    assert intvl.iter_rate == 24.0


def test_interval_duration():
    intvl = Interval(("00;09;51;03", 30), ("00;09;54;11", 30), sr=180, iter_rate=24)
    assert intvl.dur_time == pytest.approx(3.2667, rel=1e-3)
    assert intvl.dur_sample == round(intvl.dur_time * intvl.sr) + 1 == 589


def test_interval_iteration():
    intvl = Interval(("00;09;51;03", 30), ("00;09;54;11", 30), sr=180, iter_rate=24)
    samples = list(intvl)
    assert (
        len(samples) == np.ceil(intvl.dur_time * 24 + 1) == 80
    )  # ceil(3.2667 seconds * 24 fps + 1)
    assert samples[0] == (106398, pytest.approx(591.1, rel=1e-3), 0)
    assert samples[-1] == (106990, pytest.approx(594.39167, rel=1e-3), 79)


def test_interval_arithmetic():
    intvl = Interval(("00;09;51;03", 30), ("00;09;54;11", 30), sr=180, iter_rate=24)
    shifted_intvl = intvl + 1.0
    assert shifted_intvl.start.sample == 106578
    assert shifted_intvl.end.sample == 107166

    shifted_intvl = intvl - 1.0
    assert shifted_intvl.start.sample == 106218
    assert shifted_intvl.end.sample == 106806


def test_interval_union():
    intvl1 = Interval(("00;09;51;03", 30), ("00;09;52;03", 30), sr=180, iter_rate=24)
    intvl2 = Interval(("00;09;51;23", 30), ("00;09;54;11", 30), sr=180, iter_rate=24)
    union_intvl = intvl1.union(intvl2)
    assert union_intvl.start.sample == 106398
    assert union_intvl.end.sample == 106986


def test_interval_intersection():
    intvl1 = Interval(("00;09;51;03", 30), ("00;09;52;03", 30), sr=180, iter_rate=24)
    intvl2 = Interval(("00;09;51;23", 30), ("00;09;54;11", 30), sr=180, iter_rate=24)
    intersection_intvl = intvl1.intersection(intvl2)
    assert intersection_intvl.start.sample == 106518
    assert intersection_intvl.end.sample == 106578
    assert intersection_intvl.dur_time == pytest.approx(0.3333, rel=1e-3)
    assert intersection_intvl.dur_sample == 61  # interval includes start and end times


if __name__ == "__main__":
    pytest.main()
