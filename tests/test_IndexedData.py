import pytest
import numpy as np
from pysampled.core import IndexedData

@pytest.fixture(scope="module")
def data_2d():
    """Fixture for 2D signal data with shape (1000, 6)."""
    sig = np.random.random((1000, 6))
    return IndexedData(sig, sr=100, signal_names=["acc1", "acc2"], signal_coords=["x", "y", "z"])

@pytest.fixture(scope="module")
def data_2d_transposed():
    """Fixture for 2D signal data with shape (6, 1000)."""
    sig = np.random.random((6, 1000))
    return IndexedData(sig, sr=100, signal_names=["acc1", "acc2"], signal_coords=["x", "y", "z"])

@pytest.fixture(scope="module")
def data_1d():
    """Fixture for 1D signal data with shape (1000,)."""
    sig = np.random.random(1000)
    return IndexedData(sig, sr=100)

def test_initialization(data_2d, data_2d_transposed, data_1d):
    """Test initialization of IndexedData."""
    assert data_2d.n_signals() == 6
    assert data_2d_transposed.n_signals() == 6
    assert data_1d.n_signals() == 1

    assert data_2d.meta["signal_names"] == ["acc1", "acc2"]
    assert data_2d.meta["signal_coords"] == ["x", "y", "z"]

def test_access_by_signal_name(data_2d, data_2d_transposed):
    """Test accessing signals by their names."""
    acc1 = data_2d["acc1"]
    assert np.allclose(acc1(), data_2d()[:, :3])
    assert np.allclose(data_2d["x"](), data_2d()[:, ::3])
    assert np.allclose(data_2d["y"](), data_2d()[:, 1::3])
    assert np.allclose(data_2d["z"](), data_2d()[:, 2::3])
    assert acc1.n_signals() == 3
    assert acc1.meta["signal_names"] == ["acc1"]
    assert acc1.meta["signal_coords"] == ["x", "y", "z"]

    acc2 = data_2d["acc2"]
    assert np.allclose(acc2(), data_2d()[:, 3:])
    assert acc2.n_signals() == 3
    assert acc2.meta["signal_names"] == ["acc2"]
    assert acc2.meta["signal_coords"] == ["x", "y", "z"]

    assert np.allclose(data_2d_transposed["acc1"](), data_2d_transposed()[:3, :])
    assert np.allclose(data_2d_transposed["acc2"](), data_2d_transposed()[3:, :])
    assert np.allclose(data_2d_transposed["x"](), data_2d_transposed()[::3, :])
    assert np.allclose(data_2d_transposed["y"](), data_2d_transposed()[1::3, :])
    assert np.allclose(data_2d_transposed["z"](), data_2d_transposed()[2::3, :])


def test_access_by_signal_coord(data_2d):
    """Test accessing signals by their coordinates."""
    x_coord = data_2d["x"]
    assert x_coord.n_signals() == 2
    assert x_coord.meta["signal_names"] == ["acc1", "acc2"]
    assert x_coord.meta["signal_coords"] == ["x"]

    y_coord = data_2d["y"]
    assert y_coord.n_signals() == 2
    assert y_coord.meta["signal_names"] == ["acc1", "acc2"]
    assert y_coord.meta["signal_coords"] == ["y"]

def test_access_by_signal_name_and_coord(data_2d):
    """Test accessing specific signals by both names and coordinates."""
    acc1_x = data_2d["acc1"]["x"]
    assert np.allclose(acc1_x(), data_2d()[:, :1])
    assert acc1_x.n_signals() == 1
    assert acc1_x.meta["signal_names"] == ["acc1"]
    assert acc1_x.meta["signal_coords"] == ["x"]

def test_invalid_access(data_2d):
    """Test invalid access scenarios."""
    with pytest.raises(KeyError):
        data_2d["invalid"]

def test_subset_creation(data_2d):
    """Test creating subsets of IndexedData."""
    subset = data_2d["acc1"]["x"]
    assert subset.n_signals() == 1
    assert subset.meta["signal_names"] == ["acc1"]
    assert subset.meta["signal_coords"] == ["x"]

def test_transposed_data_access(data_2d_transposed):
    """Test accessing signals in transposed data."""
    acc1 = data_2d_transposed["acc1"]
    assert acc1.n_signals() == 3
    assert acc1.meta["signal_names"] == ["acc1"]
    assert acc1.meta["signal_coords"] == ["x", "y", "z"]

    x_coord = data_2d_transposed["x"]
    assert x_coord.n_signals() == 2
    assert x_coord.meta["signal_names"] == ["acc1", "acc2"]
    assert x_coord.meta["signal_coords"] == ["x"]

def test_1d_data_access(data_1d):
    """Test accessing 1D data."""
    assert data_1d.n_signals() == 1
    assert "signal_coords" in data_1d.meta
    assert "signal_names" in data_1d.meta

def test_selector(data_2d):
    """Test the selector method."""
    selector = data_2d._get_selector()
    assert "x" in selector
    assert "y" in selector
    assert "z" in selector
    assert len(selector["x"]) == 6
    assert len(selector["y"]) == 6
    assert len(selector["z"]) == 6

def test_label_to_idx(data_2d):
    """Test the label_to_idx method."""
    label_to_idx = data_2d._label_to_idx()
    assert label_to_idx["acc1"] == 0
    assert label_to_idx["acc2"] == 3

def test_abbr_to_label(data_2d):
    """Test the abbr_to_label method."""
    abbr_to_label = data_2d._abbr_to_label()
    assert abbr_to_label["s0"] == "acc1"
    assert abbr_to_label["s1"] == "acc2"