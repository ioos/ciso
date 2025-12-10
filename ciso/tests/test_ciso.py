from pathlib import Path

import numpy as np
import pytest

from ciso import zslice

data_path = Path(__file__).parent.joinpath("data")


@pytest.fixture
def data():
    p = np.linspace(-100, 0, 30)[:, None, None] * np.ones((50, 70))
    x, y = np.mgrid[0:20:50j, 0:20:70j]
    q = np.sin(x) + p
    return {"q": q, "p": p, "x": x, "y": y}


@pytest.fixture
def expected_results():
    return np.load(data_path.joinpath("fortran.npz"))["s50"]


def test_mismatch_shapes(data):
    with pytest.raises(ValueError, match="must be of the same shape"):
        zslice(data["q"], data["p"][0], p0=0)


def test_p0_wrong_shape(data):
    with pytest.raises(
        ValueError,
        match="p0 must be a float number or 0-dim array",
    ):
        zslice(data["q"], data["p"], p0=np.zeros((2, 2)))


def test_bad_dtypes(data):
    # NB: Boolean array are converted to float!  Only str fails correctly.
    with pytest.raises(ValueError, match="could not convert string to float"):
        zslice(np.empty_like(data["q"], dtype=np.str_), data["p"], p0=0)


@pytest.mark.parametrize(
    "dtype",
    [int, float, np.int32, np.int64, np.float16, np.float32, np.float64],
)
def test_good_dtypes(data, dtype):
    zslice(np.empty_like(data["q"], dtype=dtype), data["p"], p0=0)


def test_3d_input(data):
    _, I, J = data["q"].shape
    s50 = zslice(data["q"], data["p"], p0=-50)
    assert s50.shape == (I, J)


def test_2d_input(data):
    K, I, J = data["q"].shape
    s50 = zslice(data["q"].reshape(K, -1), data["p"].reshape(K, -1), p0=-50)
    assert s50.shape == (I * J,)


def test_1d_input(data):
    with pytest.raises(ValueError, match="Expected 2D"):
        zslice(data["q"].ravel(), data["p"].ravel(), p0=0)


def test_gt_3d_input(data):
    with pytest.raises(ValueError, match=r"Expected 2D."):
        zslice(data["q"][np.newaxis, ...], data["p"][np.newaxis, ...], p0=0)


def test_corret_results_3d(data, expected_results):
    s50 = zslice(data["q"], data["p"], p0=-50)
    np.testing.assert_almost_equal(s50, expected_results)


def test_corret_results_2d(data, expected_results):
    K, _, _ = data["q"].shape
    s50 = zslice(data["q"].reshape(K, -1), data["p"].reshape(K, -1), p0=-50)
    np.testing.assert_almost_equal(s50, expected_results.ravel())


def test_p0_outside_bounds(data):
    with pytest.raises(ValueError, match="is outside p bounds"):
        zslice(data["q"], data["p"], p0=50)
