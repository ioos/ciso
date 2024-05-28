import os

import numpy as np
import pytest

from ciso import zslice

data_path = os.path.join(os.path.dirname(__file__), "data")


@pytest.fixture
def data():
    p = np.linspace(-100, 0, 30)[:, None, None] * np.ones((50, 70))
    x, y = np.mgrid[0:20:50j, 0:20:70j]
    q = np.sin(x) + p
    yield {"q": q, "p": p, "x": x, "y": y}


@pytest.fixture
def expected_results():
    yield np.load(os.path.join(data_path, "fortran.npz"))["s50"]


def test_mismatch_shapes(data):
    with pytest.raises(ValueError):
        zslice(data["q"], data["p"][0], p0=0)


def test_p0_wrong_shape(data):
    with pytest.raises(ValueError):
        zslice(data["q"], data["p"], p0=np.zeros((2, 2)))


def test_bad_dtypes(data):
    # FIXME: Boolean array are converted to float!  Only str fails correctly.
    with pytest.raises(ValueError):
        zslice(np.empty_like(data["q"], dtype=np.str_), data["p"], p0=0)


@pytest.mark.parametrize(
    "dtype",
    [int, float, np.int32, np.int64, np.float16, np.float32, np.float64],
)
def test_good_dtypes(data, dtype):
    zslice(np.empty_like(data["q"], dtype=dtype), data["p"], p0=0)


def test_3D_input(data):
    K, I, J = data["q"].shape
    s50 = zslice(data["q"], data["p"], p0=-50)
    assert s50.shape == (I, J)


def test_2D_input(data):
    K, I, J = data["q"].shape
    s50 = zslice(data["q"].reshape(K, -1), data["p"].reshape(K, -1), p0=-50)
    assert s50.shape == (I * J,)


def test_1D_input(data):
    with pytest.raises(ValueError):
        zslice(data["q"].ravel(), data["p"].ravel(), p0=0)


def test_gt_3D_input(data):
    with pytest.raises(ValueError):
        zslice(data["q"][np.newaxis, ...], data["p"][np.newaxis, ...], p0=0)


def test_corret_results_3D(data, expected_results):
    s50 = zslice(data["q"], data["p"], p0=-50)
    np.testing.assert_almost_equal(s50, expected_results)


def test_corret_results_2D(data, expected_results):
    K, I, J = data["q"].shape
    s50 = zslice(data["q"].reshape(K, -1), data["p"].reshape(K, -1), p0=-50)
    np.testing.assert_almost_equal(s50, expected_results.ravel())


def test_p0_outside_bounds(data):
    with pytest.raises(ValueError):
        K, I, J = data["q"].shape
        zslice(data["q"], data["p"], p0=50)
