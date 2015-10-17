from __future__ import (absolute_import, division, print_function)

import os

import numpy as np
import pytest

from ciso import zslice

data_path = os.path.join(os.path.dirname(__file__), 'data')


@pytest.fixture
def data():
    p = np.linspace(-100, 0, 30)[:, None, None] * np.ones((50, 70))
    x, y = np.mgrid[0:20:50j, 0:20:70j]
    q = np.sin(x) + p
    return q, p, x, y


@pytest.fixture
def expected_results():
    return np.load(os.path.join(data_path, 'fortran.npz'))['s50']


def test_mismatch_shapes():
    q, p, x, y = data()
    with pytest.raises(ValueError):
        zslice(q, p[0], p0=0)


def test_p0_wrong_shape():
    q, p, x, y = data()
    with pytest.raises(ValueError):
        zslice(q, p, p0=np.zeros((2, 2)))


def test_bad_dtypes():
    # FIXME: Boolean array are converted to float!  Only str fails correctly.
    q, p, x, y = data()
    with pytest.raises(ValueError):
        zslice(np.empty_like(q, dtype=np.str_), p, p0=0)


def test_good_dtypes():
    # FIXME: Using `np.asfarray` will prevent from using complex dtypes.
    # NOTE: There is probably a more "numpy" efficient way to test this.
    q, p, x, y = data()
    dtypes = [int, float, np.integer, np.float16, np.float32,
              np.float64, np.float128, np.floating]
    for dtype in dtypes:
        zslice(np.empty_like(q, dtype=dtype), p, p0=0)


def test_3D_input():
    q, p, x, y = data()
    K, I, J = q.shape
    s50 = zslice(q, p, p0=-50)
    assert s50.shape == (I, J)


def test_2D_input():
    q, p, x, y = data()
    K, I, J = q.shape
    s50 = zslice(q.reshape(K, -1), p.reshape(K, -1), p0=-50)
    assert s50.shape == (I*J,)


def test_1D_input():
    q, p, x, y = data()
    with pytest.raises(ValueError):
        zslice(q.ravel(), p.ravel(), p0=0)


def test_gt_3D_input():
    q, p, x, y = data()
    with pytest.raises(ValueError):
        zslice(q[np.newaxis, ...], p[np.newaxis, ...], p0=0)


def test_corret_results_3D():
    q, p, x, y = data()
    s50 = zslice(q, p, p0=-50)
    f50 = expected_results()
    np.testing.assert_almost_equal(s50, f50)


def test_corret_results_2D():
    q, p, x, y = data()
    K, I, J = q.shape
    s50 = zslice(q.reshape(K, -1), p.reshape(K, -1), p0=-50)
    f50 = expected_results()
    np.testing.assert_almost_equal(s50, f50.ravel())


def test_p0_outside_bounds():
    with pytest.raises(ValueError):
        q, p, x, y = data()
        K, I, J = q.shape
        zslice(q, p, p0=50)
