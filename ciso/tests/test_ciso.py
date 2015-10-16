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
    with pytest.raises(ValueError):
        zslice(np.empty((2, 2)), np.empty((1, 2)), 0)


def test_wrong_p0():
    with pytest.raises(ValueError):
        zslice(np.empty((2, 2)), np.empty((2, 2)), np.empty((2, 2)))


def test_bad_dtypes():
    # FIXME: Boolean array are converted to float!  Only str fails correctly.
    with pytest.raises(ValueError):
        zslice(np.empty((2, 2), dtype=np.str_), np.empty((2, 2)), 0)


def test_good_dtypes():
    # FIXME: Using `np.asfarray` will prevent from using complex dtypes.
    # NOTE: There is probably a more "numpy" efficient way to test this.
    dtypes = [int, float, np.integer, np.float16, np.float32,
              np.float64, np.float128, np.floating]
    for dtype in dtypes:
        zslice(np.empty((2, 2), dtype=dtype), np.empty((2, 2)), 0)


def test_3D_input():
    q, p, x, y = data()
    K, I, J = q.shape
    s50 = zslice(q, p, -50)
    assert s50.shape == (I, J)


def test_2D_input():
    q, p, x, y = data()
    K, I, J = q.shape
    s50 = zslice(q.reshape(K, -1), p.reshape(K, -1), -50)
    assert s50.shape == (I*J,)


def test_1D_input():
    with pytest.raises(ValueError):
        zslice(np.empty((2)), np.empty((2)), 0)


def test_gt_3D_input():
    with pytest.raises(ValueError):
        zslice(np.empty((2, 2, 2, 2)), np.empty((2, 2, 2, 2)), 0)


def test_corret_results_3D():
    q, p, x, y = data()
    s50 = zslice(q, p, -50)
    f50 = expected_results()
    np.testing.assert_almost_equal(s50, f50)


def test_corret_results_2D():
    q, p, x, y = data()
    K, I, J = q.shape
    s50 = zslice(q.reshape(K, -1), p.reshape(K, -1), -50)
    f50 = expected_results()
    np.testing.assert_almost_equal(s50, f50.ravel())
