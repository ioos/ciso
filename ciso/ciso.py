from __future__ import (absolute_import, division, print_function)

import numpy as np

from ._ciso import _zslice


def zslice(q, p, p0):
    """
    Returns a 2D slice of the variable `q` from a 3D field defined by `p`,
    along an iso-surface at `p0` using a linear interpolation.

    The result `q_iso` is a projection of variable at property == iso-value
    in the first non-singleton dimension.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from ciso import zslice
    >>> z = np.linspace(-100, 0, 30)[:, None, None] * np.ones((50, 70))
    >>> x, y = np.mgrid[0:20:50j, 0:20:70j]
    >>> s = np.sin(x) + z
    >>> s50 = zslice(s, z, -50)
    >>> plt.pcolormesh(s50)

    """
    if q.shape != p.shape:
        msg = "Arrays q {} and p {} must be of the same shape.".format
        raise ValueError(msg(q.shape, p.shape))

    if np.array(p0).squeeze().ndim != 0:
        msg = "p0 must be a float number or 0-dim array.  Got {!r}.".format
        raise ValueError(msg(p0))

    if p0 < p.min() or p.max() < p0:
        msg = "p0 {} is outise p bounds ({}, {}).".format
        raise ValueError(msg(p0, p.min(), p.max()))

    q = np.asfarray(q)
    p = np.asfarray(p)

    if q.ndim == 3:
        K, J, I = q.shape
        iso = _zslice(q.reshape(K, -1), p.reshape(K, -1), p0)
        return iso.reshape(J, I)
    elif q.ndim == 2:
        return _zslice(q, p, p0)
    else:
        msg = "Expected 2D (UGRID) or 3D (S/RGRID) arrays.  Got {}D.".format
        raise ValueError(msg(q.ndim))
