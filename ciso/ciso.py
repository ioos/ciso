"""Compute iso-surface slices on 3D fields."""

import numpy as np

from ciso._ciso import _zslice


def zslice(q, p, p0):
    """Return a 2D slice of the variable `q` from a 3D field defined by `p`.

    The slice is defined along an iso-surface at `p0` via linear interpolation.
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
        msg = f"Arrays q {q.shape} and p {p.shape} must be of the same shape."
        raise ValueError(msg)

    if np.array(p0).squeeze().ndim != 0:
        msg = f"p0 must be a float number or 0-dim array.  Got {p0!r}."
        raise ValueError(msg)

    if p0 < p.min() or p.max() < p0:
        msg = f"p0 {p0} is outside p bounds ({p.min}, {p.max})."
        raise ValueError(msg)

    q = np.asarray(q, dtype=float)
    p = np.asarray(p, dtype=float)

    ndim = 3
    if q.ndim == ndim:
        K, J, I = q.shape  # noqa: E741, N806
        iso = _zslice(q.reshape(K, -1), p.reshape(K, -1), p0)
        return iso.reshape(J, I)
    ndim = 2
    if q.ndim == ndim:
        return _zslice(q, p, p0)
    msg = f"Expected 2D (UGRID) or 3D (S/RGRID) arrays.  Got {q.ndim}D."
    raise ValueError(msg)
