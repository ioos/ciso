cimport cython

import numpy as np
cimport numpy as np

NaN = np.NaN


@cython.boundscheck(False)
@cython.wraparound(False)
def zslice(double[:, :, ::1] q, double[:, :, ::1] p, double p0,
           double mask_val=NaN):
    """
    Returns a 2D slice of the variable `q` in a 3D field defined by `p`,
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
    p0 = -abs(p0)
    cdef int L = q.shape[2]
    cdef int M = q.shape[1]
    cdef int N = q.shape[0]
    cdef int i, j, k

    cdef np.ndarray[double, ndim=2, mode='c'] q_iso = np.empty((M, L), dtype=np.float64)

    with nogil:
        for i in range(L):
            for j in range(M):
                q_iso[j, i] = mask_val
                for k in range(N-1):
                    if (((p[k, j, i] < p0) and (p[k+1, j, i] > p0)) or
                       ((p[k, j, i] > p0) and (p[k+1, j, i] < p0))):
                        q_iso[j, i] = (q[k, j, i] +
                                       (q[k+1, j, i] - q[k, j, i]) *  # dq
                                       (p0 - p[k, j, i]) /  # dp0
                                       (p[k+1, j, i] - p[k, j, i]))  # dp
    return q_iso
