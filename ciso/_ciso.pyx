cimport cython

import numpy as np
cimport numpy as np

np.import_array()
NaN = np.nan


@cython.boundscheck(False)
@cython.wraparound(False)
def _zslice(double[:, ::1] q,
            double[:, ::1] p,
            double p0,
            double mask_val=NaN):
    """
    Cython version based on the original Fortran code [1]


    [1] http://pong.tamu.edu/~rob/python/class/examples/iso.f

    """
    cdef int IJ = q.shape[1]
    cdef int K = q.shape[0]
    cdef int ij, k

    cdef np.ndarray[double, ndim=1, mode='c'] q_iso = np.empty((IJ), dtype=np.float64)

    with nogil:
        for ij in range(IJ):
            q_iso[ij] = mask_val
            for k in range(K-1):
                if (((p[k, ij] < p0) and (p[k+1, ij] > p0)) or
                    ((p[k, ij] > p0) and (p[k+1, ij] < p0))):
                    q_iso[ij] = (q[k, ij] +
                                 (q[k+1, ij] - q[k, ij]) *  # dq
                                 (p0 - p[k, ij]) /          # dp0
                                 (p[k+1, ij] - p[k, ij]))   # dp
    return q_iso
