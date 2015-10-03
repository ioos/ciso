#       subroutine surface(z, q, q0, z_iso, L, M, N)
# c Assume z is sorted
#
#       implicit none
#       integer L, M, N
#       real*8 z(N,M,L)
#       real*8 q(N,M,L)
#       real*8 q0(M,L)
#       real*8 z_iso(M,L)
# cf2py intent(out) z_iso
#       integer i, j, k
#       real*8 dz, dq, dq0
#
#       do i=1,L
#         do j=1,M
#           z_iso(j,i)=1.0d20 ! default value - isoline not in profile
#           do k=1,N-1
#             if ( (q(k,j,i).lt.q0(j,i).and.q(k+1,j,i).gt.q0(j,i)).or.
#      &           (q(k,j,i).gt.q0(j,i).and.q(k+1,j,i).lt.q0(j,i)) ) then
#               dz = z(k+1,j,i) - z(k,j,i)
#               dq = q(k+1,j,i) - q(k,j,i)
#               dq0 = q0(j,i) - q(k,j,i)
#               z_iso(j,i) = z(k,j,i) + dz*dq0/dq

import numpy as np
cimport numpy as np

cdef float iso

def iso_slice(np.ndarray q, np.ndarray p, np.ndarray p0, mask_val=1e20):
    cdef int L = q.shape[2]
    cdef int M = q.shape[1]
    cdef int N = q.shape[0]
    cdef float dp, dq, dq0
    cdef int i, j, k
    
    cdef np.ndarray q_iso = np.zeros([M, L], dtype='d')
    
    # assert p0.ndims < 3  # either 0, 1 or 2D
    # p0 = p0 * np.ones(M, L)
    
    for i in range(L):
        for j in range(M):
            q_iso[j, i] = mask_val
            for k in range(N-1):
                if ( ((p[k,j,i]<p0[j,i]) and (p[k+1,j,i]>p0[j,i])) or
                     ((p[k,j,i]>p0[j,i]) and (p[k+1,j,i]<p0[j,i])) ):
                     dp = p[k+1,j,i] - p[k,j,i]
                     dp0 = p0[j,i] - p[k,j,i]
                     dq = q[k+1,j,i] - q[k,j,i]
                     q_iso[j,i] = q[k,j,i] + dq*dp0/dp
                     
    return q_iso