from __future__ import division
from numpy import (zeros, zeros_like, ones_like, 
    complex128, dot, where, sqrt, log, pi, abs)
from scipy.linalg.lapack import zggev

__all__ = ['generate_u', 'gen_amplitudes', 'solve_fdm'] 

def accurate_pow(z, n):
    '''perform fewer multiplications to calculate an integer power
    '''
    assert int(n) == n, 'only integer powers supported'
    if n < 0:
        return 1.0/accurate_pow(z, -n)
    else:
        res = 1;
        while n > 1:
            if n % 2 == 1:
                res *= z
            z *= z
            n = n >> 1 # bitwise shift
        if n > 0:
            res *= z
        return res

def generate_u(ul, signal, gen_u2=False, p=1): # generate G0, G1
    '''generate $U^0$ and $U^1$ matricies required for the method
    '''
    NPOW = 8
    assert (len(ul.shape) == 1 and len(signal.shape) == 1), "expected 1d vectors"
    N = signal.shape[0] # signal length
    L = ul.shape[0] # number of basis functions
    K = int(N /2. - p)

    assert (2*K + p <= N), "choose higher p, need more signal points"

    if gen_u2:
        assert p >= 2, "choose higher p, U2 requires p=2"
    
    ul_inv = 1./ul
    ul_invK = zeros_like(ul_inv, dtype=complex128)
    for l in xrange(L):
        ul_invK[l] = accurate_pow(ul_inv[l], K)
    
    ul_invk = ones_like(ul_inv)
    g0 = zeros((L,), dtype=complex128)
    g0_K = zeros_like(g0)
    
    D0 = zeros_like(g0)
    U0 = zeros((L, L), dtype=complex128)
    U1 = zeros_like(U0)

    if gen_u2:
        g1 = zeros((L,), dtype=complex128)
        g1_K = zeros_like(g0)
        U2 = zeros_like(U0)
    
    for k in xrange(K + 1): # iterate over signal halves and accumulate G0, G0_M
        for l in xrange(L):
            g0[l] += ul_invk[l]*signal[k]
            g0_K[l] += ul_invk[l]*signal[K + 1 + k]
            
            D0[l] += (k + 1)*signal[k]*ul_invk[l] \
                + (K - k)*signal[k + K + 1]*ul_invk[l]*ul_inv[l]*ul_invK[l]

            if gen_u2:
                g1[l] += ul_invk[l]*signal[k+1]
                g1_K[l] += ul_invk[l]*signal[K + 1 + k + 1]

            if k % NPOW == NPOW-1:
                ul_invk[l] = accurate_pow(ul_inv[l], k + 1)
            else:
                ul_invk[l] *= ul_inv[l]
    
    for l in xrange(L):
        for lp in xrange(l):
            U0[l, lp] = 1./(ul[l] - ul[lp]) * (ul[l]*g0[lp] - ul[lp]*g0[l] 
                + ul_invK[lp]*g0_K[l] - ul_invK[l]*g0_K[lp])
            U0[lp, l] = U0[l, lp]
            
            U1[l, lp] = 1./(ul[l] - ul[lp]) * (ul[l]*ul[lp]*g0[lp] - ul[lp]*ul[l]*g0[l] 
                - ul_invK[l]*ul[lp]*g0_K[lp] + ul_invK[lp]*ul[l]*g0_K[l])
            U1[lp, l] = U1[l, lp]

            if gen_u2:
                U2[l, lp] = 0.5*((ul[l] + ul[lp])*U1[l, lp] - ul[l]*g1[lp] \
                    - ul[lp]*g1[l] + ul_invK[l]*g1_K[lp] + ul_invK[lp]*g1_K[l])
                U2[lp, l] = U2[l, lp]
            
        U0[l, l] = D0[l]
        U1[l, l] = D0[l]*ul[l] - ul[l]*g0[l] + ul_invK[l]*g0_K[l]

        if gen_u2:
            U2[l, l] = ul[l]*U1[l, l] - ul[l]*g1[l] + ul_invK[l]*g1_K[l]
        
    if gen_u2:
        return (U0, U1, g0, U2)
    else:
        return (U0, U1, g0)     

def gen_amplitudes(B, g0):
    assert B.shape[0] == g0.shape[0], 'incorrect shape for B'

    L = g0.shape[0]
    N = B.shape[1]
    
    amps = zeros((N,), dtype=complex128)
    
    for n in xrange(N):
        amps[n] = dot(B[:, n], g0)**2
    return amps

def solve_fdm(ul, signal, threshold=1e-5):
    U0, U1, g0, U2 = generate_u(ul, signal, p=2, gen_u2=True)
    alpha,beta,vl,vr,work,info = zggev(U1, U0, compute_vl=False, compute_vr=True)

    assert info == 0, "zggev failed"
    
    # remove zero or infinite eigenvalues
    idx = where((abs(alpha) > 1e-14) & (abs(beta) > 1e-14))
    eigs = (log(alpha[idx]) - log(beta[idx]))/(-1j*2*pi)

    print("idx: ", idx)

    vr = vr[:, idx]
    g0 = g0[idx]
    U0 = U0[idx, :]

    print("vr_shape: ", vr.shape)
    print("g0_shape: ", g0.shape)
    print("U0_shape: ", U0.shape)
    
    for l in xrange(vr.shape[0]):
        # rescale vectors
        norm = dot(vr[:, l], dot(U0, vr[:, l]))
        vr[:, l] *= 1.0/sqrt(norm)
    #assert abs(dot(dot(vr[:, 0], U0), vr[:, 0]) - 1) < 1e-3, "incorrect normalization"
    A = gen_amplitudes(vr, g0)
    idx = where(abs(A) > threshold)
    return eigs[idx].squeeze(), A[idx].squeeze()
