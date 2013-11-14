## cython  _fdm.pyx; cc -ffp-contract=fast -fno-associative-math -Wno-unused-function -march=corei7 _fdm.c -O2 -shared -o _fdm.so -I/usr/local/Frameworks/Python.framework/Headers/ -lpython2.7 -L/usr/local/Cellar/python/2.7.5/Frameworks/Python.framework/Versions/Current/lib/

cdef extern from "complex.h":
    double cimag(double complex)
    double creal(double complex)
    double complex I
    
cdef extern from "float.h":
    double DBL_EPSILON 

ctypedef double complex cmplx

def accurate_pow(z, n):
    assert int(n) == n, 'only integer powers supported'

    if n < 0:
        return 1.0/_accurate_pow(z, -n)
    else:
        return _accurate_pow(z, n) 

cdef cmplx _accurate_pow(cmplx z, int n):
    cdef cmplx res = 1
    while n > 1:
        if n & 1 == 1:
            res *= z
        z *= z
        n = n >> 1 # bitwise shift
    if n > 0:
        res *= z
    return res
    
