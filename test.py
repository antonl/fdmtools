import pytest
import numpy as np
import _fdm as fdm

def test_cx_buf():
    from sys import getrefcount

    x = np.random.rand(2,2) +1j*np.random.rand(2,2)
    y = np.asarray(x, dtype=np.complex128, order='F')

    # make sure that cx_buf doesn't create extra references within itself
    i = getrefcount(fdm.make_buffer(y))
    assert  i == 1

    buf = fdm.make_buffer(y)

    assert np.allclose(y - np.asarray(buf).view(np.complex128), \
        np.ones(y.shape, dtype=np.complex128)) 

