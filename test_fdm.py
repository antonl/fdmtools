import pytest
import numpy as np
import fdm

TOL = 1.2e-16 # close to machine precision

@pytest.mark.parametrize('N, amplitude, frequency, dt', [
    (4096<<4, 10, 5 - 0.1j, 1e-3),
    (4096<<6, 0.05, 25 - 0.5j, 1e-3),
    (4096<<6, 5, 5 - 0.1j, 1e-3),
    ])
def test_make_lorenzian(N, amplitude, frequency, dt):
    TOL = 1e-5
    fft = np.fft.fft
    fftfreq = np.fft.fftfreq
    fftshift = np.fft.fftshift

    signal = fdm.make_lorenzian(N, amplitude, frequency, dt)
    
    assert signal.shape == (N,)
    assert signal.dtype == np.complex128

    # TODO: check fft for large N and small dt
    fsignal =  fftshift(2*np.pi*dt*fft(signal))
    df = fftshift(fftfreq(N, dt))
    analytic = amplitude*1./(np.imag(-frequency) + \
        1j*(df - np.real(-frequency)))

    assert np.argmax(analytic) == np.argmax(fsignal)


@pytest.mark.parametrize('val,n,ans', [
    (0.5, 1, 0.5),
    (0.5, 2, 0.25),
    ])
def test_accurate_pow(val, n, ans):
    # stupid check 
    assert (fdm.accurate_pow(val, n) - ans) < TOL
