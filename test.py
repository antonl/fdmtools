
import numpy as np
import _fdm as fdm

defs = np.array([ # in units 1/s
#    a, f, g, phi
    [10, 4, -1.5j, 0],
    [5, 15, -0.7j, 0],
    [10, 7, -4.01j, 0],
    ])

# Search for modes within frequencies freq in Hz
dt = 0.01

n = np.arange(100)

a_noise, p_noise =  np.random.random_sample(n.shape), \
    2*np.pi*np.random.random_sample(n.shape)
sig = np.sum([a*np.exp(-1.j*(g + 2*np.pi*f)*n*dt + phi*1.j) \
    for a,f,g,phi in defs], axis=0) 
cn = sig + a_noise*np.exp(1j*p_noise)


# canonical frequency will be positive, range 0 to 2 pi
fmin, fmax = 20,40 

diff = fmax - fmin
rho = cn.shape[0]*dt/2
#J = int(cn.shape[0]*dt*diff/2) + 2 # look for at least two
J = 10 
print("Expected spectral density is {}".format(rho))
print("Average inverse spacing is then {}".format(1/rho))
print("Expected number of spectral modes used in decomposition is {}".format(J))
print("By the way, frequencies are ambiguous outside the interval -{range:3.1f} to {range:3.1f}".format(range=1/(2*np.pi*dt)))
# Step 2
w_j = np.linspace(2*np.pi*fmin, 2*np.pi*fmax, int(J)) 
z_j = np.exp(-1j*w_j*dt)

ctx = fdm.make_ctx(cn, z_j)

print ctx

print "done"

