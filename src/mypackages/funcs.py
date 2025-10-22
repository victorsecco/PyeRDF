import numpy as np

#getting the unit cell from a lattice plane for a cubic space group
def a_from_d_hkl(d, h, k, l):
    hkl2 = h*h + k*k + l*l
    d = np.asarray(d, dtype=float)
    return d * np.sqrt(hkl2)

def two_theta_to_q(two_theta, wavelength, degrees=True):
    tt = np.asarray(two_theta, dtype=float)
    theta = np.radians(0.5 * tt) if degrees else 0.5 * tt
    return 4.0 * np.pi * np.sin(theta) / wavelength

def q_to_d(q):
    q = np.asarray(q, dtype=float)
    return 2.0 * np.pi / q

def d_to_q(d):
    d = np.asarray(d, dtype=float)
    return 2.0 * np.pi / d

def a_from_q_hkl(q, h, k, l):
    q_to_d(q)
    return a_from_d_hkl(d, h, k, l)

def lorentzian(x, x0, a, gam):
    return a * gam**2 / ( gam**2 + ( x - x0 )**2)