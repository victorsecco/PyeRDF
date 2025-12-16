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

def q_to_two_theta(q, wavelength, degrees=True):
    q = np.asarray(q, dtype=float)
    theta = np.arcsin(q * wavelength / (4.0 * np.pi))
    two_theta = 2.0 * theta
    return np.degrees(two_theta) if degrees else two_theta

def q_to_d(q):
    q = np.asarray(q, dtype=float)
    return 2.0 * np.pi / q

def d_to_q(d):
    d = np.asarray(d, dtype=float)
    return 2.0 * np.pi / d

def a_from_q_hkl(q, h, k, l):
    q_to_d(q)
    return a_from_d_hkl(d, h, k, l)

def lorentzian_height(x, x0, height, gam):
    return height * gam**2 / ( gam**2 + ( x - x0 )**2)

def lorentzian_area(x, x0, area, gam):
    return (area / np.pi) * (gam / ((x - x0)**2 + gam**2))


def electron_wavelength_angstrom(kV):
    e = 1.602176634e-19
    m0 = 9.10938356e-31
    c = 299792458
    h = 6.62607015e-34

    V = kV * 1e3
    lam = h / np.sqrt(2*m0*e*V*(1 + (e*V)/(2*m0*c**2)))
    return lam * 1e10  # in Å