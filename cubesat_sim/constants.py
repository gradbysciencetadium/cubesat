"""
Physical constants and unit conversions.

Conventions used throughout the package
---------------------------------------
* Wavenumber sigma is in cm^-1 (spectroscopists' convention, as in the
  Grainger paper, the FIInS thesis and the radiometric spreadsheet).
* Angles on the sky are in radians internally; arcsec at the interfaces.
* OPD (optical path difference) is in cm, so that the Fourier pair
  (OPD [cm]) <-> (sigma [cm^-1]) needs no conversion factors:
  a wavenumber sigma produces a fringe cos(2*pi*sigma*OPD).
* Spectral radiance B_sigma is in W m^-2 sr^-1 (cm^-1)^-1.
* Power at the detector is in W.
"""

import numpy as np

# ---- fundamental constants (SI) -------------------------------------------
H_PLANCK = 6.62607015e-34   # J s
C_LIGHT = 2.99792458e8      # m / s
K_BOLTZ = 1.380649e-23      # J / K

# ---- unit conversions ------------------------------------------------------
ARCSEC = np.pi / (180.0 * 3600.0)          # 1 arcsec in radians
JY = 1.0e-26                               # 1 Jansky in W m^-2 Hz^-1
# 1 Jy expressed per cm^-1 instead of per Hz:  S_sigma = S_nu * dnu/dsigma,
# dnu/dsigma = c * 100  (sigma in cm^-1  ->  nu = c * 100 * sigma)
JY_PER_CM1 = JY * C_LIGHT * 100.0          # W m^-2 (cm^-1)^-1


def sigma_to_hz(sigma_cm1):
    """Wavenumber [cm^-1] -> frequency [Hz]."""
    return C_LIGHT * 100.0 * np.asarray(sigma_cm1, dtype=float)


def sigma_to_lambda_m(sigma_cm1):
    """Wavenumber [cm^-1] -> wavelength [m]."""
    return 1.0 / (100.0 * np.asarray(sigma_cm1, dtype=float))


def photon_energy(sigma_cm1):
    """Photon energy [J] at wavenumber sigma [cm^-1]."""
    return H_PLANCK * sigma_to_hz(sigma_cm1)


def planck_bsigma(sigma_cm1, t_kelvin):
    """
    Planck spectral radiance per unit wavenumber.

    B_sigma(T) = B_nu(T) * dnu/dsigma          [W m^-2 sr^-1 (cm^-1)^-1]

    with B_nu = 2 h nu^3 / c^2 / (exp(h nu / k T) - 1).

    This is the same Planck function used by the SkyGenerator in PyFIInS
    (skygenerator.py) and by the 'Wavelengths' sheet of the radiometric
    spreadsheet, just kept in per-cm^-1 units end to end.
    """
    sigma = np.asarray(sigma_cm1, dtype=float)
    nu = sigma_to_hz(sigma)
    x = H_PLANCK * nu / (K_BOLTZ * float(t_kelvin))
    # expm1 keeps the Rayleigh-Jeans tail numerically accurate
    b_nu = 2.0 * H_PLANCK * nu**3 / C_LIGHT**2 / np.expm1(x)
    return b_nu * C_LIGHT * 100.0
