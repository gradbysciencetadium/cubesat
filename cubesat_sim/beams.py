"""
Primary beams and the parity-mismatch envelope.

Beam model
----------
Each 2-inch collector is modelled as a uniformly illuminated circular
aperture, so its amplitude beam is the 'jinc' pattern

    b(theta, sigma) = 2 J1(x) / x,    x = pi D sin(theta) / lambda,

with first null at 1.22 lambda / D (~50 arcsec at 8 um for D = 50.8 mm,
matching 'Inst Performance' E9 of the radiometric model). This is the
'Perfect' beam option of PyFIInS pbmodelgenerator.py; GRASP/measured
beam import can be added behind the same interface when a real optical
prescription exists.

Parity-mismatch envelope (new calculation for this design)
----------------------------------------------------------
The optical layout shows one extra fold mirror in one arm. If the total
reflection parity of the two arms differs (odd vs even number of
mirrors), arm 2 delivers a mirror-flipped image of the sky: a source at
field angle +theta_x in arm 1 lands where -theta_x lands in arm 2. The
two single-arm point-spread functions for the SAME sky point are then
displaced by 2*theta_x on the detector, and the fringe contrast for that
point is reduced by their overlap integral

    mu(theta_x) = integral p*(xi - theta_x) p(xi + theta_x) dxi
                / integral |p(xi)|^2 dxi.

By Parseval the overlap of two displaced PSFs equals the Fourier
transform of the pupil intensity |P(u)|^2 (a uniform disc of diameter D)
evaluated at the displacement 2*theta_x, giving

    mu(theta_x) = 2 J1(z) / z,    z = 2 pi D theta_x / lambda.

mu = 1 on axis (parity mismatch costs nothing at the phase centre) and
falls to zero at theta_x = 0.61 lambda / D - i.e. the usable half-field
shrinks to HALF the Airy radius in the flip direction. That is the
quantitative content of the warning 'parity mismatch shrinks the field
of view'. For matched arms mu = 1 everywhere.
"""

import numpy as np
from scipy.special import j1

from . import constants as const


def _jinc(x):
    """2 J1(x) / x, with the x -> 0 limit handled (value 1)."""
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    nz = np.abs(x) > 1.0e-12
    out[nz] = 2.0 * j1(x[nz]) / x[nz]
    return out


def amplitude_beam(theta_x_rad, theta_y_rad, sigma_cm1, d_dish_m):
    """
    Amplitude beam b(theta) of one circular aperture, normalised to 1
    on axis. theta may be scalar or array; sigma in cm^-1.
    """
    theta = np.hypot(theta_x_rad, theta_y_rad)
    lam = 1.0 / (100.0 * np.asarray(sigma_cm1, dtype=float))
    x = np.pi * d_dish_m * theta / lam
    return _jinc(x)


def parity_envelope(theta_x_rad, sigma_cm1, d_dish_m, mismatch: bool):
    """
    Fringe-contrast envelope mu(theta_x) due to arm parity.

    mismatch=False -> 1 everywhere (arms have matched handedness).
    mismatch=True  -> 2 J1(z)/z with z = 2 pi D theta_x / lambda
                      (flip assumed about the y axis, i.e. acting on x;
                      the choice of axis is a convention).
    """
    if not mismatch:
        return np.ones_like(np.asarray(theta_x_rad, dtype=float)
                            * np.asarray(sigma_cm1, dtype=float))
    lam = 1.0 / (100.0 * np.asarray(sigma_cm1, dtype=float))
    z = 2.0 * np.pi * d_dish_m * np.asarray(theta_x_rad, dtype=float) / lam
    return _jinc(z)


def fov_first_null_arcsec(sigma_cm1, d_dish_m):
    """Single-dish Airy first-null radius 1.22 lambda/D [arcsec]."""
    lam = 1.0 / (100.0 * float(sigma_cm1))
    return 1.22 * lam / d_dish_m / const.ARCSEC
