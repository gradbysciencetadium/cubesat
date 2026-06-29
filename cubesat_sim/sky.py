"""
Sky / scene model.

The scene is a list of compact sources, each with a position, an angular
size and a spectrum. Two source spectra are supported:

  * blackbody continuum, specified by a temperature and a flux density
    at a reference wavelength (in Jy) - the natural way to describe a
    star or warm dust blob;
  * the same continuum with one Gaussian emission line added, so the
    'spectroscopy' half of the double-Fourier product can be validated.

Internally a source is stored by its band-resolved flux density
S(sigma) in W m^-2 (cm^-1)^-1 and its position. The forward model never
needs a pixelated image: for a scene made of compact sources, the
van Cittert-Zernike integral

    V(u, v, sigma) = sum_k S_k(sigma) * B(theta_k, sigma)
                     * exp(-2 pi i sigma (b . theta_k))

is a finite sum over sources, evaluated exactly. This replaces the
npix x npix x nsigma sky cube + FFT machinery of PyFIInS
(skygenerator.py / observe.py): with a single-baseline two-element
instrument doing source separation rather than wide-field imaging, the
discrete-source description is both faster and free of pixelisation
error. A pixel-grid scene loader can be added later if extended
emission becomes a science target; the interface (visibility(u,v,sigma))
would not change.

Each source has finite angular size theta_s, applied as a Gaussian
visibility taper exp(-2 (pi sigma b theta_rms)^2) - the standard
resolved-source attenuation; theta_s = 0 gives an ideal point source.
"""

from dataclasses import dataclass, field
from typing import List
import numpy as np

from . import constants as const


@dataclass
class Source:
    """One compact source in the scene."""

    # position offset from the phase centre [arcsec]
    offset_x_arcsec: float = 0.0
    offset_y_arcsec: float = 0.0

    # spectrum: blackbody normalised to flux_jy at lambda_ref_um
    temperature_k: float = 600.0
    flux_jy: float = 1.0e4          # flux density at the reference wavelength
    lambda_ref_um: float = 10.0

    # angular FWHM [arcsec]; 0 = point source
    size_arcsec: float = 0.0

    # optional emission line: amplitude as a fraction of the local
    # continuum, centre [cm^-1], FWHM [cm^-1]
    line_amplitude: float = 0.0
    line_centre_cm1: float = 1000.0
    line_fwhm_cm1: float = 20.0

    def spectrum(self, sigma_cm1: np.ndarray) -> np.ndarray:
        """Flux density S(sigma) [W m^-2 (cm^-1)^-1] on the given grid."""
        sigma = np.asarray(sigma_cm1, dtype=float)
        sigma_ref = 1.0e4 / self.lambda_ref_um

        # blackbody SHAPE, normalised to 1 at the reference wavenumber
        shape = (const.planck_bsigma(sigma, self.temperature_k)
                 / const.planck_bsigma(sigma_ref, self.temperature_k))
        s = self.flux_jy * const.JY_PER_CM1 * shape

        if self.line_amplitude != 0.0:
            sig = self.line_fwhm_cm1 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            line = self.line_amplitude * np.exp(
                -0.5 * ((sigma - self.line_centre_cm1) / sig) ** 2)
            s = s * (1.0 + line)
        return s

    @property
    def offset_rad(self) -> tuple:
        return (self.offset_x_arcsec * const.ARCSEC,
                self.offset_y_arcsec * const.ARCSEC)


@dataclass
class SkyModel:
    """The scene: a list of sources."""

    sources: List[Source] = field(default_factory=list)

    def total_flux(self, sigma_cm1: np.ndarray) -> np.ndarray:
        """Sum of all source spectra [W m^-2 (cm^-1)^-1]."""
        out = np.zeros_like(np.asarray(sigma_cm1, dtype=float))
        for src in self.sources:
            out += src.spectrum(sigma_cm1)
        return out


def binary_scene(separation_arcsec=8.0, position_angle_deg=0.0,
                 t_a=600.0, t_b=450.0, flux_a_jy=5.0e4, flux_b_jy=2.0e4,
                 line_on_b=True) -> SkyModel:
    """
    Convenience constructor for the canonical test scene: an unequal
    'binary' centred on the phase centre - the double-slit case of the
    Grainger paper (Eqs. 11-13), which is the regime this instrument
    actually works in (source separation + spectroscopy, not imaging).

    Default fluxes are deliberately bright (10^4-10^5 Jy ~ the brightest
    mid-IR stars, e.g. Betelgeuse ~5e3 Jy, IRC+10216 ~5e4 Jy at 10 um):
    with the placeholder TEC-detector NEP of 5e-14 W/rtHz a small warm
    instrument only detects very bright sources. See radiometry.py.
    """
    pa = np.radians(position_angle_deg)
    dx = 0.5 * separation_arcsec * np.sin(pa)
    dy = 0.5 * separation_arcsec * np.cos(pa)
    src_a = Source(offset_x_arcsec=+dx, offset_y_arcsec=+dy,
                   temperature_k=t_a, flux_jy=flux_a_jy)
    src_b = Source(offset_x_arcsec=-dx, offset_y_arcsec=-dy,
                   temperature_k=t_b, flux_jy=flux_b_jy,
                   line_amplitude=0.5 if line_on_b else 0.0,
                   line_centre_cm1=950.0, line_fwhm_cm1=25.0)
    return SkyModel(sources=[src_a, src_b])
