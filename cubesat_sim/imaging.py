"""
Reduction step 2: visibilities -> sky (the SECOND Fourier transform),
plus the source-separation pipeline.

*** FUTURE / NON-MISSION CONFIGURATION ***
The dirty-map / source-separation products below assume MANY uv points
(~100 on 2-3 baseline rings) from baseline rotation+contraction. The
CONFIRMED mission (run_crater.py / run_cases.py) is a SINGLE FIXED
baseline and does NOT image; for it only per-baseline visibility
spectra apply. This module is retained for the future deployable-boom
(multi-baseline) scenario.

Two products:

1. dirty_cube / dirty_beam - the direct inverse van Cittert-Zernike sum
   (PyFIInS dirtyimage.py, re-derived):

      I_d(theta) = (1/N_uv) sum_uv Re{ V_hat e^{+2 pi i sigma (b.theta)} }

   With ~100 uv points on 2-3 rings the dirty beam has strong sidelobes;
   the dirty map is a DIAGNOSTIC, not the science product. (Hogbom CLEAN,
   which PyFIInS carried but had commented out of its pipeline, is
   deliberately omitted: for a two-source scene parametric fitting below
   is both simpler and statistically better.)

2. locate_sources + refine_positions + extract_spectra - the actual
   science pipeline for a two-element instrument ('source separation +
   spectroscopy'):
     * initial positions: peaks of the band-averaged dirty map (grid
       search with parabolic sub-pixel refinement);
     * position refinement: for FIXED positions the per-channel model
           V(u,v) = sum_k S_k e^{-2 pi i sigma (b . theta_k)}
       is LINEAR in the S_k, so the best-fit flux densities and the fit
       residual are cheap to evaluate; the positions are then the
       nonlinear parameters of a classic separable least-squares
       (variable projection) problem, minimised with Nelder-Mead. The
       dirty-map peaks are only accurate to a fraction of a pixel, which
       leaks per-channel phase errors into the spectra (~8% at 0.3
       arcsec error); the refinement removes this, so the noiseless
       validation loop closes to numerical precision.
     * spectra: the linear solve at the refined positions. This inverts
       the double-slit cosine of Grainger Eqs. 11-13 instead of just
       fitting its period.
   The recovered S_k are 'beam-weighted' flux densities; the known
   primary beam b^2(theta_k) is divided out at the recovered positions.
"""

from dataclasses import dataclass
from typing import List
import numpy as np
from scipy.optimize import minimize

from . import beams
from . import constants as const
from .parameters import InstrumentParams
from .reduction import UVSpectrum


@dataclass
class DirtyMap:
    x_arcsec: np.ndarray     # 1-d axis
    y_arcsec: np.ndarray
    image: np.ndarray        # [ny, nx] band-averaged dirty map
    beam: np.ndarray         # [ny, nx] dirty beam (V=1 everywhere)


def dirty_map(uvspectra: List[UVSpectrum], fov_arcsec=60.0,
              npix=121) -> DirtyMap:
    """Band-averaged dirty image and dirty beam."""
    x = np.linspace(-fov_arcsec, fov_arcsec, npix)
    y = np.linspace(-fov_arcsec, fov_arcsec, npix)
    tx = x * const.ARCSEC
    ty = y * const.ARCSEC

    image = np.zeros((npix, npix))
    beam = np.zeros((npix, npix))
    n_terms = 0
    for uvs in uvspectra:
        # delay [cm] for every pixel: 100 * (bx*tx + by*ty)
        dx = 100.0 * uvs.uv.bx * tx[None, :]     # [1, nx]
        dy = 100.0 * uvs.uv.by * ty[:, None]     # [ny, 1]
        for k, sig in enumerate(uvs.sigma):
            ph = np.exp(2.0j * np.pi * sig * (dx + dy))
            image += np.real(uvs.vhat[k] * ph)
            beam += np.real(ph)
            n_terms += 1
    image /= n_terms
    beam /= n_terms
    return DirtyMap(x_arcsec=x, y_arcsec=y, image=image, beam=beam)


def locate_sources(dmap: DirtyMap, n_sources=2, exclusion_arcsec=4.0):
    """
    Find the n brightest, mutually separated peaks of the dirty map and
    refine each with a parabolic (quadratic) sub-pixel fit.
    Returns positions [(x, y) arcsec].
    """
    img = dmap.image.copy()
    dx = dmap.x_arcsec[1] - dmap.x_arcsec[0]
    excl = max(1, int(round(exclusion_arcsec / dx)))
    positions = []
    for _ in range(n_sources):
        iy, ix = np.unravel_index(np.argmax(img), img.shape)
        # parabolic refinement in each axis (guard the array edges)
        px, py = dmap.x_arcsec[ix], dmap.y_arcsec[iy]
        if 0 < ix < img.shape[1] - 1:
            c = dmap.image[iy, ix - 1:ix + 2]
            denom = c[0] - 2 * c[1] + c[2]
            if denom != 0:
                px += 0.5 * (c[0] - c[2]) / denom * dx
        if 0 < iy < img.shape[0] - 1:
            c = dmap.image[iy - 1:iy + 2, ix]
            denom = c[0] - 2 * c[1] + c[2]
            if denom != 0:
                py += 0.5 * (c[0] - c[2]) / denom * dx
        positions.append((px, py))
        img[max(0, iy - excl):iy + excl + 1,
            max(0, ix - excl):ix + excl + 1] = -np.inf
    return positions


def _fit_residual(positions_arcsec, uvspectra):
    """
    Variable-projection residual: at fixed positions, solve the linear
    per-channel flux fit and return the total squared misfit over all
    channels and uv points.
    """
    sigma = uvspectra[0].sigma
    theta = np.asarray(positions_arcsec, dtype=float).reshape(-1, 2) \
        * const.ARCSEC
    bx = np.array([u.uv.bx for u in uvspectra])
    by = np.array([u.uv.by for u in uvspectra])
    vhat = np.array([u.vhat for u in uvspectra])

    delay_cm = 100.0 * (bx[:, None] * theta[None, :, 0]
                        + by[:, None] * theta[None, :, 1])
    total, norm = 0.0, 0.0
    # positions are common to all channels: a ~10-channel subset is
    # enough to constrain them and keeps the optimiser fast
    step = max(1, len(sigma) // 10)
    for k in range(0, len(sigma), step):
        design = np.exp(-2.0j * np.pi * sigma[k] * delay_cm)
        sol, *_ = np.linalg.lstsq(design, vhat[:, k], rcond=None)
        resid = vhat[:, k] - design @ sol
        total += float(np.sum(np.abs(resid) ** 2))
        norm += float(np.sum(np.abs(vhat[:, k]) ** 2))
    # normalised to O(1) so the optimiser's fatol is meaningful
    return total / norm


def refine_positions(uvspectra: List[UVSpectrum], positions_arcsec,
                     max_iter=300):
    """
    Refine the dirty-map source positions by minimising the separable
    least-squares residual (Nelder-Mead over the 2*n_src position
    coordinates; flux densities re-solved linearly at every step).
    """
    x0 = np.asarray(positions_arcsec, dtype=float).ravel()
    res = minimize(_fit_residual, x0, args=(uvspectra,),
                   method="Nelder-Mead",
                   options={"maxiter": max_iter, "xatol": 1e-5,
                            "fatol": 1e-14})
    return [tuple(xy) for xy in res.x.reshape(-1, 2)]


def extract_spectra(p: InstrumentParams, uvspectra: List[UVSpectrum],
                    positions_arcsec):
    """
    Per-channel complex least squares for the source flux densities at
    fixed positions; divides out the primary beam at each position.
    Returns (sigma, spectra [n_src, n_sigma]) in W m^-2 (cm^-1)^-1.

    LIMITATION (verified numerically): the model assumes POINT sources.
    A resolved source's visibilities carry the size taper, which varies
    across the uv plane and is not divided out here, so its recovered
    flux is biased LOW by roughly the uv-averaged taper (e.g. a 3 arcsec
    FWHM source over the 0.3-0.5 m multi-baseline schedule -- a future
    config, not the confirmed single ~0.4 m baseline -- comes back at
    ~0.26x its true flux). Extending the linear model with a fitted size
    parameter per source is the natural upgrade when resolved targets
    matter.

    Channels where the beam x parity factor |b^2 mu| is below a small
    threshold are returned as NaN rather than divided out: a source
    sitting at (or beyond) a beam/parity null is simply not measurable
    there, and a silent division would fabricate enormous fluxes.
    """
    sigma = uvspectra[0].sigma
    n_src = len(positions_arcsec)
    n_sig = len(sigma)
    theta = np.array(positions_arcsec) * const.ARCSEC   # [n_src, 2]

    spectra = np.zeros((n_src, n_sig))
    bx = np.array([u.uv.bx for u in uvspectra])
    by = np.array([u.uv.by for u in uvspectra])
    vhat = np.array([u.vhat for u in uvspectra])        # [n_uv, n_sig]

    delay_cm = 100.0 * (bx[:, None] * theta[None, :, 0]
                        + by[:, None] * theta[None, :, 1])  # [n_uv, n_src]
    for k in range(n_sig):
        design = np.exp(-2.0j * np.pi * sigma[k] * delay_cm)
        sol, *_ = np.linalg.lstsq(design, vhat[:, k], rcond=None)
        spectra[:, k] = np.real(sol)    # sky brightness is real

    # divide out the primary beam (and parity envelope) at each position;
    # guard channels where the correction would divide by ~zero
    d = p.telescope.d_dish_m
    for j in range(n_src):
        b = beams.amplitude_beam(theta[j, 0], theta[j, 1], sigma, d)
        mu = beams.parity_envelope(theta[j, 0], sigma, d,
                                   p.cold_optics.parity_mismatch)
        denom = b * b * mu
        ok = np.abs(denom) > 1.0e-3
        spectra[j] = np.where(ok, spectra[j] / np.where(ok, denom, 1.0),
                              np.nan)
    return sigma, spectra
