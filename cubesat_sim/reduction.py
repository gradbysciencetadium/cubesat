"""
Reduction step 1: interferogram -> visibility spectrum (the FIRST of the
two Fourier transforms; CubeSat replacement for PyFIInS
reduceinterferogram.py).

Method
------
PyFIInS shifted each interferogram so zero OPD sat at array index 0 and
applied an FFT. Here the OPD axis is known exactly (it is recorded with
the data), so each scan is reduced by LINEAR LEAST SQUARES against the
physical model instead:

    P(Delta_j) = c0 + sum_k [ a_k cos(2 pi sigma_k Delta_j)
                            + b_k sin(2 pi sigma_k Delta_j) ]

solved for (c0, a_k, b_k) in one lstsq call. The recovered complex
visibility is V_k = (a_k - i b_k) / gain_k, with the known instrument
gain (area x efficiency x sqrt(t1 t2) x dsigma x detector MTF) divided
out so V comes back in physical flux-density units.

Why least squares instead of an FFT:
  * the OPD axis is known exactly (recorded with the data), so a
    design-matrix lstsq fits the cos/sin/DC model directly, independent
    of the window symmetry. For the SINGLE-SIDED alternative this also
    cures the non-orthogonality of the DC (background) term to the odd
    channels over 0..OPD_max, where a plain 'subtract the mean then FFT'
    biases those channels; for the DEFAULT double-sided scan it simply
    gives exact fitting with no FFT-specific symmetry assumption;
  * it needs no zero-OPD shifting or phase correction (the OPD axis is
    explicit), removing the step PyFIInS marked 'must ask David N how
    to do this properly';
  * it generalises unchanged to irregular OPD sampling (delay-line
    velocity errors) when that systematic is added later.
The cost (a 500 x 85 lstsq per scan in the default double-sided mode,
250 x 85 single-sided) is negligible.
"""

from dataclasses import dataclass
from typing import List
import numpy as np

from .forward import ScanData, detector_mtf, fringe_gain
from .fts import FTSGrids
from .parameters import InstrumentParams
from .uvcoverage import UVPoint


@dataclass
class UVSpectrum:
    """Recovered complex visibility spectrum at one uv point."""
    uv: UVPoint
    sigma: np.ndarray        # wavenumber channels [cm^-1]
    vhat: np.ndarray         # complex visibility [W m^-2 (cm^-1)^-1]


def _design_matrix(sigma: np.ndarray, opd: np.ndarray) -> np.ndarray:
    """Design matrix [1, cos(2 pi sigma_k opd), sin(...)] for one scan."""
    arg = 2.0 * np.pi * sigma[None, :] * opd[:, None]
    return np.concatenate(
        [np.ones((len(opd), 1)), np.cos(arg), np.sin(arg)], axis=1)


def reduce_scan(p: InstrumentParams, grids: FTSGrids, scan: ScanData,
                design: np.ndarray = None) -> UVSpectrum:
    """
    Reduce one interferogram to a complex visibility spectrum.
    `design` may be precomputed (reduce_all does this); it depends only
    on the OPD grid, which is shared by all scans of one observation.
    """
    sigma = grids.sigma
    nsig = len(sigma)
    if design is None:
        design = _design_matrix(sigma, scan.opd)
    coef, *_ = np.linalg.lstsq(design, scan.power, rcond=None)
    a_k = coef[1:1 + nsig]
    b_k = coef[1 + nsig:1 + 2 * nsig]

    # divide out the known instrument gain (calibration) -- the SAME
    # fringe_gain the forward model used, so V comes back in physical
    # flux-density units (and the combiner factor-of-2 cancels exactly)
    gain = fringe_gain(p, grids) * detector_mtf(p, grids)
    vhat = (a_k - 1.0j * b_k) / gain

    # ZPD calibration: a known inter-arm offset Delta_0 makes the fitted
    # quadratures estimate V e^{-2 pi i sigma Delta_0}; undo it. If the
    # true offset differs from the calibrated value (p.scan.zpd_offset_cm
    # set in forward, a different value assumed here), the residual phase
    # ramp e^{-2 pi i sigma dDelta} propagates into the science - which
    # is exactly the systematic this parameter exists to study.
    if p.scan.zpd_offset_cm != 0.0:
        vhat = vhat * np.exp(2.0j * np.pi * sigma * p.scan.zpd_offset_cm)
    return UVSpectrum(uv=scan.uv, sigma=sigma.copy(), vhat=vhat)


def reduce_all(p: InstrumentParams, grids: FTSGrids,
               scans: List[ScanData]) -> List[UVSpectrum]:
    """Reduce every scan, building the shared design matrix once."""
    design = _design_matrix(grids.sigma, grids.opd)
    out = []
    for s in scans:
        # all scans of one observation share the OPD grid; fall back to
        # a per-scan design if a future systematic makes them differ
        d = design if np.array_equal(s.opd, grids.opd) else None
        out.append(reduce_scan(p, grids, s, design=d))
    return out
