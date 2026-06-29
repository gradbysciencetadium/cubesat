"""
Forward model: scene -> interferograms.

This is the CubeSat replacement for PyFIInS observe.py. The measured
power at the bright output port of the combiner, for baseline b and
delay-line OPD Delta, is the double-Fourier equation (Grainger et al.
2012 Eq. 15, written per wavenumber channel):

  P(Delta) = sum_sigma  dsigma * A * eta(sigma) *
             [ (t1 + t2)/2 * S0(sigma)  +  sqrt(t1 t2) * Re{ V(sigma)
               * e^{2 pi i sigma Delta} } ]   +   P_background

  V(sigma) = sum_k  S_k(sigma) * b1(theta_k) * b2(theta_k)
             * mu_parity(theta_k) * taper_k
             * e^{-2 pi i sigma (b . theta_k)}            (van Cittert-
                                                           Zernike sum)

with: A the area of ONE dish, eta the symmetric system efficiency,
t1/t2 the per-arm relative throughputs (extra fold mirror in arm 2),
b1/b2 the collector amplitude beams, mu_parity the field-of-view
envelope if the arms have mismatched handedness (see beams.py), and
taper_k the resolved-size attenuation of source k. The DC term S0 uses
the intensity beam b^2. Units: sigma [cm^-1], OPD [cm], so b.theta
(metres x radians) is converted to cm before entering the phase.

Two deliberate simplifications, both flagged:
  * the baseline is FROZEN during each FTS scan (0.75 s in the default
    double-sided mode, 0.375 s single-sided). The design's own
    spin-rate criterion (one dish diameter of motion per scan,
    [RM] Spacecraft E23) makes the smearing this neglects equal to the
    uv-cell size, i.e. marginal by construction. Fringe smearing within
    a scan is a known missing systematic.
  * pointing jitter is not yet modelled (PyFIInS used measured Herschel
    jitter spectra - irrelevant for a CubeSat reaction-wheel platform).
    A hook (per-scan pointing offset) exists in observe_scan for when a
    CubeSat ADCS jitter spectrum becomes available.   [PH]

The detector's finite time constant (500 us, [RM] Inst_detectors E12)
low-pass filters the fringes as the delay line scans: channel sigma
appears at temporal frequency f = sigma * v_opd (the 'audio tone' of
that colour), and is attenuated by the single-pole MTF
1/sqrt(1+(2 pi f tau)^2) - between 0.97 and 0.88 across the band at the
design scan speed. Included in the forward model and divided back out
in reduction (it is a known calibration).
"""

from dataclasses import dataclass
from typing import List, Optional
import warnings
import numpy as np

from . import beams
from . import constants as const
from .fts import FTSGrids
from .parameters import InstrumentParams
from .sky import SkyModel
from .uvcoverage import UVPoint


@dataclass
class ScanData:
    """One recorded interferogram at one uv point."""
    uv: UVPoint
    opd: np.ndarray          # OPD axis [cm]
    power: np.ndarray        # detector samples [W]


def _per_source_terms(p: InstrumentParams, sky: SkyModel, grids: FTSGrids,
                      spectral_filter=None):
    """
    Precompute per-source quantities on the sigma grid:
    spectra S_k(sigma), beam products, parity envelope. Geometry-
    independent, so computed once per observation.
    Returns (S [nsrc,nsig], beamprod_dc=b^2 [nsrc,nsig],
    beamprod_fr=b^2*mu [nsrc,nsig], theta [nsrc,2]). The DC pedestal uses
    b^2 (a translation-invariant incoherent sum: each arm collects its
    full power regardless of where the parity-flipped spot lands); the
    parity envelope mu gates ONLY the coherent fringe term.

    spectral_filter: optional callable f(sigma) -> transmission, the
    band-defining filter applied to every source spectrum. Default None
    (no filter) leaves the main pipeline unchanged; the single-baseline
    demo passes a smooth roll-off so the recovered spectrum is not
    dominated by Gibbs ringing off a razor-sharp band edge.
    """
    sigma = grids.sigma
    filt = spectral_filter(sigma) if spectral_filter is not None else 1.0
    d = p.telescope.d_dish_m
    n_src = len(sky.sources)
    spectra = np.zeros((n_src, grids.n_channels))
    beamprod_dc = np.zeros((n_src, grids.n_channels))   # b^2: incoherent DC pedestal
    beamprod_fr = np.zeros((n_src, grids.n_channels))   # b^2 * mu: coherent fringe
    theta = np.zeros((n_src, 2))
    for k, src in enumerate(sky.sources):
        tx, ty = src.offset_rad
        theta[k] = (tx, ty)
        spectra[k] = src.spectrum(sigma) * filt
        b = beams.amplitude_beam(tx, ty, sigma, d)       # same for both arms
        mu = beams.parity_envelope(tx, sigma, d,
                                   p.cold_optics.parity_mismatch)
        beamprod_dc[k] = b * b
        beamprod_fr[k] = b * b * mu
    return spectra, beamprod_dc, beamprod_fr, theta


def visibility(p: InstrumentParams, sky: SkyModel, grids: FTSGrids,
               uv: UVPoint, _cache=None):
    """
    Complex source visibility V(sigma) [W m^-2 (cm^-1)^-1] and the DC
    spectrum S0(sigma) at one uv point - the van Cittert-Zernike sum
    over the discrete sources.
    """
    spectra, beamprod_dc, beamprod_fr, theta = (
        _cache if _cache is not None else _per_source_terms(p, sky, grids))
    sigma = grids.sigma

    # geometric delay (b . theta) in cm -> phase 2 pi sigma * delay
    delay_cm = 100.0 * (uv.bx * theta[:, 0] + uv.by * theta[:, 1])
    phase = np.exp(-2.0j * np.pi * sigma[None, :] * delay_cm[:, None])

    # resolved-size Gaussian taper per source
    taper = np.ones_like(spectra)
    for k, src in enumerate(sky.sources):
        if src.size_arcsec > 0.0:
            theta_rms = src.size_arcsec * const.ARCSEC / 2.355
            u_cyc = sigma * 100.0 * uv.baseline_m       # cycles / radian
            taper[k] = np.exp(-2.0 * np.pi**2 * (u_cyc * theta_rms) ** 2)

    # fringe (coherent) carries b^2*mu; the DC pedestal carries b^2 only
    v = np.sum(spectra * beamprod_fr * taper * phase, axis=0)
    s0 = np.sum(spectra * beamprod_dc, axis=0)
    return v, s0


def detector_mtf(p: InstrumentParams, grids: FTSGrids):
    """Single-pole detector MTF per channel at the design scan speed."""
    v_opd_cm_s = grids.delta_opd * p.detector.acquisition_hz   # cm/s
    f = grids.sigma * v_opd_cm_s                               # Hz
    return 1.0 / np.sqrt(1.0 + (2.0 * np.pi * f
                                * p.detector.time_constant_s) ** 2)


# ---------------------------------------------------------------------------
# Combiner gains -- the SINGLE source of truth, used by the forward model,
# the reduction calibration, and the sensitivity/noise estimates. They
# implement the two-beam combiner of Grainger Eq.(1):
#     I_port = T*I1 + R*I2 + 2*sqrt(T R)*sqrt(I1 I2)*|gamma|*cos(...),
# with T = R folded into `system_efficiency` (the 0.49 beam-combiner
# transmission). So the DC term carries (t1+t2) [= T t1 + R t2] and the
# fringe term carries the factor 2 [= 2 sqrt(T R)]. Getting either factor
# wrong rescales every ABSOLUTE flux/sensitivity number; the roundtrip is
# blind to it because the reduction divides the same gain back out.
def dc_gain(p: InstrumentParams, grids: FTSGrids):
    """DC (un-fringing) gain: A * eta * (t1 + t2) * dsigma."""
    t1, t2 = p.cold_optics.arm_throughput
    return (p.telescope.area_single_m2 * p.system_efficiency
            * (t1 + t2) * grids.delta_sigma)


def fringe_gain(p: InstrumentParams, grids: FTSGrids):
    """Fringe gain: 2 * A * eta * sqrt(t1 t2) * dsigma (detector MTF is
    applied per channel by the caller). The leading 2 is the combiner
    fringe coefficient 2 sqrt(T R)."""
    t1, t2 = p.cold_optics.arm_throughput
    return (2.0 * p.telescope.area_single_m2 * p.system_efficiency
            * np.sqrt(t1 * t2) * grids.delta_sigma)


def observe_scan(p: InstrumentParams, sky: SkyModel, grids: FTSGrids,
                 uv: UVPoint, background_w: float, noise_sigma_w: float,
                 rng: Optional[np.random.Generator] = None,
                 _cache=None, spectral_filter=None) -> ScanData:
    """Simulate one interferogram scan at one uv point (discrete sources)."""
    cache = (_cache if _cache is not None
             else _per_source_terms(p, sky, grids, spectral_filter))
    v, s0 = visibility(p, sky, grids, uv, _cache=cache)
    power = interferogram_from_visibility(p, grids, v, s0, background_w,
                                          noise_sigma_w, rng=rng)
    return ScanData(uv=uv, opd=grids.opd.copy(), power=power)


def interferogram_from_visibility(p: InstrumentParams, grids: FTSGrids,
                                  v, s0, background_w, noise_sigma_w,
                                  rng: Optional[np.random.Generator] = None):
    """
    Assemble the recorded interferogram from a complex visibility V(sigma)
    and DC spectrum S0(sigma). This is the SINGLE source of truth for the
    detector-power formula (gains, detector MTF, ZPD offset, background,
    noise, saturation), shared by both the discrete-source path
    (observe_scan) and the extended-scene path (the brightness-map loader
    in extended.py). Any correction to the combiner normalisation lives
    here and propagates to both.
    """
    mtf = detector_mtf(p, grids)
    gain_dc = dc_gain(p, grids)
    gain_fr = fringe_gain(p, grids)

    # fringe phase relative to the TRUE zero path difference: a fixed
    # inter-arm offset (residual ZPD error, parameters.FTSScan) shifts
    # where the white-light fringe sits in the recorded scan
    opd_true = grids.opd[:, None] - p.scan.zpd_offset_cm
    fringe_phase = np.exp(2.0j * np.pi * grids.sigma[None, :]
                          * opd_true)                     # [nopd, nsig]
    power = (np.sum(gain_dc * np.asarray(s0)[None, :], axis=1)
             + np.sum(gain_fr * mtf[None, :]
                      * np.real(np.asarray(v)[None, :] * fringe_phase), axis=1)
             + background_w)

    if rng is not None and noise_sigma_w > 0.0:
        power = power + rng.normal(0.0, noise_sigma_w, size=power.shape)

    # saturation check ([RM] Inst Performance B70 - empty there, [PH]):
    # warn rather than clip, so the user decides how to handle it
    sat = p.detector.saturation_power_w
    if sat is not None and np.max(power) > sat:
        warnings.warn(
            f"detector saturated: peak sample {np.max(power):.3e} W "
            f"exceeds saturation_power_w = {sat:.3e} W", RuntimeWarning)
    return power


def observe(p: InstrumentParams, sky: SkyModel, grids: FTSGrids,
            uv_points: List[UVPoint], background_w: float,
            noise_sigma_w: float, seed: Optional[int] = 0):
    """
    Simulate the full observation: one scan per scheduled uv point.
    seed=None disables noise (the noiseless validation mode).
    """
    rng = None if seed is None else np.random.default_rng(seed)
    cache = _per_source_terms(p, sky, grids)
    return [observe_scan(p, sky, grids, uv, background_w,
                         noise_sigma_w, rng=rng, _cache=cache)
            for uv in uv_points]
