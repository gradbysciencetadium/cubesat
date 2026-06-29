"""
Radiometric model: thermal background, photon noise, sensitivity.

This module is the seam where the MSc radiometric model plugs into the
simulator: it produces the background power on the detector and the
total NEP, which the forward model turns into noise on every
interferogram sample.

Background components (mirroring the spreadsheet rows, [RM] = sheet/cell
of Cubesat_interferometer_v0.2.xlsx):

  * environment  - the 250 K Earth/Sun term ('SUN', [RM] SKY&Sources E13,
                   loading 5.4e-9 W = the dominant background);
  * telescope    - warm (293 K) telescope mirrors, equivalent emissivity
                   0.0199 ([RM] Telescopes E21, loading 2.5e-10 W);
  * cold optics  - the 293 K combiner box, equivalent emissivity 0.081
                   ([RM] Inst_ColdOptics E103, loading 1.03e-9 W).

Each emitter contributes

    P = integral  eps_eq(sigma) * B_sigma(T) * AOmega(sigma)
                  * eta_downstream * eta_det   dsigma

with eta_downstream the throughput between the emitting surface and the
detector. 'Incident' loadings (before detector absorption and downstream
losses) are also reported because that is the convention that reproduces
the spreadsheet's 'Background Loading' rows.

Photon NEP: Poisson term only,  NEP^2 = sum 2 E_ph(sigma) P_abs(sigma).
The Bose (wave bunching) excess is < 1% at these occupations
(h nu / k T ~ 6 at 10 um / 293 K) and is neglected, as in the
spreadsheet ('Photon Noise (Poisson only)', [RM] Inst Performance B34).

The noise model is WHITE: every OPD sample gets an independent Gaussian
draw at the NEP level. Real detectors add 1/f drift below some knee
frequency; the design relies on the OPD scan itself for modulation (a
fringe at channel sigma appears at f = sigma * v_opd ~ 50-170 Hz, above
typical knees, and there is no chopper between telescope and sky), so
white noise is the right first model -- but a 1/f component is a known
missing systematic, and would enter here as a coloured-noise generator
in place of the white sigma_sample.

CROSS-CHECK FLAG (found while writing this module - take to GS):
the spreadsheet's own numbers are internally inconsistent. Its Band-1
background loadings (e.g. SUN: 5.37e-9 W absorbed-or-incident power)
and its photon NEPs (SUN: 6.5e-17 W/rtHz) do not satisfy
NEP = sqrt(2 P E_ph): that relation applied to 5.37e-9 W gives
~4.6e-15 W/rtHz, a factor ~70 larger. Equivalently, the NEP quoted
implies a background power of only ~1e-13 W. The 'Wavelengths' sheet
that feeds those NEP cells mixes per-cm^2 flux units with m^2 etendue
in at least one place (its NEP columns are labelled [1/cm^2/sr]^0.5),
which is exactly a (100x)^2 power error. This module therefore computes
the photon NEP from first principles and prints both its own value and
the spreadsheet's for comparison; expect ours to be ~1e-14, not ~1e-16.
The conclusion 'detector-noise limited with a TEC detector' survives
either way, but the photon floor itself matters for any future
cryogenic-detector trade.
"""

from dataclasses import dataclass
import numpy as np

from . import constants as const
from .parameters import InstrumentParams


# spreadsheet target values for the cross-check printout
RM_TARGETS = {
    "loading_environment_W": 5.370852106152927e-09,   # [RM] InstPerf E25
    "loading_telescope_W": 2.514201901091539e-10,     # [RM] InstPerf E26
    "loading_cold_optics_W": 1.0251340903729366e-09,  # [RM] InstPerf E27
    "nep_total_W_rtHz": 9.211939600667909e-17,        # [RM] InstPerf E41
}


@dataclass
class BackgroundReport:
    sigma: np.ndarray                 # wavenumber grid [cm^-1]
    incident_W: dict                  # per-component incident power [W]
    absorbed_W: dict                  # per-component absorbed power [W]
    nep_photon: float                 # photon NEP [W/rtHz]
    nep_detector: float               # detector dark NEP [W/rtHz]
    nep_total: float                  # quadrature sum [W/rtHz]


def background(p: InstrumentParams, sigma=None) -> BackgroundReport:
    """Compute background loadings and NEPs on the wavenumber grid."""
    if sigma is None:
        sigma = np.linspace(p.band.sigma_min, p.band.sigma_max, 200)
    sigma = np.asarray(sigma, dtype=float)
    dsig = np.gradient(sigma)
    a_omega = p.etendue_m2sr(sigma)
    eta_det = p.detector.absorption_efficiency

    # (emissivity spectrum, temperature, downstream throughput to detector)
    chain = {
        "telescope": (np.full_like(sigma, p.telescope.equivalent_emissivity),
                      p.telescope.temperature_k,
                      p.cold_optics.efficiency_common * eta_det),
        "cold_optics": (np.full_like(sigma, p.cold_optics.equivalent_emissivity),
                        p.cold_optics.box_temperature_k,
                        eta_det),
    }
    if p.environment.include_environment:
        chain["environment"] = (
            np.full_like(sigma, p.environment.environment_emissivity),
            p.environment.t_environment_k,
            p.optical_efficiency * eta_det)

    incident, absorbed = {}, {}
    nep2 = 0.0
    for name, (eps, temp, downstream) in chain.items():
        spec_inc = eps * const.planck_bsigma(sigma, temp) * a_omega
        spec_abs = spec_inc * downstream
        incident[name] = float(np.sum(spec_inc * dsig))
        absorbed[name] = float(np.sum(spec_abs * dsig))
        nep2 += float(np.sum(2.0 * const.photon_energy(sigma)
                             * spec_abs * dsig))

    nep_ph = np.sqrt(nep2)
    nep_det = p.detector.nep_dark
    return BackgroundReport(sigma=sigma, incident_W=incident,
                            absorbed_W=absorbed, nep_photon=nep_ph,
                            nep_detector=nep_det,
                            nep_total=float(np.hypot(nep_ph, nep_det)))


def noise_sigma_per_sample(p: InstrumentParams, report: BackgroundReport):
    """
    RMS noise on one OPD sample [W].

    NEP is per sqrt(Hz) of post-detection bandwidth; a sample integrated
    for t_samp has bandwidth 1/(2 t_samp), so
        sigma_sample = NEP_total * sqrt(1 / (2 t_samp)).
    With t_samp = 1.5 ms (666.67 Hz) the factor is sqrt(333) ~ 18.3.
    """
    t_samp = 1.0 / p.detector.acquisition_hz
    return report.nep_total * np.sqrt(1.0 / (2.0 * t_samp))


def sensitivity_summary(p: InstrumentParams, report: BackgroundReport,
                        grids, n_uv_points: int) -> dict:
    """
    Point-source sensitivity, following the (currently empty) sensitivity
    rows of [RM] Inst Performance B61-B76.

    A source of flux density S_nu [W m^-2 Hz^-1] produces a coherent
    fringe amplitude per spectral channel of

        P_ch = A_dish * eta_sys * sqrt(t1 t2) * S_sigma * delta_sigma

    (A_dish = ONE dish area: the correlated signal scales with the
    geometric mean of the two collector areas, equal dishes -> A_dish).
    The DFT of one scan estimates that amplitude with noise
    sigma_ch = sigma_sample * sqrt(2 / N_samples); co-adding N
    independent scans drops it by sqrt(N) (white per-sample noise).
    For the CONFIRMED single fixed-baseline mission, N is the number of
    REPEATED scans at the same baseline (it raises the SNR of that one
    uv point; it adds no spatial/uv information). The same sqrt(N) law
    describes a FUTURE multi-baseline imaging pass, where N would be the
    number of DISTINCT uv points. The 5-sigma minimum detectable flux
    density per channel is reported in Jy, both for one scan and after
    co-adding N scans (the n_uv_points argument).
    """
    from .forward import fringe_gain, detector_mtf  # shared combiner gain (incl. factor 2) + per-channel MTF

    sig_samp = noise_sigma_per_sample(p, report)
    n = grids.n_samples
    sigma_ch_1scan = sig_samp * np.sqrt(2.0 / n)
    sigma_ch_alluv = sigma_ch_1scan / np.sqrt(max(n_uv_points, 1))

    # 1 Jy -> fitted fringe-coefficient amplitude per channel [W], via the
    # SAME gain the reduction calibrates with: fringe_gain * detector_mtf
    # (reduction.py). The MTF must appear here because the forward model
    # shapes the fringe by it; the detector NEP noise, however, is white and
    # added AFTER that shaping, so it is NOT attenuated by the MTF -- the
    # signal carries the MTF, the noise does not. MTF varies across the band,
    # so p_ch_1jy and the MDFD are per-channel; the MDFD keys below report the
    # worst (band-edge, lowest-MTF) channel as the conservative sensitivity.
    p_ch_1jy = fringe_gain(p, grids) * detector_mtf(p, grids) * const.JY_PER_CM1

    return {
        "noise_per_sample_W": sig_samp,
        "noise_per_channel_1scan_W": sigma_ch_1scan,
        # 'alluv' = co-add of all N scans; for the confirmed single fixed
        # baseline these are REPEATED scans (temporal), NOT distinct uv
        # points. Key name kept for caller compatibility (run_simulation.py).
        "noise_per_channel_alluv_W": sigma_ch_alluv,
        "fringe_amplitude_per_Jy_W": float(np.mean(p_ch_1jy)),
        "mdfd_5sigma_1scan_Jy": float(np.max(5.0 * sigma_ch_1scan / p_ch_1jy)),
        "mdfd_5sigma_alluv_Jy": float(np.max(5.0 * sigma_ch_alluv / p_ch_1jy)),
    }


def crosscheck_report(p: InstrumentParams) -> str:
    """Human-readable comparison against the spreadsheet targets."""
    rep = background(p)
    lines = ["Radiometric cross-check vs Cubesat_interferometer_v0.2.xlsx",
             "-" * 62]
    pairs = [("environment", "loading_environment_W"),
             ("telescope", "loading_telescope_W"),
             ("cold_optics", "loading_cold_optics_W")]
    for name, key in pairs:
        if name in rep.incident_W:
            ours, target = rep.incident_W[name], RM_TARGETS[key]
            lines.append(f"{name:12s} incident: {ours:9.3e} W   "
                         f"spreadsheet: {target:9.3e} W   "
                         f"ratio {ours / target:5.2f}")
    lines.append(f"photon NEP (ours, absorbed): {rep.nep_photon:9.3e} W/rtHz")
    lines.append(f"photon NEP (spreadsheet):    "
                 f"{RM_TARGETS['nep_total_W_rtHz']:9.3e} W/rtHz")
    ratio = rep.nep_photon / RM_TARGETS["nep_total_W_rtHz"]
    lines.append(f"ratio {ratio:.1f} -- see module docstring: the "
                 "spreadsheet's loading and NEP rows are mutually "
                 "inconsistent (suspected cm^2/m^2 unit slip); "
                 "our NEP follows from our loadings via NEP=sqrt(2 P E).")
    lines.append(f"detector NEP (PLACEHOLDER):  {rep.nep_detector:9.3e} W/rtHz")
    lines.append(f"total NEP:                   {rep.nep_total:9.3e} W/rtHz")
    return "\n".join(lines)
