"""
Single-baseline analysis: one interferogram, one Fourier transform.

The full pipeline (run_simulation.py) rotates the baseline to fill the
uv plane. This module does the BUILDING BLOCK the professor described -
hold ONE baseline, scan the delay line once, and look at what comes out:

  * the INTERFEROGRAM: detector power vs OPD. A central PEAK (the white-
    light fringe, where every colour adds in phase at zero OPD) with
    fringes/RIPPLES around it. The ripples are the source's spectrum and
    spatial structure imprinted onto the delay axis.

  * its FOURIER TRANSFORM: the recovered spectrum. The scan is FINITE
    (length OPD_max), which is equivalent to multiplying the true
    interferogram by a boxcar window - and a boxcar in OPD becomes a
    SINC in wavenumber. So every true spectral peak comes back CONVOLVED
    with a sinc that has ripples (sidelobes at ~13%). That convolution
    is the 'ripples' the professor referred to; APODIZATION (tapering
    the window) trades the ripples for a wider peak, shown here too.

  The ratio (scan length)/(feature scale) sets how many ripples and how
  sharp the recovered peak is - the professor's "target 10 m, mirror
  100 m" is exactly this scale ratio, on whichever axis (a longer scan
  resolves a narrower line; a longer baseline resolves a smaller angle).

Honest limitation of a SINGLE baseline: the recovered spectrum is
S(sigma) * V(sigma) - the true spectrum times the spatial visibility at
that one baseline. The two cannot be separated from one baseline alone:
a cosine ripple across the band could be a binary (spatial) OR a genuine
spectral modulation. Disentangling them is exactly what baseline
diversity (the full uv rotation) is for. This module is the per-baseline
building block; the full pipeline stacks many of these.

To make the line-shape ripples visible the source is simulated on a grid
finer than the FTS channels (Case.spectral_oversample) - this is also
why the result here exposes systematics that the on-channel-grid
validation loop deliberately cannot.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import beams
from . import constants
from . import forward
from . import radiometry
from .fts import build_grids, FTSGrids
from .parameters import InstrumentParams
from .uvcoverage import UVPoint


# ---------------------------------------------------------------------------
def _fine_grids(p: InstrumentParams, base: FTSGrids,
                oversample: int) -> FTSGrids:
    """FTS grids with a spectral grid `oversample` times finer than the
    channels (same OPD axis), so off-channel content is represented."""
    step = base.delta_sigma / oversample
    sigma = np.arange(p.band.sigma_min, p.band.sigma_max + step / 2, step)
    return FTSGrids(sigma=sigma, delta_sigma=step, opd=base.opd,
                    delta_opd=base.delta_opd, opd_max=base.opd_max,
                    t_scan=base.t_scan, t_sample=base.t_sample)


def band_filter(p: InstrumentParams, sigma, edge_frac=0.06):
    """
    Smooth band-defining filter transmission: flat in the band interior
    with raised-cosine (Tukey-style) roll-off over the outer edge_frac of
    the band, zero outside. This represents the real LPE band filter,
    whose FINITE roll-off replaces the razor-sharp grid edge that would
    otherwise ring (Gibbs) all across the recovered spectrum. Edge
    ringing in the recovered spectrum is then the genuine source/line
    line shape, not an artefact of a hard cutoff.
    """
    sigma = np.asarray(sigma, float)
    lo, hi = p.band.sigma_min, p.band.sigma_max
    w = edge_frac * (hi - lo)
    t = np.ones_like(sigma)
    t = np.where(sigma < lo + w,
                 0.5 * (1.0 - np.cos(np.pi * np.clip((sigma - lo) / w, 0, 1))),
                 t)
    t = np.where(sigma > hi - w,
                 0.5 * (1.0 - np.cos(np.pi * np.clip((hi - sigma) / w, 0, 1))),
                 t)
    return np.where((sigma < lo) | (sigma > hi), 0.0, t)


def measure(p: InstrumentParams, sky, baseline_m, angle_rad=0.0,
            oversample=16, with_noise=False, seed=0):
    """
    Record one interferogram at one baseline. Returns (base_grids, scan).
    Noiseless by default (with_noise=False); with noise it draws the
    background and per-sample sigma from the radiometric model. The
    band-defining filter (smooth edges) is applied to the source spectrum.
    """
    base = build_grids(p)
    fine = _fine_grids(p, base, oversample)
    if with_noise:
        rep = radiometry.background(p)
        bg = sum(rep.absorbed_W.values())
        noise = radiometry.noise_sigma_per_sample(p, rep)
        rng = np.random.default_rng(seed)
    else:
        bg, noise, rng = 0.0, 0.0, None
    uv = UVPoint(baseline_m=baseline_m, angle_rad=angle_rad, time_s=0.0)
    scan = forward.observe_scan(p, sky, fine, uv, bg, noise, rng=rng,
                                spectral_filter=lambda s: band_filter(p, s))
    return base, scan


def measure_extended(p: InstrumentParams, scene, baseline_m, angle_rad=0.0,
                     with_noise=False, seed=0):
    """
    Record one interferogram at one fixed baseline for an EXTENDED scene
    (a brightness map, extended.ExtendedScene). No baseline movement: the
    delay line scans, the baseline is held. Returns (base_grids, scan,
    V_true, S0_true), where V_true/S0_true are the analytic visibility and
    DC spectrum on the FTS channel grid (the noiseless truth the recovered
    spectrum should match).

    The blackbody surface has no narrow spectral features, so the FTS
    channel grid suffices (no fine spectral oversampling needed) - what
    matters here is the spatial visibility, not the line shape.
    """
    base = build_grids(p)
    V, S0 = scene.visibility(p, baseline_m, angle_rad, base.sigma)
    if with_noise:
        rep = radiometry.background(p)
        bg = sum(rep.absorbed_W.values())
        noise = radiometry.noise_sigma_per_sample(p, rep)
        rng = np.random.default_rng(seed)
    else:
        bg, noise, rng = 0.0, 0.0, None
    power = forward.interferogram_from_visibility(p, base, V, S0, bg, noise,
                                                  rng=rng)
    scan = forward.ScanData(
        uv=UVPoint(baseline_m=baseline_m, angle_rad=angle_rad, time_s=0.0),
        opd=base.opd.copy(), power=power)
    return base, scan, V, S0


# ---------------------------------------------------------------------------
def recover(interferogram, opd, sigma_out, apodize=None,
            subtract_mean=True):
    """
    Fourier-transform an interferogram onto a chosen (fine) wavenumber
    grid - a direct DFT, so the output grid is arbitrary and the sinc
    instrument line shape is sampled finely enough to see its ripples.

    apodize: None (boxcar), 'triangle', or 'hann' - tapers the OPD window
    to suppress the sinc sidelobes at the cost of a broader peak.
    Returns the complex recovered spectrum on sigma_out.
    """
    opd = np.asarray(opd, float)
    y = np.asarray(interferogram, float).copy()
    if subtract_mean:
        y = y - y.mean()                 # drop the DC + background pedestal
    L = np.max(np.abs(opd))
    if apodize == "triangle":
        w = 1.0 - np.abs(opd) / L
    elif apodize == "hann":
        w = 0.5 * (1.0 + np.cos(np.pi * opd / L))
    elif apodize is None:
        w = np.ones_like(opd)
    else:
        raise ValueError(f"unknown apodization {apodize!r}")
    y = y * w
    kernel = np.exp(2.0j * np.pi * np.outer(np.asarray(sigma_out, float), opd))
    return kernel @ y


def baseline_spectrum(p: InstrumentParams, sky, baseline_m, angle_rad,
                      sigma_out, spectral_filter=None):
    """
    Analytic S(sigma) * V(sigma) at one baseline (the noiseless,
    infinite-resolution truth the recovered spectrum should converge to,
    before the instrument line shape). Mirrors forward.visibility exactly.
    spectral_filter (callable f(sigma)) applies the band filter so the
    overlay matches the recovered spectrum; default None gives the
    geometric (unfiltered) result used by the visibility tests.
    """
    sigma = np.asarray(sigma_out, float)
    filt = spectral_filter(sigma) if spectral_filter is not None else 1.0
    bx = baseline_m * np.cos(angle_rad)
    by = baseline_m * np.sin(angle_rad)
    d = p.telescope.d_dish_m
    V = np.zeros(len(sigma), complex)
    for src in sky.sources:
        tx, ty = src.offset_rad
        S = src.spectrum(sigma) * filt
        b = beams.amplitude_beam(tx, ty, sigma, d)
        mu = beams.parity_envelope(tx, sigma, d, p.cold_optics.parity_mismatch)
        taper = np.ones_like(sigma)
        if src.size_arcsec > 0.0:
            th_rms = src.size_arcsec * constants.ARCSEC / 2.355
            u = sigma * 100.0 * baseline_m
            taper = np.exp(-2.0 * np.pi**2 * (u * th_rms) ** 2)
        delay_cm = 100.0 * (bx * tx + by * ty)
        V += S * b * b * mu * taper * np.exp(-2.0j * np.pi * sigma * delay_cm)
    return V


def analyse(case, p: InstrumentParams = None, with_noise=False,
            sigma_out=None):
    """Run one case end to end at its single baseline. Returns a dict."""
    if p is None:
        p = InstrumentParams()
    base, scan = measure(p, case.sky, case.baseline_m, case.angle_rad,
                         case.spectral_oversample, with_noise)
    if sigma_out is None:
        sigma_out = np.linspace(p.band.sigma_min - 30.0,
                                p.band.sigma_max + 30.0, 1400)
    filt = lambda s: band_filter(p, s)
    # truth overlay matches the recovered spectrum (same band filter);
    # the visibility panel uses the geometric (unfiltered) V/S so the
    # band edges don't show up as a spurious 0/0
    truth_filt = baseline_spectrum(p, case.sky, case.baseline_m,
                                   case.angle_rad, sigma_out, spectral_filter=filt)
    truth_geom = baseline_spectrum(p, case.sky, case.baseline_m,
                                   case.angle_rad, sigma_out)
    src = case.sky.total_flux(sigma_out)
    vis = np.where(src > 0, np.abs(truth_geom) / np.where(src > 0, src, 1.0),
                   np.nan)
    return {
        "params": p, "case": case, "scan": scan,
        "sigma_out": sigma_out,
        "recovered": recover(scan.power, scan.opd, sigma_out),
        "recovered_apod": recover(scan.power, scan.opd, sigma_out,
                                  apodize="hann"),
        "baseline_truth": truth_filt,
        "visibility": vis,
        "source_spectrum": src,
    }


# ---------------------------------------------------------------------------
def plot_case(result, outdir):
    """Three-panel figure: interferogram, recovered spectrum, visibility."""
    os.makedirs(outdir, exist_ok=True)
    case = result["case"]
    scan = result["scan"]
    sig = result["sigma_out"]
    p = result["params"]

    # normalise the spectral curves to peak: the POINT is the SHAPE
    # (ripples, tilt, line width), not absolute Jy (the channelised
    # lstsq reduction handles calibration, validated separately)
    def norm(a):
        a = np.abs(a)
        m = np.max(a)
        return a / m if m > 0 else a

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))

    # (1) the interferogram: peak + ripples
    ax[0].plot(scan.opd * 1e4, scan.power * 1e9, lw=0.7, color="#482661")
    ax[0].set_xlabel("OPD [um]")
    ax[0].set_ylabel("detector power [nW]")
    ax[0].set_title("interferogram (peak + ripples)")

    # (2) recovered spectrum vs truth: the convolution / line shape
    ax[1].plot(sig, norm(result["baseline_truth"]), color="#963A8A",
               lw=1.6, label="true $S\\,|V|$ at this baseline")
    ax[1].plot(sig, norm(result["recovered"]), color="#482661", lw=1.0,
               label="recovered (boxcar FT)")
    ax[1].plot(sig, norm(result["recovered_apod"]), color="#7A4FB0",
               lw=1.0, ls="--", label="recovered (apodised)")
    ax[1].axvspan(p.band.sigma_min, p.band.sigma_max, color="0.9", zorder=0)
    ax[1].set_xlabel("wavenumber [cm$^{-1}$]")
    ax[1].set_ylabel("normalised spectral power")
    ax[1].set_title("recovered spectrum = truth $\\ast$ line shape")
    ax[1].legend(fontsize=7)

    # (3) the visibility at this baseline (the spatial information)
    ax[2].plot(sig, result["visibility"], color="#1E6E50", lw=1.4)
    ax[2].axvspan(p.band.sigma_min, p.band.sigma_max, color="0.9", zorder=0)
    ax[2].set_ylim(0, 1.05)
    ax[2].set_xlabel("wavenumber [cm$^{-1}$]")
    ax[2].set_ylabel("fringe visibility $|V|/S$")
    ax[2].set_title("visibility at this baseline (spatial info)")

    fig.suptitle(f"{case.name}: {case.description}", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(outdir, f"case_{case.name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
