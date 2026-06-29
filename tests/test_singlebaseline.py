"""
Tests for the single-baseline analysis and the target cases. Each checks
a hand-computable property, so a pass means the FTS / visibility physics
is right, not merely self-consistent. Run with: python -m pytest tests
"""

import numpy as np
import pytest

import cubesat_sim as cs
from cubesat_sim import scenes, singlebaseline as sb


@pytest.fixture()
def params():
    return cs.InstrumentParams()


@pytest.fixture()
def opd(params):
    return cs.build_grids(params).opd


# ---------------------------------------------------------------------------
def test_recover_monochromatic_peak(opd):
    """A pure cosine interferogram at sigma0 transforms to a peak at
    sigma0 (the FTS inversion lands the line at the right wavenumber)."""
    sigma0 = 1000.0
    interf = np.cos(2.0 * np.pi * sigma0 * opd)
    sig_out = np.linspace(900.0, 1100.0, 4001)
    spec = np.abs(sb.recover(interf, opd, sig_out, subtract_mean=False))
    assert abs(sig_out[np.argmax(spec)] - sigma0) < 1.0


def test_apodization_reduces_sidelobes(opd):
    """Hann apodization must suppress the sinc sidelobes (the 'ripples'),
    relative to the boxcar transform, at the cost of a wider peak."""
    sigma0 = 1000.0
    interf = np.cos(2.0 * np.pi * sigma0 * opd)
    sig_out = np.linspace(900.0, 1100.0, 4001)
    box = np.abs(sb.recover(interf, opd, sig_out, subtract_mean=False))
    han = np.abs(sb.recover(interf, opd, sig_out, apodize="hann",
                            subtract_mean=False))
    sidelobe = np.abs(sig_out - sigma0) > 25.0   # outside the main lobe
    box_rel = box[sidelobe].max() / box.max()
    han_rel = han[sidelobe].max() / han.max()
    assert han_rel < 0.5 * box_rel


def test_binary_ripple_matches_separation(params):
    """A binary imprints a cosine ripple on the single-baseline spectrum;
    the number of visibility minima across the band must match
    (sigma_max-sigma_min) * b * theta_sep half-cycles."""
    case = scenes.binary(baseline_m=0.4, separation_arcsec=24.0)
    sig = np.linspace(params.band.sigma_min, params.band.sigma_max, 3000)
    V = sb.baseline_spectrum(params, case.sky, 0.4, 0.0, sig)
    vis = np.abs(V) / case.sky.total_flux(sig)
    minima = int(np.sum((vis[1:-1] < vis[:-2]) & (vis[1:-1] < vis[2:])))
    sep = 24.0 * cs.constants.ARCSEC
    expected = (params.band.sigma_max - params.band.sigma_min) * 40.0 * sep
    assert abs(minima - expected) <= 1     # ~2 minima for these numbers


def test_resolved_source_blue_suppression(params):
    """A resolved source's visibility falls toward bluer wavenumbers
    (it is more resolved at shorter wavelength)."""
    case = scenes.resolved_disk(baseline_m=0.4, size_arcsec=2.5)
    sig = np.linspace(params.band.sigma_min, params.band.sigma_max, 500)
    V = sb.baseline_spectrum(params, case.sky, 0.4, 0.0, sig)
    vis = np.abs(V) / case.sky.total_flux(sig)
    assert vis[0] > vis[-1]
    assert vis[-1] < 0.5


def test_point_source_visibility_flat(params):
    """An on-axis point source has flat, unit visibility (never
    resolved) - the reference against which the other two are read."""
    case = scenes.compact_line()
    sig = np.linspace(params.band.sigma_min, params.band.sigma_max, 400)
    V = sb.baseline_spectrum(params, case.sky, case.baseline_m, 0.0, sig)
    vis = np.abs(V) / case.sky.total_flux(sig)
    assert np.allclose(vis, 1.0, atol=1e-6)


def test_all_cases_run(params):
    """Every registered case runs end to end and returns finite arrays."""
    for name, ctor in scenes.CASES.items():
        res = sb.analyse(ctor())
        assert np.all(np.isfinite(res["recovered"]))
        assert np.all(np.isfinite(res["baseline_truth"]))
        assert np.all(np.isfinite(res["scan"].power))
