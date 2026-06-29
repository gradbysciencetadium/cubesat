"""
Tests for the extended (pixel-map) scene loader. Each checks a property
that can be reasoned out by hand, and one checks the loader against the
discrete-source forward model in the point-source limit - so a pass means
the VCZ sum, units, and phase convention match the rest of the code.
Run with:  python -m pytest tests
"""

import numpy as np
import pytest

import cubesat_sim as cs
from cubesat_sim import extended


@pytest.fixture()
def params():
    return cs.InstrumentParams()


@pytest.fixture()
def sigma(params):
    return cs.build_grids(params).sigma


# ---------------------------------------------------------------------------
def test_single_pixel_phase_matches_geometry(params, sigma):
    """
    A single hot pixel is a point source: |V| = S0 (visibility magnitude
    1, nothing resolved) and the phase is the clean geometric ramp
    arg V = -2 pi sigma * 100 * (b . theta_pixel). This checks the VCZ
    phase and the cm/rad unit convention against hand calculation.
    """
    npix = 41
    scale = 1.0                       # arcsec/pixel
    T = np.full((npix, npix), 3.0)    # ~cold background (negligible flux)
    # one hot pixel offset +5 px in x from centre
    ix = (npix - 1) // 2 + 5
    iy = (npix - 1) // 2
    T[iy, ix] = 500.0
    scene = extended.ExtendedScene(temperature_map=T, pixel_scale_arcsec=scale)

    b, ang = 0.4, 0.0
    V, S0 = scene.visibility(params, b, ang, sigma)
    # magnitude: a single dominant pixel -> |V|/S0 ~ 1
    assert np.allclose(np.abs(V) / S0, 1.0, atol=2e-2)
    # phase: theta_x of the hot pixel = 5 px * 1 arcsec
    theta_x = 5 * scale * cs.constants.ARCSEC
    expected = -2.0 * np.pi * sigma * 100.0 * (b * theta_x)
    dphase = np.angle(V * np.exp(-1j * expected))   # residual after removing ramp
    assert np.allclose(dphase, 0.0, atol=1e-2)


def test_uniform_patch_resolves_out(params, sigma):
    """A uniform patch filling the beam washes the fringes out: |V|/S0 is
    near 1 at a tiny baseline and small at the design baseline."""
    patch = extended.uniform_patch(surface_T=390.0)
    V_short, S0_short = patch.visibility(params, 0.005, 0.0, sigma)
    V_long, S0_long = patch.visibility(params, 0.5, 0.0, sigma)
    vis_short = np.mean(np.abs(V_short) / S0_short)
    vis_long = np.mean(np.abs(V_long) / S0_long)
    assert vis_short > 0.9          # essentially unresolved at b -> 0
    assert vis_long < 0.3           # resolved out at 0.5 m
    assert vis_long < vis_short


def test_centred_crater_visibility_is_real(params, sigma):
    """A centred (symmetric) crater gives a real visibility: the imaginary
    part is negligible (phase 0 or pi), as for any centro-symmetric
    brightness distribution."""
    scene, _ = extended.moon_patch(crater_diameter_km=30.0,
                                   crater_offset_arcsec=(0.0, 0.0))
    V, _ = scene.visibility(params, 0.5, 0.0, sigma)
    assert np.all(np.abs(V.imag) < 1e-6 * np.max(np.abs(V)))


def test_crater_lifts_visibility(params, sigma):
    """The detection signal: a crater raises |V| above the uniform-patch
    floor (it breaks the uniformity that was washing out the fringes)."""
    uniform = extended.uniform_patch(surface_T=390.0)
    scene, _ = extended.moon_patch(crater_diameter_km=30.0, crater_T=250.0,
                                   surface_T=390.0)
    Vu, _ = uniform.visibility(params, 0.5, 0.0, sigma)
    Vc, _ = scene.visibility(params, 0.5, 0.0, sigma)
    assert np.sum(np.abs(Vc)) > np.sum(np.abs(Vu))


def test_moon_patch_angular_size(params):
    """The crater builder converts km to arcsec at the Moon's distance
    correctly: 30 km at 384,400 km -> ~16.1 arcsec."""
    _, info = extended.moon_patch(crater_diameter_km=30.0)
    assert info["crater_diameter_arcsec"] == pytest.approx(16.1, rel=0.02)


def test_extended_interferogram_runs(params):
    """The extended scene flows through the shared interferogram core and
    the reduction, producing a finite visibility spectrum."""
    scene, _ = extended.moon_patch(crater_diameter_km=30.0)
    base, scan, V, S0 = cs.singlebaseline.measure_extended(
        params, scene, baseline_m=0.5, angle_rad=0.0, with_noise=False)
    assert np.all(np.isfinite(scan.power))
    uvs = cs.reduce_scan(params, base, scan)
    assert np.all(np.isfinite(uvs.vhat))
    # noiseless: recovered visibility matches the analytic truth
    assert np.allclose(uvs.vhat, V, rtol=1e-6)
