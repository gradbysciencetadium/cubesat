"""
Physics sanity tests. Each test checks the code against a result that
can be computed by hand, so a pass means the implementation matches the
textbook physics, not just itself. Run with:  python -m pytest tests
"""

import numpy as np
import pytest

import cubesat_sim as cs
from cubesat_sim import beams, constants as const


@pytest.fixture()
def params():
    return cs.InstrumentParams()


@pytest.fixture()
def grids(params):
    return cs.build_grids(params)


# ---------------------------------------------------------------------------
def test_fts_grids_match_spreadsheet(params):
    """
    In single-sided mode the grids must reproduce 'Inst Spectral'
    E43/E50/E53/E54 (250 samples, 0.375 s). The default double-sided
    scan doubles both (the complex-visibility design finding).
    """
    p_ss = cs.InstrumentParams()
    p_ss.scan.single_sided = True
    g = cs.build_grids(p_ss)
    assert g.delta_sigma == pytest.approx(10.0)          # cm^-1
    assert g.opd_max == pytest.approx(0.05)              # cm
    assert g.delta_opd == pytest.approx(2.0e-4)          # cm (2 um)
    assert g.n_samples == 250
    assert g.t_scan == pytest.approx(0.375, rel=1e-3)    # s

    g2 = cs.build_grids(params)                          # default
    assert g2.n_samples == 500
    assert g2.t_scan == pytest.approx(0.75, rel=1e-3)


def test_resolution_and_fov(params):
    """
    1.22 lambda/b at 10 um = 5.03 arcsec ([RM] InstPerf E7) and
    1.22 lambda/d at 10 um = 49.5 arcsec (the [RM] E9 cached value;
    its '@ 8um' label is stale - at 8 um the true value is 39.6).
    """
    lam = 10.0e-6
    res = 1.22 * lam / params.geometry.baseline_max_m / const.ARCSEC
    assert res == pytest.approx(5.0328, rel=1e-3)
    fov10 = beams.fov_first_null_arcsec(1000.0, params.telescope.d_dish_m)
    assert fov10 == pytest.approx(49.536, rel=1e-3)
    fov8 = beams.fov_first_null_arcsec(1250.0, params.telescope.d_dish_m)
    assert fov8 == pytest.approx(39.63, rel=1e-3)


# ---------------------------------------------------------------------------
def test_point_source_visibility_is_flat(params, grids):
    """An on-axis point source has |V| = S(sigma): never resolved."""
    sky = cs.SkyModel(sources=[cs.Source(flux_jy=1.0e4)])
    uv = cs.UVPoint(baseline_m=0.5, angle_rad=0.3, time_s=0.0)
    v, s0 = cs.visibility(params, sky, grids, uv)
    s_true = sky.sources[0].spectrum(grids.sigma)
    assert np.allclose(np.abs(v), s_true, rtol=1e-10)
    assert np.allclose(np.angle(v), 0.0, atol=1e-12)


def test_binary_visibility_cosine(params, grids):
    """
    Equal binary separated by theta along the baseline:
    |V| = 2 S b^2 |cos(pi sigma_cm * 100 * b * theta)|  (Grainger Eq. 11
    structure: separation -> cosine in the visibility).
    """
    sep = 8.0 * const.ARCSEC
    sky = cs.SkyModel(sources=[
        cs.Source(offset_x_arcsec=+4.0, flux_jy=1.0e4, temperature_k=600.0),
        cs.Source(offset_x_arcsec=-4.0, flux_jy=1.0e4, temperature_k=600.0)])
    uv = cs.UVPoint(baseline_m=0.5, angle_rad=0.0, time_s=0.0)
    v, _ = cs.visibility(params, sky, grids, uv)

    s = sky.sources[0].spectrum(grids.sigma)
    b = beams.amplitude_beam(4.0 * const.ARCSEC, 0.0, grids.sigma,
                             params.telescope.d_dish_m)
    expected = 2.0 * s * b**2 * np.cos(
        np.pi * grids.sigma * 100.0 * uv.baseline_m * sep)
    # phase factors of the two mirrored sources combine to a real cosine
    assert np.allclose(np.real(v), expected, rtol=1e-8)
    assert np.allclose(np.imag(v), 0.0, atol=1e-12 * np.max(np.abs(v)))


def test_parity_envelope_limits(params):
    """mu = 1 on axis; first zero at 0.61 lambda/D (half the Airy null)."""
    d = params.telescope.d_dish_m
    sigma = 1000.0                      # 10 um
    lam = 1.0e-5
    assert beams.parity_envelope(0.0, sigma, d, True) == pytest.approx(1.0)
    theta_zero = 3.83171 * lam / (2.0 * np.pi * d)   # first jinc zero
    val = beams.parity_envelope(theta_zero, sigma, d, True)
    assert abs(val) < 1e-5
    # matched parity: exactly 1 everywhere
    assert beams.parity_envelope(theta_zero, sigma, d, False) == 1.0


# ---------------------------------------------------------------------------
def test_noiseless_roundtrip(params, grids):
    """
    THE validation-loop test: forward model then reduction must return
    the input visibility spectrum to numerical precision when there is
    no noise. Any failure here is an architecture error.
    """
    sky = cs.binary_scene()
    uv = cs.UVPoint(baseline_m=0.4, angle_rad=1.1, time_s=0.0)
    scan = cs.observe_scan(params, sky, grids, uv,
                           background_w=1.0e-9, noise_sigma_w=0.0)
    uvspec = cs.reduce_scan(params, grids, scan)
    v_true, _ = cs.visibility(params, sky, grids, uv)
    assert np.allclose(uvspec.vhat, v_true, rtol=1e-6)


def test_zpd_offset_roundtrip(params, grids):
    """
    A fixed inter-arm ZPD offset (residual after the path-balanced
    design of the optical layout, where the C+D budgets sum to 29 = 29)
    must (a) shift the white-light fringe in the recorded scan and
    (b) be removed exactly by the reduction's calibration when known.
    """
    p = cs.InstrumentParams()
    p.scan.zpd_offset_cm = 30.0e-4          # 30 um residual offset
    g = cs.build_grids(p)
    sky = cs.SkyModel(sources=[cs.Source(flux_jy=5.0e4)])  # on-axis point
    uv = cs.UVPoint(baseline_m=0.4, angle_rad=0.7, time_s=0.0)
    scan = cs.observe_scan(p, sky, g, uv, background_w=0.0,
                           noise_sigma_w=0.0)
    # (a) the white-light fringe sits at the offset, not at index of 0
    peak_opd = scan.opd[int(np.argmax(scan.power))]
    assert peak_opd == pytest.approx(p.scan.zpd_offset_cm, abs=g.delta_opd)
    # (b) calibrated reduction recovers the true visibility exactly
    uvs = cs.reduce_scan(p, g, scan)
    v_true, _ = cs.visibility(p, sky, g, uv)
    assert np.allclose(uvs.vhat, v_true, rtol=1e-6)


def test_spectrum_extraction_roundtrip(params, grids):
    """Full pipeline, noiseless: recovered spectra match inputs <1%."""
    sky = cs.binary_scene(separation_arcsec=8.0, position_angle_deg=30.0)
    uv_points, _ = cs.build_schedule(params, grids.t_scan)
    scans = cs.observe(params, sky, grids, uv_points, 1.0e-9, 0.0,
                       seed=None)
    uvspec = cs.reduce_all(params, grids, scans)
    dmap = cs.dirty_map(uvspec, fov_arcsec=25.0, npix=151)
    positions = cs.locate_sources(dmap, n_sources=2)
    positions = cs.refine_positions(uvspec, positions)
    sigma, spectra = cs.extract_spectra(params, uvspec, positions)

    rows = cs.validate.position_metrics(sky, positions)
    for r in rows:
        assert r["error_arcsec"] < 0.05
    specm = cs.validate.spectrum_metrics(sky, sigma, spectra, rows)
    for m in specm:
        assert m["fractional_rms"] < 0.01


# ---------------------------------------------------------------------------
def test_background_reproduces_spreadsheet_loadings(params):
    """
    Incident loadings must reproduce the spreadsheet rows (E25-E27) to a
    few percent (residual difference: integration grid vs their 200-pt
    trapezoid). A looser tolerance would let unit slips through.
    """
    rep = cs.radiometry.background(params)
    targets = cs.radiometry.RM_TARGETS
    assert rep.incident_W["environment"] == pytest.approx(
        targets["loading_environment_W"], rel=0.05)
    assert rep.incident_W["telescope"] == pytest.approx(
        targets["loading_telescope_W"], rel=0.05)
    assert rep.incident_W["cold_optics"] == pytest.approx(
        targets["loading_cold_optics_W"], rel=0.05)


def test_uv_schedule_geometry(params, grids):
    """
    Schedule invariants: 3 rings for the default geometry (the
    spreadsheet's 2 contraction steps), arc step d/b (uv Nyquist),
    angles within the half-circle, innermost ring clamped at b_min.
    """
    points, info = cs.build_schedule(params, grids.t_scan)
    assert info["n_rings"] == 3
    b = info["baselines_m"]
    assert b[0] == pytest.approx(params.geometry.baseline_max_m)
    assert b[-1] == pytest.approx(params.geometry.baseline_min_m)
    d = params.telescope.d_dish_m
    for ring in b:
        angles = sorted(pt.angle_rad for pt in points
                        if pt.baseline_m == pytest.approx(ring))
        dphi = np.diff(angles)
        assert np.allclose(dphi, d / ring, rtol=1e-9)   # uv Nyquist step
        assert angles[-1] < np.pi
    assert info["total_time_s"] == pytest.approx(
        info["n_uv_points"] * grids.t_scan)


def test_noise_propagates_at_predicted_level(params, grids):
    """
    Statistical closure: inject pure noise (empty sky), reduce, and
    check the per-channel visibility scatter matches the analytic
    prediction sigma_sample * sqrt(2/N) / gain. Catches any silent
    mis-calibration between the noise injection and the reduction.
    """
    bg = cs.radiometry.background(params)
    sig_samp = cs.radiometry.noise_sigma_per_sample(params, bg)
    empty = cs.SkyModel(sources=[])
    uv = cs.UVPoint(baseline_m=0.5, angle_rad=0.0, time_s=0.0)

    rng_seeds = range(8)
    reals = []
    for s in rng_seeds:
        scan = cs.observe_scan(params, empty, grids, uv,
                               background_w=1.0e-9, noise_sigma_w=sig_samp,
                               rng=np.random.default_rng(s))
        uvs = cs.reduce_scan(params, grids, scan)
        reals.append(np.real(uvs.vhat))
    measured = float(np.std(np.concatenate(reals)))

    gain = cs.forward.fringe_gain(params, grids)   # shared gain; MTF ~ 1, omitted
    predicted = sig_samp * np.sqrt(2.0 / grids.n_samples) / gain
    assert measured == pytest.approx(predicted, rel=0.25)


def test_rigid_bench_configuration(params, grids):
    """
    The 125 mm fixed-baseline 6U configuration (open question 1) must
    run end to end: one ring, exact noiseless recovery of a wide binary.
    """
    p = cs.InstrumentParams()
    p.geometry.baseline_min_m = 0.125
    p.geometry.baseline_max_m = 0.125
    g = cs.build_grids(p)
    pts, info = cs.build_schedule(p, g.t_scan)
    assert info["n_rings"] == 1
    sky = cs.binary_scene(separation_arcsec=20.0, position_angle_deg=30.0)
    scans = cs.observe(p, sky, g, pts, 1e-9, 0.0, seed=None)
    uvspec = cs.reduce_all(p, g, scans)
    dmap = cs.dirty_map(uvspec, fov_arcsec=40.0, npix=161)
    pos = cs.locate_sources(dmap, n_sources=2)
    pos = cs.refine_positions(uvspec, pos)
    for r in cs.validate.position_metrics(sky, pos):
        assert r["error_arcsec"] < 0.05


def test_beam_null_guard_no_flux_blowup(params, grids):
    """
    A source near the parity-envelope null must NOT come back with a
    fabricated huge flux: channels where |b^2 mu| ~ 0 are NaN, the rest
    stay within a sane factor of the truth.
    """
    p = cs.InstrumentParams()
    p.cold_optics.parity_mismatch = True
    g = cs.build_grids(p)
    pts, _ = cs.build_schedule(p, g.t_scan)
    sky = cs.SkyModel(sources=[cs.Source(offset_x_arcsec=24.8,
                                         flux_jy=5.0e4)])
    scans = cs.observe(p, sky, g, pts[:5], 1e-9, 0.0, seed=None)
    uvspec = cs.reduce_all(p, g, scans)
    sigma, spec = cs.extract_spectra(p, uvspec, [(24.8, 0.0)])
    s_true = sky.sources[0].spectrum(sigma)
    finite = np.isfinite(spec[0])
    assert np.all(np.abs(spec[0][finite]) < 5.0 * np.max(s_true))


def test_saturation_warning(params, grids):
    """The [PH] saturation hook must fire when a sample exceeds it."""
    p = cs.InstrumentParams()
    p.detector.saturation_power_w = 1.0e-12   # well below the ~3 nW bg
    g = cs.build_grids(p)
    uv = cs.UVPoint(baseline_m=0.5, angle_rad=0.0, time_s=0.0)
    with pytest.warns(RuntimeWarning, match="saturated"):
        cs.observe_scan(p, cs.SkyModel(sources=[]), g, uv,
                        background_w=3.0e-9, noise_sigma_w=0.0)


def test_throughput_chain(params):
    """Overall optical efficiency = 0.4042 [RM] Inst_ColdOptics E13."""
    assert params.optical_efficiency == pytest.approx(0.40415, rel=1e-3)
    t1, t2 = params.cold_optics.arm_throughput
    assert t1 == 1.0
    assert t2 == pytest.approx(0.98)
    # intensity-mismatch visibility factor ~ 0.99995: negligible, which
    # is itself the quantitative point about the extra mirror
    v_loss = 2 * np.sqrt(t1 * t2) / (t1 + t2)
    assert v_loss > 0.9999
