"""
Run the single-baseline target cases.

Usage (from the cubesat_codebase directory):
    python run_cases.py                 # run all three cases
    python run_cases.py binary          # run one case by name
    python run_cases.py compact_line resolved_disk   # run a subset

Each case is independent. For each it records one interferogram at one
baseline, Fourier-transforms it to recover the spectrum, and writes a
three-panel figure (interferogram / recovered spectrum / visibility) to
outputs/cases/. A short physics read-out is printed per case.
"""

import os
import sys
import numpy as np

import cubesat_sim as cs


def physics_readout(result):
    """One-line, calibration-independent diagnostic per case."""
    case = result["case"]
    sig = result["sigma_out"]
    p = result["params"]
    inband = (sig >= p.band.sigma_min) & (sig <= p.band.sigma_max)
    vis = result["visibility"][inband]

    if case.name == "compact_line":
        # locate the recovered line as a sharp feature ABOVE the smooth
        # continuum: subtract a rolling-median continuum (window ~40 cm^-1
        # removes the narrow line, keeps the blackbody trend), then take
        # the peak of the residual. A plain global max fails because the
        # Wien continuum peak can exceed the ILS-suppressed narrow line.
        from scipy.ndimage import median_filter
        rec = np.abs(result["recovered"])
        dsig = sig[1] - sig[0]
        win = max(5, int(round(40.0 / dsig)))
        resid = rec - median_filter(rec, size=win, mode="nearest")
        margin = 30.0
        interior = ((sig > p.band.sigma_min + margin)
                    & (sig < p.band.sigma_max - margin))
        peak_sigma = sig[interior][np.argmax(resid[interior])]
        return (f"recovered line at {peak_sigma:.1f} cm^-1 "
                f"(true 1003.0); visibility flat at "
                f"{np.mean(vis):.2f} (unresolved point).")
    if case.name == "binary":
        # count visibility minima across the band -> ripple count
        mins = np.sum((vis[1:-1] < vis[:-2]) & (vis[1:-1] < vis[2:]))
        b = case.baseline_m
        # expected ripples across band = (sigma_max-sigma_min)*b*theta_sep
        sep = abs(case.sky.sources[0].offset_x_arcsec
                  - case.sky.sources[1].offset_x_arcsec) * cs.constants.ARCSEC
        exp_ripples = (p.band.sigma_max - p.band.sigma_min) * (b * 100.0) * sep
        return (f"visibility shows ~{mins} minimum(a) across the band; "
                f"expected ~{exp_ripples:.1f} cosine half-cycles for "
                f"the {sep/cs.constants.ARCSEC:.0f}\" separation.")
    if case.name == "resolved_disk":
        return (f"visibility falls {vis[0]:.2f} (red) -> {vis[-1]:.2f} "
                f"(blue): source more resolved at shorter wavelength.")
    return "ok"


def main(names=None):
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(here, "outputs", "cases")
    names = names or list(cs.scenes.CASES)

    p = cs.InstrumentParams()
    grids = cs.build_grids(p)
    print(f"Single-baseline analysis  |  band "
          f"{p.band.lambda_min_um:.0f}-{p.band.lambda_max_um:.0f} um, "
          f"R={p.band.spectral_resolution:.0f}, OPD scan +-"
          f"{grids.opd_max*1e4:.0f} um")
    print(f"Resolution lambda/B at b=0.4 m, 10 um: "
          f"{1e-5/0.4/cs.constants.ARCSEC:.1f} arcsec; "
          f"single-dish FoV {cs.beams.fov_first_null_arcsec(1000.0, 0.0508):.0f}"
          f" arcsec\n")

    for nm in names:
        if nm not in cs.scenes.CASES:
            print(f"  unknown case '{nm}' (have: "
                  f"{', '.join(cs.scenes.CASES)})")
            continue
        case = cs.scenes.CASES[nm]()
        result = cs.singlebaseline.analyse(case)
        path = cs.singlebaseline.plot_case(result, outdir)
        print(f"[{nm}] baseline {case.baseline_m} m  ({case.target_note})")
        print(f"   {physics_readout(result)}")
        print(f"   figure -> {os.path.relpath(path, here)}\n")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
