"""
Baseline schedule / uv coverage.

*** FUTURE / NON-MISSION CONFIGURATION ***
The CONFIRMED mission (GS, June 2026) is a SINGLE FIXED ~0.4 m baseline
for detection + one-axis sizing, with NO baseline rotation or
contraction and NO imaging (see run_crater.py / run_cases.py). This
module implements a multi-baseline rotation+contraction schedule for a
FUTURE deployable-boom scenario only; it is not the confirmed design.

The schedule fills the uv plane the way a two-element instrument with a
MOVABLE baseline would: by ROTATING the baseline (spacecraft spin about
the line of sight) and stepping its LENGTH between half-turns. This
replaces the spiral patterns of PyFIInS uvmapgenerator.py, which assumed
a flagship with telescopes trolleying along a boom.

The schedule implements the spreadsheet's strategy ('Spacecraft' rows
22-31, 'Interferometer' rows 15-18):

  * spin rate set so the dish moves one full diameter per interferogram
    scan:  omega(b) = (d/2) / (t_scan * b/2)     [RM Spacecraft E23/E24]
    -> uv samples along each arc are spaced ~ one aperture diameter
       (the uv-plane Nyquist criterion: delta_phi = d / b);
  * one half-circle (180 deg) per baseline length - the other half is
    free because the sky is real: V(-u,-v) = V*(u,v) (Hermitian);
  * baseline length stepped by radial_step_factor * d_dish between
    half-turns (spreadsheet value 2*d: each dish moves one diameter,
    which UNDERSAMPLES radially by 2x vs Nyquist - flagged in
    parameters.py).

Each scheduled uv point freezes the baseline for the duration of one
FTS scan (0.75 s in the default double-sided mode, 0.375 s single-
sided). During a real scan the baseline rotates by omega * t_scan = d/b
~ 5.8 deg (independent of scan mode, since omega scales as 1/t_scan);
the design criterion is precisely that this smearing stays within one
aperture diameter, so the frozen-baseline approximation is consistent
with the design's own tolerance. Fringe smearing within a scan is NOT
modelled - noted as a future systematic.
"""

from dataclasses import dataclass
import numpy as np

from .parameters import InstrumentParams


@dataclass
class UVPoint:
    baseline_m: float        # baseline length [m]
    angle_rad: float         # position angle of the baseline [rad]
    time_s: float            # scheduled start time of the scan [s]

    @property
    def bx(self) -> float:
        return self.baseline_m * np.cos(self.angle_rad)

    @property
    def by(self) -> float:
        return self.baseline_m * np.sin(self.angle_rad)


def build_schedule(p: InstrumentParams, t_scan: float):
    """
    Build the list of UVPoints for one full uv-plane pass.

    Returns (points, info) where info is a small dict of derived
    quantities for logging / cross-checks against the spreadsheet.
    """
    geo, tel = p.geometry, p.telescope
    d = tel.d_dish_m

    # baseline rings, stepped inwards from b_max by factor * d.
    # The number of contraction steps follows the spreadsheet's rounding
    # ([RM] Interferometer E18: ROUND((b_max/2 - b_min/2)/d_dish) = 2
    # steps -> 3 rings for the default geometry); the innermost ring is
    # clamped at b_min where the step overshoots it.
    step = geo.radial_step_factor * d
    n_steps = max(0, int(round((geo.baseline_max_m - geo.baseline_min_m)
                               / step)))
    baselines = geo.baseline_max_m - step * np.arange(n_steps + 1)
    baselines = np.unique(np.maximum(baselines, geo.baseline_min_m))[::-1]
    n_rings = len(baselines)

    points = []
    t = 0.0
    for b in baselines:
        # spin rate for this ring [RM Spacecraft E23]
        omega = (d / 2.0) / (t_scan * b / 2.0)
        # angular step between scans = rotation during one scan
        dphi = omega * t_scan                       # = d / b  (uv Nyquist)
        n_arc = max(1, int(np.floor(np.radians(geo.rotation_deg) / dphi)))
        for k in range(n_arc):
            points.append(UVPoint(baseline_m=float(b),
                                  angle_rad=float(k * dphi),
                                  time_s=float(t)))
            t += t_scan

    # science data volume for the pass: one detector, f_acq samples/s,
    # 16 bit each ([RM] Electronics&Data E12/E17: N_pix * nu_acq * bits
    # = 10.7 kbps generation rate)
    n_samples_total = t * p.detector.acquisition_hz
    data_mbit = (n_samples_total * p.detector.n_pixels
                 * p.detector.bits_per_sample / 1.0e6)

    info = {
        "n_rings": n_rings,
        "baselines_m": baselines,
        "n_uv_points": len(points),
        "total_time_s": t,
        "data_volume_Mbit": data_mbit,
        # Cross-checks against the spreadsheet for the default geometry:
        # 30 arc points at b_max vs [RM] Interferometer E15 ~ 28 (they use
        # an averaged radius); 72 points total vs Electronics&Data E11's
        # 97 (their count assumes finer sampling); total time = n_points *
        # t_scan, where t_scan doubled with the double-sided scan finding.
        # Slew/settle time between rings is NOT included.
    }
    return points, info
