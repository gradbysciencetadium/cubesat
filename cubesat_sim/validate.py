"""
Validation loop: compare the recovered scene with the input scene.

This module is the point of the whole simulator (thesis Ch.4 / Grainger
bench philosophy): a KNOWN scene goes in, the forward model + reduction
produce a recovered scene, and the difference measures what the
instrument design can actually do. Outputs: a metrics dict and a set of
diagnostic figures written to an output directory.
"""

import os
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use("Agg")          # headless backend: figures go to files
import matplotlib.pyplot as plt

from . import constants as const
from .imaging import DirtyMap
from .sky import SkyModel


def position_metrics(sky: SkyModel, recovered_positions):
    """
    Match recovered positions to true sources ONE TO ONE (optimal
    assignment, not greedy nearest-neighbour: with greedy matching two
    recovered positions can both claim the same true source and the
    spectrum comparison then uses the wrong truth) and report the
    offsets [arcsec]. Each row carries 'true_index' so downstream
    metrics stay aligned with the recovered-position ordering.
    """
    true = [(s.offset_x_arcsec, s.offset_y_arcsec) for s in sky.sources]
    cost = np.array([[np.hypot(rx - tx, ry - ty) for tx, ty in true]
                     for rx, ry in recovered_positions])
    rec_idx, true_idx = linear_sum_assignment(cost)
    rows = [None] * len(recovered_positions)
    for r, j in zip(rec_idx, true_idx):
        rows[r] = {"true_index": int(j), "true_xy": true[j],
                   "recovered_xy": tuple(recovered_positions[r]),
                   "error_arcsec": float(cost[r, j])}
    return [row for row in rows if row is not None]


def spectrum_metrics(sky: SkyModel, sigma, spectra, matched_rows):
    """
    Fractional rms difference between each recovered spectrum and the
    matched true source spectrum (matching via 'true_index' from
    position_metrics, so the pairing is guaranteed one-to-one).
    """
    out = []
    for j, row in enumerate(matched_rows):
        k = row["true_index"]
        s_true = sky.sources[k].spectrum(sigma)
        resid = spectra[j] - s_true
        out.append({"source": k,
                    "fractional_rms": float(np.sqrt(np.mean(resid**2))
                                            / np.mean(s_true))})
    return out


# --------------------------------------------------------------------------
def plot_uv(uv_points, outdir):
    fig, ax = plt.subplots(figsize=(5, 5))
    bx = [u.bx for u in uv_points]
    by = [u.by for u in uv_points]
    ax.plot(bx, by, ".", ms=4, label="sampled")
    ax.plot([-x for x in bx], [-y for y in by], ".", ms=4, alpha=0.4,
            label="Hermitian mirror")
    ax.set_xlabel("baseline x [m]")
    ax.set_ylabel("baseline y [m]")
    ax.set_title("uv coverage (baseline plane)")
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "uv_coverage.png"), dpi=150)
    plt.close(fig)


def plot_interferogram(scan, outdir, label="interferogram"):
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(scan.opd * 1.0e4, scan.power * 1e9, lw=0.8)
    ax.set_xlabel("OPD [um]")
    ax.set_ylabel("detector power [nW]")
    ax.set_title(f"{label}  (baseline {scan.uv.baseline_m:.2f} m, "
                 f"angle {np.degrees(scan.uv.angle_rad):.1f} deg)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{label}.png"), dpi=150)
    plt.close(fig)


def plot_dirty_map(dmap: DirtyMap, sky: SkyModel, recovered_positions,
                   outdir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    extent = [dmap.x_arcsec[0], dmap.x_arcsec[-1],
              dmap.y_arcsec[0], dmap.y_arcsec[-1]]
    im0 = axes[0].imshow(dmap.image, origin="lower", extent=extent)
    for s in sky.sources:
        axes[0].plot(s.offset_x_arcsec, s.offset_y_arcsec, "wx", ms=10,
                     mew=2, label="true")
    for rx, ry in recovered_positions:
        axes[0].plot(rx, ry, "r+", ms=12, mew=1.5, label="recovered")
    handles, labels = axes[0].get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    axes[0].legend(uniq.values(), uniq.keys(), fontsize=8)
    axes[0].set_title("dirty map (band-averaged)")
    axes[0].set_xlabel("x [arcsec]")
    axes[0].set_ylabel("y [arcsec]")
    fig.colorbar(im0, ax=axes[0], shrink=0.85)

    im1 = axes[1].imshow(dmap.beam, origin="lower", extent=extent)
    axes[1].set_title("dirty beam")
    axes[1].set_xlabel("x [arcsec]")
    fig.colorbar(im1, ax=axes[1], shrink=0.85)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "dirty_map.png"), dpi=150)
    plt.close(fig)


def plot_spectra(sky: SkyModel, sigma, spectra, matched_rows, outdir):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    colors = plt.cm.tab10.colors
    for j, row in enumerate(matched_rows):
        k = row["true_index"]
        c = colors[j % 10]
        ax.plot(sigma, sky.sources[k].spectrum(sigma) / const.JY_PER_CM1,
                "-", color=c, label=f"source {k} input")
        ax.plot(sigma, spectra[j] / const.JY_PER_CM1, "o", color=c, ms=4,
                label=f"source {k} recovered")
    ax.set_xlabel("wavenumber [cm$^{-1}$]")
    ax.set_ylabel("flux density [Jy]")
    ax.set_title("input vs recovered spectra (the validation loop)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "spectra_validation.png"), dpi=150)
    plt.close(fig)
