"""
Demonstration: single baseline vs many baselines.

This is a TEACHING demo, not the mission mode (the real instrument is a
single fixed baseline -- detection only). It shows *why* you need many
baselines to make an image: each baseline measures one Fourier
"ingredient" of the sky (one stripe pattern of a given scale + direction),
and only by collecting many do you accumulate enough ingredients to
rebuild the picture.

It reconstructs the SAME scene (three point sources) from an increasing
number of baselines and shows, side by side:
  * top row    -- the uv coverage so far (which Fourier ingredients we have)
  * bottom row -- the resulting reconstruction (dirty image), with the true
                  source positions marked.

N = 1   : one baseline  -> one stripe pattern. You see fringes, not sources.
N = 6   : a few angles at one baseline length -> sources start to localise.
N = 24  : a full half-rotation (one ring) -> a ring-shaped point-spread.
N = all : rotation + length steps -> the three sources cleanly recovered.

Run:  python run_synthesis_demo.py   ->  outputs/synthesis/
"""

import os
import numpy as np

import cubesat_sim as cs
from cubesat_sim.fts import FTSGrids
from cubesat_sim.reduction import UVSpectrum


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(here, "outputs", "synthesis")
    os.makedirs(outdir, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = cs.InstrumentParams()
    base = cs.build_grids(p)

    # monochromatic (band-centre) grid -> a clean, smear-free imaging demo
    gmono = FTSGrids(sigma=np.array([p.band.sigma_centre]),
                     delta_sigma=base.delta_sigma, opd=base.opd,
                     delta_opd=base.delta_opd, opd_max=base.opd_max,
                     t_scan=base.t_scan, t_sample=base.t_sample)

    # the scene: three point sources in an asymmetric triangle
    sky = cs.SkyModel(sources=[
        cs.Source(offset_x_arcsec=0.0, offset_y_arcsec=7.0, flux_jy=5.0e4),
        cs.Source(offset_x_arcsec=8.0, offset_y_arcsec=-5.0, flux_jy=3.0e4),
        cs.Source(offset_x_arcsec=-8.0, offset_y_arcsec=-5.0, flux_jy=3.0e4)])
    truth = [(s.offset_x_arcsec, s.offset_y_arcsec) for s in sky.sources]

    # full baseline schedule (rotation + length steps), then take subsets
    uv_points, info = cs.build_schedule(p, base.t_scan)
    print(f"full schedule: {info['n_uv_points']} baselines on "
          f"{info['n_rings']} rings")

    # one analytic (noiseless) visibility per baseline -> the ideal data
    uvspec = []
    for uv in uv_points:
        V, _ = cs.visibility(p, sky, gmono, uv)
        uvspec.append(UVSpectrum(uv=uv, sigma=gmono.sigma.copy(), vhat=V))

    n_total = len(uvspec)
    stages = [("N = 1\n(one baseline)", 1),
              ("N = 6\n(a few angles)", 6),
              ("N = 24\n(one half-rotation)", 24),
              (f"N = {n_total}\n(rotation + lengths)", n_total)]

    fig, ax = plt.subplots(2, 4, figsize=(15, 7.6))
    for col, (label, n) in enumerate(stages):
        subset = uvspec[:n]
        bx = [u.uv.bx for u in subset]
        by = [u.uv.by for u in subset]

        # top: uv coverage so far (+ Hermitian mirror, faint)
        a0 = ax[0, col]
        a0.plot(bx, by, ".", ms=5, color="#482661")
        a0.plot([-x for x in bx], [-y for y in by], ".", ms=5,
                color="#963A8A", alpha=0.35)
        a0.set_title(label, fontsize=10)
        a0.set_aspect("equal")
        a0.set_xlim(-0.6, 0.6)
        a0.set_ylim(-0.6, 0.6)
        if col == 0:
            a0.set_ylabel("uv coverage\nbaseline y [m]")
        a0.set_xlabel("baseline x [m]")

        # bottom: the reconstruction (dirty image) + true positions
        dmap = cs.dirty_map(subset, fov_arcsec=22.0, npix=141)
        a1 = ax[1, col]
        ext = [dmap.x_arcsec[0], dmap.x_arcsec[-1],
               dmap.y_arcsec[0], dmap.y_arcsec[-1]]
        a1.imshow(dmap.image, origin="lower", extent=ext, cmap="inferno")
        for tx, ty in truth:
            a1.plot(tx, ty, "+", color="cyan", ms=11, mew=1.6)
        if col == 0:
            a1.set_ylabel("reconstruction\ny [arcsec]")
        a1.set_xlabel("x [arcsec]")

    fig.suptitle("Single vs many baselines: each baseline is one Fourier "
                 "ingredient; the image only appears once you have many\n"
                 "(cyan + = the three true source positions)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(outdir, "baseline_synthesis.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)

    print(f"true sources at {truth} arcsec")
    print("N=1: one stripe pattern (can't see the sources)")
    print(f"N={n_total}: three sources recovered")
    print(f"figure -> {os.path.relpath(path, here)}")


if __name__ == "__main__":
    main()
