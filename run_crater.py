"""
Single-baseline lunar-crater detection / sizing study.

The real instrument is one FIXED baseline, single axis, no movement (the
delay line scans for the spectrum; the baseline is held). So this is NOT
imaging - it is detection and one-axis sizing, exactly the regime worked
through with the ship-on-sea example.

The study, for each candidate (fixed) baseline:
  * point at a lunar patch (uniform 390 K surface) containing one crater;
  * also point at a featureless reference patch;
  * record one interferogram each, reduce to the visibility V(sigma);
  * the crater raises |V| above the uniform-patch floor -> that lift,
    relative to the radiometric noise, is the DETECTION;
  * the SHAPE of |V|(sigma) (a sinc fall-off across the band, because
    u = b/lambda sweeps a radial line) gives the crater's diameter.

Repeating over crater sizes and baselines shows which feature sizes each
fixed baseline can detect/size - the evidence for choosing the one
baseline to fly. Run:  python run_crater.py

Because the Moon is extremely bright at 10 um, detectability is set by
RESOLUTION (size vs lambda/B), not by noise - so the detector-NEP
placeholder and the combiner factor-of-2 do not change the conclusion.
"""

import os
import numpy as np

import cubesat_sim as cs
from cubesat_sim import extended


BASELINES_M = [0.2, 0.4, 0.5]    # GS: undeployed ~0.2, deployed nominal ~0.4, ideal 0.5
CRATER_KM = [10.0, 30.0, 90.0]            # small / medium / Copernicus-class
SURFACE_T = 390.0                          # sunlit lunar dayside [K]
CRATER_T = 250.0                           # cooler crater floor [K]


def visibility_noise_per_channel(p, grids):
    """RMS noise on one recovered visibility channel [W m^-2 (cm^-1)^-1],
    from the radiometric model and the reduction gain (matches
    reduction.reduce_scan's calibration)."""
    rep = cs.radiometry.background(p)
    sig_samp = cs.radiometry.noise_sigma_per_sample(p, rep)
    gain = cs.forward.fringe_gain(p, grids)                 # shared gain; MTF ~ 1
    return sig_samp * np.sqrt(2.0 / grids.n_samples) / gain


def resolution_arcsec(baseline_m, lam_um=10.0):
    return 1.22 * (lam_um * 1e-6) / baseline_m / cs.constants.ARCSEC


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(here, "outputs", "crater")
    os.makedirs(outdir, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = cs.InstrumentParams()
    grids = cs.build_grids(p)
    sigma = grids.sigma
    sigV = visibility_noise_per_channel(p, grids)
    fov = cs.beams.fov_first_null_arcsec(1000.0, p.telescope.d_dish_m)

    print("Single-baseline lunar-crater detection study")
    print(f"  band 8-12 um, single-dish field of view ~{fov:.0f} arcsec "
          f"(~{fov*cs.constants.ARCSEC*3.844e8/1e3:.0f} km patch at the Moon)")
    print(f"  per-channel visibility noise: {sigV:.3e} W m^-2 (cm^-1)^-1\n")
    print(f"{'baseline':>9} {'resolution':>11} {'crater':>8} "
          f"{'det. SNR':>9}  verdict")
    print("-" * 60)

    uniform = extended.uniform_patch(surface_T=SURFACE_T)
    snr_table = np.zeros((len(CRATER_KM), len(BASELINES_M)))

    for jb, b in enumerate(BASELINES_M):
        res = resolution_arcsec(b)
        res_km = res * cs.constants.ARCSEC * 3.844e8 / 1e3
        # uniform reference visibility at this baseline
        Vu, _ = uniform.visibility(p, b, 0.0, sigma)
        for ic, ck in enumerate(CRATER_KM):
            scene, info = extended.moon_patch(crater_diameter_km=ck,
                                              crater_T=CRATER_T,
                                              surface_T=SURFACE_T)
            Vc, _ = scene.visibility(p, b, 0.0, sigma)
            signal = Vc - Vu                       # the crater's contribution
            # Vc and Vu are two INDEPENDENT measured visibilities (one
            # interferogram each), so the noise on their difference is
            # sqrt(2)*sigV, not sigV (Var(Vc-Vu) = Var(Vc)+Var(Vu)).
            snr = np.sqrt(np.sum(np.abs(signal) ** 2)) / (np.sqrt(2.0) * sigV)
            snr_table[ic, jb] = snr
            verdict = ("detect+size" if snr > 5 and info["crater_diameter_arcsec"] > res
                       else "detect only" if snr > 5
                       else "not detected")
            print(f"{b:>7.3f} m {res:>8.1f}\" {ck:>6.0f} km "
                  f"{snr:>9.1e}  {verdict}")
    print()

    # --- figure 1: the patches we point at -------------------------------
    scene30, info30 = extended.moon_patch(crater_diameter_km=30.0,
                                          crater_T=CRATER_T, surface_T=SURFACE_T)
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.2))
    ext = [-info30["fov_arcsec"] / 2, info30["fov_arcsec"] / 2] * 2
    for a, sc, ttl in [(ax[0], uniform, "uniform reference patch"),
                       (ax[1], scene30, "patch with 30 km crater")]:
        im = a.imshow(sc.temperature_map, origin="lower", extent=ext,
                      cmap="inferno")
        a.set_title(ttl)
        a.set_xlabel("x [arcsec]")
        a.set_ylabel("y [arcsec]")
        fig.colorbar(im, ax=a, shrink=0.8, label="T [K]")
    fig.suptitle("What the single baseline points at (Moon, ~92 km patch)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "crater_patches.png"), dpi=150)
    plt.close(fig)

    # --- figure 2: visibility signature, 30 km crater, vs baseline -------
    fig, ax = plt.subplots(1, len(BASELINES_M), figsize=(14, 4.0),
                           sharey=True)
    for jb, b in enumerate(BASELINES_M):
        Vu, S0u = uniform.visibility(p, b, 0.0, sigma)
        Vc, S0c = scene30.visibility(p, b, 0.0, sigma)
        ax[jb].plot(sigma, np.abs(Vu) / S0u, color="#7A4FB0", lw=1.4,
                    label="uniform (no crater)")
        ax[jb].plot(sigma, np.abs(Vc) / S0c, color="#963A8A", lw=1.6,
                    label="with 30 km crater")
        ax[jb].set_title(f"b = {b} m  (res {resolution_arcsec(b):.0f}\")")
        ax[jb].set_xlabel("wavenumber [cm$^{-1}$]")
        if jb == 0:
            ax[jb].set_ylabel("fringe visibility $|V|/S_0$")
            ax[jb].legend(fontsize=8)
    fig.suptitle("Crater detection signature: |V| lifted above the "
                 "uniform-surface floor")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(outdir, "crater_visibility.png"), dpi=150)
    plt.close(fig)

    # --- figure 3: detection SNR vs size and baseline --------------------
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    im = ax.imshow(np.log10(snr_table), origin="lower", cmap="viridis",
                   aspect="auto")
    ax.set_xticks(range(len(BASELINES_M)))
    ax.set_xticklabels([f"{b} m" for b in BASELINES_M])
    ax.set_yticks(range(len(CRATER_KM)))
    ax.set_yticklabels([f"{c:.0f} km" for c in CRATER_KM])
    ax.set_xlabel("baseline")
    ax.set_ylabel("crater diameter")
    for ic in range(len(CRATER_KM)):
        for jb in range(len(BASELINES_M)):
            ax.text(jb, ic, f"{snr_table[ic, jb]:.0e}", ha="center",
                    va="center", color="w", fontsize=8)
    fig.colorbar(im, ax=ax, label="log10 detection SNR")
    ax.set_title("Detection SNR vs crater size and (fixed) baseline")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "crater_snr.png"), dpi=150)
    plt.close(fig)

    print(f"figures -> {os.path.relpath(outdir, here)}")


if __name__ == "__main__":
    main()
