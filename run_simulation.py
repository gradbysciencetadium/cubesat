"""
End-to-end demonstration run of the CubeSat double-Fourier simulator.

*** FUTURE / NON-MISSION CONFIGURATION ***
This demo runs the MULTI-BASELINE imaging pipeline (baseline rotation +
contraction + dirty-map reconstruction). That is NOT the confirmed
mission design. The CONFIRMED mission is a SINGLE FIXED ~0.4 m baseline
for detection + one-axis sizing only (no imaging): see run_crater.py
and run_cases.py.

Pipeline:  parameters -> grids -> uv schedule -> scene
           -> forward model (interferograms, with and without noise)
           -> reduction (visibility spectra)
           -> dirty map -> source positions -> source spectra
           -> validation metrics + figures in ./outputs

Run from the cubesat_codebase directory:   python run_simulation.py
"""

import os
import numpy as np

import cubesat_sim as cs


def main():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "outputs")
    os.makedirs(outdir, exist_ok=True)

    # ---- 1. instrument ----------------------------------------------------
    p = cs.InstrumentParams()
    grids = cs.build_grids(p)
    print(f"Band {p.band.lambda_min_um}-{p.band.lambda_max_um} um, "
          f"R={p.band.spectral_resolution:.0f} -> "
          f"{grids.n_channels} channels of {grids.delta_sigma:.1f} cm^-1")
    print(f"OPD scan: {grids.n_samples} samples of "
          f"{grids.delta_opd*1e4:.1f} um up to {grids.opd_max*1e4:.0f} um, "
          f"t_scan = {grids.t_scan:.3f} s")

    # ---- 2. radiometric background and noise -------------------------------
    print()
    print(cs.radiometry.crosscheck_report(p))
    bg = cs.radiometry.background(p)
    bg_power = sum(bg.absorbed_W.values())
    noise_sigma = cs.radiometry.noise_sigma_per_sample(p, bg)
    print(f"\nbackground power on detector: {bg_power:9.3e} W")
    print(f"noise per OPD sample:         {noise_sigma:9.3e} W rms")

    # ---- 3. uv schedule -----------------------------------------------------
    uv_points, info = cs.build_schedule(p, grids.t_scan)
    print(f"\nuv schedule: {info['n_uv_points']} points on "
          f"{info['n_rings']} baseline rings "
          f"{np.round(info['baselines_m'], 3)} m, "
          f"total {info['total_time_s']:.1f} s, "
          f"{info['data_volume_Mbit']:.2f} Mbit science data")

    sens = cs.radiometry.sensitivity_summary(p, bg, grids,
                                             info["n_uv_points"])
    print(f"5-sigma min detectable flux density (1 scan):    "
          f"{sens['mdfd_5sigma_1scan_Jy']:9.1f} Jy")
    print(f"5-sigma min detectable flux density (full pass): "
          f"{sens['mdfd_5sigma_alluv_Jy']:9.1f} Jy")

    # ---- 4. scene -----------------------------------------------------------
    sky = cs.binary_scene(separation_arcsec=8.0, position_angle_deg=30.0,
                          flux_a_jy=5.0e4, flux_b_jy=2.0e4)
    print("\nscene: unequal binary, separation 8 arcsec, PA 30 deg, "
          "5e4 + 2e4 Jy, line on source B")

    # ---- 5. forward model ---------------------------------------------------
    # noiseless run = the architecture validation (any input/output
    # mismatch here is a code/physics error, not statistics)
    scans_clean = cs.observe(p, sky, grids, uv_points, bg_power,
                             0.0, seed=None)
    # noisy run = the performance estimate
    scans_noisy = cs.observe(p, sky, grids, uv_points, bg_power,
                             noise_sigma, seed=42)
    cs.validate.plot_interferogram(scans_clean[0], outdir,
                                   label="interferogram_noiseless")
    cs.validate.plot_interferogram(scans_noisy[0], outdir,
                                   label="interferogram_noisy")
    cs.validate.plot_uv(uv_points, outdir)

    # ---- 6+7. reduction and source separation, both runs -------------------
    for tag, scans in [("noiseless", scans_clean), ("noisy", scans_noisy)]:
        uvspec = cs.reduce_all(p, grids, scans)
        dmap = cs.dirty_map(uvspec, fov_arcsec=25.0, npix=151)
        positions = cs.locate_sources(dmap, n_sources=2)
        positions = cs.refine_positions(uvspec, positions)
        sigma, spectra = cs.extract_spectra(p, uvspec, positions)

        rows = cs.validate.position_metrics(sky, positions)
        specm = cs.validate.spectrum_metrics(sky, sigma, spectra, rows)
        print(f"\n--- {tag} run ---")
        for r in rows:
            tx, ty = r["true_xy"]
            print(f"  source at ({tx:.2f}, {ty:.2f}) arcsec recovered at "
                  f"({r['recovered_xy'][0]:.2f}, {r['recovered_xy'][1]:.2f}),"
                  f" error {r['error_arcsec']:.3f} arcsec")
        for m in specm:
            print(f"  source {m['source']} spectrum fractional rms: "
                  f"{m['fractional_rms']:.3%}")

        if tag == "noisy":
            cs.validate.plot_dirty_map(dmap, sky, positions, outdir)
            cs.validate.plot_spectra(sky, sigma, spectra, rows, outdir)

    print(f"\nfigures written to {outdir}")


if __name__ == "__main__":
    main()
