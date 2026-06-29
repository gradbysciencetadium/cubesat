# cubesat_sim — CubeSat Double-Fourier Interferometer Simulator

End-to-end instrument simulator for the UCL CubeSat spatio-spectral
(double-Fourier) interferometer. A known scene is pushed through a software
model of the instrument to produce realistic detector interferograms.

In the **multi-baseline pipeline** (`run_simulation.py`, a FUTURE /
NON-MISSION deployable-boom configuration) the scene is then
reconstructed from those interferograms — the source positions and a spectrum
per source (parametric *source separation*, not full image synthesis) — and
compared with the input scene. Where they match, the design works; where they
diverge, an instrument limitation has been found. The CONFIRMED mission
(`run_crater.py`, `run_cases.py`) uses a SINGLE FIXED ~0.4 m baseline for
detection and one-axis sizing only, with NO multi-baseline imaging.

In the **single-baseline mode** (`run_cases.py`) there is no scene
reconstruction: it characterises one baseline's visibility and spectrum
(detection and sizing — "is something there, and how big or separated") — the
building block the full pipeline stacks many of.

Heritage: PyFIInS / FIInS (R. Juanola-Parramon, FISICA project),
re-architected in Python 3 for a two-element, single-baseline, warm 6U
platform. Physics per Grainger et al. 2012 (arXiv:1203.2144); parameters
per `Cubesat_interferometer_v0.2.xlsx` (cell references in
`parameters.py`) and the optical layout slides.

## Quick start

```
pip install -r requirements.txt
python run_simulation.py          # full multi-baseline demo -> ./outputs
python run_cases.py               # single-baseline target cases -> ./outputs/cases
python run_crater.py              # single-baseline crater study -> ./outputs/crater
python -m pytest tests            # physics sanity tests
```

## Layout

```
cubesat_sim/
  constants.py      physical constants, Planck function, unit conversions
  parameters.py     ALL instrument numbers, each traced to a source
                    document; placeholders/flags marked [PH]/FLAG
  fts.py            wavenumber + OPD sampling grids (Nyquist logic)
  sky.py            scene model: compact sources, blackbody + line spectra
  scenes.py         three runnable target cases (the building blocks)
  extended.py       pixel-map scenes (Moon patch, crater) for detection
  beams.py          Airy primary beam; parity-mismatch FoV envelope
  uvcoverage.py     spin + baseline-contraction uv schedule (FUTURE/non-mission; confirmed mission is single fixed baseline)
  radiometry.py     background loading, photon/detector NEP, sensitivity
  forward.py        scene -> visibilities -> interferograms (+ noise)
  reduction.py      interferograms -> visibility spectra (1st transform)
  imaging.py        dirty map; source positions + spectra (2nd transform; FUTURE/non-mission multi-baseline config)
  singlebaseline.py one baseline: interferogram <-> spectrum, the cases
  validate.py       input-vs-recovered metrics and figures
run_simulation.py   full multi-baseline end-to-end demo
run_cases.py        single-baseline demo over the three target cases
run_crater.py       single-baseline lunar-crater detection/sizing study
tests/              physics tests (test_physics, test_singlebaseline, test_extended)
docs/               LaTeX documentation (cubesat_simulator.tex)
```

