"""
cubesat_sim - end-to-end instrument simulator for the UCL CubeSat
double-Fourier (spatio-spectral) interferometer.

Forward model:   sky -> visibilities -> interferograms (+ noise)
Inverse model:   interferograms -> visibility spectra -> positions+spectra
Validation:      recovered scene vs input scene

Heritage: PyFIInS / FIInS (R. Juanola-Parramon, FISICA) re-architected
for a two-element, single-baseline, warm 6U platform; physics per
Grainger et al. 2012; parameters per Cubesat_interferometer_v0.2.xlsx.
"""

from .parameters import (Band, Telescope, ColdOptics, Detector, FTSScan,
                         Geometry, Environment, InstrumentParams)
from .fts import build_grids, FTSGrids
from .sky import Source, SkyModel, binary_scene
from .uvcoverage import build_schedule, UVPoint
from . import radiometry
from . import forward
from .forward import (observe, observe_scan, visibility, ScanData,
                     fringe_gain, dc_gain, interferogram_from_visibility)
from .reduction import reduce_all, reduce_scan, UVSpectrum
from .imaging import (dirty_map, locate_sources, refine_positions,
                      extract_spectra, DirtyMap)
from . import validate
from . import beams
from . import constants
from . import scenes
from .scenes import Case, CASES
from . import singlebaseline
from . import extended
from .extended import ExtendedScene, moon_patch, uniform_patch

__version__ = "0.1.0"

__all__ = [
    "Band", "Telescope", "ColdOptics", "Detector", "FTSScan", "Geometry",
    "Environment", "InstrumentParams", "build_grids", "FTSGrids",
    "Source", "SkyModel", "binary_scene", "build_schedule", "UVPoint",
    "radiometry", "forward", "observe", "observe_scan", "visibility",
    "ScanData", "fringe_gain", "dc_gain", "interferogram_from_visibility",
    "reduce_all", "reduce_scan", "UVSpectrum", "dirty_map",
    "locate_sources", "refine_positions", "extract_spectra", "DirtyMap",
    "validate", "beams", "constants",
    "scenes", "Case", "CASES", "singlebaseline",
    "extended", "ExtendedScene", "moon_patch", "uniform_patch",
]
