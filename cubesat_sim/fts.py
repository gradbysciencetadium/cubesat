"""
FTS spectral/OPD sampling grids.

This is the CubeSat equivalent of PyFIInS fts.py, with the same Nyquist
logic but the numbers driven by the radiometric spreadsheet:

    delta_sigma = sigma_centre / R            (spectral channel width)
    opd_max     = 1 / (2 * delta_sigma)       (max OPD extent; single-sided scan length, half the double-sided travel)
    delta_opd   = 1 / (2 * sigma_max * oversample)   (OPD step)

For the spreadsheet parameters (8-12 um, R = 100, oversample 2) this
gives delta_sigma = 10 cm^-1, opd_max = 0.05 cm, delta_opd = 2 um. The
spreadsheet BUDGETS a single-sided scan of 250 samples (0.375 s),
matching 'Inst Spectral' E43/E50/E53/E54 exactly. The DEFAULT scan
implemented here is double-sided (500 samples, 0.75 s): see the design
finding in parameters.FTSScan - complex visibilities need the symmetric
scan.

The sky is simulated on the *natural* FTS wavenumber grid
sigma_k = k * delta_sigma (k integer), truncated to the band. Simulating
on the natural grid means a noiseless interferogram transforms back to
the input spectrum with no interpolation error, which makes the
validation loop airtight: any input/output difference is then physics
(noise, asymmetry, sampling), never gridding artefacts.
"""

from dataclasses import dataclass
import numpy as np

from .parameters import InstrumentParams


@dataclass
class FTSGrids:
    sigma: np.ndarray         # wavenumber channels [cm^-1], band only
    delta_sigma: float        # channel width [cm^-1]
    opd: np.ndarray           # OPD sample positions [cm]
    delta_opd: float          # OPD step [cm]
    opd_max: float            # maximum OPD [cm]
    t_scan: float             # duration of one scan [s]
    t_sample: float           # integration time per OPD sample [s]

    @property
    def n_samples(self) -> int:
        return len(self.opd)

    @property
    def n_channels(self) -> int:
        return len(self.sigma)


def build_grids(p: InstrumentParams) -> FTSGrids:
    """Construct the wavenumber and OPD grids from the parameters."""
    band, scan, det = p.band, p.scan, p.detector

    delta_sigma = band.sigma_centre / band.spectral_resolution      # 10 cm^-1
    opd_max = 1.0 / (2.0 * delta_sigma)                             # 0.05 cm
    delta_opd = 1.0 / (2.0 * band.sigma_max * scan.nyquist_oversample)  # 2 um

    # natural (unaliased) wavenumber grid, truncated to the band
    k_min = int(np.ceil(band.sigma_min / delta_sigma))
    k_max = int(np.floor(band.sigma_max / delta_sigma))
    sigma = np.arange(k_min, k_max + 1, dtype=float) * delta_sigma

    if scan.single_sided:
        # single-sided alternative: OPD = 0 .. opd_max, the spreadsheet budget (250 samples)
        n = int(round(opd_max / delta_opd))
        opd = np.arange(n) * delta_opd
    else:
        # symmetric scan, same OPD step: twice the samples, R unchanged
        n = int(round(opd_max / delta_opd))
        opd = np.arange(-n, n) * delta_opd

    t_sample = 1.0 / det.acquisition_hz
    t_scan = len(opd) * t_sample        # double-sided default 500/666.67 Hz = 0.75 s; single-sided 250/666.67 Hz = 0.375 s

    return FTSGrids(sigma=sigma, delta_sigma=delta_sigma,
                    opd=opd, delta_opd=delta_opd, opd_max=opd_max,
                    t_scan=t_scan, t_sample=t_sample)
