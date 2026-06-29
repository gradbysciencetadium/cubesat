"""
Scene library: three ready-to-run target cases.

Each Case bundles a SkyModel with a recommended single baseline and a
short description of (a) the candidate science target it stands for and
(b) the physics regime it exercises. The cases are INDEPENDENT: pick one
by name from CASES and push it through singlebaseline.analyse() without
touching the others, exactly as the professor framed it ("first select
what we are pointing at, then simulate that one case").

The three span the three things a double-Fourier instrument can be
limited by, each framed as a candidate CubeSat target:

  compact_line   - an UNRESOLVED source carrying a spectral feature
                   (a star, or an atmospheric/gas emission line). It is
                   a point on the sky, so it exercises the SPECTRAL axis
                   and shows the instrument line shape (the sinc 'ripples
                   in the convolution' the professor described).

  binary         - TWO compact sources a few arcsec apart (two craters,
                   a binary star, two thermal hotspots). It exercises
                   SOURCE SEPARATION: at a single baseline the angular
                   separation imprints a cosine RIPPLE across the
                   recovered spectrum, whose period reads off the
                   separation - the double-slit result of Grainger
                   Eqs. 11-13 seen on one baseline.

  resolved_disk  - one EXTENDED source comparable in size to lambda/B
                   (a planetary/lunar disk or limb, a wide thermal
                   region). It exercises RESOLUTION: the fringe contrast
                   (visibility) falls as the source is resolved, more so
                   at bluer wavenumbers, tilting the recovered spectrum.

The confirmed design is a SINGLE FIXED deployed baseline of ~0.4 m. All
three cases use b = 0.4 m by default; pass a different baseline to
analyse() to explore alternatives (e.g. the 0.125 m rigid-bench value),
noting these are comparisons, not multiple operating baselines.
"""

from dataclasses import dataclass
import numpy as np

from .sky import SkyModel, Source


@dataclass
class Case:
    """One runnable target case for the single-baseline analysis."""
    name: str
    description: str
    sky: SkyModel
    baseline_m: float
    angle_rad: float = 0.0
    # spectral oversampling of the simulated sky relative to the FTS
    # channels: the source is built on a grid this many times finer than
    # delta_sigma, so off-channel spectral content is represented and the
    # instrument line shape (the sinc ripples) is actually visible rather
    # than hidden by on-grid sampling.
    spectral_oversample: int = 16
    target_note: str = ""


# ---------------------------------------------------------------------------
def compact_line(baseline_m: float = 0.4) -> Case:
    """
    Unresolved warm source with a narrow emission line.

    On-axis point source (so the spatial visibility is flat and the
    SPECTRAL behaviour is isolated), 300 K continuum at 5e4 Jy, with a
    narrow line at 1003 cm^-1 (deliberately BETWEEN the 1000 and 1010
    channels, and FWHM 2.5 cm^-1, narrower than the 10 cm^-1 resolution)
    so the recovered line IS the instrument line shape - a sinc with
    sidelobe ripples.
    """
    src = Source(offset_x_arcsec=0.0, offset_y_arcsec=0.0,
                 temperature_k=300.0, flux_jy=5.0e4, lambda_ref_um=10.0,
                 line_amplitude=0.8, line_centre_cm1=1003.0,
                 line_fwhm_cm1=2.5)
    return Case(
        name="compact_line",
        description="Unresolved source + narrow spectral line "
                    "(spectral axis / instrument line shape).",
        sky=SkyModel(sources=[src]),
        baseline_m=baseline_m, angle_rad=0.0, spectral_oversample=24,
        target_note="A star, or an atmospheric gas line: a point on the "
                    "sky whose colours we want to measure.")


def binary(baseline_m: float = 0.4, separation_arcsec: float = 24.0) -> Case:
    """
    Two equal compact sources separated along the baseline.

    The separation imprints a cosine modulation on the single-baseline
    spectrum with period 1/(b*theta_sep) in wavenumber; at b = 0.4 m and
    24 arcsec that is ~2 ripples across the 8-12 um band. Reading the
    ripple period gives the separation.
    """
    half = 0.5 * separation_arcsec
    s1 = Source(offset_x_arcsec=+half, temperature_k=300.0, flux_jy=3.0e4)
    s2 = Source(offset_x_arcsec=-half, temperature_k=300.0, flux_jy=3.0e4)
    return Case(
        name="binary",
        description="Two compact sources (source separation; the "
                    "separation makes a cosine ripple across the spectrum).",
        sky=SkyModel(sources=[s1, s2]),
        baseline_m=baseline_m, angle_rad=0.0, spectral_oversample=16,
        target_note="Two craters, a binary star, or two thermal hotspots: "
                    "can the instrument tell there are two, and how far "
                    "apart?")


def resolved_disk(baseline_m: float = 0.4, size_arcsec: float = 2.5) -> Case:
    """
    One extended source of angular size comparable to lambda/B.

    A Gaussian source (FWHM 2.5 arcsec) on the b = 0.4 m baseline, where
    lambda/B ~ 5 arcsec at 10 um, so the source is partially resolved.
    The Gaussian visibility taper falls with wavenumber, suppressing the
    blue end of the recovered spectrum - the signature of resolving a
    source out.
    """
    src = Source(offset_x_arcsec=0.0, temperature_k=300.0, flux_jy=5.0e4,
                 size_arcsec=size_arcsec)
    return Case(
        name="resolved_disk",
        description="Extended source ~lambda/B across (resolution; fringe "
                    "contrast falls, bluer wavenumbers suppressed).",
        sky=SkyModel(sources=[src]),
        baseline_m=baseline_m, angle_rad=0.0, spectral_oversample=16,
        target_note="A planetary/lunar disk or limb, a wide thermal "
                    "region: is the target resolved, and at which colours?")


# registry: pick a case by name and run it independently
CASES = {
    "compact_line": compact_line,
    "binary": binary,
    "resolved_disk": resolved_disk,
}
