"""
Instrument and mission parameters.

Every number in this file is traceable to one of three sources:

  [RM]   Cubesat_interferometer_v0.2.xlsx  (the radiometric model, v0.2,
         G. Savini, 27/10/2025) - sheet and cell noted next to each value.
  [OPT]  cubesat_interferometer.pptx       (optical layout slides).
  [PH]   PLACEHOLDER - no value exists yet in any source document.
         These are the open design questions; each carries a comment
         saying what is missing and who/what decides it.

The dataclasses are deliberately flat and dumb: they hold numbers and
derive a few obvious quantities. All physics lives in the other modules.
"""

from dataclasses import dataclass, field
import numpy as np


# ---------------------------------------------------------------------------
@dataclass
class Band:
    """Spectral band definition. [RM] 'Inst Spectral' sheet."""

    lambda_min_um: float = 8.0      # [RM] Inst Spectral E8  (channel B1 cut-on)
    lambda_max_um: float = 12.0     # [RM] Inst Spectral E9  (channel B1 cut-off)
    spectral_resolution: float = 100.0   # [RM] Inst Spectral E36 (R at band centre)

    # CONFIRMED by GS (June 2026): 8-12 um mid-infrared IS the intended
    # band; the spreadsheet's "Far-IR" header is a leftover FISICA label
    # (far-IR = 25-400 um). Mid-IR drives the mirror/beamsplitter/detector
    # choices, so this is the design band, not a stray value.

    @property
    def sigma_min(self) -> float:
        """Band cut-on wavenumber [cm^-1]. [RM] Inst Spectral E17 = 833.33."""
        return 1.0e4 / self.lambda_max_um

    @property
    def sigma_max(self) -> float:
        """Band cut-off wavenumber [cm^-1]. [RM] Inst Spectral E18 = 1250."""
        return 1.0e4 / self.lambda_min_um

    @property
    def sigma_centre(self) -> float:
        """Band centre wavenumber [cm^-1]. [RM] Inst Spectral I26 = 1000."""
        return 1.0e4 / (0.5 * (self.lambda_min_um + self.lambda_max_um))


# ---------------------------------------------------------------------------
@dataclass
class Telescope:
    """Collector parameters. [RM] 'Telescopes' sheet."""

    n_dish: int = 2                 # [RM] Telescopes E8
    d_dish_m: float = 0.0508        # [RM] Telescopes E9  (2 inch)
    n_mirrors: int = 2              # [RM] Telescopes E13 (per telescope)
    temperature_k: float = 293.0    # [RM] Telescopes E14 (warm instrument!)
    emissivity_per_mirror: float = 0.01   # [RM] Telescopes E15

    @property
    def area_single_m2(self) -> float:
        """Collecting area of ONE dish [m^2]."""
        return np.pi * (self.d_dish_m / 2.0) ** 2

    @property
    def optical_efficiency(self) -> float:
        """(1-eps)^N_m = 0.9801. [RM] Telescopes E19."""
        return (1.0 - self.emissivity_per_mirror) ** self.n_mirrors

    @property
    def equivalent_emissivity(self) -> float:
        """1-(1-eps)^N_m = 0.0199. [RM] Telescopes E21."""
        return 1.0 - self.optical_efficiency


# ---------------------------------------------------------------------------
@dataclass
class ColdOptics:
    """
    Combiner-box optics. [RM] 'Inst_ColdOptics' sheet.

    'Cold' is the inherited FISICA name only: this box is at 293 K
    ([RM] Inst_ColdOptics E26), which is why the thermal background
    matters so much at 10 um (Wien peak of 293 K is 9.9 um).
    """

    n_cold_mirrors: int = 5           # [RM] E18
    mirror_reflectivity: float = 0.98  # [RM] E29
    mirror_emissivity: float = 0.02    # [RM] E28
    combiner_t: float = 0.49           # [RM] E52 beam combiner <T>
    combiner_r: float = 0.49           # [RM] E53 beam combiner <R>
    combiner_emissivity: float = 0.02  # [RM] E54
    lyot_aperture: float = 0.95        # [RM] E23
    filter_t: float = 0.98             # [RM] E47 band-defining filter
    box_temperature_k: float = 293.0   # [RM] E26
    box_emissivity: float = 0.1        # [RM] E27
    stray_fraction: float = 0.01       # [RM] E25

    # --- the extra-fold-mirror asymmetry --------------------------------
    # [OPT] slides 3-4: the right-hand routing to the combiner carries one
    # extra diagonal fold (path budgets '10+10' and '3+5+6+d' vs '10+5'
    # and '5+4'). The radiometric model treats both arms as identical
    # ([RM] uses a single Opteff_cold for everything) and parks
    # 'visibility loss due to intensity mismatch' under THINGS TO ADD
    # ([RM] Inst Performance B50). Here each arm gets its own throughput.
    # Mirror-count consistency with the layout: each telescope unit is a
    # 2-mirror OAP beam reducer ([OPT] slide 1, = N_m of the Telescopes
    # sheet); the combiner-box chain (folds + roof pair + relay) is the
    # spreadsheet's N_cold = 5, with this extra fold the +1 on arm 2.
    extra_mirrors_arm2: int = 1        # [OPT] set 0 to model symmetric arms

    # GS (June 2026): the optical design is NOT finalised, and "in
    # principle you do not need an extra mirror -- if there is one the
    # design needs to be tweaked." Parity is to be checked on the FINAL
    # design by propagating a letter 'P' (or an arrow) through every
    # surface and confirming it matches at the combiner/detector. So
    # MATCHED parity is the correct working assumption for now; the
    # extra_mirrors_arm2 above is a worst-case knob, not a fixed feature.
    # parity_mismatch shrinks the usable field of view off-axis
    # (beams.parity_envelope) but does nothing on-axis, so it is
    # negligible for the single-baseline on-axis detection scope.
    parity_mismatch: bool = False      # GS: assume matched (design not final)

    @property
    def efficiency_common(self) -> float:
        """
        Cold-optics throughput common to both arms = 0.4124.
        [RM] Inst_ColdOptics E8:
        (1-eps_cm)^N_cold * (T_BC = 0.49) * filter * Lyot
        = 0.98^5 * 0.49 * 0.98 * 0.95 = 0.41236
        NOTE the 0.49 beam-combiner term IS included here, exactly as the
        spreadsheet does, so it must not be applied again elsewhere.
        """
        return (self.mirror_reflectivity ** self.n_cold_mirrors
                * self.combiner_t * self.filter_t * self.lyot_aperture)

    @property
    def arm_throughput(self) -> tuple:
        """
        (t1, t2): relative throughput of each arm with the asymmetric
        extra fold applied to arm 2 only.  t1 = 1, t2 = rho^extra.
        These multiply efficiency_common in the forward model.
        """
        return (1.0, self.mirror_reflectivity ** self.extra_mirrors_arm2)

    @property
    def equivalent_emissivity(self) -> float:
        """
        Overall cold-optics emissivity referred to the detector = 0.0811.
        [RM] Inst_ColdOptics E103 (eq_em_cold_B1):
        box stray + cold-mirror chain + beam-combiner terms.
        Reproduced from the spreadsheet's own decomposition:
          emiss_cm_B1 = 0.07083  [RM] E87
          emiss_BC_B1 = 0.00931  [RM] E97
          box stray   = eps_box * stray = 0.001
        """
        emiss_cm = ((1.0 - self.mirror_reflectivity ** (self.n_cold_mirrors - 1))
                    * (self.combiner_t + self.combiner_r) * self.filter_t
                    * self.lyot_aperture)
        emiss_bc = (self.combiner_emissivity * self.filter_t
                    * 0.49 * self.lyot_aperture)   # 0.49 = FTS modulation eff [RM] E31
        return self.box_emissivity * self.stray_fraction + emiss_cm + emiss_bc


# ---------------------------------------------------------------------------
@dataclass
class Detector:
    """Detector parameters. [RM] 'Inst_detectors' sheet."""

    absorption_efficiency: float = 0.9   # [RM] E7 / E14
    time_constant_s: float = 500e-6      # [RM] E12
    acquisition_hz: float = 666.666666   # [RM] E18 (sample every 3 tau)
    n_pixels: int = 1                    # [RM] E24 - single-pixel instrument
    n_modes: int = 1                     # [RM] E16

    # Digitisation: [RM] Electronics&Data E12 (16 bit/sample). Used only
    # for the data-volume estimate in the uv schedule; quantisation noise
    # itself is not modelled (negligible against the detector NEP).
    bits_per_sample: int = 16

    # PLACEHOLDER [PH]: detector saturation power. [RM] Inst Performance
    # B70 / Inst_detectors B47 carry the row ('Source flux that would
    # saturate the detector ... iterate on this') but no value. When set,
    # the forward model warns if any sample exceeds it - the warm
    # background (~3 nW) plus a bright source is exactly the regime where
    # a small-well mid-IR detector could clip.
    saturation_power_w: float = None   # [PH] None = no check

    # PLACEHOLDER [PH]: detector dark NEP. The spreadsheet rows
    # 'NEP (Dark) - B1..B4' ([RM] Inst_detectors B36-B39) are EMPTY -
    # this is the one missing number that decides whether the system is
    # photon-noise or detector-noise limited. The default below is a
    # representative TEC-cooled mid-IR photodetector:
    #   D* ~ 1e11 cm sqrt(Hz)/W, 50 um pixel  ->  NEP = sqrt(A)/D* ~ 5e-14
    # That is ~500x LARGER than the photon-background NEP (9.2e-17,
    # [RM] Inst Performance E41), i.e. with this detector the instrument
    # is detector-noise limited, not background limited. A cryogenic
    # detector (NEP ~ 1e-16) would change that conclusion completely.
    # >>> Gate question for GS: what detector, what NEP? <<<
    nep_dark: float = 5.0e-14            # [PH] W / sqrt(Hz)


# ---------------------------------------------------------------------------
@dataclass
class FTSScan:
    """
    Delay-line scan definition. [RM] 'Inst Spectral' sheet.

    Derivations (standard FTS sampling, identical logic to fts.py in
    PyFIInS, numbers from the spreadsheet):
      delta_sigma = sigma_centre / R          = 10 cm^-1
      opd_max     = 1 / (2 delta_sigma)       = 0.05 cm   [RM] E50
      delta_opd   = 1/(2 sigma_max)/oversamp  = 2 um      [RM] E43
      n_samples   = opd_max / delta_opd       = 250       [RM] E54   (single-sided)
      t_scan      = n_samples / f_acq         = 0.375 s   [RM] E53   (single-sided)
    NOTE: these spreadsheet values are the SINGLE-SIDED budget. The
    DEFAULT here is double-sided (single_sided=False below), giving
    500 samples and t_scan = 0.75 s.
    """

    nyquist_oversample: int = 2      # [RM] Inst Spectral E41
    scans_per_uv_point: int = 1      # [RM] Inst Spectral E56

    # The moving stage is a roof-mirror pair: mirror travel x gives OPD 2x,
    # so mechanical travel = OPD/2. [RM] E51 (MPD_Max = OPD_Max/2) and
    # [OPT] slide 4 ('1/2 dOPD' on the Mov.Stage).
    # NOTE this differs from FIInS, whose cat's-eye design had OPD = 4 x
    # mechanical (smec_opd_to_mpd = 0.25 in the old fts.py).
    opd_to_mechanical: float = 0.5

    # Fixed inter-arm path offset (where zero path difference sits
    # relative to the delay line's nominal zero), in cm.
    # [OPT] slides 3-4: the path budgets BALANCE by design - summing the
    # C and D rows gives right arm 10+10+5+4 = 29 and left arm
    # 10+5+3+5+6+delta = 29+delta, i.e. the extra fold mirror is
    # path-length compensated and ZPD sits at the stage's nominal zero.
    # The DESIGN value is therefore 0; the [PH] placeholder here is the
    # RESIDUAL offset (manufacturing tolerance, thermal drift of the
    # bench) which is not specified in any source document. The forward
    # model applies it as a real instrument error; the reduction removes
    # it only if told its value (a calibration), so setting it nonzero
    # here and zero in the reduction simulates an uncalibrated ZPD error.
    zpd_offset_cm: float = 0.0          # [PH] residual; design value 0

    # Scan geometry. A double-Fourier interferometer measures COMPLEX
    # visibilities (the fringe phase carries the source position), which a
    # single-sided scan cannot recover -- the per-channel cos/sin
    # quadratures are degenerate over a one-sided window, so recovery fails
    # at the 10-100% level even noiseless. GS (June 2026) CONFIRMS:
    # "the plan was to have this double-sided anyway" (as SPIRE did, for
    # risk management); a single-sided + short phase-correction segment is
    # the lighter alternative they sometimes use, but double-sided is the
    # baseline plan. Default here: double-sided (matches GS).
    single_sided: bool = False

    # GS sizing constraint (June 2026): the delay-line mechanism must have
    # enough TRAVEL to REACH zero path difference, not just to cover the
    # spectral OPD range. A static OPD difference between the two arms
    # (baseline geometry, thermal/mechanical) shifts where ZPD sits, and
    # can be mm-scale -- far larger than the +-0.05 cm (0.5 mm) spectral
    # scan. GS: "if you end up with an OPD on the two arms which is a
    # (huge) 3 mm and our intended OPD is only 2 mm, goodbye interferometry
    # without the ZPD position." So total travel = spectral OPD range +
    # |zpd_offset| + margin. zpd_offset_cm above models this static shift;
    # the simulator assumes the mechanism can reach it (i.e. travel is
    # adequately sized) -- the sizing itself is a hardware requirement.


# ---------------------------------------------------------------------------
@dataclass
class Geometry:
    """
    Baseline geometry and uv-coverage strategy.
    [RM] 'Interferometer' + 'Spacecraft' sheets.
    """

    baseline_min_m: float = 0.3     # [RM] Interferometer E8
    baseline_max_m: float = 0.5     # [RM] Interferometer E9

    # RESOLVED by GS (June 2026): a "simple" design has a SINGLE baseline
    # value once deployed. Each telescope is a ~10 cm construct (two
    # beam-contraction mirrors) and the two cannot interpenetrate, so the
    # UNDEPLOYED centre-to-centre distance (= the baseline) is ~20 cm.
    # Each arm then deploys outward by p (7 < p < 10 cm), giving a nominal
    # DEPLOYED baseline of ~40 cm (50 cm is ideal, not impossible, but less
    # likely). So model a single fixed deployed baseline ~0.4 m
    # (undeployed ~0.2 m, ideal 0.5 m) -- the single-baseline crater study
    # (run_crater.py) sweeps 0.2/0.4/0.5 m to bracket this.
    # FUTURE (GS, deferred): if a mirror-pair relay is aligned with the
    # deployment direction, interference can be held at every point of the
    # p stroke -> the deployment itself would scan the baseline (0.2->0.4 m)
    # and give genuine radial uv coverage. Not modelled yet; would revive a
    # (limited) imaging mode.
    #
    # CONFIRMED design = SINGLE FIXED baseline ~0.4 m: baseline_min_m,
    # baseline_max_m and radial_step_factor below are NOT used by it.
    # They are consumed ONLY by uvcoverage.py (the FUTURE multi-baseline
    # schedule) and by the schedule tests. They are kept at the
    # spreadsheet's 0.3-0.5 m so that schedule remains runnable; do NOT
    # read them as the mission baseline. The single-baseline studies
    # (run_crater.py / run_cases.py) set their own fixed baseline.

    # Baseline rotation: one interferogram scan per 'dish diameter' of
    # arc motion. [RM] Spacecraft E23/E24: omega = (d/2) / t_scan / (b/2).
    # uv samples along the arc are then spaced by ~the aperture diameter.
    # Radial contraction step: each dish moves one full diameter per
    # half-turn -> baseline step 2*d_dish. [RM] Spacecraft E25, and
    # Interferometer E18 ('max radial density' = 2 extra rings).
    radial_step_factor: float = 2.0   # baseline step = factor * d_dish
    # FLAG: 2*d_dish radial spacing undersamples the uv plane radially by
    # 2x relative to the Nyquist criterion (step = d_dish). It is the
    # spreadsheet's time-saving choice. Set to 1.0 for Nyquist rings.

    rotation_deg: float = 180.0       # half-circle per ring (Hermitian
    # symmetry V(-u) = V*(u) fills the other half for free)


# ---------------------------------------------------------------------------
@dataclass
class Environment:
    """
    Thermal environment for the background model.
    [RM] 'SKY & Sources' sheet.

    NOTE: the spreadsheet is currently configured as a LAB TESTBED run -
    its 'source' is an 1800 K circuit ([RM] E18) against a 293 K lab
    background with emissivity 1 ([RM] E16/E17). For flight, the relevant
    background terms are the warm telescope, the warm combiner box, and
    the 250 K Earth/solar environment term the spreadsheet carries as
    'SUN' ([RM] E13, loading 5.4e-9 W = the dominant background).
    """

    t_environment_k: float = 250.0    # [RM] SKY&Sources E13 (T_SUN-EARTH)
    include_environment: bool = True
    # PLACEHOLDER [PH]: the effective emissivity/dilution with which the
    # 250 K environment couples into the beam is not stated anywhere in
    # the spreadsheet (the SUN column uses emissivity 1 through the full
    # optical chain). Kept at 1.0 to reproduce the spreadsheet's numbers;
    # a baffled flight design would reduce this a lot.
    environment_emissivity: float = 1.0

    t_cmb_k: float = 2.726            # [RM] E8 (utterly negligible at 10 um)


# ---------------------------------------------------------------------------
@dataclass
class InstrumentParams:
    """Top-level container; build one of these and pass it around."""

    band: Band = field(default_factory=Band)
    telescope: Telescope = field(default_factory=Telescope)
    cold_optics: ColdOptics = field(default_factory=ColdOptics)
    detector: Detector = field(default_factory=Detector)
    scan: FTSScan = field(default_factory=FTSScan)
    geometry: Geometry = field(default_factory=Geometry)
    environment: Environment = field(default_factory=Environment)

    # Etendue convention for the background model:
    #  'spreadsheet_pixel' reproduces [RM] Inst Performance E11:
    #      AOmega = pi (d/2)^2 * pi * theta_pix^2 = 3.67e-10 m^2 sr,
    #      theta_pix = 1.22 lambda_CENTRE / d  (lambda-independent).
    #      FLAG: the spreadsheet labels this row '@ 8um' and its formula
    #      references lambda_L1, but its cached VALUE (3.6725e-10, and
    #      FoV 0.8256 arcmin in E9) corresponds to the 10 um band centre
    #      - the label/formula and the stored number disagree. We follow
    #      the stored numbers (10 um) so the cross-checks reproduce them.
    #  'single_mode' uses the physically standard diffraction-limited
    #      single-mode etendue AOmega = lambda^2 (~1.0e-10 m^2 sr at 10 um,
    #      i.e. ~3.7x smaller -> ~1.9x lower photon NEP).
    etendue_convention: str = "spreadsheet_pixel"

    @property
    def optical_efficiency(self) -> float:
        """
        Overall warm+cold optical efficiency, both arms symmetric part:
        0.9801 * 0.41236 = 0.4042.  [RM] Inst_ColdOptics E13.
        (Detector absorption is applied separately.)
        """
        return (self.telescope.optical_efficiency
                * self.cold_optics.efficiency_common)

    @property
    def system_efficiency(self) -> float:
        """Optical efficiency x detector absorption = 0.364."""
        return self.optical_efficiency * self.detector.absorption_efficiency

    def etendue_m2sr(self, sigma_cm1):
        """Etendue [m^2 sr] under the chosen convention (see above)."""
        sigma = np.asarray(sigma_cm1, dtype=float)
        if self.etendue_convention == "single_mode":
            lam = 1.0 / (100.0 * sigma)
            return lam ** 2
        # spreadsheet pixel etendue, fixed at the band-centre beam width
        # (see FLAG above on the spreadsheet's stale '@ 8um' label)
        d = self.telescope.d_dish_m
        lam_centre_um = 0.5 * (self.band.lambda_min_um
                               + self.band.lambda_max_um)
        theta_pix = 1.22 * (lam_centre_um * 1e-6) / d
        a_omega = (np.pi * (d / 2.0) ** 2) * np.pi * theta_pix ** 2
        return np.full_like(sigma, a_omega, dtype=float)
