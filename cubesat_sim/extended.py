"""
Extended (pixel-map) scenes.

The discrete-source sky (sky.py) is the right model for a few compact
sources. An extended target -- a patch of the Moon with a crater, a
thermal region on Earth -- is instead a 2-D brightness MAP, and its
visibility is the van Cittert-Zernike integral over that map rather than
a finite sum over sources. This module adds that map as a new scene
type, behind the same (visibility -> interferogram) interface, so the
forward model, noise, and single-baseline analysis are all reused
unchanged.

A scene is stored as a temperature map T(x, y) [K] on an angular pixel
grid centred on the pointing; each pixel radiates as a blackbody at its
temperature (times an emissivity). At one baseline the complex
visibility and DC spectrum are

    V(sigma)  = sum_pix  eps B_sigma(T_pix) b^2(theta_pix) mu(theta_pix)
                * exp(-2 pi i sigma (b . theta_pix)) dOmega
    S0(sigma) = sum_pix  eps B_sigma(T_pix) b^2(theta_pix) dOmega

(the parity envelope mu gates only the coherent fringe term V, not the
incoherent DC pedestal S0)

i.e. the same VCZ sum forward.visibility uses, with the discrete sources
replaced by pixels of solid angle dOmega. Units match the discrete path
(W m^-2 (cm^-1)^-1), so the result feeds forward.interferogram_from_
visibility directly.

IMPORTANT (single-baseline scope): a uniform patch filling the beam
washes the fringes out (V -> 0, all power in the DC term); a compact
feature (a crater) breaks that uniformity and lifts |V| above zero.
That is the DETECTION signal. The fixed baseline sets which feature
SIZES are visible -- a crater near lambda/B gives the strongest, most
size-diagnostic signal; one much smaller is unresolved (tiny flat V),
one filling the beam washes out again. There is no imaging here (one
baseline, no rotation): the product is detection + one-axis sizing.
"""

from dataclasses import dataclass
import numpy as np

from . import beams
from . import constants as const
from .parameters import InstrumentParams


def _planck_Tmap(sigma_scalar, t_map):
    """Planck spectral radiance for an ARRAY of temperatures at one
    wavenumber [W m^-2 sr^-1 (cm^-1)^-1]. (constants.planck_bsigma casts
    T to a scalar, so it can't take a map; this is the array-T twin.)"""
    nu = const.C_LIGHT * 100.0 * float(sigma_scalar)
    x = const.H_PLANCK * nu / (const.K_BOLTZ * np.asarray(t_map, float))
    b_nu = 2.0 * const.H_PLANCK * nu**3 / const.C_LIGHT**2 / np.expm1(x)
    return b_nu * const.C_LIGHT * 100.0


@dataclass
class ExtendedScene:
    """A 2-D blackbody brightness map on an angular grid."""
    temperature_map: np.ndarray      # [ny, nx] in K
    pixel_scale_arcsec: float         # angular size of one pixel
    emissivity: float = 1.0
    name: str = "extended"

    def _angular_grid(self):
        ny, nx = self.temperature_map.shape
        sc = self.pixel_scale_arcsec * const.ARCSEC
        xs = (np.arange(nx) - (nx - 1) / 2.0) * sc
        ys = (np.arange(ny) - (ny - 1) / 2.0) * sc
        tx, ty = np.meshgrid(xs, ys)
        return tx, ty, sc * sc          # theta_x, theta_y [rad], dOmega [sr]

    def visibility(self, p: InstrumentParams, baseline_m, angle_rad, sigma):
        """
        Complex visibility V(sigma) and DC spectrum S0(sigma) at one
        baseline, by direct VCZ summation over the map pixels.
        Returns (V [nsig] complex, S0 [nsig] float), in W m^-2 (cm^-1)^-1.
        """
        sigma = np.atleast_1d(np.asarray(sigma, float))
        tx, ty, dOmega = self._angular_grid()
        d = p.telescope.d_dish_m
        bx = baseline_m * np.cos(angle_rad)
        by = baseline_m * np.sin(angle_rad)
        mismatch = p.cold_optics.parity_mismatch

        V = np.empty(sigma.size, complex)
        S0 = np.empty(sigma.size, float)
        for k, s in enumerate(sigma):
            I = self.emissivity * _planck_Tmap(s, self.temperature_map)
            beam = beams.amplitude_beam(tx, ty, s, d)        # [ny,nx]
            mu = beams.parity_envelope(tx, s, d, mismatch)
            w_dc = I * beam * beam * dOmega                  # incoherent DC weight
            w_fr = w_dc * mu                                 # coherent fringe weight
            phase = np.exp(-2.0j * np.pi * s * 100.0 * (bx * tx + by * ty))
            V[k] = np.sum(w_fr * phase)
            S0[k] = np.sum(w_dc)
        return V, S0


# ---------------------------------------------------------------------------
def moon_patch(crater_diameter_km=30.0, crater_T=250.0, surface_T=390.0,
               crater_offset_arcsec=(0.0, 0.0), fov_arcsec=110.0, npix=161,
               moon_distance_m=3.844e8):
    """
    A patch of sunlit lunar surface (uniform surface_T) with one circular
    crater (crater_T, cooler) of the given physical diameter. The patch
    spans fov_arcsec (a little larger than the ~50 arcsec single-dish
    beam). Returns (ExtendedScene, info) where info carries the crater's
    angular size for reference.

    NOTE: a real crater has rim/floor/central-peak thermal structure; a
    uniform cool disk is the minimal model that exercises the detection.
    The Moon is so bright that detectability is set by RESOLUTION (size
    vs lambda/B), not by the surface/crater contrast.
    """
    diam_arcsec = (crater_diameter_km * 1.0e3 / moon_distance_m) / const.ARCSEC
    sc = fov_arcsec / npix
    xs = (np.arange(npix) - (npix - 1) / 2.0) * sc
    X, Y = np.meshgrid(xs, xs)
    T = np.full((npix, npix), float(surface_T))
    cx, cy = crater_offset_arcsec
    mask = (X - cx) ** 2 + (Y - cy) ** 2 <= (diam_arcsec / 2.0) ** 2
    T[mask] = float(crater_T)
    scene = ExtendedScene(temperature_map=T, pixel_scale_arcsec=sc,
                          emissivity=1.0,
                          name=f"moon_{crater_diameter_km:.0f}km")
    info = {"crater_diameter_km": crater_diameter_km,
            "crater_diameter_arcsec": diam_arcsec,
            "fov_arcsec": fov_arcsec, "pixel_scale_arcsec": sc}
    return scene, info


def uniform_patch(surface_T=390.0, fov_arcsec=110.0, npix=161):
    """A featureless surface patch -- the reference against which the
    crater's visibility signal is measured."""
    sc = fov_arcsec / npix
    T = np.full((npix, npix), float(surface_T))
    return ExtendedScene(temperature_map=T, pixel_scale_arcsec=sc,
                         emissivity=1.0, name="uniform")
