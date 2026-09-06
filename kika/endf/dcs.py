r"""Differential cross sections from ENDF MF4 (angular) + MF3 (cross section).

This module is the **single definition** of the physics behind the desktop
app's MF4 differential view.  Each piece used to exist twice — once here in
Python and once as a TypeScript reimplementation in the app — which is exactly
how the two drifted apart (see ``delta_t_is_fwhm`` below).  Anything that
encodes a *choice* now lives here:

- the angular reconstruction :math:`f(\mu)` from Legendre coefficients,
- the interpolation of those coefficients **in energy**, honouring the MF4
  ``(NBT, INT)`` law rather than assuming lin-lin,
- the elastic LAB :math:`\leftrightarrow` CM frame transform,
- the three readings of :math:`\sigma(E)`: nominal, bin-averaged, TOF-folded.

The two plot products the app asks for are :func:`differential_xs_vs_angle`
(:math:`d\sigma/d\Omega` against :math:`\mu` at fixed :math:`E`) and
:func:`differential_xs_vs_energy` (against :math:`E` at fixed :math:`\mu`).

Everything takes plain arrays rather than an ENDF object, so the API layer can
feed it already-extracted data, and the tests can feed it literals.

Conventions
-----------
Energies are in **eV** unless a name says ``_mev``; cross sections in **barns**;
:math:`d\sigma/d\Omega` in **barn/sr** and :math:`d\sigma/d\mu` in **barn**.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from kika._constants import FWHM_TO_SIGMA, NEUTRON_MASS_AMU
from kika.processing.interpolation import interpolate_1d
from kika.utils.energy_folding import tof_energy_resolution
from kika.utils.numerics import fold_tabulated

__all__ = [
    "TofResolution",
    "legendre_basis",
    "angular_pdf",
    "angular_pdf_uncertainty",
    "frame_alpha",
    "cos_lab_from_cos_cm",
    "cos_cm_from_cos_lab",
    "jacobian_cm_to_lab",
    "transform_angular_curve",
    "interpolate_log_log",
    "bin_edges_for_energy",
    "bin_edges_for_width",
    "sigma_nominal",
    "sigma_bin_averaged",
    "sigma_folded",
    "resolve_sigma",
    "max_legendre_order",
    "legendre_provenance",
    "LegendreProvenance",
    "DEFAULT_PROJECTION_ORDER",
    "coefficients_at_energies",
    "coefficients_bin_averaged",
    "coefficients_folded",
    "resolve_coefficients",
    "differential_xs_factor",
    "differential_xs_vs_angle",
    "differential_xs_vs_energy",
]

Frame = Literal["lab", "cm"]
XsMode = Literal["nominal", "binavg", "folded"]
Weighting = Literal["lethargy", "constant"]


# =============================================================================
# TOF resolution
# =============================================================================

@dataclass(frozen=True)
class TofResolution:
    r"""Time-of-flight energy resolution for the folded :math:`\sigma(E)`.

    Parameters
    ----------
    flight_path_m : float
        Flight path :math:`L` in metres (GELINA default 27.037 m).
    delta_t_ns : float
        Timing spread :math:`\delta t` in nanoseconds.
    delta_t_is_fwhm : bool, default True
        Whether ``delta_t_ns`` is a **FWHM** (the usual experimental
        convention, and what :func:`kika.utils.energy_folding.tof_energy_resolution`
        assumes) or already a standard deviation.

        .. warning::
           This flag is the divergence that motivated this module.  The app's
           TypeScript reimplementation applied no FWHM→σ conversion, so its
           folding kernel was :data:`kika._constants.FWHM_TO_SIGMA` ≈ 2.355×
           wider than the library's for the same inputs.  There is no safe
           default that suits both readings; callers that care should pass it
           explicitly.
    min_sigma_e_kev : float, default 1.0
        Floor on :math:`\sigma_E`, in keV.  Keeps the kernel from collapsing to
        a delta at low energy where the TOF relation gives an unusably narrow
        width.
    """

    flight_path_m: float = 27.037
    delta_t_ns: float = 10.0
    delta_t_is_fwhm: bool = True
    min_sigma_e_kev: float = 1.0

    #: A resolution the experiment declared itself, as a **FWHM in eV**,
    #: constant in energy.  When set it replaces the flight-path geometry: see
    #: the class note below.
    declared_fwhm_ev: Optional[float] = None

    #: The same, as a fraction of the incident energy (EXFOR's PER-CENT form).
    #: Ignored when :attr:`declared_fwhm_ev` is set.
    declared_fwhm_fraction: Optional[float] = None

    def sigma_e_mev(self, energy_mev: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        r""":math:`\sigma_E` in MeV at one or many incident energies.

        A declared width wins over the flight-path geometry when one is given.
        The two are not interchangeable and one cannot be re-expressed as the
        other: the TOF relation makes :math:`\sigma_E` grow as
        :math:`E^{3/2}`, whereas a declared resolution is typically quoted as a
        constant or as a fixed fraction of :math:`E`.  Fitting an
        :math:`(L, \delta t)` pair to a declared width would therefore only
        agree at the single energy it was fitted at.
        """
        e = np.atleast_1d(np.asarray(energy_mev, dtype=float))
        out = np.empty_like(e)

        if self.declared_fwhm_ev is not None and self.declared_fwhm_ev > 0:
            out[:] = (self.declared_fwhm_ev / 1e6) / FWHM_TO_SIGMA
            out[e <= 0] = 0.0
        elif self.declared_fwhm_fraction is not None and self.declared_fwhm_fraction > 0:
            out[:] = np.where(e > 0, e * self.declared_fwhm_fraction / FWHM_TO_SIGMA, 0.0)
        else:
            for i, ei in enumerate(e):
                if ei <= 0 or self.flight_path_m <= 0 or self.delta_t_ns <= 0:
                    out[i] = 0.0
                else:
                    out[i] = tof_energy_resolution(
                        float(ei),
                        flight_path_m=self.flight_path_m,
                        delta_t_ns=self.delta_t_ns,
                        delta_t_is_fwhm=self.delta_t_is_fwhm,
                    )

        np.maximum(out, self.min_sigma_e_kev / 1000.0, out=out)
        return float(out[0]) if np.ndim(energy_mev) == 0 else out


# =============================================================================
# Angular reconstruction
# =============================================================================

def legendre_basis(mu: np.ndarray, max_order: int) -> np.ndarray:
    r"""Legendre polynomials :math:`P_l(\mu)` for :math:`l = 0 \ldots L`.

    Evaluated by Bonnet's recurrence
    :math:`(l+1)P_{l+1} = (2l+1)\mu P_l - l P_{l-1}`, which is what the app's
    client-side kernel uses; the shared basis keeps the two bit-comparable.

    Returns
    -------
    np.ndarray
        Shape ``(max_order + 1, len(mu))``.
    """
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    basis = np.empty((max_order + 1, mu.size), dtype=float)
    basis[0] = 1.0
    if max_order >= 1:
        basis[1] = mu
    for l in range(2, max_order + 1):
        basis[l] = ((2 * l - 1) * mu * basis[l - 1] - (l - 1) * basis[l - 2]) / l
    return basis


def angular_pdf(mu: np.ndarray, a_coeffs: Sequence[float]) -> np.ndarray:
    r"""Normalized angular distribution :math:`f(\mu)` from ENDF coefficients.

    .. math::
        f(\mu) = \tfrac{1}{2} \sum_{l=0}^{L} (2l+1)\, a_l\, P_l(\mu)

    with :math:`a_0 = 1` implicit by the MF4 normalization convention, so
    ``a_coeffs`` is :math:`[a_1, a_2, \ldots, a_L]` and :math:`\int f\,d\mu = 1`.

    Equivalent to :func:`kika.utils.energy_folding.endf_angular_distribution`
    (a test pins them together); this one evaluates the basis by recurrence
    instead of ``scipy.special.legendre``, which matters when sweeping
    thousands of energies.
    """
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    a = np.asarray(a_coeffs, dtype=float)
    basis = legendre_basis(mu, a.size)
    orders = np.arange(1, a.size + 1)
    result = 0.5 * basis[0]
    if a.size:
        result = result + 0.5 * ((2 * orders + 1) * a) @ basis[1:]
    return result


def angular_pdf_uncertainty(
    mu: np.ndarray,
    a_coeffs: Sequence[float],
    a_sigmas: Sequence[float],
) -> np.ndarray:
    r"""Diagonal (uncorrelated) uncertainty on :math:`f(\mu)`.

    .. math::
        \sigma_f(\mu)^2 = \sum_l \left[\tfrac{2l+1}{2} P_l(\mu)\right]^2
                          \sigma_{a_l}^2

    Diagonal-only: it ignores the MF34 off-diagonal blocks, so it is a display
    aid rather than a defensible uncertainty on the reconstructed shape.
    """
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    a = np.asarray(a_coeffs, dtype=float)
    s = np.asarray(a_sigmas, dtype=float)
    if a.size != s.size:
        raise ValueError(f"a_coeffs ({a.size}) and a_sigmas ({s.size}) must have equal length")
    if a.size == 0:
        return np.zeros_like(mu)
    basis = legendre_basis(mu, a.size)
    orders = np.arange(1, a.size + 1)
    weights = (0.5 * (2 * orders + 1) * s) ** 2
    return np.sqrt(weights @ (basis[1:] ** 2))


# =============================================================================
# Elastic frame transform
# =============================================================================

def frame_alpha(*, awr: Optional[float] = None, mass_number: Optional[float] = None) -> float:
    r"""Mass ratio :math:`\alpha = m_n / m_{target}` for the elastic transform.

    Prefers the ENDF atomic weight ratio (:math:`AWR = m_{target}/m_n`, so
    :math:`\alpha = 1/AWR`); falls back to the mass number
    (:math:`\alpha \approx m_n / A`), good to ~0.1%.  Returns ``0.0`` when
    neither is known, which every function here treats as "no transform".
    """
    if awr and awr > 0:
        return 1.0 / float(awr)
    if mass_number and mass_number > 0:
        return NEUTRON_MASS_AMU / float(mass_number)
    return 0.0


def cos_lab_from_cos_cm(mu_cm, alpha: float):
    r""":math:`\mu_L = (\mu_C + \alpha)/\sqrt{1 + 2\alpha\mu_C + \alpha^2}`."""
    mu_cm = np.asarray(mu_cm, dtype=float)
    return (mu_cm + alpha) / np.sqrt(1.0 + 2.0 * alpha * mu_cm + alpha * alpha)


def cos_cm_from_cos_lab(mu_lab, alpha: float):
    r"""Inverse of :func:`cos_lab_from_cos_cm` (forward branch).

    .. math::
        \mu_C = -\alpha(1 - \mu_L^2) + \mu_L\sqrt{1 - \alpha^2(1 - \mu_L^2)}

    Single-valued only for :math:`\alpha < 1` (target heavier than the
    neutron), which holds for every nuclide but hydrogen.
    """
    mu_lab = np.asarray(mu_lab, dtype=float)
    s = 1.0 - mu_lab * mu_lab
    return -alpha * s + mu_lab * np.sqrt(np.maximum(1.0 - alpha * alpha * s, 0.0))


def jacobian_cm_to_lab(mu_cm, alpha: float):
    r"""Solid-angle Jacobian :math:`J = d\Omega_C/d\Omega_L` at :math:`\mu_C`.

    .. math::
        J = \frac{(1 + \alpha^2 + 2\alpha\mu_C)^{3/2}}{|1 + \alpha\mu_C|}
    """
    mu_cm = np.asarray(mu_cm, dtype=float)
    return np.power(1.0 + alpha * alpha + 2.0 * alpha * mu_cm, 1.5) / np.abs(1.0 + alpha * mu_cm)


def transform_angular_curve(
    mu: np.ndarray,
    y: np.ndarray,
    alpha: float,
    direction: Literal["cm2lab", "lab2cm"],
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Move a per-solid-angle curve between the LAB and CM frames.

    A PDF :math:`f(\mu)` and a :math:`d\sigma/d\Omega` are both per-solid-angle
    densities, so the same Jacobian applies::

        cm2lab:  y_L(\mu_L) = y_C(\mu_C) · J,   \mu_L = cos_lab_from_cos_cm(\mu_C)
        lab2cm:  y_C(\mu_C) = y_L(\mu_L) / J,   \mu_C = cos_cm_from_cos_lab(\mu_L)

    **Elastic two-body kinematics only** (MT=2 with a neutron projectile).  For
    a discrete inelastic level the mapping also depends on the incident energy
    through the Q value, which this does not model.

    The returned cosine grid is generally non-uniform.  ``alpha = 0`` is a
    passthrough.
    """
    mu = np.asarray(mu, dtype=float)
    y = np.asarray(y, dtype=float)
    if not alpha or alpha <= 0:
        return mu.copy(), y.copy()
    if direction == "cm2lab":
        return cos_lab_from_cos_cm(mu, alpha), y * jacobian_cm_to_lab(mu, alpha)
    if direction == "lab2cm":
        mu_cm = cos_cm_from_cos_lab(mu, alpha)
        return mu_cm, y / jacobian_cm_to_lab(mu_cm, alpha)
    raise ValueError(f"direction must be 'cm2lab' or 'lab2cm', got {direction!r}")


# =============================================================================
# sigma(E)
# =============================================================================

def interpolate_log_log(x_grid, y_grid, xq):
    """Log-log interpolation of ``y(x)``, clamped to the endpoints.

    The ENDF convention for pointwise cross sections (INT=5).  Falls back to
    linear on any interval where a value is non-positive.  ``x_grid`` ascending.
    """
    x = np.asarray(x_grid, dtype=float)
    y = np.asarray(y_grid, dtype=float)
    scalar = np.ndim(xq) == 0
    q = np.atleast_1d(np.asarray(xq, dtype=float))
    if x.size == 0:
        out = np.full(q.shape, np.nan)
        return float(out[0]) if scalar else out
    if x.size == 1:
        out = np.full(q.shape, y[0])
        return float(out[0]) if scalar else out

    k = np.clip(np.searchsorted(x, q, side="right") - 1, 0, x.size - 2)
    x0, x1 = x[k], x[k + 1]
    y0, y1 = y[k], y[k + 1]

    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(x1 == x0, 0.0, (q - x0) / (x1 - x0))
        out = y0 + t * (y1 - y0)
        log_ok = (x0 > 0) & (x1 > 0) & (q > 0) & (y0 > 0) & (y1 > 0)
        if np.any(log_ok):
            tt = (np.log(q[log_ok]) - np.log(x0[log_ok])) / (np.log(x1[log_ok]) - np.log(x0[log_ok]))
            out[log_ok] = np.exp(
                (1.0 - tt) * np.log(y0[log_ok]) + tt * np.log(y1[log_ok])
            )

    out = np.where(q <= x[0], y[0], out)
    out = np.where(q >= x[-1], y[-1], out)
    return float(out[0]) if scalar else out


def bin_edges_for_energy(energies: Sequence[float], index: int) -> Tuple[float, float]:
    """Bin ``[lo, hi]`` around ``energies[index]``, as midpoints to its neighbours.

    The first/last point takes a one-sided half-bin mirrored to the other side.
    A single-point grid yields a degenerate bin.
    """
    e = np.asarray(energies, dtype=float)
    n = e.size
    if n == 0:
        raise ValueError("energies is empty")
    ei = float(e[index])
    if n == 1:
        return ei, ei
    if index <= 0:
        hi = 0.5 * (e[0] + e[1])
        lo = ei - (hi - ei)
    elif index >= n - 1:
        lo = 0.5 * (e[n - 2] + e[n - 1])
        hi = ei + (ei - lo)
    else:
        lo = 0.5 * (e[index - 1] + ei)
        hi = 0.5 * (ei + e[index + 1])
    return max(float(lo), 0.0), float(hi)


BinWindow = Literal["mf4grid", "relative", "tof"]


def bin_edges_for_width(
    energy_ev: float,
    *,
    mode: BinWindow = "relative",
    relative_width: float = 0.02,
    tof: Optional["TofResolution"] = None,
    n_sigma: float = 1.0,
) -> Tuple[float, float]:
    r"""Bin ``[lo, hi]`` around ``energy_ev`` from an explicit width.

    The companion to :func:`bin_edges_for_energy`, which takes its width from
    the spacing of the MF4 incident-energy grid.  That is the right default —
    it is the resolution at which the angular distribution is actually
    tabulated — but it also means the window is as coarse as that grid, and for
    a nuclide whose angular data is far sparser than its cross section it can
    span a wide range with nothing on screen to say so.

    ``mode``:

    ``'relative'``
        A fixed :math:`\Delta E / E`, centred on ``energy_ev``.
    ``'tof'``
        ``n_sigma`` times the time-of-flight resolution width at this energy,
        so a bin average and a fold can be compared on equal terms.
    ``'mf4grid'``
        Not handled here — it needs the grid, so call
        :func:`bin_edges_for_energy` instead.  Passing it raises, rather than
        silently substituting a different window.
    """
    if mode == "mf4grid":
        raise ValueError(
            "mode='mf4grid' takes its width from the energy grid; "
            "call bin_edges_for_energy(energies, index) instead"
        )
    e = float(energy_ev)
    if not (e > 0.0):
        return 0.0, 0.0
    if mode == "relative":
        half = 0.5 * e * float(relative_width)
    elif mode == "tof":
        resolution = tof or TofResolution()
        half = float(n_sigma) * resolution.sigma_e_mev(e / 1e6) * 1e6
    else:  # pragma: no cover - guarded by the Literal
        raise ValueError(f"unknown bin window mode {mode!r}")
    return max(e - half, 0.0), e + half


def sigma_nominal(xs_energies_ev, xs_values, energy_ev):
    r"""Pointwise :math:`\sigma(E)`, log-log interpolated (ENDF convention)."""
    return interpolate_log_log(xs_energies_ev, xs_values, energy_ev)


def sigma_bin_averaged(
    xs_energies_ev,
    xs_values,
    e_lo_ev: float,
    e_hi_ev: float,
    weighting: Weighting = "lethargy",
) -> float:
    r"""Flux-averaged :math:`\sigma` over ``[e_lo, e_hi]``.

    .. math::
        \langle\sigma\rangle = \frac{\int \sigma(E) w(E)\,dE}{\int w(E)\,dE}

    Trapezoidal on the union of the bin edges and the grid points strictly
    inside the bin, with :math:`\sigma` sampled log-log.  ``weighting``:
    ``'lethargy'`` uses :math:`w = 1/E` (flux flat per unit lethargy, the
    neutronics default); ``'constant'`` uses :math:`w = 1`.  A degenerate bin
    returns :math:`\sigma(e_{lo})`.

    The lethargy weight is not integrable at :math:`E = 0`, and the lowest bin
    of an MF4 grid does reach zero once its half-width is mirrored.  Both
    integrals diverge there while their ratio tends to :math:`\sigma` at the
    lower end, so such a bin is truncated at the first positive grid point
    inside it rather than evaluated at the singularity.
    """
    grid = np.asarray(xs_energies_ev, dtype=float)
    if not (e_hi_ev > e_lo_ev):
        return float(sigma_nominal(grid, xs_values, e_lo_ev))

    if weighting == "lethargy" and e_lo_ev <= 0.0:
        positive = grid[(grid > 0.0) & (grid < e_hi_ev)]
        e_lo_ev = float(positive[0]) if positive.size else e_hi_ev * 1e-6
        if not (e_hi_ev > e_lo_ev):
            return float(sigma_nominal(grid, xs_values, e_hi_ev))

    # Slice the bin by binary search rather than masking the whole grid: this is
    # called once per output energy, and an MF3 grid runs to hundreds of
    # thousands of points, so a full scan here is the difference between a
    # sub-second sweep and a three-minute one.
    lo_i = int(np.searchsorted(grid, e_lo_ev, side="right"))
    hi_i = int(np.searchsorted(grid, e_hi_ev, side="left"))
    inside = grid[lo_i:hi_i]
    points = np.concatenate(([e_lo_ev], inside, [e_hi_ev]))
    values = np.atleast_1d(interpolate_log_log(grid, xs_values, points))
    w = 1.0 / points if weighting == "lethargy" else np.ones_like(points)

    d_e = np.diff(points)
    keep = d_e > 0
    if not np.any(keep):
        return float(sigma_nominal(grid, xs_values, e_lo_ev))
    fw = values * w
    num = np.sum(0.5 * (fw[:-1] + fw[1:])[keep] * d_e[keep])
    den = np.sum(0.5 * (w[:-1] + w[1:])[keep] * d_e[keep])
    return float(num / den) if den > 0 else float(sigma_nominal(grid, xs_values, e_lo_ev))


def sigma_folded(xs_energies_ev, xs_values, energy_ev, tof: TofResolution):
    r"""TOF-resolution-folded :math:`\sigma(E)`.

    Averages :math:`\sigma` over a Gaussian kernel
    :math:`N(E_0, \sigma_E^2)` by Gauss-Hermite quadrature
    (:func:`kika.utils.numerics.fold_tabulated`), with :math:`\sigma_E` from
    ``tof``.  Vectorized over ``energy_ev``.

    Note the folding samples :math:`\sigma` **linearly** (``numpy.interp``
    semantics), unlike :func:`sigma_nominal`.  That is deliberate: the fold is
    an integral of the tabulated interpolant, and log-log sampling inside it
    would bias the quadrature near zeros.
    """
    e0 = np.asarray(energy_ev, dtype=float)
    sigma_e_ev = np.asarray(tof.sigma_e_mev(e0 / 1e6), dtype=float) * 1e6
    return fold_tabulated(
        np.asarray(xs_energies_ev, dtype=float),
        np.asarray(xs_values, dtype=float),
        e0,
        sigma_e_ev,
    )


def resolve_sigma(
    xs_energies_ev,
    xs_values,
    energy_ev,
    *,
    mode: XsMode = "nominal",
    bin_edges: Optional[Tuple[float, float]] = None,
    tof: Optional[TofResolution] = None,
    weighting: Weighting = "lethargy",
):
    r"""Dispatch to the requested reading of :math:`\sigma(E)`.

    ``'nominal'`` → :func:`sigma_nominal`; ``'binavg'`` → :func:`sigma_bin_averaged`
    over ``bin_edges``; ``'folded'`` → :func:`sigma_folded` with ``tof``.  A
    mode whose inputs are missing falls back to nominal rather than raising, so
    a UI toggle can never produce a hole in the curve.
    """
    if mode == "binavg" and bin_edges is not None:
        return sigma_bin_averaged(xs_energies_ev, xs_values, bin_edges[0], bin_edges[1], weighting)
    if mode == "folded" and tof is not None:
        return sigma_folded(xs_energies_ev, xs_values, energy_ev, tof)
    return sigma_nominal(xs_energies_ev, xs_values, energy_ev)


# =============================================================================
# Coefficients in energy
# =============================================================================

def max_legendre_order(section) -> int:
    r"""Highest Legendre order :math:`L` an MF4 section actually carries.

    Per ENDF-6, an MF4 LTT=1 LIST record holds exactly ``NL`` values,
    :math:`a_1 \ldots a_{NL}` — :math:`a_0 = 1` is the normalization convention
    and is **never stored**.  So the row length *is* the order, with no
    off-by-one to correct.  Sections vary ``NL`` with incident energy (a
    forward-peaked distribution at 45 MeV needs far more orders than a nearly
    isotropic one at thermal), so the answer is the maximum over the grid.

    Returns ``0`` for a section with no Legendre part, e.g. a purely isotropic
    one, which is the correct order for :math:`f(\mu) = 1/2`.
    """
    rows = getattr(section, "legendre_coefficients", None)
    if rows:
        lengths = [len(r) for r in rows if r is not None and len(r)]
        if lengths:
            return int(max(lengths))
    # Some representations expose only a count rather than the rows.
    count = getattr(section, "num_legendre_coefficients", None)
    try:
        return max(0, int(count or 0))
    except (TypeError, ValueError):
        return 0


DEFAULT_PROJECTION_ORDER = 20
r"""Orders to offer for a distribution the file stores as :math:`f(\mu)`.

A tabulated section carries no :math:`a_\ell`, so any order shown for it is one
we computed by projection.  How far to go is a choice, not a property of the
file: the projection is defined for every :math:`\ell` and the coefficients
just get small.

Twenty, because the forward peak is what sets the answer and it sharpens with
energy.  Reconstructing JEFF-4.0 U-235 elastic from its own projection and
comparing against the stored 91-point table, worst error relative to the peak:

=========  ==========  ==========
energy     L = 10      L = 20
=========  ==========  ==========
0.1 MeV    0.00 %      0.00 %
1 MeV      0.03 %      0.04 %
10 MeV     9.66 %      0.26 %
18 MeV     25.73 %     0.47 %
=========  ==========  ==========

Ten is exact where the distribution is nearly flat and visibly wrong where it
is not.  Past 20 nothing moves: the half-percent left at 18 MeV is the table's
own kink structure, not a missing order.

Deliberately *not* measured per file at parse time: that would mean projecting
every tabulated section of every file the moment it is opened, on the read path
that already dominates the cost of opening a heavy evaluation.
"""


@dataclass(frozen=True)
class LegendreProvenance:
    r"""Where an MF4 section's :math:`a_\ell(E)` come from.

    A viewer built only around stored coefficients shows nothing for the half
    of the evaluations that tabulate :math:`f(\mu)` instead, and shows a mixed
    section as if its whole energy range were stored.  Both are answered by
    carrying the provenance next to the numbers rather than inferring it.

    Attributes
    ----------
    ltt
        The file's own representation flag: 0 isotropic, 1 Legendre,
        2 tabulated, 3 mixed.
    kind
        ``ltt`` as a name, for anything that has to say it out loud.
    stored_max_order
        Highest :math:`\ell` the file actually writes.  0 for a tabulated or
        isotropic section, which store no coefficients at all.
    max_order
        Highest :math:`\ell` that can be *asked* for -- stored where the file
        stores them, projected where it does not.  This is the one a list of
        available orders should be built from.
    projected
        ``"none"``, ``"all"``, or ``"above_boundary"`` for a mixed section,
        whose coefficients are read below ``boundary_energy`` and projected
        above it.
    boundary_energy
        Where a mixed section switches representation, in eV; None otherwise.
    """

    ltt: int
    kind: str
    stored_max_order: int
    max_order: int
    projected: str
    boundary_energy: Optional[float] = None

    @property
    def is_projected(self) -> bool:
        return self.projected != "none"


_LTT_KINDS = {0: "isotropic", 1: "legendre", 2: "tabulated", 3: "mixed"}


def legendre_provenance(
    section, *, projection_order: int = DEFAULT_PROJECTION_ORDER
) -> LegendreProvenance:
    r"""Describe what :math:`a_\ell(E)` a section can produce, and from what.

    Pairs with :func:`max_legendre_order`, which stays a statement about the
    file: this one is a statement about what can be shown.  For a tabulated
    section the two differ by exactly the point of this function -- the file
    stores nothing, and the projection can still give you :math:`a_\ell` up to
    whatever order you ask for.
    """
    ltt = int(getattr(section, "ltt", 0) or 0)
    stored = max_legendre_order(section)
    kind = _LTT_KINDS.get(ltt, "legendre" if stored else "isotropic")

    if ltt == 2:
        return LegendreProvenance(
            ltt=ltt,
            kind=kind,
            stored_max_order=0,
            max_order=int(projection_order),
            projected="all",
        )
    if ltt == 3:
        legendre_energies = getattr(section, "legendre_energies", None) or []
        boundary = float(legendre_energies[-1]) if legendre_energies else None
        return LegendreProvenance(
            ltt=ltt,
            kind=kind,
            stored_max_order=stored,
            max_order=max(stored, int(projection_order)),
            projected="above_boundary",
            boundary_energy=boundary,
        )
    # LTT=0 stores nothing but needs nothing: f(mu) = 1/2 is a_0 = 1 exactly,
    # and every higher order is zero rather than unknown.
    return LegendreProvenance(
        ltt=ltt,
        kind=kind,
        stored_max_order=stored,
        max_order=stored,
        projected="none",
    )


def _as_coefficient_matrix(
    coefficients: Union[Mapping[Union[int, str], Sequence[float]], np.ndarray],
    n_energies: int,
    max_order: Optional[int] = None,
) -> np.ndarray:
    """Normalize the app's ``{order: [a_l(E)]}`` wire shape to ``(L, n_energies)``.

    Order 0 is dropped when present: MF4 normalizes :math:`a_0 = 1`, and
    :func:`angular_pdf` puts it back implicitly.  Rows are indexed by
    :math:`l - 1`, and a missing order becomes a zero row rather than shifting
    the ones above it down.
    """
    if isinstance(coefficients, Mapping):
        orders = sorted(int(k) for k in coefficients.keys())
        orders = [o for o in orders if o >= 1]
        if max_order is not None:
            orders = [o for o in orders if o <= max_order]
        if not orders:
            return np.zeros((0, n_energies), dtype=float)
        top = max(orders)
        matrix = np.zeros((top, n_energies), dtype=float)
        for o in orders:
            row = np.asarray(
                coefficients.get(o, coefficients.get(str(o))), dtype=float
            )
            if row.size != n_energies:
                raise ValueError(
                    f"order {o} has {row.size} values but the energy grid has {n_energies}"
                )
            matrix[o - 1] = row
        return matrix

    matrix = np.atleast_2d(np.asarray(coefficients, dtype=float))
    if matrix.shape[1] != n_energies:
        raise ValueError(
            f"coefficient matrix is {matrix.shape} but the energy grid has {n_energies} points"
        )
    if max_order is not None:
        matrix = matrix[:max_order]
    return matrix


def coefficients_at_energies(
    energies_ev: Sequence[float],
    coefficients: Union[Mapping[Union[int, str], Sequence[float]], np.ndarray],
    query_ev: Union[float, Sequence[float]],
    *,
    nbt_int_pairs: Optional[Sequence[Tuple[int, int]]] = None,
    max_order: Optional[int] = None,
) -> np.ndarray:
    r"""Interpolate :math:`a_l(E)` onto ``query_ev``, honouring the MF4 law.

    ``nbt_int_pairs`` are the MF4 ``(NBT, INT)`` regions for the incident-energy
    grid.  When omitted the ENDF default of lin-lin (INT=2) over the whole grid
    is assumed — which is what the app's client-side kernel implicitly did by
    only ever evaluating *at* grid points.  Outside the tabulated range the edge
    value is held; callers that would rather show a gap should mask on
    ``energies_ev[0] / [-1]`` themselves.

    Returns
    -------
    np.ndarray
        Shape ``(L, len(query_ev))``, rows indexed by :math:`l - 1`.
    """
    grid = np.asarray(energies_ev, dtype=float)
    matrix = _as_coefficient_matrix(coefficients, grid.size, max_order)
    q = np.atleast_1d(np.asarray(query_ev, dtype=float))
    if matrix.shape[0] == 0:
        return np.zeros((0, q.size), dtype=float)

    pairs = list(nbt_int_pairs) if nbt_int_pairs else [(grid.size, 2)]
    out = np.empty((matrix.shape[0], q.size), dtype=float)
    for i, row in enumerate(matrix):
        out[i] = np.atleast_1d(
            interpolate_1d(grid, row, pairs, q, out_of_range="hold")
        )
    return out


# =============================================================================
# Plot products
# =============================================================================

def coefficients_folded(
    energies_ev,
    coefficients,
    query_ev,
    tof: TofResolution,
    *,
    nbt_int_pairs=None,
    max_order: Optional[int] = None,
) -> np.ndarray:
    r"""TOF-resolution-folded :math:`a_\ell(E)`.

    The angular counterpart of :func:`sigma_folded`: each coefficient is
    averaged over the same Gaussian energy-resolution kernel, on the same
    Gauss-Hermite quadrature.

    Why this exists.  Folding only :math:`\sigma` gives
    :math:`\langle\sigma\rangle\,f(\mu, E)` — the measured normalization with
    the evaluated *shape*.  A detector integrating over its own resolution sees
    the shape smeared too, and where the angular distribution changes quickly
    with energy the difference is not small.

    ``a_0 = 1`` identically, so it is neither stored nor folded; ``coefficients``
    is :math:`a_1 \ldots a_L` per energy, exactly as
    :func:`coefficients_at_energies` returns them.

    Returns an ``(L, nq)`` array.
    """
    grid = np.asarray(energies_ev, dtype=float)
    matrix = np.atleast_2d(np.asarray(coefficients, dtype=float))
    q = np.atleast_1d(np.asarray(query_ev, dtype=float))
    if max_order is not None:
        matrix = matrix[:max_order]

    sigma_e_ev = np.asarray(tof.sigma_e_mev(q / 1e6), dtype=float) * 1e6
    out = np.empty((matrix.shape[0], q.size), dtype=float)
    for i, row in enumerate(matrix):
        out[i] = np.atleast_1d(fold_tabulated(grid, row, q, sigma_e_ev))
    return out


def coefficients_bin_averaged(
    energies_ev,
    coefficients,
    bin_edges: Sequence[Tuple[float, float]],
    *,
    weighting: Weighting = "lethargy",
    max_order: Optional[int] = None,
) -> np.ndarray:
    r"""Flux-averaged :math:`a_\ell` over one window per query point.

    The angular counterpart of :func:`sigma_bin_averaged`, and averaged the same
    way — trapezoidal over the window edges plus every grid point strictly
    inside, under the same flux weight.

    ``bin_edges`` is one ``(lo, hi)`` per output point.  Returns an
    ``(L, len(bin_edges))`` array.
    """
    grid = np.asarray(energies_ev, dtype=float)
    matrix = np.atleast_2d(np.asarray(coefficients, dtype=float))
    if max_order is not None:
        matrix = matrix[:max_order]

    edges = list(bin_edges)
    out = np.empty((matrix.shape[0], len(edges)), dtype=float)
    for i, row in enumerate(matrix):
        for j, (lo, hi) in enumerate(edges):
            out[i, j] = sigma_bin_averaged(grid, row, lo, hi, weighting)
    return out


def resolve_coefficients(
    energies_ev,
    coefficients,
    query_ev,
    *,
    mode: XsMode = "nominal",
    bin_edges: Optional[Sequence[Tuple[float, float]]] = None,
    tof: Optional[TofResolution] = None,
    weighting: Weighting = "lethargy",
    nbt_int_pairs=None,
    max_order: Optional[int] = None,
) -> np.ndarray:
    r"""Dispatch to the requested reading of :math:`a_\ell(E)`.

    Deliberately the same three modes, the same argument names and the same
    fallback behaviour as :func:`resolve_sigma`, because the two are meant to
    be chosen independently:

    ======================  ======================  ==================================
    :math:`\sigma` mode     :math:`a_\ell` mode     result
    ======================  ======================  ==================================
    ``nominal``             ``nominal``             the evaluation as written
    ``folded``              ``nominal``             measured normalization, evaluated shape
    ``nominal``             ``folded``              evaluated normalization, smeared shape
    ``folded``              ``folded``              both, the *factor average*
    ======================  ======================  ==================================

    The last row is :math:`\langle\sigma\rangle\,F(\langle a_\ell\rangle)`, not
    :math:`\langle\sigma\,F(a_\ell)\rangle`.  The two differ by the
    :math:`\mathrm{Cov}(\sigma, a_\ell)` term across the window, and this module
    follows the factor average because MF3 :math:`\sigma` and MF4
    :math:`a_\ell` come from different underlying data — a product fold would
    impose a point-by-point energy correlation the evaluation does not carry.
    That is the same choice ``scripts/precompute_chi2_folded_al_c0.py`` makes,
    and the two must not disagree.
    """
    if mode == "binavg" and bin_edges is not None:
        return coefficients_bin_averaged(
            energies_ev, coefficients, bin_edges, weighting=weighting, max_order=max_order
        )
    if mode == "folded" and tof is not None:
        return coefficients_folded(
            energies_ev, coefficients, query_ev, tof,
            nbt_int_pairs=nbt_int_pairs, max_order=max_order,
        )
    return coefficients_at_energies(
        energies_ev, coefficients, query_ev,
        nbt_int_pairs=nbt_int_pairs, max_order=max_order,
    )


def differential_xs_factor(sigma, per_steradian: bool):
    r"""Scale turning :math:`f(\mu)` into a differential cross section.

    :math:`d\sigma/d\Omega = \sigma/(2\pi)\,f(\mu)` [barn/sr] when
    ``per_steradian``, else :math:`d\sigma/d\mu = \sigma\,f(\mu)` [barn].
    Both follow from :math:`\int f\,d\mu = 1`.
    """
    return (np.asarray(sigma, dtype=float) / (2.0 * np.pi)) if per_steradian else np.asarray(sigma, dtype=float)


def differential_xs_vs_angle(
    mu: Sequence[float],
    a_coeffs: Optional[Sequence[float]] = None,
    sigma: Optional[float] = None,
    *,
    pdf_values: Optional[Sequence[float]] = None,
    per_steradian: bool = True,
    alpha: float = 0.0,
    native_frame: Frame = "cm",
    output_frame: Optional[Frame] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    r""":math:`d\sigma/d\Omega` (or the bare PDF) against :math:`\mu`.

    ``mu`` is interpreted in ``native_frame`` — the frame the MF4 coefficients
    are given in, from the section's LCT flag.  With ``sigma=None`` the
    normalized PDF is returned instead, which is the panel's "PDF" view.

    When ``output_frame`` differs from ``native_frame`` the curve is moved with
    :func:`transform_angular_curve` (elastic only), so the returned cosine grid
    is non-uniform and is returned alongside the values.

    ``pdf_values`` short-circuits the reconstruction: a section that stores
    :math:`f(\mu)` as a table can hand its own values in and skip the round
    trip through a truncated expansion.  Everything after that point -- the
    cross section factor, the frame transform, the units -- is a property of
    the view and not of the representation, which is why the two paths meet
    here rather than in two copies of this function.
    """
    mu = np.asarray(mu, dtype=float)
    if pdf_values is not None:
        values = np.asarray(pdf_values, dtype=float)
        if values.shape != mu.shape:
            raise ValueError(
                f"pdf_values has shape {values.shape}, expected {mu.shape} to match mu"
            )
    elif a_coeffs is not None:
        values = angular_pdf(mu, a_coeffs)
    else:
        raise ValueError("pass either a_coeffs or pdf_values")
    if sigma is not None:
        values = values * differential_xs_factor(sigma, per_steradian)

    if output_frame is not None and output_frame != native_frame and alpha > 0:
        direction = "cm2lab" if native_frame == "cm" else "lab2cm"
        return transform_angular_curve(mu, values, alpha, direction)
    return mu, values


def differential_xs_vs_energy(
    *,
    energies_ev: Sequence[float],
    coefficients: Union[Mapping[Union[int, str], Sequence[float]], np.ndarray],
    mu: float,
    xs_energies_ev: Optional[Sequence[float]] = None,
    xs_values: Optional[Sequence[float]] = None,
    mu_frame: Frame = "lab",
    native_frame: Frame = "cm",
    alpha: float = 0.0,
    per_steradian: bool = True,
    xs_mode: XsMode = "nominal",
    tof: Optional[TofResolution] = None,
    weighting: Weighting = "lethargy",
    query_energies_ev: Optional[Sequence[float]] = None,
    nbt_int_pairs: Optional[Sequence[Tuple[int, int]]] = None,
    max_order: Optional[int] = None,
    pdf_at_energies: Optional[Callable[[float, np.ndarray], np.ndarray]] = None,
) -> dict:
    r""":math:`d\sigma/d\Omega` at a **fixed angle**, against incident energy.

    The complement of :func:`differential_xs_vs_angle`: pin :math:`\mu` and
    sweep :math:`E`.  Because the elastic frame relation depends only on the
    mass ratio and not on energy, a fixed ``mu`` in ``mu_frame`` maps to a
    single fixed cosine in ``native_frame``, evaluated once up front.

    Parameters
    ----------
    energies_ev, coefficients
        The MF4 incident-energy grid and its :math:`a_l(E)`, in the app's
        ``{order: [...]}`` wire shape or as an ``(L, n_energies)`` matrix.
    mu
        Target cosine, expressed in ``mu_frame``.
    xs_energies_ev, xs_values
        MF3 :math:`\sigma(E)`.  Omit both to get the bare PDF :math:`f(\mu, E)`,
        which is the useful view when only the *shape*'s energy dependence
        matters.
    query_energies_ev
        Output grid.  Defaults to the MF4 grid; a denser grid is interpolated
        through :func:`coefficients_at_energies` under ``nbt_int_pairs`` and
        clipped to the MF4 range.
    pdf_at_energies
        ``f(mu_native, energies) -> array``, for a section that stores the
        distribution itself and would otherwise be routed through a truncated
        expansion.  Given as a callable rather than an array because the output
        grid is decided here -- it is clipped to where MF4 has data -- so only
        this function knows the energies the values have to line up with.
        Everything downstream (the sigma modes, the frame jacobian, the units)
        is the same either way, which is why both representations meet here.

    Returns
    -------
    dict
        ``energies`` (eV), ``values``, ``mu_native``/``mu_requested``,
        ``jacobian``, ``y_unit``, and ``sigma`` — the :math:`\sigma(E)` actually
        used, so a caller can show what the reconstruction mode did.
    """
    grid = np.asarray(energies_ev, dtype=float)
    if grid.size == 0:
        raise ValueError("energies_ev is empty")

    # 1. Fixed angle: map once into the frame the coefficients live in.
    jacobian = 1.0
    mu_native = float(mu)
    if mu_frame != native_frame and alpha > 0:
        if native_frame == "cm":
            # `mu` is a LAB cosine; evaluate the CM-frame expansion at the
            # matching CM cosine, then push the density out with y_L = y_C · J.
            mu_native = float(cos_cm_from_cos_lab(mu, alpha))
            jacobian = float(jacobian_cm_to_lab(mu_native, alpha))
        else:
            # `mu` is already the CM cosine; the expansion is in LAB, so
            # evaluate there and pull the density back with y_C = y_L / J.
            mu_native = float(cos_lab_from_cos_cm(mu, alpha))
            jacobian = 1.0 / float(jacobian_cm_to_lab(float(mu), alpha))

    # 2. Output grid, clipped to where MF4 actually has data.
    if query_energies_ev is None:
        out_e = grid.copy()
    else:
        out_e = np.asarray(query_energies_ev, dtype=float)
        out_e = out_e[(out_e >= grid[0]) & (out_e <= grid[-1])]

    # 3. f(mu_native, E) for every energy.
    if pdf_at_energies is not None:
        # The section holds the distribution; ask it, and never build the
        # expansion at all.
        pdf = np.asarray(pdf_at_energies(mu_native, out_e), dtype=float).ravel()
        if pdf.size != out_e.size:
            raise ValueError(
                f"pdf_at_energies returned {pdf.size} values for {out_e.size} energies"
            )
    else:
        # One Legendre basis, reused across the sweep.
        if query_energies_ev is None:
            coeffs = _as_coefficient_matrix(coefficients, grid.size, max_order)
        else:
            coeffs = coefficients_at_energies(
                grid, coefficients, out_e, nbt_int_pairs=nbt_int_pairs, max_order=max_order
            )
        basis = legendre_basis(np.array([mu_native]), coeffs.shape[0])[:, 0]
        orders = np.arange(1, coeffs.shape[0] + 1)
        pdf = 0.5 * basis[0] + 0.5 * ((2 * orders + 1) * basis[1:]) @ coeffs

    # 4. sigma(E) under the requested reconstruction mode.
    sigma = None
    if xs_energies_ev is not None and xs_values is not None and len(xs_values):
        if xs_mode == "binavg":
            # Bin edges come from the MF4 grid, so each output energy borrows
            # the bin of the MF4 point it falls in.
            idx = np.clip(np.searchsorted(grid, out_e), 0, grid.size - 1)
            sigma = np.array([
                sigma_bin_averaged(
                    xs_energies_ev, xs_values, *bin_edges_for_energy(grid, int(i)), weighting
                )
                for i in idx
            ])
        elif xs_mode == "folded" and tof is not None:
            sigma = np.atleast_1d(sigma_folded(xs_energies_ev, xs_values, out_e, tof))
        else:
            sigma = np.atleast_1d(sigma_nominal(xs_energies_ev, xs_values, out_e))

    if sigma is None:
        values = pdf
        y_unit = "Probability Density"
    else:
        values = pdf * differential_xs_factor(sigma, per_steradian)
        y_unit = "barn/sr" if per_steradian else "barn"

    values = values * jacobian

    return {
        "energies": out_e,
        "values": values,
        "sigma": sigma,
        "mu_requested": float(mu),
        "mu_native": mu_native,
        "jacobian": jacobian,
        "frame": mu_frame,
        "native_frame": native_frame,
        "y_unit": y_unit,
        "xs_mode": xs_mode,
    }
