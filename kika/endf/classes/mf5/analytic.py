"""The MF5 laws given by parameters rather than by a table (ENDF-6 §5.1.1).

LF=1 writes chi(E->E') out point by point; LF=5, 7, 9 and 11 write a handful of
energy-dependent parameters and leave the shape to a formula. This module reads
those four.

**Decoded for reading, emitted from bytes.** Every class here subclasses
:class:`~kika.endf.classes.mf5.partials.MF5PartialRaw` and so keeps
``raw_lines`` and inherits its :meth:`emit`. The decode is purely additive: the
records that go back onto the tape are the records that came off it, which is
what makes this change unable to disturb the byte gate in
``test_mf5_roundtrip.py``. Reconstructing the TAB1s on the way out would have
re-opened the ``format_interp_pairs`` padding defect that ENDF/B-VIII.1 already
pins as a strict xfail.

**One normalisation rule, not four.** ENDF-6 §5.1 requires
``int_0^(E-U) f(E->E') dE' = 1`` of every law, so each class here states only an
unnormalised *shape* and the closed form of ``I(E) = int_0^(E-U) shape dE'``,
and the division happens once in the base class. That is not tidiness: it is
what makes LF=5 unambiguous. The manual writes LF=5 as ``f = g(E'/theta)``, and
that reading cannot hold ``int f dE' = 1`` for all E when a single g is paired
with an energy-dependent theta -- ``f = g(x)/theta`` is the reading that can.
Normalising by the measured integral of the shape satisfies §5.1 under either,
and reduces to ``f = g`` exactly on the one witness available here (Cf-252
MT455, where theta == 1 and ``int g dx == 1``), so nothing has to be guessed.

**LF=12 (Madland-Nix) is not here.** It needs a numerical double integral and
there is no tape on this machine to close it against; a plausible implementation
would be silently wrong, which is worse than none. It stays an
:class:`MF5PartialRaw` and ``report_gaps`` says so.

**What is witnessed and what is not.** LF=5 is read off a committed fixture
(``micro_cf252_pfns.endf``, MT455, six subsections). LF=7, 9 and 11 have no tape
here: their record *layout* is the one the walker has always used, and their
formulae are gated against numerical quadrature of their own shape, which
catches a typo in either the shape or its normalisation constant but cannot
catch a misreading of the manual. See ``test_mf5_analytic.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ...utils import interp_energy_values, segment_int_codes
from .partials import (
    MF5PartialRaw,
    cumulative_integral,
    evaluate_table,
    exact_segment_codes,
    integral_to,
)

#: Points in the display grid a law builds for itself when the caller does not
#: supply one. Only ever a rendering choice -- an analytic law has no grid of
#: its own, so no result depends on this beyond how smooth the curve looks.
DEFAULT_GRID_POINTS = 400

#: How far below the upper bound the default log grid starts. Nine decades is
#: enough to show a Maxwellian's rise and cheap enough not to think about.
_DEFAULT_GRID_DECADES = 9.0


def tab1_at(x: Sequence[float], y: Sequence[float],
            interp: Sequence[Tuple[int, int]], at: float) -> float:
    """A TAB1 evaluated at one abscissa, under its own INT codes.

    Uses the library's generic ENDF interpolators rather than the exact-panel
    pair in :mod:`~kika.endf.classes.mf5.partials`, and the difference is
    deliberate: theta(E), a(E) and b(E) are only ever *evaluated*, so all five
    INT codes are fine, while g(x) is *integrated* and so is held to the two
    codes that integrate exactly.
    """
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    if xs.size == 0:
        raise ValueError("cannot evaluate an empty TAB1")
    if xs.size == 1 or at <= xs[0]:
        return float(ys[0])
    if at >= xs[-1]:
        return float(ys[-1])

    codes = segment_int_codes(xs.size, list(interp))
    i = int(np.searchsorted(xs, at, side="right")) - 1
    i = min(max(i, 0), xs.size - 2)
    value = interp_energy_values(
        float(xs[i]), np.array([ys[i]]),
        float(xs[i + 1]), np.array([ys[i + 1]]),
        float(at), int(codes[i]),
    )
    return float(value[0])


@dataclass
class MF5PartialAnalytic(MF5PartialRaw):
    """Base of the laws stated by parameters: shape over its own integral.

    Subclasses supply :meth:`shape` and :meth:`normalisation_at`; everything a
    caller touches is here, and has the same signatures as
    :class:`~kika.endf.classes.mf5.partials.MF5PartialTabulated` so that code
    drawing a spectrum never has to ask which law it is holding.
    """

    @property
    def is_decoded(self) -> bool:
        return True

    # ------------------------------------------------------------------
    def upper_bound(self, energy: float) -> float:
        """``E - U``: the largest outgoing energy this law admits.

        ``U`` is often negative -- Cf-252's delayed spectra write -30 MeV -- so
        this is not a truncation of the incident energy but a genuine bound of
        its own.
        """
        return float(energy) - float(self.u)

    def shape(self, energy: float, e_out: np.ndarray) -> np.ndarray:
        """The law's unnormalised shape at outgoing energies *e_out*."""
        raise NotImplementedError

    def normalisation_at(self, energy: float) -> float:
        """``I(E) = int_0^(E-U) shape dE'`` -- in closed form where there is one."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    def evaluate_on_grid(self, energy: float,
                         points: Sequence[float]) -> np.ndarray:
        """chi(E->E') at *energy*, on a grid the caller chose.

        Zero outside ``[0, E - U]``: the bound is part of the law, not a
        plotting range, and a Maxwellian continued past it would integrate to
        more than one.
        """
        points = np.asarray(points, dtype=float)
        hi = self.upper_bound(energy)
        out = np.zeros(points.shape, dtype=float)
        if hi <= 0.0:
            return out
        norm = self.normalisation_at(energy)
        if not np.isfinite(norm) or norm <= 0.0:
            return out
        inside = (points >= 0.0) & (points <= hi)
        if np.any(inside):
            out[inside] = self.shape(energy, points[inside]) / norm
        return out

    def default_grid(self, energy: float,
                     n_points: int = DEFAULT_GRID_POINTS) -> np.ndarray:
        """A display grid over ``[0, E - U]``, log-spaced with 0 kept.

        Log because every consumer of a fission spectrum plots it log-log, and
        the interesting decade is the low-energy rise, which a linear grid of
        the same size renders as a single segment.
        """
        hi = self.upper_bound(energy)
        if hi <= 0.0:
            return np.zeros(0, dtype=float)
        lo = hi * 10.0 ** (-_DEFAULT_GRID_DECADES)
        return np.concatenate(([0.0], np.geomspace(lo, hi, max(n_points - 1, 2))))

    def evaluate_at_incident(
        self, energy: float, e_out: Optional[Sequence[float]] = None,
        n_points: int = DEFAULT_GRID_POINTS,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """``(E' grid, chi)`` at *energy*.

        Same return shape as
        :meth:`~kika.endf.classes.mf5.partials.MF5PartialTabulated.evaluate_at_incident`.
        The extra arguments are optional because an analytic law has no grid of
        its own to hand back, so one has to be chosen; a caller with a grid in
        mind passes it.
        """
        grid = (self.default_grid(energy, n_points) if e_out is None
                else np.asarray(e_out, dtype=float))
        return grid, self.evaluate_on_grid(energy, grid)

    def normalisation(self, energy: float) -> float:
        """``int chi dE'`` over ``[0, E - U]`` -- 1 by construction.

        Kept, rather than dropped as a tautology, because it is the cheap check
        that the closed form in :meth:`normalisation_at` and the shape it is
        supposed to normalise have not drifted apart. Anything displaying it
        next to an LF=1 partial's own normalisation is comparing like with like.
        """
        hi = self.upper_bound(energy)
        if hi <= 0.0:
            return 0.0
        norm = self.normalisation_at(energy)
        return 1.0 if np.isfinite(norm) and norm > 0.0 else 0.0

    def normalisation_at_incident(self, energy: float) -> float:
        """The same number as :meth:`normalisation`, under the name LF=1 uses.

        A caller holding a subsection should not have to ask which law it is to
        ask what it integrates to. The tabulated partial needs two methods --
        one by node index for the sampler, one by energy -- and this is the
        second name, so the by-energy one is spelled the same on both.
        """
        return self.normalisation(energy)


@dataclass
class MF5GeneralEvaporation(MF5PartialAnalytic):
    """LF=5: a tabulated shape ``g(x)`` in the reduced variable ``x = E'/theta(E)``."""

    lf: int = 5
    theta_interp: List[Tuple[int, int]] = field(default_factory=list)
    theta_energies: List[float] = field(default_factory=list)
    theta_values: List[float] = field(default_factory=list)
    g_interp: List[Tuple[int, int]] = field(default_factory=list)
    g_x: List[float] = field(default_factory=list)
    g_values: List[float] = field(default_factory=list)

    def theta(self, energy: float) -> float:
        return tab1_at(self.theta_energies, self.theta_values,
                       self.theta_interp, energy)

    def _g_codes(self) -> np.ndarray:
        return exact_segment_codes(self.g_x, self.g_interp, "MF5 LF=5 g(x)")

    def shape(self, energy: float, e_out: np.ndarray) -> np.ndarray:
        theta = self.theta(energy)
        if theta <= 0.0:
            return np.zeros(np.shape(e_out), dtype=float)
        x = np.asarray(e_out, dtype=float) / theta
        gx = np.asarray(self.g_x, dtype=float)
        gy = np.asarray(self.g_values, dtype=float)
        if gx.size < 2:
            return np.zeros(np.shape(e_out), dtype=float)
        return evaluate_table(gx, gy, self._g_codes(), x) / theta

    def normalisation_at(self, energy: float) -> float:
        """``int_0^(E-U) g(E'/theta)/theta dE'`` = ``int_0^{(E-U)/theta} g dx``.

        Integrated exactly over g's own panels rather than by quadrature: g is
        histogram-interpolated on the reference tape, and a trapezoid over a
        histogram is simply a different number.
        """
        theta = self.theta(energy)
        hi = self.upper_bound(energy)
        if theta <= 0.0 or hi <= 0.0:
            return 0.0
        gx = np.asarray(self.g_x, dtype=float)
        gy = np.asarray(self.g_values, dtype=float)
        if gx.size < 2:
            return 0.0
        codes = self._g_codes()
        cumulative = cumulative_integral(gx, gy, codes)
        return integral_to(gx, gy, codes, cumulative, hi / theta)

    def default_grid(self, energy: float,
                     n_points: int = DEFAULT_GRID_POINTS) -> np.ndarray:
        """``theta * x`` over g's own abscissae -- the law's exact break points.

        Overridden because unlike the closed-form laws this one *does* have a
        natural grid, and sampling it anywhere else would round its corners.
        """
        theta = self.theta(energy)
        hi = self.upper_bound(energy)
        if theta <= 0.0 or hi <= 0.0:
            return np.zeros(0, dtype=float)
        full = np.asarray(self.g_x, dtype=float) * theta
        grid = full[full <= hi]
        if full.size and full[-1] > hi:
            # The bound cuts the table: keep it, so the last panel is the
            # partial one the law actually has.
            grid = np.concatenate((grid, [hi]))
        elif grid.size == 0:
            grid = np.array([0.0, hi])
        # When g's support ends *before* the bound the grid stops there, and
        # deliberately so. Reaching on to ``hi`` would add one panel spanning
        # everything in between, and under the histogram convention that panel
        # holds the last tabulated value rather than zero -- on Cf-252 MT455
        # that single stretch is worth 3e-4 of a spectrum that integrates to 1.
        return grid

    def describe(self) -> str:
        return (f"LF=5, general evaporation, {len(self.g_x)}-point g(x), "
                f"{len(self.theta_energies)}-point theta(E)")


@dataclass
class MF5Maxwellian(MF5PartialAnalytic):
    """LF=7: ``sqrt(E') exp(-E'/theta(E))``, the simple Maxwellian fission spectrum."""

    lf: int = 7
    theta_interp: List[Tuple[int, int]] = field(default_factory=list)
    theta_energies: List[float] = field(default_factory=list)
    theta_values: List[float] = field(default_factory=list)

    def theta(self, energy: float) -> float:
        return tab1_at(self.theta_energies, self.theta_values,
                       self.theta_interp, energy)

    def shape(self, energy: float, e_out: np.ndarray) -> np.ndarray:
        theta = self.theta(energy)
        if theta <= 0.0:
            return np.zeros(np.shape(e_out), dtype=float)
        e_out = np.asarray(e_out, dtype=float)
        return np.sqrt(e_out) * np.exp(-e_out / theta)

    def normalisation_at(self, energy: float) -> float:
        """``theta^(3/2) [sqrt(pi)/2 erf(sqrt(y)) - sqrt(y) exp(-y)]``, ``y = (E-U)/theta``."""
        theta = self.theta(energy)
        hi = self.upper_bound(energy)
        if theta <= 0.0 or hi <= 0.0:
            return 0.0
        y = hi / theta
        return theta ** 1.5 * (0.5 * math.sqrt(math.pi) * math.erf(math.sqrt(y))
                               - math.sqrt(y) * math.exp(-y))

    def describe(self) -> str:
        return (f"LF=7, simple Maxwellian, "
                f"{len(self.theta_energies)}-point theta(E)")


@dataclass
class MF5Evaporation(MF5PartialAnalytic):
    """LF=9: ``E' exp(-E'/theta(E))``, the evaporation spectrum."""

    lf: int = 9
    theta_interp: List[Tuple[int, int]] = field(default_factory=list)
    theta_energies: List[float] = field(default_factory=list)
    theta_values: List[float] = field(default_factory=list)

    def theta(self, energy: float) -> float:
        return tab1_at(self.theta_energies, self.theta_values,
                       self.theta_interp, energy)

    def shape(self, energy: float, e_out: np.ndarray) -> np.ndarray:
        theta = self.theta(energy)
        if theta <= 0.0:
            return np.zeros(np.shape(e_out), dtype=float)
        e_out = np.asarray(e_out, dtype=float)
        return e_out * np.exp(-e_out / theta)

    def normalisation_at(self, energy: float) -> float:
        """``theta^2 [1 - exp(-y)(1 + y)]``, ``y = (E-U)/theta``."""
        theta = self.theta(energy)
        hi = self.upper_bound(energy)
        if theta <= 0.0 or hi <= 0.0:
            return 0.0
        y = hi / theta
        return theta * theta * (1.0 - math.exp(-y) * (1.0 + y))

    def describe(self) -> str:
        return (f"LF=9, evaporation, "
                f"{len(self.theta_energies)}-point theta(E)")


@dataclass
class MF5Watt(MF5PartialAnalytic):
    """LF=11: ``exp(-E'/a(E)) sinh(sqrt(b(E) E'))``, the energy-dependent Watt."""

    lf: int = 11
    a_interp: List[Tuple[int, int]] = field(default_factory=list)
    a_energies: List[float] = field(default_factory=list)
    a_values: List[float] = field(default_factory=list)
    b_interp: List[Tuple[int, int]] = field(default_factory=list)
    b_energies: List[float] = field(default_factory=list)
    b_values: List[float] = field(default_factory=list)

    def a(self, energy: float) -> float:
        return tab1_at(self.a_energies, self.a_values, self.a_interp, energy)

    def b(self, energy: float) -> float:
        return tab1_at(self.b_energies, self.b_values, self.b_interp, energy)

    def shape(self, energy: float, e_out: np.ndarray) -> np.ndarray:
        a, b = self.a(energy), self.b(energy)
        if a <= 0.0 or b < 0.0:
            return np.zeros(np.shape(e_out), dtype=float)
        e_out = np.asarray(e_out, dtype=float)
        return np.exp(-e_out / a) * np.sinh(np.sqrt(b * e_out))

    def normalisation_at(self, energy: float) -> float:
        """ENDF-6 §5.1.1.5's closed form for the Watt integral over ``[0, E-U]``."""
        a, b = self.a(energy), self.b(energy)
        hi = self.upper_bound(energy)
        if a <= 0.0 or b < 0.0 or hi <= 0.0:
            return 0.0
        root = math.sqrt(a * b / 4.0)
        y = math.sqrt(hi / a)
        return (0.5 * math.sqrt(math.pi * b * a ** 3 / 4.0)
                * math.exp(a * b / 4.0)
                * (math.erf(y - root) + math.erf(y + root))
                - a * math.exp(-hi / a) * math.sinh(math.sqrt(b * hi)))

    def describe(self) -> str:
        return (f"LF=11, energy-dependent Watt, "
                f"{len(self.a_energies)}-point a(E), "
                f"{len(self.b_energies)}-point b(E)")


#: ``LF`` -> the class that reads it. A law absent from here keeps its bytes and
#: is reported as a gap; see this module's docstring on LF=12.
ANALYTIC_LAWS = {
    5: MF5GeneralEvaporation,
    7: MF5Maxwellian,
    9: MF5Evaporation,
    11: MF5Watt,
}

#: ``LF`` -> the ``(interp, abscissa, ordinate)`` field names of each TAB1 record
#: after the subsection header, **in tape order**.
#:
#: Separate from :data:`ANALYTIC_LAWS` because it says something the classes do
#: not: which record is which. LF=11 writes ``a(E)`` then ``b(E)`` and swapping
#: them produces a spectrum that is wrong and looks plausible, so the order is
#: written down once, here, and
#: ``test_analytic_record_names_match_the_walker`` holds it to the same counts
#: the walker in :data:`~kika.endf.classes.mf5.partials.TAB1_RECORDS_AFTER_HEADER`
#: uses.
ANALYTIC_RECORDS = {
    5: (("theta_interp", "theta_energies", "theta_values"),
        ("g_interp", "g_x", "g_values")),
    7: (("theta_interp", "theta_energies", "theta_values"),),
    9: (("theta_interp", "theta_energies", "theta_values"),),
    11: (("a_interp", "a_energies", "a_values"),
         ("b_interp", "b_energies", "b_values")),
}
