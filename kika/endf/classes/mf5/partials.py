"""The subsections of an MF5 section: one energy-distribution law each.

**Why there are three classes and not one per LF.** ENDF-6 §5 gives MF5 seven
laws. Exactly one of them — LF=1, arbitrary tabulated — is what the fission
spectra this library perturbs are written in: NK=1, LF=1, lin-lin, on both
ENDF/B-VIII.1 and JEFF-4.0 U-235 and on ENDF/B-VIII.1 Cf-252. So LF=1 is
modelled in full and every other law is kept **verbatim**, as the source bytes
of columns 1-66.

That is not a shortcut, it is the correct trade. A raw partial round-trips
byte-for-byte for free, which buys MT455 (LF=5 in ENDF/B-VIII.1) at no cost;
and it answers the case that decides whether the pipeline is safe on a tape
nobody has read yet — an NK>1 section where only one partial is tabulated. That
one is perturbed and the rest pass through, and no LF=11 emitter ever has to be
written to make it work.

**Columns 1-66 only, and re-stamped on emit.** A verbatim partial can follow a
tabulated one that has grown by thousands of lines, and the section's sequence
numbers have to stay monotone through to the SEND record. Storing the identity
field would freeze the numbers the source happened to have. It also makes the
zero-versus-blank padding of the two tape dialects round-trip for nothing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ....processing.panel_integrals import (
    EXACT_INT_CODES,
    cumulative_integral,
    evaluate_table,
    exact_segment_codes,
    integral_to,
)
from ...utils import (
    ENDF_FORMAT_FLOAT,
    ENDF_FORMAT_INT,
    format_endf_data_line,
    format_tab1,
    format_tab2,
)

#: Interpolation codes the tabulated partial can integrate and refine exactly.
#: Kept as a name here because this module's own error messages and tests speak
#: of it; the list itself now lives with the arithmetic it constrains.
_EXACT_INT_CODES = EXACT_INT_CODES

#: How many TAB1 records follow the subsection header, per law (ENDF-6 §5.1).
#: Used only to walk past a law this module stores verbatim, so it needs the
#: record *count* and nothing about the meaning.
#:
#:   LF=5  general evaporation      theta(E), g(x)
#:   LF=7  simple Maxwellian        theta(E)
#:   LF=9  evaporation              theta(E)
#:   LF=11 energy-dependent Watt    a(E), b(E)
#:   LF=12 Madland-Nix              T_M(E)
TAB1_RECORDS_AFTER_HEADER = {5: 2, 7: 1, 9: 1, 11: 2, 12: 1}


# ---------------------------------------------------------------------------
# The exact panel arithmetic, shared -- and no longer defined here
#
# LF=1 owns a table in E' and every analytic law owns at least one table too --
# theta(E), g(x), a(E), b(E). They need the same three things: the INT code of
# every interval, an exact cumulative integral, and evaluation at arbitrary
# points. They were written at this module's scope so
# :mod:`~kika.endf.classes.mf5.analytic` could reuse them instead of growing a
# second implementation of the same integral.
#
# **The same argument then reached one layer further out.** The model's energy
# distribution node needs those integrals too -- MF35 is the covariance of the
# group-integrated probabilities of an MF5 table, so perturbing the node means
# integrating it -- and ``kika/nuclear_data`` may not import ``kika.endf``. So
# the four moved down to :mod:`kika.processing.panel_integrals`, which is the
# calculation layer both sides may read, and are imported back here under their
# original names. Nothing about them changed in the move.
# ---------------------------------------------------------------------------

@dataclass
class MF5Partial:
    """The part every law shares: ``p_k(E)``, the weight of this partial.

    ENDF-6 writes it as the subsection's own TAB1 header record, with ``U`` in
    C1 and ``LF`` in L2 — so parsing the header and parsing ``p_k(E)`` are one
    call, not two.
    """

    u: float = 0.0
    lf: int = 1
    p_interp: List[Tuple[int, int]] = field(default_factory=list)
    p_energies: List[float] = field(default_factory=list)
    p_values: List[float] = field(default_factory=list)

    def emit_header(self, mat: int, mf: int, mt: int,
                    line_num: int) -> Tuple[List[str], int]:
        """The subsection header TAB1: ``U, 0.0, 0, LF, NR, NP`` then p(E)."""
        return format_tab1(
            self.u, 0.0, 0, self.lf,
            self.p_interp, self.p_energies, self.p_values,
            mat, mf, mt, line_num,
        )

    def emit(self, mat: int, mf: int, mt: int,
             line_num: int) -> Tuple[List[str], int]:
        raise NotImplementedError(
            f"{type(self).__name__} does not know how to emit itself"
        )

    def describe(self) -> str:
        return f"LF={self.lf}"

    @property
    def is_decoded(self) -> bool:
        """Whether this class *reads* the law, or only carries its bytes.

        The flag every consumer branches on. It exists so that
        :meth:`kika.endf.classes.mf5.base.MF5MT.report_gaps` and anything
        drawing a spectrum ask the same question, rather than each testing a
        different ``isinstance`` and drifting apart the first time a law moves
        from passed-through to decoded.
        """
        return False


@dataclass
class MF5PartialRaw(MF5Partial):
    """Any law but LF=1, kept as the bytes the evaluator wrote.

    ``raw_lines`` holds columns 1-66 of every record after the subsection
    header. Nothing interprets them; they are re-stamped with the running MAT /
    MF / MT / sequence number on the way out.
    """

    raw_lines: List[str] = field(default_factory=list)

    def emit(self, mat: int, mf: int, mt: int,
             line_num: int) -> Tuple[List[str], int]:
        lines, line_num = self.emit_header(mat, mf, mt, line_num)
        for body in self.raw_lines:
            lines.append(f"{body[:66]:<66}" + f"{mat:4d}{mf:2d}{mt:3d}{line_num:5d}")
            line_num += 1
        return lines, line_num

    def describe(self) -> str:
        return f"LF={self.lf} ({len(self.raw_lines)} records kept verbatim)"


@dataclass
class MF5PartialTabulated(MF5Partial):
    """LF=1: a TAB2 over incident energy, each node carrying a TAB1 in E'.

    ``outgoing_grids`` is a list because ENDF allows — and ENDF/B-VIII.1 U-235
    uses — a different outgoing grid at every incident node. Code that assumes
    one shared grid is wrong on the reference library, which is why the grid is
    stored per node rather than once.
    """

    tab2_interp: List[Tuple[int, int]] = field(default_factory=list)
    incident_energies: List[float] = field(default_factory=list)
    outgoing_grids: List[List[float]] = field(default_factory=list)
    chi: List[List[float]] = field(default_factory=list)
    outgoing_interp: List[List[Tuple[int, int]]] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Emission
    # ------------------------------------------------------------------
    def emit(self, mat: int, mf: int, mt: int,
             line_num: int) -> Tuple[List[str], int]:
        lines, line_num = self.emit_header(mat, mf, mt, line_num)

        tab2_lines, line_num = format_tab2(
            0.0, 0.0, 0, 0, self.tab2_interp, len(self.incident_energies),
            mat, mf, mt, line_num,
        )
        lines.extend(tab2_lines)

        for k, energy in enumerate(self.incident_energies):
            sub_lines, line_num = format_tab1(
                0.0, energy, 0, 0,
                self.outgoing_interp[k], self.outgoing_grids[k], self.chi[k],
                mat, mf, mt, line_num,
            )
            lines.extend(sub_lines)

        return lines, line_num

    def describe(self) -> str:
        return (f"LF=1, {len(self.incident_energies)} incident nodes, "
                f"{sum(len(g) for g in self.outgoing_grids)} outgoing points")

    # ------------------------------------------------------------------
    # Table access
    # ------------------------------------------------------------------
    @property
    def is_decoded(self) -> bool:
        return True

    def table(self, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Outgoing grid and chi values at incident node *k*, as arrays."""
        return (np.asarray(self.outgoing_grids[k], dtype=float),
                np.asarray(self.chi[k], dtype=float))

    def _segment_codes(self, k: int) -> np.ndarray:
        """The ENDF INT code of every interval of table *k*."""
        return exact_segment_codes(
            self.outgoing_grids[k], self.outgoing_interp[k],
            f"MF5 LF=1 outgoing table {k}",
        )

    def replace_table(self, k: int, x: Sequence[float], y: Sequence[float],
                      interp: Optional[Sequence[Tuple[int, int]]] = None) -> None:
        """Overwrite incident node *k*'s outgoing table."""
        x = [float(v) for v in x]
        y = [float(v) for v in y]
        if len(x) != len(y):
            raise ValueError(f"{len(x)} outgoing energies for {len(y)} values")
        self.outgoing_grids[k] = x
        self.chi[k] = y
        self.outgoing_interp[k] = (
            [(int(a), int(b)) for a, b in interp] if interp else [(len(x), 2)]
        )

    # ------------------------------------------------------------------
    # Integrals
    # ------------------------------------------------------------------
    def _cumulative_integral(self, k: int) -> np.ndarray:
        """``[0, ∫_{x0}^{x1}, ∫_{x0}^{x2}, …]`` — exact on the stated law."""
        x, y = self.table(k)
        if x.size < 2:
            return np.zeros(x.size, dtype=float)
        return cumulative_integral(x, y, self._segment_codes(k))

    def normalisation(self, k: int) -> float:
        """``∫ chi_k(E') dE'`` over the whole table, exactly."""
        cumulative = self._cumulative_integral(k)
        return float(cumulative[-1]) if cumulative.size else 0.0

    def _integral_to(self, k: int, cumulative: np.ndarray,
                     limit: float) -> float:
        """``∫_{x0}^{limit}``, with *limit* anywhere inside or outside the table."""
        x, y = self.table(k)
        if x.size < 2:
            return 0.0
        return integral_to(x, y, self._segment_codes(k), cumulative, limit)

    def group_integrals(self, k: int, boundaries: Sequence[float]) -> np.ndarray:
        """``P_j = ∫_{g_j}^{g_j+1} chi_k dE'`` for the given boundaries.

        Exact: the limits are clipped into the table's own panels rather than
        the grid being refined first. This is what MF35's LB=7 matrix is the
        covariance *of* — see the roadmap's measured fact 4 — so it has to be
        the same integral the evaluator took, not a quadrature of it.
        """
        cumulative = self._cumulative_integral(k)
        edges = np.asarray(boundaries, dtype=float)
        totals = np.array(
            [self._integral_to(k, cumulative, edge) for edge in edges],
            dtype=float,
        )
        return np.diff(totals)

    # ------------------------------------------------------------------
    # The incident axis
    # ------------------------------------------------------------------
    def _incident_codes(self) -> np.ndarray:
        ne = len(self.incident_energies)
        pairs = self.tab2_interp or [(ne, 2)]
        codes = np.empty(max(ne - 1, 0), dtype=int)
        start = 0
        for nbt, code in pairs:
            stop = min(int(nbt) - 1, ne - 1)
            if stop > start:
                codes[start:stop] = int(code)
            start = max(start, stop)
        if start < ne - 1:
            codes[start:] = int(pairs[-1][1])
        return codes

    def evaluate_at_incident(self, energy: float) -> Tuple[np.ndarray, np.ndarray]:
        """chi at *energy*, on the union of the two bracketing outgoing grids.

        **This is an exact refinement of the ENDF interpolant, not an
        approximation**, and the PFNS sampler depends on that. With lin-lin on
        both axes, chi(E, E') is linear in E at fixed E'. Evaluating both
        bracketing tables on their union grid loses nothing (the union refines
        each, and each is lin-lin), and blending them linearly reproduces the
        interpolant at the new node exactly. Interpolating between the original
        table and the new one then agrees with the original everywhere, because
        a linear function restricted to a sub-interval is the same linear
        function.

        Copying the neighbouring table instead — the obvious shortcut — is
        about 0.1 % wrong and shows up as a central-value shift in a zero-delta
        regression run, which is precisely the bug the shift would hide.
        """
        energies = np.asarray(self.incident_energies, dtype=float)
        if energies.size == 0:
            return np.empty(0), np.empty(0)
        if energy <= energies[0]:
            return self.table(0)
        if energy >= energies[-1]:
            return self.table(energies.size - 1)

        upper = int(np.searchsorted(energies, energy, side="right"))
        lower = upper - 1
        if energies[lower] == energy:
            return self.table(lower)

        code = int(self._incident_codes()[lower])
        if code not in (1, 2):
            raise NotImplementedError(
                f"MF5 LF=1 incident interpolation code {code} between "
                f"{energies[lower]:.6e} and {energies[upper]:.6e}; only "
                f"histogram and lin-lin refine exactly"
            )

        x_lo, y_lo = self.table(lower)
        if code == 1:                            # histogram: hold the lower table
            return x_lo.copy(), y_lo.copy()

        x_hi, y_hi = self.table(upper)
        union = np.union1d(x_lo, x_hi)
        f_lo = self._interpolate_table(lower, union)
        f_hi = self._interpolate_table(upper, union)
        weight = (energy - energies[lower]) / (energies[upper] - energies[lower])
        return union, f_lo + weight * (f_hi - f_lo)

    def normalisation_at_incident(self, energy: float) -> float:
        """``int chi dE'`` at any incident energy, exactly.

        :meth:`normalisation` takes a node index and is what the sampler uses.
        This takes an energy, and is not a quadrature of the curve
        :meth:`evaluate_at_incident` returns: with lin-lin on the incident axis
        the interpolant is ``(1-w) chi_lo + w chi_hi`` at every fixed E', so its
        integral is the same blend of the two nodes' integrals. Each of those is
        exact on its own panels, so the blend is too.

        It exists because integrating the returned curve is *not* the same
        number. A histogram-interpolated table integrates by left rectangles,
        and a trapezoid over the same points comes out a few parts in a
        thousand short -- which is exactly the size of a real normalisation
        defect, so a display that used one could not be told from a file with a
        problem.
        """
        energies = np.asarray(self.incident_energies, dtype=float)
        if energies.size == 0:
            return 0.0
        if energy <= energies[0]:
            return self.normalisation(0)
        if energy >= energies[-1]:
            return self.normalisation(energies.size - 1)

        upper = int(np.searchsorted(energies, energy, side="right"))
        lower = upper - 1
        if energies[lower] == energy:
            return self.normalisation(lower)

        code = int(self._incident_codes()[lower])
        if code == 1:                            # histogram: hold the lower node
            return self.normalisation(lower)
        if code != 2:
            raise NotImplementedError(
                f"MF5 LF=1 incident interpolation code {code} between "
                f"{energies[lower]:.6e} and {energies[upper]:.6e}; only "
                f"histogram and lin-lin refine exactly"
            )
        weight = (energy - energies[lower]) / (energies[upper] - energies[lower])
        return ((1.0 - weight) * self.normalisation(lower)
                + weight * self.normalisation(upper))

    def evaluate_on_grid(self, energy: float, points: Sequence[float]) -> np.ndarray:
        """chi at *energy*, evaluated on a grid the caller chose.

        **Exact, and by a shorter route than :meth:`evaluate_at_incident`.**
        That method has to return a *table*, so it must first build a grid the
        blended interpolant is piecewise-linear on -- the union of the two
        bracketing grids. Here the abscissae are given, so the union is not
        needed: evaluating each bracketing table at the point and blending the
        two values *is* the definition of the lin-lin incident interpolant at
        fixed E'.

        It exists because a caller comparing several incident energies wants
        them on one shared axis -- a plot's data table, a CSV -- and doing that
        by resampling this class's output would put a second, mirrored copy of
        the file's interpolation rule in the caller. There is one rule, and it
        lives here.
        """
        points = np.asarray(points, dtype=float)
        energies = np.asarray(self.incident_energies, dtype=float)
        if energies.size == 0:
            return np.zeros(points.shape, dtype=float)
        if energy <= energies[0]:
            return self._interpolate_table(0, points)
        if energy >= energies[-1]:
            return self._interpolate_table(energies.size - 1, points)

        upper = int(np.searchsorted(energies, energy, side="right"))
        lower = upper - 1
        if energies[lower] == energy:
            return self._interpolate_table(lower, points)

        code = int(self._incident_codes()[lower])
        if code == 1:                            # histogram: hold the lower table
            return self._interpolate_table(lower, points)
        if code != 2:
            raise NotImplementedError(
                f"MF5 LF=1 incident interpolation code {code} between "
                f"{energies[lower]:.6e} and {energies[upper]:.6e}; only "
                f"histogram and lin-lin refine exactly"
            )

        f_lo = self._interpolate_table(lower, points)
        f_hi = self._interpolate_table(upper, points)
        weight = (energy - energies[lower]) / (energies[upper] - energies[lower])
        return f_lo + weight * (f_hi - f_lo)

    def _interpolate_table(self, k: int, points: np.ndarray) -> np.ndarray:
        """Table *k* evaluated at *points*, zero outside its own support."""
        x, y = self.table(k)
        return evaluate_table(x, y, self._segment_codes(k), points)

    def insert_incident_node(self, energy: float) -> int:
        """Add an incident node at *energy*, refining exactly. Returns its index.

        If the node already exists it is not duplicated and its index comes
        back unchanged, so callers can be careless about band edges that are
        already MF5 nodes — which, measured on both target libraries, all of
        them are.
        """
        energies = list(self.incident_energies)
        for index, existing in enumerate(energies):
            if existing == energy:
                return index

        x, y = self.evaluate_at_incident(energy)
        position = int(np.searchsorted(np.asarray(energies, dtype=float), energy))
        self.incident_energies.insert(position, float(energy))
        self.outgoing_grids.insert(position, [float(v) for v in x])
        self.chi.insert(position, [float(v) for v in y])
        self.outgoing_interp.insert(position, [(len(x), 2)])
        self._renumber_tab2(len(energies) + 1)
        return position

    def _renumber_tab2(self, ne: int) -> None:
        """Push every TAB2 NBT out to the new NE.

        Single-region is the measured case on every tape read (INT=2, one
        region), and there the only correct new NBT is NE. A multi-region TAB2
        would need the insertion point's region to grow and the later ones to
        shift, which no tape has asked for; it raises rather than guessing.
        """
        if len(self.tab2_interp) <= 1:
            code = self.tab2_interp[0][1] if self.tab2_interp else 2
            self.tab2_interp = [(ne, int(code))]
            return
        raise NotImplementedError(
            f"inserting an incident node into a multi-region TAB2 "
            f"({len(self.tab2_interp)} regions) is not implemented; no tape "
            f"read so far uses one for MF5/LF=1"
        )


def make_partial(u: float, lf: int, p_interp, p_energies, p_values,
                 **law: Any) -> MF5Partial:
    """Build the right partial class for *lf*.

    Three outcomes: LF=1 is the tabulated law, LF=5/7/9/11 are read by
    :mod:`~kika.endf.classes.mf5.analytic`, and anything else keeps its bytes.
    All three carry ``raw_lines`` except LF=1, so an analytic partial can be
    built with or without them and still emit whatever it was given.
    """
    if lf == 1:
        return MF5PartialTabulated(
            u=u, lf=lf, p_interp=list(p_interp),
            p_energies=list(p_energies), p_values=list(p_values), **law,
        )

    # Imported here and not at module scope: ``analytic`` needs this module's
    # panel arithmetic, so the two would form a cycle. The factory is the only
    # place that needs to know both.
    from .analytic import ANALYTIC_LAWS

    cls = ANALYTIC_LAWS.get(lf, MF5PartialRaw)
    return cls(
        u=u, lf=lf, p_interp=list(p_interp),
        p_energies=list(p_energies), p_values=list(p_values), **law,
    )
