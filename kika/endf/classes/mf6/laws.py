"""The law bodies of an MF6 product subsection (ENDF-6 §6.2).

**Why every law is modelled here and MF5 models only one.** MF5 can walk past a
law it does not decode from a table of record counts, because every MF5 law body
is a fixed number of TAB1s. MF6's bodies are *data-dependent* — how many records
a LAW=1 body occupies is written inside its own TAB2 — so a walker that does not
understand a law cannot find the end of it, and a wrong length silently swallows
the product after it. There is no cheap "keep it verbatim" for MF6. Given that
the walker has to know every law's shape anyway, decoding it costs the emitter
and the tests and nothing else, which is why all of them are here.

**Negative LAW is not an error.** ENDF-6 lets a product defer its distribution to
another file, and writes that as a negative LAW naming the file: −4 → File 4,
−5 → File 5, −14 → File 14, −15 → File 15. It consumes no records. Measured on
ENDF/B-VIII.1, ``LAW=-5`` occurs only with ``ZAP=1`` (the neutron, sent to MF5)
and ``LAW=-15`` only with ``ZAP=0`` (photons, sent to MF15) — and MF14/MF15 are
files kika does not read, so :class:`MF6LawElsewhere` reports itself as a gap
rather than pretending the distribution is in hand.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from ...utils import (
    ENDF_FORMAT_FLOAT,
    ENDF_FORMAT_INT,
    PAD_BLANK,
    PadStyle,
    format_data_values,
    format_endf_data_line,
    format_tab1,
    format_tab2,
)

#: LAW values whose body is empty because the distribution is simply not given
#: (0), is isotropic two-body (3), or is derived from the partner product's
#: two-body kinematics (4).
NO_BODY_LAWS = (0, 3, 4)

#: ``|LAW|`` → the file the distribution was deferred to, and whether kika reads
#: that file today. Kept as data because the second column is the whole reason
#: :meth:`MF6LawElsewhere.report_gaps` exists.
DEFERRED_FILES = {4: ("MF4", True), 5: ("MF5", True),
                  14: ("MF14", False), 15: ("MF15", False)}


def _emit_list(c1, c2, l1, l2, values, mat, mf, mt, line_num,
               n2=0, pad=PAD_BLANK):
    """A LIST record: the CONT header carrying NW, then NW values six per line.

    MF6 writes three different LIST records (LAW=1, 2 and 5) that differ only in
    what their header's integer fields mean, so the record itself is written
    once here rather than three times.
    """
    header = format_endf_data_line(
        [c1, c2, l1, l2, len(values), n2],
        mat, mf, mt, line_num,
        formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT,
                 ENDF_FORMAT_INT, ENDF_FORMAT_INT,
                 ENDF_FORMAT_INT, ENDF_FORMAT_INT],
    )
    body, line_num = format_data_values(list(values), mat, mf, mt,
                                        line_num + 1, pad=pad)
    return [header] + body, line_num


@dataclass
class MF6Law:
    """What every law body shares: its LAW number and how to write itself back."""

    law: int = 0

    def emit(self, mat: int, mf: int, mt: int, line_num: int,
             pad: PadStyle) -> Tuple[List[str], int]:
        """The records *after* the product's header TAB1. May be none."""
        return [], line_num

    def describe(self) -> str:
        return f"LAW={self.law}"

    def report_gaps(self, mt: int, index: int) -> List[str]:
        """One line per thing this body carries but this class does not model."""
        return []


@dataclass
class MF6LawNoBody(MF6Law):
    """LAW=0, 3, 4 — the distribution is stated by the LAW number alone.

    Distinct from :class:`MF6LawElsewhere`, which also has no body: these three
    *are* the answer (unknown, isotropic, recoil), while a negative LAW is a
    pointer to a different file.
    """

    def describe(self) -> str:
        return {0: "LAW=0 (distribution not given)",
                3: "LAW=3 (isotropic two-body)",
                4: "LAW=4 (recoil, from the two-body partner)"}.get(
                    self.law, f"LAW={self.law}")


@dataclass
class MF6LawElsewhere(MF6Law):
    """A negative LAW: the distribution lives in the file ``|LAW|`` names."""

    @property
    def file_number(self) -> int:
        return abs(self.law)

    @property
    def is_reachable(self) -> bool:
        """Whether kika has a parser for the file this defers to."""
        return DEFERRED_FILES.get(self.file_number, ("", False))[1]

    def describe(self) -> str:
        name, reachable = DEFERRED_FILES.get(
            self.file_number, (f"MF{self.file_number}", False))
        return (f"LAW={self.law} (distribution given in {name}"
                f"{'' if reachable else ', which kika does not read'})")

    def report_gaps(self, mt: int, index: int) -> List[str]:
        if self.is_reachable:
            return []
        name = DEFERRED_FILES.get(self.file_number, (f"MF{self.file_number}",))[0]
        return [f"MF6/MT{mt} product {index} defers its distribution to "
                f"{name} (LAW={self.law}), which kika does not parse: the "
                f"yield is read and the distribution is not"]


@dataclass
class MF6LawContinuum(MF6Law):
    """LAW=1: a TAB2 over incident energy, each node a LIST of E' and its shape.

    Each node's LIST holds ``NEP`` groups of ``NA+2`` numbers — the outgoing
    energy, ``f0``, then ``NA`` further coefficients whose meaning is ``LANG``'s
    to say (Legendre for LANG=1, Kalbach ``r`` and ``a`` for LANG=2, tabulated
    cosines for LANG=11-15). The values are kept **flat, exactly as written**;
    the shape is a view taken by :meth:`block`, so round-tripping never depends
    on the reshape being right.

    The first ``ND`` of the ``NEP`` groups are *discrete* lines rather than
    points of a continuum — this is where the resolved inelastic levels sit, and
    losing the count silently turns a line spectrum into a continuum.
    """

    law: int = 1
    lang: int = 1
    lep: int = 2
    tab2_interp: List[Tuple[int, int]] = field(default_factory=list)
    incident_energies: List[float] = field(default_factory=list)
    nd: List[int] = field(default_factory=list)
    na: List[int] = field(default_factory=list)
    values: List[List[float]] = field(default_factory=list)

    # ------------------------------------------------------------------
    def emit(self, mat, mf, mt, line_num, pad):
        lines, line_num = format_tab2(
            0.0, 0.0, self.lang, self.lep, self.tab2_interp,
            len(self.incident_energies), mat, mf, mt, line_num,
            interp_pad=pad.interp,
        )
        for k, energy in enumerate(self.incident_energies):
            block, line_num = _emit_list(
                0.0, energy, self.nd[k], self.na[k], self.values[k],
                mat, mf, mt, line_num, n2=self.nep(k), pad=pad.values,
            )
            lines.extend(block)
        return lines, line_num

    def describe(self) -> str:
        return (f"LAW=1 LANG={self.lang} LEP={self.lep}, "
                f"{len(self.incident_energies)} incident nodes")

    # ------------------------------------------------------------------
    # Views over the flat values
    # ------------------------------------------------------------------
    def nep(self, k: int) -> int:
        """Number of outgoing-energy groups at incident node *k*."""
        return len(self.values[k]) // (self.na[k] + 2)

    def block(self, k: int) -> np.ndarray:
        """Node *k* as ``(NEP, NA+2)``: E', f0, then NA coefficients."""
        width = self.na[k] + 2
        return np.asarray(self.values[k], dtype=float).reshape(-1, width)

    def spectrum(self, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """``(E', f0)`` at incident node *k* — the outgoing energy spectrum.

        ``f0`` is the angle-integrated shape for every LANG, which is why this
        one accessor is meaningful without asking what LANG is.
        """
        table = self.block(k)
        return table[:, 0], table[:, 1]

    def discrete_lines(self, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """The ``ND`` discrete lines at node *k*, split off from the continuum."""
        e_out, f0 = self.spectrum(k)
        return e_out[:self.nd[k]], f0[:self.nd[k]]

    def legendre(self, k: int) -> np.ndarray:
        """``(NEP, NA+1)`` Legendre coefficients ``f_l(E')``. LANG=1 only."""
        if self.lang != 1:
            raise ValueError(
                f"LAW=1 LANG={self.lang} does not carry Legendre coefficients; "
                f"only LANG=1 does (LANG=2 is Kalbach-Mann, 11-15 tabulated)"
            )
        return self.block(k)[:, 1:]

    def kalbach(self, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(E', r, a)`` at node *k*. LANG=2 only.

        ``a`` is absent when ``NA=1``: ENDF-6 lets the evaluator write only the
        pre-compound fraction ``r`` and leave the slope to be recomputed from
        Kalbach's systematics. It is returned as NaN rather than 0.0, because a
        zero slope is a physically meaningful isotropic value and would be taken
        for one.
        """
        if self.lang != 2:
            raise ValueError(
                f"LAW=1 LANG={self.lang} is not Kalbach-Mann; only LANG=2 is"
            )
        table = self.block(k)
        e_out, r = table[:, 0], table[:, 2]
        a = table[:, 3] if self.na[k] >= 2 else np.full(e_out.shape, np.nan)
        return e_out, r, a

    def evaluate_at_incident(self, energy: float):
        """Not provided, on purpose — and this says why rather than being absent.

        Every accessor above is a *view* of one tabulated node: exact, and wrong
        only if the reshape is wrong, which
        ``test_law1_list_widths_are_nep_times_na_plus_two`` gates. Interpolating
        **between** nodes is a different kind of thing, because the outgoing
        grids differ from node to node — Fe-56 and C-12 both change ``NEP``
        along the incident axis. Interpolating at fixed ``E'`` across two nodes
        whose spectra end at different maximum outgoing energies smears the
        end-point and is not what a processing code does: NJOY interpolates
        LAW=1 continua on a *unit base*, rescaling each node's outgoing range to
        [0, 1] first.

        ENDF-6 has codes for saying which is meant — 11-15 corresponding-point,
        21-25 unit-base — and **the tapes do use them**, which corrects what
        this docstring said until 2026-08-24. Measured over the 402
        ENDF/B-VIII.1 neutron tapes under 2.5 MB: of 9 604 LAW=1 TAB2 records,
        5 025 write a plain ``INT=2``, **4 498 write ``INT=22`` (unit base) and
        81 write ``INT=12`` (corresponding points)** — 229 of the 402 tapes
        carry at least one. So the file usually *does* say.

        That changes the reason and not the answer. The qualifier is read and
        re-emitted (``splitEndfTab2Code``, and ``XYs2d.interpolationQualifier``
        on the model side, so a GNDS file written from a tape carries it), but
        **kika implements neither rule**: rescaling each node's outgoing range
        to [0, 1] before interpolating is a processing step this library does
        not do, and there is nothing here to gate one against. A plausible
        implementation would be silently wrong, which is worse than none.

        Use :meth:`spectrum` at a node, or reconstruct through NJOY.
        """
        # §0.5.2.1's tens digit: 0 plain, 1 corresponding points, 2 unit base.
        # Spelled out rather than imported from `kika.nuclear_data.model` --
        # this is three words in an error message, not a second mapping table.
        qualifier = {1: "corresponding-point", 2: "unit-base"}.get(
            (self.tab2_interp[0][1] // 10) if self.tab2_interp else 0, "")
        says = (f"this record asks for {qualifier} interpolation "
                f"(INT={self.tab2_interp[0][1]})" if qualifier else
                "this record writes a plain INT, so it does not say which")
        raise NotImplementedError(
            f"MF6 LAW=1 has no between-node evaluator in kika: the outgoing "
            f"grid changes from node to node, so interpolating at fixed E' "
            f"smears the end point, and {says}. Neither ENDF's "
            f"corresponding-point nor its unit-base rule is implemented here. "
            f"Use spectrum(k) at an incident node, or go through NJOY. See "
            f"this method's docstring and docs/library/mf6_notes.md."
        )

    def report_gaps(self, mt: int, index: int) -> List[str]:
        if 11 <= self.lang <= 15:
            return [f"MF6/MT{mt} product {index} is LAW=1 LANG={self.lang}: "
                    f"the tabulated cosine grids are read and re-written, and "
                    f"no accessor interprets them yet"]
        return []


@dataclass
class MF6LawTwoBody(MF6Law):
    """LAW=2: two-body scattering, angle only — E' follows from kinematics.

    Each incident node is a LIST whose ``LANG`` says how the angular shape is
    written: 0 is Legendre coefficients ``A_1..A_NL`` (``A_0`` is 1 by
    normalisation and is not on the tape), 12 and 14 are ``NL`` (μ, f) pairs,
    linearly and log-linearly interpolated.
    """

    law: int = 2
    tab2_interp: List[Tuple[int, int]] = field(default_factory=list)
    incident_energies: List[float] = field(default_factory=list)
    lang: List[int] = field(default_factory=list)
    values: List[List[float]] = field(default_factory=list)

    def emit(self, mat, mf, mt, line_num, pad):
        lines, line_num = format_tab2(
            0.0, 0.0, 0, 0, self.tab2_interp,
            len(self.incident_energies), mat, mf, mt, line_num,
            interp_pad=pad.interp,
        )
        for k, energy in enumerate(self.incident_energies):
            block, line_num = _emit_list(
                0.0, energy, self.lang[k], 0, self.values[k],
                mat, mf, mt, line_num, n2=self.nl(k), pad=pad.values,
            )
            lines.extend(block)
        return lines, line_num

    def describe(self) -> str:
        langs = sorted(set(self.lang))
        return (f"LAW=2, {len(self.incident_energies)} incident nodes, "
                f"LANG={','.join(str(v) for v in langs)}")

    def nl(self, k: int) -> int:
        """``NL``: the coefficient count, or the number of (μ, f) pairs."""
        return (len(self.values[k]) if self.lang[k] == 0
                else len(self.values[k]) // 2)

    def legendre(self, k: int) -> np.ndarray:
        """``A_1..A_NL`` at node *k*. LANG=0 only; ``A_0`` is 1 and unwritten."""
        if self.lang[k] != 0:
            raise ValueError(
                f"LAW=2 node {k} is LANG={self.lang[k]}, a tabulated (mu, f) "
                f"record; only LANG=0 carries Legendre coefficients"
            )
        return np.asarray(self.values[k], dtype=float)

    def tabulated(self, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """``(mu, f)`` at node *k*. LANG=12 and 14."""
        if self.lang[k] == 0:
            raise ValueError(
                f"LAW=2 node {k} is LANG=0, Legendre coefficients, not a table"
            )
        table = np.asarray(self.values[k], dtype=float).reshape(-1, 2)
        return table[:, 0], table[:, 1]


@dataclass
class MF6LawChargedElastic(MF6Law):
    """LAW=5: charged-particle elastic scattering (ENDF-6 §6.2.5).

    ``LTP`` on each node says whether the LIST is a nuclear-amplitude expansion
    (``LTP=1``, ``LTP=2``) or a table of (μ, P) pairs (``LTP>2``). ``values``
    stays **flat**, as it arrived: re-emitting what was read cannot be wrong,
    and that is what makes the byte gate on the four committed charged-particle
    fixtures pass. :meth:`amplitudes` and :meth:`tabulated` do the reshape for
    callers who want the physics.

    The reshape used to be refused, on the grounds that *"no tape on this
    machine carries a LAW=5 to check a reshape against"*. That was true of
    ``endfb81/``, which is neutron-only, and false of the share: ENDF/B-VIII.0
    ships its charged-particle sublibraries and 63 of those tapes carry a
    LAW=5. Measured over their 4 710 nodes, ``NW`` is fixed by ``LTP`` and
    ``LIDP`` with no exceptions:

    ==============  ==============  ==================
    ``LTP``         ``LIDP``        ``NW``
    ==============  ==============  ==================
    1 or 2          0               ``4·NL + 3``
    1 or 2          1               ``3·NL + 3``
    >2              either          ``2·NL``
    ==============  ==============  ==================

    and the arithmetic says why, which is what makes it an identity rather than
    a fit. Distinguishable particles need the nuclear expansion out to order
    ``2NL`` (``2NL+1`` reals) beside ``NL+1`` complex interference amplitudes
    (``2NL+2`` reals). Identical particles keep only the even orders (``NL+1``
    reals) beside the same ``2NL+2``. A table is (μ, P) pairs, ``2NL``.

    ``LTP=2``, ``LTP=14`` and ``LTP=15`` occur zero times in those 63 tapes, so
    the ``LTP=2`` branch of :meth:`amplitudes` and the log-interpolated tables
    are reshaped by the same identity but witnessed by nothing. See
    ``docs/library/mf6_witness_hunt.md`` in kika-workspace.
    """

    law: int = 5
    spi: float = 0.0
    lidp: int = 0
    tab2_interp: List[Tuple[int, int]] = field(default_factory=list)
    incident_energies: List[float] = field(default_factory=list)
    ltp: List[int] = field(default_factory=list)
    values: List[List[float]] = field(default_factory=list)
    #: ``NL`` as written, per node. Kept rather than derived because what it
    #: counts is a different thing in each branch — a Legendre order under
    #: ``LTP<=2``, a number of table points under ``LTP>2`` — so there is no one
    #: expression to derive it by, and re-emitting what was read cannot be wrong.
    nl_values: List[int] = field(default_factory=list)

    def emit(self, mat, mf, mt, line_num, pad):
        lines, line_num = format_tab2(
            self.spi, 0.0, self.lidp, 0, self.tab2_interp,
            len(self.incident_energies), mat, mf, mt, line_num,
            interp_pad=pad.interp,
        )
        for k, energy in enumerate(self.incident_energies):
            block, line_num = _emit_list(
                0.0, energy, self.ltp[k], 0, self.values[k],
                mat, mf, mt, line_num, n2=self.nl(k), pad=pad.values,
            )
            lines.extend(block)
        return lines, line_num

    def nl(self, k: int) -> int:
        return self.nl_values[k] if self.nl_values else len(self.values[k])

    def amplitudes(self, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """``(nuclear, interference)`` at node *k*. ``LTP=1`` and ``LTP=2`` only.

        *nuclear* is the real Legendre expansion of the nuclear scattering —
        ``2NL+1`` coefficients with distinguishable particles, ``NL+1`` even
        orders with identical ones. *interference* is the ``NL+1`` complex
        amplitudes, which the tape writes as interleaved (real, imaginary)
        reals. The split point is the identity in this class's docstring; the
        interleaving is ENDF-6 §6.2.5 and is not separately witnessed.
        """
        if self.ltp[k] > 2:
            raise ValueError(
                f"LAW=5 node {k} is LTP={self.ltp[k]}, a table of (mu, P) "
                f"pairs, not an amplitude expansion; use tabulated()"
            )
        nl = self.nl(k)
        split = 2 * nl + 1 if self.lidp == 0 else nl + 1
        flat = np.asarray(self.values[k], dtype=float)
        self._check_width(k, flat, split + 2 * nl + 2)
        pairs = flat[split:].reshape(-1, 2)
        return flat[:split], pairs[:, 0] + 1j * pairs[:, 1]

    def tabulated(self, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """``(mu, p)`` at node *k*. ``LTP>2`` only — 12 lin-lin, 14/15 log.

        **Normalised, but not non-negative — do not treat it as a density.**
        Measured over the 3 779 ``LTP>2`` nodes of ENDF/B-VIII.0's
        charged-particle sublibraries:

        * **3 777 integrate to 1 ± 1 %.** The two that do not are
          ``p-080_Hg_199`` and ``p-080_Hg_204`` MT2 near 83 MeV, where the table
          spans six orders of magnitude and a trapezoid over 150 points is
          simply inaccurate — not a different normalisation.
        * **3 280 take negative values somewhere**, reaching -5.4e+05.

        Those two facts together are the whole warning. Per ENDF-6 §6.2.5 the
        Rutherford term is excluded from this table because it is analytic and
        diverges as μ → 1, so what is tabulated is the signed remainder; that
        mechanism is the manual's account, while the two counts above are
        measured here. Either way a caller must not clip it at zero, renormalise
        it, or sample it as a probability.

        Contrast :meth:`MF6LawTwoBody.tabulated`, which returns an ``f(mu)``
        that *is* an ordinary density. The two share a name and not a meaning.
        """
        if self.ltp[k] <= 2:
            raise ValueError(
                f"LAW=5 node {k} is LTP={self.ltp[k]}, a nuclear-amplitude "
                f"expansion, not a table; use amplitudes()"
            )
        flat = np.asarray(self.values[k], dtype=float)
        self._check_width(k, flat, 2 * self.nl(k))
        table = flat.reshape(-1, 2)
        return table[:, 0], table[:, 1]

    def _check_width(self, k: int, flat: np.ndarray, expected: int) -> None:
        """Refuse to reshape a body ENDF-6 does not admit.

        A LAW=5 body whose length disagrees with its ``LTP``/``LIDP`` is either
        a mis-walked record or a hand-built section, and reshaping it would turn
        either into plausible numbers. The identity is cheap enough to check
        every time.
        """
        if len(flat) != expected:
            raise ValueError(
                f"LAW=5 node {k}: LTP={self.ltp[k]} LIDP={self.lidp} "
                f"NL={self.nl(k)} requires NW={expected}, body has {len(flat)}"
            )

    def describe(self) -> str:
        return (f"LAW=5 SPI={self.spi} LIDP={self.lidp}, "
                f"{len(self.incident_energies)} incident nodes")


@dataclass
class MF6LawPhaseSpace(MF6Law):
    """LAW=6: n-body phase space, described by two numbers and nothing else."""

    law: int = 6
    apsx: float = 0.0
    npsx: int = 0

    def emit(self, mat, mf, mt, line_num, pad):
        line = format_endf_data_line(
            [self.apsx, 0.0, 0, 0, 0, self.npsx],
            mat, mf, mt, line_num,
            formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT,
                     ENDF_FORMAT_INT, ENDF_FORMAT_INT,
                     ENDF_FORMAT_INT, ENDF_FORMAT_INT],
        )
        return [line], line_num + 1

    def describe(self) -> str:
        return f"LAW=6 (n-body phase space, APSX={self.apsx}, NPSX={self.npsx})"


@dataclass
class LawSevenAngle:
    """One cosine of a LAW=7 node: a TAB1 of ``f(E')`` at fixed ``mu``."""

    mu: float = 0.0
    interp: List[Tuple[int, int]] = field(default_factory=list)
    e_out: List[float] = field(default_factory=list)
    f: List[float] = field(default_factory=list)


@dataclass
class LawSevenEnergy:
    """One incident energy of a LAW=7 body: a TAB2 over ``mu``."""

    energy: float = 0.0
    mu_interp: List[Tuple[int, int]] = field(default_factory=list)
    angles: List[LawSevenAngle] = field(default_factory=list)


@dataclass
class MF6LawLabAngleEnergy(MF6Law):
    """LAW=7: ``f(mu, E')`` in the laboratory frame, nested three deep.

    TAB2 over incident energy → TAB2 over cosine → TAB1 over outgoing energy.
    It is the only MF6 law with three levels, and the reason the walker is
    written as recursion over record kinds rather than a record-count table.
    """

    law: int = 7
    tab2_interp: List[Tuple[int, int]] = field(default_factory=list)
    blocks: List[LawSevenEnergy] = field(default_factory=list)

    def emit(self, mat, mf, mt, line_num, pad):
        lines, line_num = format_tab2(
            0.0, 0.0, 0, 0, self.tab2_interp, len(self.blocks),
            mat, mf, mt, line_num, interp_pad=pad.interp,
        )
        for block in self.blocks:
            inner, line_num = format_tab2(
                0.0, block.energy, 0, 0, block.mu_interp, len(block.angles),
                mat, mf, mt, line_num, interp_pad=pad.interp,
            )
            lines.extend(inner)
            for angle in block.angles:
                tab1, line_num = format_tab1(
                    0.0, angle.mu, 0, 0, angle.interp, angle.e_out, angle.f,
                    mat, mf, mt, line_num, pad=pad.pairs,
                    interp_pad=pad.interp,
                )
                lines.extend(tab1)
        return lines, line_num

    def describe(self) -> str:
        n_mu = sum(len(b.angles) for b in self.blocks)
        return (f"LAW=7, {len(self.blocks)} incident nodes, "
                f"{n_mu} cosine tables")

    @property
    def incident_energies(self) -> List[float]:
        return [block.energy for block in self.blocks]

    def table(self, k: int, m: int) -> Tuple[np.ndarray, np.ndarray]:
        """``(E', f)`` at incident node *k* and cosine *m*."""
        angle = self.blocks[k].angles[m]
        return (np.asarray(angle.e_out, dtype=float),
                np.asarray(angle.f, dtype=float))


@dataclass
class MF6LawRaw(MF6Law):
    """A law whose records were walked but whose meaning is not modelled.

    Nothing produces one today — every law ENDF-6 defines has a class above.
    It exists for the case a future revision adds a law whose *shape* we can
    walk from an existing record kind: the section still round-trips, and
    :meth:`report_gaps` says so out loud. A law whose shape is unknown cannot
    be walked at all and raises in the parser instead.
    """

    raw_lines: List[str] = field(default_factory=list)

    def emit(self, mat, mf, mt, line_num, pad):
        lines = []
        for body in self.raw_lines:
            lines.append(f"{body[:66]:<66}"
                         + f"{mat:4d}{mf:2d}{mt:3d}{line_num:5d}")
            line_num += 1
        return lines, line_num

    def describe(self) -> str:
        return f"LAW={self.law} ({len(self.raw_lines)} records kept verbatim)"

    def report_gaps(self, mt: int, index: int) -> List[str]:
        return [f"MF6/MT{mt} product {index} is stored verbatim: "
                f"{self.describe()}"]
