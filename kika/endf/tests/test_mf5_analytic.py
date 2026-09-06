"""MF5's parametrised laws: what the formulae owe, and what the tape witnesses.

Three separate things are being held down here, and they are not equally strong
evidence. Saying which is which is the point of this module.

**The witness.** LF=5 is read off ``micro_cf252_pfns.endf`` MT455 -- six real
subsections from ENDF/B-VIII.1 Cf-252. That gates the record layout, the field
order and the normalisation reading against a tape an evaluator wrote.

**The identity.** LF=7, 9 and 11 have no tape on this machine. Their closed-form
normalisation constants are gated against numerical quadrature of the very shape
they are supposed to normalise, which catches a typo in either but *cannot*
catch a misreading of ENDF-6 §5.1.1. Their record layout is gated by
round-tripping a hand-built section through the parser -- which proves the
walker and the field names agree, not that a real evaluator writes them that
way. Both limits are real and neither is hidden.

**The gate that must not move.** The analytic partials keep their bytes and emit
them, which is what lets ``test_mf5_roundtrip.py`` go on gating this fixture
byte for byte now that its subsections are decoded. Decoding a law changed what
kika can *say* about it and nothing about what it writes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

from kika.endf import read_endf
from kika.endf.classes.mf5.analytic import (
    ANALYTIC_LAWS,
    ANALYTIC_RECORDS,
    MF5Evaporation,
    MF5GeneralEvaporation,
    MF5Maxwellian,
    MF5PartialAnalytic,
    MF5Watt,
)
from kika.endf.classes.mf5.partials import (
    TAB1_RECORDS_AFTER_HEADER,
    MF5PartialRaw,
    MF5PartialTabulated,
)
from kika.endf.parsers.parse_mf5 import parse_mf5_mt
from kika.endf.utils import (
    ENDF_FORMAT_FLOAT,
    ENDF_FORMAT_INT,
    format_endf_data_line,
    format_endf_send_record,
    format_tab1,
)

DATA = Path(__file__).parent / "data"
CF252 = DATA / "micro_cf252_pfns.endf"

MAT, MT = 9999, 18


# ---------------------------------------------------------------------------
# Building a section by hand, for the laws no tape here carries
# ---------------------------------------------------------------------------

def build_section(lf: float, u: float, law_tab1s, mt: int = MT):
    """One MF5 section with a single subsection of law *lf*, as tape lines.

    ``law_tab1s`` is ``[(interp, x, y), ...]`` in tape order. Written with the
    library's own formatters so the parser is being fed the dialect the library
    emits, not one invented here.
    """
    lines = [format_endf_data_line(
        [92235.0, 233.0248, 0, 0, 1, 0], MAT, 5, mt, 1,
        formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT,
                 ENDF_FORMAT_INT, ENDF_FORMAT_INT, ENDF_FORMAT_INT],
    )]
    line_num = 2
    header, line_num = format_tab1(
        u, 0.0, 0, int(lf), [(2, 2)], [1.0e-5, 2.0e7], [1.0, 1.0],
        MAT, 5, mt, line_num,
    )
    lines.extend(header)
    for interp, x, y in law_tab1s:
        block, line_num = format_tab1(
            0.0, 0.0, 0, 0, interp, list(x), list(y), MAT, 5, mt, line_num,
        )
        lines.extend(block)
    lines.append(format_endf_send_record(MAT, 5))
    return lines


def one_partial(lf, u, law_tab1s):
    """Parse a hand-built section and hand back its only subsection."""
    section = parse_mf5_mt(build_section(lf, u, law_tab1s), MT)
    assert section.num_partials == 1
    return section.partials[0]


THETA = ([(2, 2)], [1.0e-5, 2.0e7], [1.3e6, 1.3e6])
A_TAB = ([(2, 2)], [1.0e-5, 2.0e7], [9.88e5, 9.88e5])
B_TAB = ([(2, 2)], [1.0e-5, 2.0e7], [2.249e-6, 2.249e-6])


# ---------------------------------------------------------------------------
# The layout table and the walker have to agree
# ---------------------------------------------------------------------------

def test_analytic_record_names_match_the_walker():
    """Every decoded law names exactly as many records as the walker consumes.

    The two tables are separate on purpose -- one says how far to move, the
    other says what was passed -- so nothing but this test stops them drifting,
    and a drift shows up as a law silently falling back to verbatim.
    """
    assert set(ANALYTIC_RECORDS) == set(ANALYTIC_LAWS)
    for lf, names in ANALYTIC_RECORDS.items():
        assert len(names) == TAB1_RECORDS_AFTER_HEADER[lf], f"LF={lf}"


def test_every_analytic_law_reports_itself_as_decoded():
    for lf, cls in ANALYTIC_LAWS.items():
        assert issubclass(cls, MF5PartialAnalytic)
        assert cls(lf=lf).is_decoded is True
    assert MF5PartialRaw(lf=12).is_decoded is False


# ---------------------------------------------------------------------------
# LF=5 -- the witnessed law
# ---------------------------------------------------------------------------

def test_cf252_mt455_is_read_as_six_general_evaporation_partials():
    section = read_endf(str(CF252), mf_numbers=[5]).files[5].sections[455]
    assert section.num_partials == 6
    assert all(isinstance(p, MF5GeneralEvaporation) for p in section.partials)
    # The whole point of decoding them: they stop being reported as a gap.
    assert section.report_gaps() == []


def test_cf252_delayed_spectra_are_normalised_as_written():
    """theta == 1 and ``int g dx == 1`` -- so ``f == g``, with nothing rescaled.

    This is the measurement the module's normalisation argument rests on. If a
    future fixture had theta != 1 the two candidate readings of §5.1.1.2 would
    separate here, and this assertion is where that would surface.
    """
    section = read_endf(str(CF252), mf_numbers=[5]).files[5].sections[455]
    for index, partial in enumerate(section.partials):
        theta = partial.theta(1.0e6)
        assert theta == pytest.approx(1.0), f"partial {index}"
        # U is -30 MeV, so the bound is far beyond g's support and the
        # normalisation integral covers the whole table.
        assert partial.upper_bound(1.0e6) > partial.g_x[-1]
        assert partial.normalisation_at(1.0e6) == pytest.approx(1.0, abs=1e-6)

        grid, chi = partial.evaluate_at_incident(1.0e6)
        # theta == 1 collapses x -> E', so the law is g itself: every tabulated
        # point must come back untouched, not merely close.
        n = len(partial.g_x)
        assert np.allclose(grid[:n], partial.g_x, rtol=0, atol=0)
        assert np.allclose(chi[:n], np.asarray(partial.g_values)
                           / partial.normalisation_at(1.0e6), rtol=1e-12)


def test_cf252_general_evaporation_integrates_to_one():
    section = read_endf(str(CF252), mf_numbers=[5]).files[5].sections[455]
    for partial in section.partials:
        grid, chi = partial.evaluate_at_incident(1.0e6)
        # g is histogram-interpolated, so the exact integral is a left-rectangle
        # sum, not a trapezoid -- that difference is why the class integrates
        # over g's own panels instead of calling a quadrature.
        total = float(np.sum(chi[:-1] * np.diff(grid)))
        assert total == pytest.approx(1.0, abs=1e-6)


def test_analytic_partials_still_carry_their_bytes():
    """The decode is additive: every analytic partial kept its raw records.

    The byte gate itself is ``test_mf5_roundtrip.py``'s
    ``test_micro_cf252_mt455_roundtrip``, which now runs against decoded
    partials and is where a regression would be caught. What is asserted here
    is the mechanism that makes it hold -- that reading a law did not replace
    the emit path with a reconstruction.
    """
    section = read_endf(str(CF252), mf_numbers=[5]).files[5].sections[455]
    for index, partial in enumerate(section.partials):
        assert isinstance(partial, MF5PartialRaw), f"partial {index}"
        assert partial.raw_lines, f"partial {index} lost its bytes"


# ---------------------------------------------------------------------------
# LF=7, 9, 11 -- closed forms against quadrature
# ---------------------------------------------------------------------------

CLOSED_FORM_CASES = [
    pytest.param(7.0, [THETA], 1.0e7, 0.0, id="maxwellian"),
    pytest.param(7.0, [THETA], 2.0e7, 3.0e5, id="maxwellian-with-U"),
    pytest.param(7.0, [THETA], 1.0e5, 0.0, id="maxwellian-truncated"),
    pytest.param(9.0, [THETA], 1.0e7, 0.0, id="evaporation"),
    pytest.param(9.0, [THETA], 2.0e7, 3.0e5, id="evaporation-with-U"),
    pytest.param(9.0, [THETA], 1.0e5, 0.0, id="evaporation-truncated"),
    pytest.param(11.0, [A_TAB, B_TAB], 1.0e7, 0.0, id="watt"),
    pytest.param(11.0, [A_TAB, B_TAB], 2.0e7, 3.0e5, id="watt-with-U"),
    pytest.param(11.0, [A_TAB, B_TAB], 1.0e6, 0.0, id="watt-truncated"),
]


@pytest.mark.parametrize("lf,tab1s,energy,u", CLOSED_FORM_CASES)
def test_closed_form_normalisation_matches_quadrature(lf, tab1s, energy, u):
    """``I(E)`` from the manual equals the integral of the shape it normalises.

    The strongest statement available without a tape: it gates the shape and
    its normalisation constant *against each other*, so a sign, a factor or a
    swapped erf argument fails. It says nothing about whether the shape is the
    one ENDF-6 means.
    """
    partial = one_partial(lf, u, tab1s)
    hi = partial.upper_bound(energy)
    numeric, error = quad(
        lambda e: float(partial.shape(energy, np.array([e]))[0]),
        0.0, hi, limit=400,
    )
    assert partial.normalisation_at(energy) == pytest.approx(numeric, rel=1e-8)
    assert error < abs(numeric) * 1e-6


@pytest.mark.parametrize("lf,tab1s,energy,u", CLOSED_FORM_CASES)
def test_normalised_spectrum_integrates_to_one(lf, tab1s, energy, u):
    partial = one_partial(lf, u, tab1s)
    hi = partial.upper_bound(energy)
    total, _ = quad(
        lambda e: float(partial.evaluate_on_grid(energy, np.array([e]))[0]),
        0.0, hi, limit=400,
    )
    assert total == pytest.approx(1.0, rel=1e-8)
    assert partial.normalisation(energy) == pytest.approx(1.0)


@pytest.mark.parametrize("lf,tab1s", [
    pytest.param(7.0, [THETA], id="maxwellian"),
    pytest.param(9.0, [THETA], id="evaporation"),
    pytest.param(11.0, [A_TAB, B_TAB], id="watt"),
    pytest.param(5.0, [THETA, ([(4, 2)], [0.0, 1.0e6, 5.0e6, 2.0e7],
                               [0.0, 1.0e-7, 4.0e-8, 0.0])], id="general-evap"),
])
def test_spectrum_vanishes_above_the_upper_bound(lf, tab1s):
    """``E - U`` is part of the law, not a plotting range."""
    partial = one_partial(lf, 3.0e5, tab1s)
    energy = 5.0e6
    hi = partial.upper_bound(energy)
    beyond = np.array([hi * 1.000001, hi * 2.0, hi * 10.0])
    assert np.all(partial.evaluate_on_grid(energy, beyond) == 0.0)
    assert partial.evaluate_on_grid(energy, np.array([-1.0]))[0] == 0.0


# ---------------------------------------------------------------------------
# Record order -- the swap a plausible-looking spectrum would hide
# ---------------------------------------------------------------------------

def test_watt_reads_a_then_b_and_not_the_other_way_round():
    """LF=11 writes a(E) first. Swapped, it still draws a curve.

    a is an energy in eV and b its reciprocal, so the two differ by twelve
    orders of magnitude and a swap is not subtle in the numbers -- but it is
    invisible in the *shape*, which is why the order is asserted rather than
    eyeballed.
    """
    partial = one_partial(11.0, 0.0, [A_TAB, B_TAB])
    assert isinstance(partial, MF5Watt)
    assert partial.a(1.0e6) == pytest.approx(A_TAB[2][0])
    assert partial.b(1.0e6) == pytest.approx(B_TAB[2][0])


def test_general_evaporation_reads_theta_then_g():
    g = ([(4, 2)], [0.0, 1.0e6, 5.0e6, 2.0e7], [0.0, 1.0e-7, 4.0e-8, 0.0])
    partial = one_partial(5.0, 0.0, [THETA, g])
    assert isinstance(partial, MF5GeneralEvaporation)
    assert partial.theta(1.0e6) == pytest.approx(THETA[2][0])
    assert partial.g_x == list(g[1])


@pytest.mark.parametrize("lf,cls", [(7.0, MF5Maxwellian), (9.0, MF5Evaporation)])
def test_theta_only_laws_parse_to_their_class(lf, cls):
    partial = one_partial(lf, 0.0, [THETA])
    assert isinstance(partial, cls)
    assert partial.theta(1.0e6) == pytest.approx(THETA[2][0])


def test_undecoded_law_is_still_kept_verbatim():
    """LF=12 has no evaluator, and says so rather than going quiet."""
    partial = one_partial(12.0, 0.0, [THETA])
    assert type(partial) is MF5PartialRaw
    assert partial.is_decoded is False
    assert partial.raw_lines


# ---------------------------------------------------------------------------
# evaluate_on_grid on the tabulated law
# ---------------------------------------------------------------------------

def tabulated_partial() -> MF5PartialTabulated:
    return read_endf(str(CF252), mf_numbers=[5]).files[5].sections[18].partials[0]


def test_evaluate_on_grid_reproduces_a_node_exactly():
    partial = tabulated_partial()
    for k in (0, 3, len(partial.incident_energies) - 1):
        x, y = partial.table(k)
        got = partial.evaluate_on_grid(partial.incident_energies[k], x)
        assert np.array_equal(got, y)


def test_evaluate_on_grid_agrees_with_the_union_table_between_nodes():
    """The short route and the long one are the same interpolant.

    ``evaluate_at_incident`` builds the union grid because it has to return a
    table; ``evaluate_on_grid`` blends the two bracketing tables point by point.
    They must agree to the last bit on the union's own abscissae, and that is
    the whole justification for the second method existing.
    """
    partial = tabulated_partial()
    lo, hi = partial.incident_energies[2], partial.incident_energies[3]
    energy = 0.5 * (lo + hi)
    grid, expected = partial.evaluate_at_incident(energy)
    assert np.allclose(partial.evaluate_on_grid(energy, grid), expected,
                       rtol=1e-14, atol=0.0)


def test_normalisation_at_incident_matches_the_node_it_lands_on():
    partial = tabulated_partial()
    for k in (0, 5, len(partial.incident_energies) - 1):
        assert (partial.normalisation_at_incident(partial.incident_energies[k])
                == partial.normalisation(k))


def test_normalisation_at_incident_blends_between_nodes():
    """The integral of the blend is the blend of the integrals.

    Exactly, not nearly: with lin-lin on the incident axis the interpolant is
    ``(1-w) chi_lo + w chi_hi`` at every fixed E', and integration is linear.
    Asserted against the arithmetic rather than against a quadrature, because a
    quadrature would agree with a wrong answer to five digits.
    """
    partial = tabulated_partial()
    lo_index, hi_index = 2, 3
    lo = partial.incident_energies[lo_index]
    hi = partial.incident_energies[hi_index]
    for weight in (0.25, 0.5, 0.9):
        energy = lo + weight * (hi - lo)
        expected = ((1.0 - weight) * partial.normalisation(lo_index)
                    + weight * partial.normalisation(hi_index))
        assert partial.normalisation_at_incident(energy) == pytest.approx(
            expected, rel=1e-14)


def test_a_trapezoid_over_a_histogram_law_is_not_the_normalisation():
    """Why the method exists, stated as the difference it avoids.

    Cf-252 MT455 is histogram-interpolated. Integrating the curve
    ``evaluate_at_incident`` hands back, with the trapezoid rule any consumer
    would reach for, comes out five parts in ten thousand short of one on a
    single subsection -- and about six times that once the six are summed. That
    is the size of a real normalisation defect, so a readout computed by
    quadrature could not be told from a file with a problem. The exact answer
    is one.
    """
    section = read_endf(str(CF252), mf_numbers=[5]).files[5].sections[455]
    partial = section.partials[0]
    grid, chi = partial.evaluate_at_incident(1.0e6)
    trapezoid = float(np.trapezoid(chi, grid))

    assert partial.normalisation_at_incident(1.0e6) == pytest.approx(1.0, abs=1e-9)
    assert trapezoid == pytest.approx(0.99947, abs=1e-4)
    assert abs(trapezoid - 1.0) > 1e-4, "the two must not have converged"


def test_evaluate_on_grid_holds_the_ends():
    partial = tabulated_partial()
    x, y = partial.table(0)
    below = partial.incident_energies[0] * 0.5
    assert np.array_equal(partial.evaluate_on_grid(below, x), y)

    last = len(partial.incident_energies) - 1
    x, y = partial.table(last)
    above = partial.incident_energies[last] * 2.0
    assert np.array_equal(partial.evaluate_on_grid(above, x), y)
