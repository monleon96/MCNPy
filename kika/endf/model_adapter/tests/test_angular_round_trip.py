"""MF4 → ``angularTwoBody`` → MF4, byte for byte, against the **file**.

Everywhere else in phase 3c the gate is "the model reproduces what the flat
class produces", because the flat class is the code being replaced. MF4 is the
one place that gate would be *weaker* than the truth:
``AngularDistribution.to_endf()`` cannot reproduce an LTT=3 section, so matching
it would mean reproducing its losses. Both losses are pinned below, as xfails,
so that when phase 3d reimplements the flat bodies over the model they turn into
XPASS and the suite tells someone to delete them.

The four representations behind one MT — LTT=0/1/2/3 — are all exercised: the
committed slice carries LTT=3, and under ``--deep`` the real tapes bring the
other three.
"""
from __future__ import annotations

import numpy as np
import pytest

from kika.endf.model_adapter import decodeMF4MT, encodeMF4MT
from kika.endf.read_endf import read_endf
from kika.nuclear_data import AngularDistribution
from kika.nuclear_data.model import (AngularTwoBody, Isotropic2d, Legendre,
                                     Regions2d, XYs2d)

REAL_TAPES = ["fe56_host_tape", "fe57_host_tape", "fe56_jendl_tape",
              "u235_tape", "th232_tape", "pu241_tape"]


@pytest.fixture(scope="module")
def mf4(micro_tape):
    return read_endf(str(micro_tape)).mf[4].mt[2]


def _roundTrip(section, mt):
    distribution, provenance, report = decodeMF4MT(section)
    encoded, report = encodeMF4MT(distribution, provenance, mt, report)
    return encoded, distribution, report


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

def test_the_fixture_is_the_hard_case(mf4):
    """LTT=3 — both representations in one section. A tape with only LTT=1
    would leave the interesting half of this module untested."""
    assert mf4._ltt == 3


def test_the_section_encodes_byte_identically_to_the_file(mf4):
    encoded, _, _ = _roundTrip(mf4, 2)
    assert str(encoded) == str(mf4)


@pytest.mark.parametrize("tape", REAL_TAPES)
def test_every_mf4_section_of_a_real_tape_encodes_byte_identically(request, tape):
    """Under ``--deep``. Six evaluations, LTT 0, 1, 2 and 3 between them."""
    endf = read_endf(str(request.getfixturevalue(tape)))
    if 4 not in endf.mf:
        pytest.skip(f"{tape} carries no MF4")

    seen = set()
    for mt in sorted(endf.mf[4].mt):
        section = endf.mf[4].mt[mt]
        seen.add(section._ltt)
        encoded, _, _ = _roundTrip(section, mt)
        assert str(encoded) == str(section), f"{tape} MT{mt} (LTT={section._ltt})"
    assert seen, f"{tape} MF4 is empty"


# ---------------------------------------------------------------------------
# The shape — a dict here would have been silently lossy
# ---------------------------------------------------------------------------

def test_a_repeated_incident_energy_survives(mf4):
    """The finding that made ``byEnergy`` a list.

    3.905 MeV appears twice in this tape's Legendre grid — ENDF's way of writing
    a discontinuity. A ``Dict[float, Function1d]`` keeps one of them, raises
    nothing, and writes out a valid file with one distribution missing.
    """
    _, distribution, _ = _roundTrip(mf4, 2)
    energies = distribution.energies
    assert energies.count(3905000.0) == 2, (
        "the duplicated incident energy was collapsed; the outer axis is not a "
        "set and must not be stored as one"
    )
    assert len(energies) == len(mf4.legendre_energies) + len(mf4.tabulated_energies)


def test_the_ltt3_boundary_energy_belongs_to_both_regions(mf4):
    """45 MeV is the last Legendre energy *and* the first tabulated one."""
    _, distribution, _ = _roundTrip(mf4, 2)
    assert isinstance(distribution.angular, Regions2d)
    legendre, tabulated = distribution.angular[0], distribution.angular[1]
    assert legendre.domainMax == tabulated.domainMin
    assert distribution.energies.count(legendre.domainMax) == 2


def test_ltt3_is_two_regions_and_not_a_flattened_list(mf4):
    """The split has to stay, because it is what says where the TAB2s divide."""
    _, distribution, _ = _roundTrip(mf4, 2)
    assert len(distribution.angular) == 2
    assert isinstance(distribution.angular[0], XYs2d)
    assert all(isinstance(f, Legendre) for f in distribution.angular[0])


def test_a_legendre_child_carries_a0_which_endf_leaves_implicit(mf4):
    """ENDF writes ``a_1..a_NL``; GNDS writes every coefficient, so ``a_0`` = 1."""
    _, distribution, _ = _roundTrip(mf4, 2)
    first = distribution.angular[0][0]
    assert first.coefficients[0] == 1.0
    assert first.maxOrder == len(mf4.legendre_coefficients[0])


def test_a_trailing_zero_coefficient_is_not_trimmed(mf4):
    """NL is what the evaluator declared, not what is numerically necessary."""
    withTrailingZero = [
        i for i, row in enumerate(mf4.legendre_coefficients) if row and row[-1] == 0.0
    ]
    assert withTrailingZero, "this tape has no trailing zero to test against"

    _, distribution, _ = _roundTrip(mf4, 2)
    legendre = distribution.angular[0]
    for i in withTrailingZero:
        assert legendre[i].maxOrder == len(mf4.legendre_coefficients[i])


# ---------------------------------------------------------------------------
# The flat path's two losses, pinned
# ---------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason=(
    "AngularDistribution.to_endf() cannot reproduce an LTT=3 section: metadata "
    "has no home for NM, so the second CONT record's N2 is written as 0, and "
    "the coefficient rows are rebuilt from a dense {order: array} then trimmed "
    "of trailing zeros, which lowers NL. Phase 3d reimplements these bodies "
    "over the model, at which point this XPASSes and should be deleted."
))
def test_the_flat_path_round_trips_mf4(mf4):
    assert str(AngularDistribution.from_endf(mf4).to_endf()) == str(mf4)


def test_the_flat_losses_are_exactly_the_two_described(mf4):
    """Pins *what* is lost, so the xfail above is a claim and not a shrug."""
    rebuilt = str(AngularDistribution.from_endf(mf4).to_endf()).split("\n")
    original = str(mf4).split("\n")
    differing = [i for i, (a, b) in enumerate(zip(original, rebuilt)) if a != b]

    # NM, in the second CONT record's N2 field (columns 56-66).
    assert 1 in differing
    assert mf4._nm is not None and int(original[1][55:66]) == mf4._nm
    assert rebuilt[1][55:66].strip() == "", (
        "expected the rebuilt section to have lost NM entirely, leaving the "
        "field blank; it wrote something instead"
    )

    # And every other difference is an NL, or a coefficient line under one.
    assert len(differing) > 1


# ---------------------------------------------------------------------------
# The other three representations, and the refusals
# ---------------------------------------------------------------------------

def test_an_isotropic_section_becomes_isotropic2d_not_an_empty_distribution():
    """LI=1 states the distribution positively; it is not an absence."""
    from kika.endf.classes.mf4.isotropic import MF4MTIsotropic

    section = MF4MTIsotropic(number=102)
    section._za, section._awr, section._mat = 26056.0, 55.45, 2631
    section._li, section._lct = 1, 2

    distribution, provenance, report = decodeMF4MT(section)
    assert isinstance(distribution, Isotropic2d)
    encoded, _ = encodeMF4MT(distribution, provenance, 102, report)
    assert str(encoded) == str(section)


def test_encoding_without_the_endf_provenance_refuses():
    """LTT, LI and LCT are not recoverable from the model, so it must not guess."""
    with pytest.raises(ValueError, match="EndfProvenance"):
        encodeMF4MT(AngularTwoBody(angular=XYs2d()), None, 2)


def test_an_unknown_ltt_is_declared_rather_than_guessed():
    from kika.endf.classes.mf4.base import MF4MT

    section = MF4MT(number=2)
    section._za, section._awr, section._ltt, section._li, section._lct = (
        26056.0, 55.45, 9, 0, 2
    )
    distribution, _, report = decodeMF4MT(section)
    assert distribution is None
    assert any("LTT=9" in entry for entry in report.unsupported)


def test_a_bad_lct_warns_and_does_not_invent_a_frame():
    from kika.endf.classes.mf4.polynomial import MF4MTLegendre

    section = MF4MTLegendre(number=2)
    section._za, section._awr, section._li, section._lct = 26056.0, 55.45, 0, 7
    section._energies = [1.0, 10.0]
    section._legendre_coeffs = [[0.1], [0.2]]
    section._interpolation = [(2, 2)]

    _, _, report = decodeMF4MT(section)
    assert any("LCT=7" in entry for entry in report.warnings)


def test_a_multi_region_tab2_becomes_regions2d():
    """NR > 1 on the outer axis is a ``regions2d``, not a dominant scheme."""
    from kika.endf.classes.mf4.polynomial import MF4MTLegendre

    section = MF4MTLegendre(number=2)
    section._za, section._awr, section._mat = 26056.0, 55.45, 2631
    section._li, section._lct = 0, 2
    section._energies = [1.0, 10.0, 100.0, 1000.0]
    section._legendre_coeffs = [[0.1], [0.2], [0.3], [0.4]]
    section._interpolation = [(2, 2), (4, 5)]
    section._nr, section._ne = 2, 4

    distribution, provenance, report = decodeMF4MT(section)
    assert isinstance(distribution.angular, Regions2d)
    assert [len(r) for r in distribution.angular] == [2, 2]

    encoded, _ = encodeMF4MT(distribution, provenance, 2, report)
    assert str(encoded) == str(section)
    assert encoded._interpolation == [(2, 2), (4, 5)]


def test_the_tab2_split_does_not_duplicate_the_boundary_record():
    """``regions1d`` shares its boundary *point*; a TAB2 stores each record once.

    Copying the boundary into both regions would add an angular distribution to
    the file that the evaluation never wrote.
    """
    from kika.nuclear_data.model import fromEndfTab2, toEndfTab2

    functions = [Legendre(coefficients=[1.0, 0.1 * i], outerDomainValue=float(i))
                 for i in range(5)]
    form = fromEndfTab2(functions, [(2, 2), (5, 1)])
    rebuilt, pairs = toEndfTab2(form)

    assert len(rebuilt) == 5
    assert pairs == [(2, 2), (5, 1)]
    assert [f.outerDomainValue for f in rebuilt] == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_the_angular_probabilities_are_the_arrays_the_file_carries(mf4):
    """The tabulated half, element for element rather than by re-encoding."""
    _, distribution, _ = _roundTrip(mf4, 2)
    tabulated = distribution.angular[1]
    for i, function in enumerate(tabulated):
        mu, p, _ = function.toEndfRegions()
        np.testing.assert_array_equal(mu, np.asarray(mf4.tabulated_cosines[i], dtype=float))
        np.testing.assert_array_equal(
            p, np.asarray(mf4.tabulated_probabilities[i], dtype=float)
        )
