"""GNDS §3.5 units: the tables, the grammar, and the one prohibition.

The registry exists because Juan wanted every quantity to carry its unit, from
prior experience with unit bugs. The bug class is concrete and still in the
tree: ``* 1e6  # MeV -> eV`` written inline, where nothing records what the
number meant on either side. Every conversion below goes through
``conversion_factor``, which refuses when the dimensions disagree.
"""
from __future__ import annotations

import pytest

from kika.nuclear_data.model.units import (
    DERIVED_SI_UNITS,
    PREFIXES,
    SI_UNITS,
    UnitError,
    check_mass_unit,
    conversion_factor,
    parse_unit,
)


# ---------------------------------------------------------------------------
# The tables, against the document
# ---------------------------------------------------------------------------

def test_the_prefix_table_is_table_3_4():
    assert PREFIXES["Y"] == 1e24 and PREFIXES["y"] == 1e-24
    assert len(PREFIXES) == 20
    # The two the spec spells with two characters, and the two that are easy to
    # get wrong: micro is 'mu' (not 'u', not the Greek letter) and deka is 'da'.
    assert PREFIXES["mu"] == pytest.approx(1e-6)
    assert PREFIXES["da"] == pytest.approx(1e1)
    assert "u" not in PREFIXES


def test_the_si_table_is_table_3_6():
    assert set(SI_UNITS) == {"m", "kg", "s", "A", "K", "mol", "cd"}


def test_the_derived_table_is_table_3_7():
    assert DERIVED_SI_UNITS["N"] == "m*kg/s**2"
    assert DERIVED_SI_UNITS["ohm"] == "V/A"  # spelled out, not the Greek letter
    assert len(DERIVED_SI_UNITS) == 18


# ---------------------------------------------------------------------------
# The grammar
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text,factor",
    [
        ("", 1.0),
        ("eV", 1.0),
        ("MeV", 1e6),
        ("keV", 1e3),
        ("mb", 1e-3),
        ("MeV/c**2", 1e6),
        ("kg*m**2/s**2", 1.0),  # exactly a Joule; kg is a base symbol, not kilo+g
        ("kg * m**2 / s**2", 1.0),  # §3.5: spaces are ignored
    ],
)
def test_parse_unit_scale_factors(text, factor):
    assert parse_unit(text).factor == pytest.approx(factor, rel=1e-12)


def test_spaces_are_insignificant():
    assert parse_unit("kg*m**2/s**2") .dimensions == parse_unit("kg * m**2 / s**2").dimensions


def test_a_bare_known_symbol_beats_a_prefix_reading_of_it():
    """``m`` is metre, ``cd`` is candela, ``T`` is tesla.

    Reading the prefix first would make ``m`` milli-nothing and ``cd``
    centi-day. The parser checks the whole token against the symbol tables
    before it tries to peel a prefix, and this is the test that says so.
    """
    assert parse_unit("m").factor == 1.0
    assert dict(parse_unit("m").dimensions) == {"m": 1}
    assert parse_unit("cd").factor == 1.0
    assert parse_unit("T").factor == 1.0


def test_powers_and_division_accumulate():
    unit = parse_unit("m**2/s**2")
    assert {str(k): int(v) for k, v in unit.dimensions.items()} == {"m": 2, "s": -2}


def test_dimensionally_incompatible_conversion_raises():
    with pytest.raises(UnitError, match="cannot convert"):
        conversion_factor("eV", "b")


def test_conversion_between_compatible_units():
    assert conversion_factor("MeV", "eV") == pytest.approx(1e6)
    assert conversion_factor("eV", "MeV") == pytest.approx(1e-6)
    assert conversion_factor("b", "b") == 1.0


@pytest.mark.parametrize("bad", ["zz", "eV**", "eV+MeV", "eV/", "1/", "s/1"])
def test_unspellable_units_raise(bad):
    with pytest.raises(UnitError):
        parse_unit(bad)


@pytest.mark.parametrize("good, dimensions", [
    ("1/s", {"s": -1}),
    ("1/s**2", {"s": -2}),
    ("1", {}),
])
def test_a_bare_one_numerator_is_spellable(good, dimensions):
    """``1/s`` used to raise, and it was ``1/s`` this test used to pin as bad.

    The pin was wrong, and this module was the place it was least likely to be
    noticed: :data:`DERIVED_SI_UNITS` in ``units.py`` spells the admixture of
    ``Hz`` as ``"1/s"``, so the parser rejected a string its own table
    contains — and GNDS files write ``unit="1/s"`` for a decay rate, which is
    what §18.4's ``rate`` needed when MF1/455's precursor constants landed. A
    parser that cannot read what the format writes is not strict, it is broken.
    """
    unit = parse_unit(good)
    assert {str(k): int(v) for k, v in unit.dimensions.items()} == dimensions
    assert unit.factor == 1.0


# ---------------------------------------------------------------------------
# The prohibition
# ---------------------------------------------------------------------------

def test_a_mass_may_not_be_spelled_as_an_energy():
    """§3.5, verbatim: *"the common nuclear physics convention of expressing
    masses in 'MeV' (rather than 'MeV/c**2') should not be allowed."*"""
    with pytest.raises(UnitError, match="is an energy, not a mass"):
        check_mass_unit("MeV")
    with pytest.raises(UnitError, match="is an energy, not a mass"):
        check_mass_unit("eV")


@pytest.mark.parametrize("good", ["MeV/c**2", "eV/c**2", "kg", "amu", "g"])
def test_the_permitted_mass_spellings(good):
    check_mass_unit(good)


def test_the_prohibition_is_opt_in():
    """``parse_unit`` must not enforce it — 'MeV' is a perfectly good energy.

    Whether a number is a mass is the caller's knowledge, not the unit string's,
    so the check is a separate function nothing calls implicitly.
    """
    assert parse_unit("MeV").factor == 1e6
