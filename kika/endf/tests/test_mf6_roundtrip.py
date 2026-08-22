"""MF6: the read-write gate, and the record walk it rests on.

**The gate.** An MF6 section must parse and re-emit as the bytes it arrived as.
That is asserted on nine committed fixtures which between them carry every LAW
that occurs on a tape on this machine, and — ``tape``-marked — on the full
evaluations the cut ones were cut from.

**The walk is the thing being tested.** MF6's law bodies are data-dependent, so
unlike MF5 there is no record-count table to fall back on and no verbatim
escape: a parser that mis-walks one law body reads the next product out of the
middle of the previous one, and every value it reports is plausible. Two things
catch that. :func:`~kika.endf.parsers.parse_mf6._check_consumed` requires the
walk to land exactly on the last record of the section; and the byte gate
requires what comes back out to be what went in. Neither alone is enough — a
walk can consume the right number of records and still assign them to the wrong
product.

**Why these four fixtures and not others.** Swept over all 557 ENDF/B-VIII.1
neutron tapes (12 388 MF6 sections): **LAW=7 occurs twice in the whole library
and both are Be-9's MT16**, which is also one of only 29 ``LCT=1`` sections;
LAW=6 occurs five times, three of them Li-6's MT41; the negative LAWs occur 172
times and all of them are actinide fission, of which U-235's MT18 is the
smallest carrier at 591 records; and C-12's MT5 is the readiest ``LCT=3``. The
census is in ``docs/mf6_notes.md`` in the workspace repo.

**LAW=5 is gated against tapes, and the reason it once was not was wrong.**
Charged-particle elastic scattering does not occur in a neutron sublibrary —
it needs a charged projectile — and this module used to say that settled it,
because ``/share_snc/lib/endf`` was believed to hold neutron and TSL
evaluations only. It does not: ``endfb8/`` is ENDF/B-VIII.0 entire, with
``protons/``, ``deuterons/``, ``tritons/``, ``helium3s/`` and ``alphas/``
alongside ``neutrons/``. The 2026-08-18 sweep looked at ``endfb81/``, which
*is* neutron-only, and generalised from it. 63 charged-particle tapes carry a
LAW=5 each, all in MT2; the five whole ones committed here cover both axes it
splits on and the fifth adds LAW=2/LANG=12. Census in
``docs/mf6_witness_hunt.md`` in kika-workspace.

**What is still unwitnessed, and stays claimed as such.** ``LTP=2``, ``LTP=14``
and ``LTP=15`` occur zero times in those 63 tapes, and ``LAW=1``/``LANG=11-15``
occurs zero times in any library on this machine. Those branches are
implemented, exercised against kika's own emitter, and gated by nothing else —
which is what :func:`test_law5_ltp2_roundtrips_through_our_own_emitter` says in
its name and must keep saying.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from kika.endf import read_endf
from kika.endf.classes.mf6.laws import (
    MF6LawChargedElastic,
    MF6LawContinuum,
    MF6LawElsewhere,
    MF6LawLabAngleEnergy,
    MF6LawNoBody,
    MF6LawPhaseSpace,
    MF6LawTwoBody,
)
from kika.endf.parsers.parse_endf import MF_PARSERS
from kika.endf.parsers.parse_mf6 import parse_mf6_mt
from kika.endf.utils import parse_endf_id

DATA = Path(__file__).resolve().parent / "data"
FIXTURES = {key: DATA / f"micro_{key}_mf6.endf"
            for key in ("be9", "li6", "c12", "u235")}

#: The charged-particle fixtures, kept apart from ``FIXTURES`` because they are
#: whole ENDF/B-VIII.0 evaluations rather than cuts, and because the coverage
#: claim each set makes is a different one. Keyed by the ``(LTP, LIDP)`` cell of
#: LAW=5 they witness; ``t_li7`` carries no cell of its own and is here for
#: LAW=2/LANG=12 and for the padding divergence.
CP_FIXTURES = {key: DATA / f"micro_{key}_mf6.endf"
               for key in ("p_he3", "d_h2", "h3_he4", "a_he4", "t_li7")}

#: ``(fixture key, LTP, LIDP, node count)`` — the four cells LAW=5 splits into,
#: and the smallest tape in ENDF/B-VIII.0 carrying each. All four are occupied,
#: which was not obvious in advance: LIDP=1 needs a target identical to the
#: projectile, and that is five reactions in the whole library.
LAW5_CELLS = [("p_he3", 1, 0, 42), ("d_h2", 1, 1, 67),
              ("h3_he4", 12, 0, 4), ("a_he4", 12, 1, 6)]


def data_lines(text: str, mf: int, mt: int) -> list[str]:
    return [
        line for line in text.splitlines()
        if len(line) >= 75 and parse_endf_id(line)[1:] == (mf, mt)
    ]


def first_difference(expected: list[str], got: list[str]) -> str:
    """Same contract as the MF5 module's: never build a whole-file diff."""
    if expected == got:
        return ""
    if len(expected) != len(got):
        return f"line count {len(expected)} -> {len(got)}"
    for i, (want, have) in enumerate(zip(expected, got)):
        if want != have:
            n_diff = sum(1 for a, b in zip(expected, got) if a != b)
            return (
                f"{n_diff} of {len(expected)} lines differ; first at index {i}\n"
                f"  source: {want!r}\n"
                f"  kika  : {have!r}"
            )
    return "lists differ but no differing line was found"


def roundtrip(path, mt: int, columns: int = 75, skip_head: bool = False) -> str:
    """Parse one MF6 section out of *path* and report how its bytes differ."""
    text = Path(path).read_text()
    endf = read_endf(str(path), mf_numbers=[6])
    section = endf.files[6].sections[mt]
    start = 1 if skip_head else 0
    source = [line[:columns] for line in data_lines(text, 6, mt)][start:]
    got = [line[:columns] for line in data_lines(str(section), 6, mt)][start:]
    return first_difference(source, got)


def all_sections(path) -> dict:
    return read_endf(str(path), mf_numbers=[6]).files[6].sections


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def test_mf6_is_registered():
    assert 6 in MF_PARSERS, "MF6 is not registered in MF_PARSERS"


def test_every_fixture_yields_an_mf6_file():
    for key, path in FIXTURES.items():
        endf = read_endf(str(path))
        assert 6 in endf.files, f"{key}: MF6 did not reach the ENDF object"


# ---------------------------------------------------------------------------
# The byte gate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key,mt", [
    ("be9", 16), ("be9", 600), ("be9", 650), ("be9", 700),
    ("be9", 701), ("be9", 800),
    ("li6", 41), ("li6", 52), ("li6", 103),
    ("u235", 18), ("u235", 800),
])
def test_fixture_sections_roundtrip(key, mt):
    assert not roundtrip(FIXTURES[key], mt)


def test_c12_roundtrips_apart_from_its_plain_decimal_head():
    """C-12 writes ``6012.00000`` where kika re-emits ``6.012000+3``.

    The same measured divergence MF5 pins on Pu-239, in the same field of the
    same kind of record, and there is nothing here to fix: re-emitting a
    non-canonical field verbatim would mean storing the source text of every
    number on the tape. Asserted explicitly so that the day it changes, this
    says so — and so that the 4 657 records *after* the HEAD stay gated.
    """
    text = FIXTURES["c12"].read_text()
    section = all_sections(FIXTURES["c12"])[5]

    source_head = data_lines(text, 6, 5)[0]
    kika_head = data_lines(str(section), 6, 5)[0]
    assert source_head[:11].strip() == "6012.00000"
    assert kika_head[:11].strip() == "6.012000+3"
    assert section._za == pytest.approx(6012.0)

    assert not roundtrip(FIXTURES["c12"], 5, skip_head=True)


# ---------------------------------------------------------------------------
# The walk
# ---------------------------------------------------------------------------

def test_every_fixture_section_consumes_all_its_records():
    """``_check_consumed`` fires inside the parser; this is what notices.

    The per-MT ``try/except`` in ``parse_mf6`` turns a mis-walk into a warning
    and a *missing section*, not an exception — so the assertion that matters
    is that every section in the file came back.
    """
    expected = {"be9": 6, "li6": 3, "c12": 1, "u235": 2}
    for key, n in expected.items():
        sections = all_sections(FIXTURES[key])
        assert len(sections) == n, (
            f"{key}: {n} MF6 sections on the tape, {len(sections)} parsed — "
            f"a dropped section means the record walk raised"
        )


def test_declared_nk_matches_the_products_read():
    for key, path in FIXTURES.items():
        for mt, section in all_sections(path).items():
            assert section._nk == len(section.products), (
                f"{key} MT{mt}: HEAD declares NK={section._nk}, "
                f"{len(section.products)} products were built"
            )


def test_law1_list_widths_are_nep_times_na_plus_two():
    """``NW = NEP*(NA+2)`` — the identity the LAW=1 reshape depends on.

    A byte gate cannot see this: the values round-trip whatever the reshape
    thinks their shape is, and a wrong ``NA`` would only surface as nonsense
    from :meth:`MF6LawContinuum.block`.
    """
    checked = 0
    for path in FIXTURES.values():
        for section in all_sections(path).values():
            for product in section.products:
                law = product.law_data
                if not isinstance(law, MF6LawContinuum):
                    continue
                for k in range(len(law.incident_energies)):
                    assert len(law.values[k]) == law.nep(k) * (law.na[k] + 2)
                    assert law.nd[k] <= law.nep(k), "ND cannot exceed NEP"
                    assert law.block(k).shape == (law.nep(k), law.na[k] + 2)
                    checked += 1
    # 357 on the fixtures as committed. Asserted as a floor so that a fixture
    # regenerated without its LAW=1 sections cannot make this test vacuous.
    assert checked >= 357, f"only {checked} LAW=1 records reached the check"


def test_law7_is_nested_three_deep():
    """Be-9 MT16 carries the library's only two LAW=7 bodies."""
    section = all_sections(FIXTURES["be9"])[16]
    bodies = [p.law_data for p in section.products
              if isinstance(p.law_data, MF6LawLabAngleEnergy)]
    assert len(bodies) == 2
    for body in bodies:
        assert body.blocks, "TAB2 over incident energy is empty"
        for block in body.blocks:
            assert block.angles, "TAB2 over cosine is empty"
            for angle in block.angles:
                assert len(angle.e_out) == len(angle.f)
                assert len(angle.e_out) > 1
        e_out, f = body.table(0, 0)
        assert e_out.shape == f.shape


def test_jp_survives_the_round_trip():
    """U-235's MT18 HEAD writes ``JP=11`` in a field that is 0 everywhere else.

    It was written back as 0 until the byte gate on this fixture said otherwise.
    A HEAD field that is almost always zero is exactly the kind that gets
    hard-coded, and on MT18 it is not decoration — it says whether the products
    listed are the prompt ones or the total.
    """
    assert all_sections(FIXTURES["u235"])[18].jp == 11
    assert all_sections(FIXTURES["be9"])[16].jp == 0


def test_lct_covers_all_three_frames():
    """LCT=1, 2 and 3 all occur, and all three fixtures that carry them say so."""
    assert all_sections(FIXTURES["be9"])[16]._lct == 1
    assert all_sections(FIXTURES["li6"])[41]._lct == 2
    assert all_sections(FIXTURES["c12"])[5]._lct == 3
    assert "lab" in all_sections(FIXTURES["c12"])[5].frame


# ---------------------------------------------------------------------------
# Which laws the fixtures actually carry
# ---------------------------------------------------------------------------

def test_the_fixtures_between_them_carry_every_law_on_a_tape_here():
    """The coverage claim this module's docstring makes, asserted.

    If a fixture is ever regenerated from a different evaluation, this is what
    notices that a law lost its only witness. LAW=5 is not in this set and is
    not expected to be — :func:`test_law5_fixtures_cover_all_four_cells` does
    the same job for the charged-particle fixtures.
    """
    seen = set()
    for path in FIXTURES.values():
        for section in all_sections(path).values():
            seen.update(p.law for p in section.products)
    assert seen == {-15, -5, 0, 1, 2, 3, 4, 6, 7}, (
        f"law coverage drifted: {sorted(seen)}. LAW=5 is absent from this set "
        f"because it cannot occur in a neutron sublibrary; its witnesses are "
        f"CP_FIXTURES."
    )


def test_law_bodies_get_the_class_their_law_says():
    expected = {
        -15: MF6LawElsewhere, -5: MF6LawElsewhere,
        0: MF6LawNoBody, 3: MF6LawNoBody, 4: MF6LawNoBody,
        1: MF6LawContinuum, 2: MF6LawTwoBody,
        6: MF6LawPhaseSpace, 7: MF6LawLabAngleEnergy,
    }
    for path in FIXTURES.values():
        for section in all_sections(path).values():
            for product in section.products:
                assert isinstance(product.law_data, expected[product.law])


# ---------------------------------------------------------------------------
# What the section says it does not have
# ---------------------------------------------------------------------------

def test_deferred_photon_distributions_are_reported_as_gaps():
    """``LAW=-15`` sends the distribution to MF15, which kika does not read.

    Every law is decoded and ``report_gaps()`` is still not empty, which is the
    whole point of it: the section is complete as *read*, and incomplete as
    *data*, and only the second is a statement about the evaluation.
    """
    section = all_sections(FIXTURES["u235"])[18]
    gaps = section.report_gaps()
    assert len(gaps) == sum(1 for p in section.products if p.law == -15)
    assert all("MF15" in g for g in gaps)
    assert all("does not parse" in g for g in gaps)


def test_neutron_deferrals_are_not_reported_because_mf5_is_read():
    """``LAW=-5`` points at MF5, which kika parses — so it is not a gap.

    The asymmetry is the information. A reader that reported both would be
    saying it cannot follow a pointer it can follow.
    """
    section = all_sections(FIXTURES["u235"])[18]
    assert any(p.law == -5 for p in section.products)
    assert not any("MF5 " in g or "(LAW=-5)" in g for g in section.report_gaps())


def test_sections_with_everything_decoded_report_nothing():
    for key in ("be9", "li6", "c12"):
        for mt, section in all_sections(FIXTURES[key]).items():
            assert section.report_gaps() == [], f"{key} MT{mt}"


# ---------------------------------------------------------------------------
# Accessors over the flat values
# ---------------------------------------------------------------------------

def test_kalbach_returns_r_and_a_and_nan_where_a_is_not_given():
    """C-12's MT5 is LANG=2 throughout — the Kalbach-Mann representation.

    ``a`` is optional in ENDF-6: with ``NA=1`` the evaluator gives only the
    pre-compound fraction and leaves the slope to Kalbach's systematics. It is
    returned as NaN and not 0.0, because a zero slope is a physically
    meaningful isotropic value and would be taken for one.
    """
    section = all_sections(FIXTURES["c12"])[5]
    body = section.products[0].law_data
    assert isinstance(body, MF6LawContinuum) and body.lang == 2

    e_out, r, a = body.kalbach(0)
    assert e_out.shape == r.shape == a.shape
    assert np.all(np.diff(e_out) > 0), "outgoing energies must ascend"
    if body.na[0] == 1:
        assert np.all(np.isnan(a))
    else:
        assert np.all(np.isfinite(a))


def test_spectrum_is_available_whatever_lang_says():
    """``f0`` is the angle-integrated shape for every LANG, so one accessor does."""
    for key in ("be9", "c12"):
        for section in all_sections(FIXTURES[key]).values():
            for product in section.products:
                body = product.law_data
                if not isinstance(body, MF6LawContinuum):
                    continue
                e_out, f0 = body.spectrum(0)
                assert e_out.shape == f0.shape
                assert np.all(f0 >= 0), "a probability density cannot be negative"


def test_legendre_and_kalbach_refuse_the_wrong_lang():
    """Asking a Kalbach body for Legendre coefficients is a mistake, not a view."""
    kalbach = all_sections(FIXTURES["c12"])[5].products[0].law_data
    with pytest.raises(ValueError, match="LANG=2"):
        kalbach.legendre(0)

    legendre = next(
        (p.law_data
         for section in all_sections(FIXTURES["be9"]).values()
         for p in section.products
         if isinstance(p.law_data, MF6LawContinuum) and p.law_data.lang == 1),
        None,
    )
    assert legendre is not None, "no LANG=1 body in the Be-9 fixture"
    with pytest.raises(ValueError, match="not Kalbach-Mann"):
        legendre.kalbach(0)


def test_discrete_lines_split_off_from_the_continuum():
    """``ND`` is where the resolved inelastic levels are; losing it makes a line
    spectrum into a continuum."""
    seen = 0
    for path in FIXTURES.values():
        for section in all_sections(path).values():
            for product in section.products:
                body = product.law_data
                if not isinstance(body, MF6LawContinuum):
                    continue
                for k in range(len(body.incident_energies)):
                    if body.nd[k]:
                        e_lines, f_lines = body.discrete_lines(k)
                        assert len(e_lines) == body.nd[k] == len(f_lines)
                        seen += 1
    assert seen, "no fixture carries a discrete line — ND is untested"


def test_two_body_legendre_and_tabulated_split_on_lang():
    section = all_sections(FIXTURES["li6"])[52]
    body = next(p.law_data for p in section.products
                if isinstance(p.law_data, MF6LawTwoBody))
    for k, lang in enumerate(body.lang):
        if lang == 0:
            assert body.legendre(k).shape == (body.nl(k),)
            with pytest.raises(ValueError, match="LANG=0"):
                body.tabulated(k)
        else:
            mu, f = body.tabulated(k)
            assert mu.shape == f.shape == (body.nl(k),)
            assert np.all(mu >= -1) and np.all(mu <= 1)


# ---------------------------------------------------------------------------
# Parsing behaviour
# ---------------------------------------------------------------------------

def test_unknown_law_raises_naming_the_product():
    """A LAW with no record layout must stop, not guess.

    Sharper than MF5's equivalent: MF5 can step over an unknown LF from a record
    count, so guessing there is merely optional. MF6 bodies are self-describing,
    so there is no length to guess *with* — the parser cannot find where the
    next product starts, and anything it does after this point is fiction.
    """
    lines = [
        " 9.223500+4 2.330248+2          0          2          1          09228 6  5",
        " 1.000000+0 1.000000+0          0         42          1          29228 6  5",
        "          2          2                                            9228 6  5",
        " 1.000000-5 1.000000+0 2.000000+7 1.000000+0                      9228 6  5",
    ]
    with pytest.raises(ValueError, match=r"MF6/MT5 product 0 uses LAW=42"):
        parse_mf6_mt(lines, 5)


def test_a_bad_section_costs_only_its_own_mt(tmp_path):
    """One unreadable MT must not take the rest of the file with it."""
    text = FIXTURES["li6"].read_text().splitlines(keepends=True)
    out = []
    for line in text:
        if len(line) >= 75 and parse_endf_id(line)[1:] == (6, 41) and line[:11].strip():
            # Corrupt MT41's HEAD so it claims far more products than it has.
            line = line[:44] + f"{999:>11}" + line[55:]
            out.append(line)
            continue
        out.append(line)
    broken = tmp_path / "broken.endf"
    broken.write_text("".join(out))

    sections = read_endf(str(broken), mf_numbers=[6]).files[6].sections
    assert 41 not in sections, "the corrupted section should not have parsed"
    assert {52, 103} <= set(sections), "the healthy sections were lost with it"


# ---------------------------------------------------------------------------
# LAW=5 — charged-particle elastic scattering, against real tapes
# ---------------------------------------------------------------------------

def law5_body(path):
    """The one LAW=5 product of a charged-particle tape.

    Every one of the 63 carriers has exactly one, and every one is in MT2 —
    which is what LAW=5 *is*, so a tape with two would mean the walk went
    wrong, and that is asserted rather than assumed.
    """
    sections = all_sections(path)
    found = [(mt, product.law_data)
             for mt, section in sections.items()
             for product in section.products
             if product.law == 5]
    assert len(found) == 1, f"{Path(path).name}: {len(found)} LAW=5 products"
    mt, body = found[0]
    assert mt == 2, f"{Path(path).name}: LAW=5 in MT{mt}, not MT2"
    assert isinstance(body, MF6LawChargedElastic)
    return body


@pytest.mark.parametrize("key,ltp,lidp,nodes", LAW5_CELLS)
def test_law5_fixtures_cover_all_four_cells(key, ltp, lidp, nodes):
    """One fixture per ``(LTP, LIDP)``, and each is the cell it claims.

    If a fixture is ever re-pointed at another evaluation, this is what notices
    that a cell lost its only witness — the same job
    :func:`test_the_fixtures_between_them_carry_every_law_on_a_tape_here` does
    for the neutron set.
    """
    body = law5_body(CP_FIXTURES[key])
    assert body.lidp == lidp
    assert sorted(set(body.ltp)) == [ltp]
    assert len(body.ltp) == nodes


@pytest.mark.parametrize("key", [c[0] for c in LAW5_CELLS])
def test_law5_sections_roundtrip(key):
    """The gate that the synthetic test could not be: the evaluator's bytes.

    All four are byte-identical in columns 1-66 — no format divergence at all,
    not even the seven-digit rounding that costs C-12 its head line. Over the
    whole 63-tape sweep 62 LAW=5 sections are byte-identical and the 63rd
    (``h-002_He_003`` MT2) differs only by that rounding, in 846 lines of
    3 410; there is not one record-layout difference in 683 204 lines.
    """
    assert not roundtrip(CP_FIXTURES[key], 2, columns=66)


@pytest.mark.parametrize("key,ltp,lidp,nodes", LAW5_CELLS)
def test_law5_body_length_is_the_identity_its_ltp_and_lidp_dictate(
        key, ltp, lidp, nodes):
    """``NW`` is not free: ``LTP`` and ``LIDP`` fix it, and arithmetic says why.

    With distinguishable particles the nuclear-amplitude expansion runs to
    order ``2NL`` (``2NL+1`` reals) and carries ``NL+1`` complex interference
    amplitudes (``2NL+2`` reals): ``NW = 4NL+3``. With identical particles only
    the even orders survive (``NL+1`` reals) beside the same ``2NL+2``:
    ``NW = 3NL+3``. For ``LTP>2`` the body is (μ, P) pairs: ``NW = 2NL``.

    This is the assertion that would have caught the synthetic test this
    replaces, which built a ``LTP=1``/``LIDP=0`` node with ``NL=3`` and three
    values where ENDF-6 requires fifteen. It held on all 4 710 nodes of the
    63-tape sweep, 931 of them ``LTP=1`` and 3 779 ``LTP=12``.
    """
    body = law5_body(CP_FIXTURES[key])
    for k in range(len(body.ltp)):
        nl = body.nl(k)
        if body.ltp[k] == 1:
            expected = 4 * nl + 3 if body.lidp == 0 else 3 * nl + 3
        else:
            expected = 2 * nl
        assert len(body.values[k]) == expected, (
            f"{key} node {k}: LTP={body.ltp[k]} LIDP={body.lidp} NL={nl} "
            f"wants NW={expected}, body has {len(body.values[k])}"
        )


@pytest.mark.parametrize("key,ltp,lidp,nodes", LAW5_CELLS)
def test_law5_accessors_reshape_the_body_its_ltp_says(key, ltp, lidp, nodes):
    """The reshape the class declined to do until there was a tape to check it.

    ``values`` stays flat — re-emitting what was read cannot be wrong, and that
    is why the byte gate above passes — but a caller asking for the physics
    should not have to redo this arithmetic. The accessors are gated here
    against every node of the cell, not against a constructed example.
    """
    body = law5_body(CP_FIXTURES[key])
    for k in range(len(body.ltp)):
        if body.ltp[k] == 1:
            nuclear, interference = body.amplitudes(k)
            nl = body.nl(k)
            assert len(nuclear) == (2 * nl + 1 if body.lidp == 0 else nl + 1)
            assert interference.shape == (nl + 1,)
            assert interference.dtype == complex
            with pytest.raises(ValueError, match="LTP=1"):
                body.tabulated(k)
        else:
            mu, sigma = body.tabulated(k)
            assert len(mu) == len(sigma) == body.nl(k)
            assert np.all(np.diff(mu) > 0), "cosines must ascend"
            assert mu[0] >= -1.0 and mu[-1] <= 1.0
            # Deliberately no positivity check — see the next test.
            with pytest.raises(ValueError, match="LTP="):
                body.amplitudes(k)


def test_law5_tabulated_is_normalised_but_not_a_density():
    """Both halves of the warning on the accessor, gated on committed bytes.

    The ``LTP>2`` table integrates to 1 and still takes negative values. Over
    the 3 779 nodes on the share, 3 777 integrate to 1 ± 1 % — the two that do
    not are ``p-080_Hg_199`` and ``p-080_Hg_204`` near 83 MeV, where the table
    spans six orders of magnitude and the trapezoid is just inaccurate — and
    3 280 go negative, down to -5.4e+05.

    Worth its own test because the shape is a trap: it looks exactly like
    ``MF6LawTwoBody.tabulated``'s ``f(mu)``, which *is* an ordinary density. The
    normalisation half is what a reader checks and finds reassuring; the
    signedness half is what makes clipping, renormalising or sampling it wrong.
    Asserting both here stops either from being "fixed" into the other.
    """
    # Normalised: both committed LTP>2 fixtures, every node.
    for key in ("h3_he4", "a_he4"):
        body = law5_body(CP_FIXTURES[key])
        for k in range(len(body.ltp)):
            mu, values = body.tabulated(k)
            assert np.trapezoid(values, mu) == pytest.approx(1.0, abs=0.01), (
                f"{key} node {k} is not normalised"
            )

    # Not a density: the committed fixtures happen to be non-negative, so the
    # signedness is gated on the share instead of quietly going unasserted.
    # 45 kB of p-001_H_002 would buy only this one fact.
    tape = Path("/share_snc/lib/endf/endfb8/protons/p-001_H_002.endf")
    if not tape.exists():
        pytest.skip("ENDF/B-VIII.0 protons not reachable")
    body = law5_body(tape)
    minima = [body.tabulated(k)[1].min() for k in range(len(body.ltp))]
    assert min(minima) < 0, (
        "p-001_H_002 MT2 has no negative value — either the reshape changed or "
        "this is not the nuclear-plus-interference remainder after all"
    )


def test_law5_accessors_refuse_a_body_endf6_does_not_admit():
    """The reshape checks the identity every time, and this is why.

    The synthetic test these replace built a ``LTP=1``/``LIDP=0`` node with
    ``NL=3`` and three values, where ENDF-6 requires ``4·3+3 = 15``. It passed
    for two days because nothing looked. Reshaping such a body would not raise
    — three values divide into a (μ, P) table just fine — it would return
    plausible numbers from a section that cannot exist, which is the failure
    mode the whole MF6 module is built to refuse.
    """
    body = MF6LawChargedElastic(
        spi=0.5, lidp=0, incident_energies=[1.0e5],
        ltp=[1], nl_values=[3], values=[[1.0, 2.0, 3.0]],
    )
    with pytest.raises(ValueError, match="requires NW=15, body has 3"):
        body.amplitudes(0)

    # And the same body relabelled as a table is refused on its own identity:
    # NL=3 wants six values, not three.
    body.ltp = [12]
    with pytest.raises(ValueError, match="requires NW=6, body has 3"):
        body.tabulated(0)


def test_law5_ltp2_roundtrips_through_our_own_emitter():
    """``LTP=2``, ``LTP=14`` and ``LTP=15`` have no witness — self-consistency only.

    The four committed cells settle ``LTP=1`` and ``LTP=12`` against ENDF-6.
    They do not touch the other three: ``LTP=2`` (nuclear amplitude expansion
    with the interference term summed rather than tabulated) and ``LTP=14/15``
    (log interpolation on the table) occur **zero times in all 63
    charged-particle tapes**, 4 710 nodes. So this proves the reader and the
    writer agree with each other and **nothing about whether either agrees with
    ENDF-6**. That distinction is the point of the name and must not be quietly
    upgraded; if a tape carrying one ever lands, replace this with a fixture.

    The body is sized by the same identity the witnessed cells obey, which is
    the one thing the old synthetic test got wrong.
    """
    from kika.endf.classes.mf6.base import MF6MT
    from kika.endf.classes.mf6.products import MF6Product

    nl = 3
    body = MF6LawChargedElastic(
        spi=0.5, lidp=0, tab2_interp=[(2, 2)],
        incident_energies=[1.0e5, 2.0e7],
        ltp=[2, 14], nl_values=[nl, nl],
        # LTP=2: 4*NL+3 = 15 reals.  LTP=14: 2*NL = 6, three (mu, P) pairs.
        values=[[float(i) for i in range(4 * nl + 3)],
                [-1.0, 0.25, 0.0, 0.5, 1.0, 0.25]],
    )
    section = MF6MT(
        number=5, _za=1001.0, _awr=0.9991673, _lct=2, _nk=1, _mat=125,
        products=[MF6Product(
            zap=1001.0, awp=0.9991673, lip=0,
            y_interp=[(2, 2)], y_energies=[1.0e5, 2.0e7], y_values=[1.0, 1.0],
            law_data=body,
        )],
    )
    text = str(section)
    reparsed = parse_mf6_mt(text.splitlines()[:-1], 5)

    assert str(reparsed) == text
    again = reparsed.products[0].law_data
    assert isinstance(again, MF6LawChargedElastic)
    assert again.spi == pytest.approx(0.5)
    assert again.ltp == [2, 14]
    assert again.values == body.values


# ---------------------------------------------------------------------------
# LAW=2/LANG=12, and the padding divergence that comes with it
# ---------------------------------------------------------------------------

def test_law2_lang12_has_a_witness_at_last():
    """The tabulated two-body form, which no neutron tape here carries.

    ``docs/mf6_notes.md`` recorded LANG=0 on 209 623 nodes and LANG=14 on 490
    over all 557 ENDF/B-VIII.1 neutron tapes, and **no LANG=12 at all**. The 63
    charged-particle tapes carry 503 LANG=12 nodes against 1 201 LANG=0; nine
    of them are this fixture's MT50. So ``MF6LawTwoBody.tabulated`` is now read
    from an evaluator's bytes on both of the branches it claims to serve.
    """
    section = all_sections(CP_FIXTURES["t_li7"])[50]
    body = section.products[0].law_data
    assert isinstance(body, MF6LawTwoBody)
    assert set(body.lang) == {12}
    assert len(body.lang) == 9

    for k in range(len(body.lang)):
        mu, f = body.tabulated(k)
        assert len(mu) == len(f) == body.nl(k)
        assert np.all(np.diff(mu) > 0)
        with pytest.raises(ValueError, match="LANG=12"):
            body.legendre(k)


def test_interp_padding_is_probed_once_per_section_and_that_is_wrong():
    """One line in 361 re-emitted wrongly, and the reason is a stated assumption.

    :class:`~kika.endf.utils.PaddingProbe` says of the convention it probes that
    it is "a property that does not change within a section", and takes the
    first short record of each kind. This section falsifies that. Its two
    products disagree: the LAW=2 product's interpolation records end in blanks,
    and the LAW=4 product's ends in four explicit zeros::

        source:           2          2          0          0          0          0
        kika  :           2          2

    Five lines across the 63 charged-particle tapes do this, one in each of
    ``d-003_Li_007`` MT700, ``p-003_Li_007`` MT50 and MT650, ``p-004_Be_009``
    MT50 and this one — and in all five the split falls on a product boundary.

    **Not fixed here, and deliberately.** The probe serves MF3, MF5 and MF7 on
    the same assumption, so a per-product grain would be an MF6-shaped patch to
    a shared claim — and per-product is a guess of exactly the same shape as the
    per-section one that just failed. The grain that is actually right is per
    record, replayed positionally, and that is its own increment across four
    files. Pinned rather than left silent, in the way C-12's head line is: the
    day the emitter learns this, this test fails and says so.
    """
    text = CP_FIXTURES["t_li7"].read_text()
    section = all_sections(CP_FIXTURES["t_li7"])[50]

    source = [line[:66] for line in data_lines(text, 6, 50)]
    got = [line[:66] for line in data_lines(str(section), 6, 50)]
    assert len(source) == len(got) == 41

    differing = [i for i, (a, b) in enumerate(zip(source, got)) if a != b]
    assert differing == [39], (
        f"the padding divergence moved or was fixed: lines {differing} differ"
    )
    assert source[39] == "          2          2          0          0          0          0"
    assert got[39] == "          2          2" + " " * 44

    assert [p.law for p in section.products] == [2, 4], (
        "the divergence is a product boundary — if the products changed, so did "
        "what this test is pinning"
    )


def test_charged_particle_fixtures_roundtrip_apart_from_that_one_line():
    """Everything else in the five whole tapes comes back as it went in.

    Eleven MF6 sections over 138 kB of evaluator bytes, and the only divergence
    in the lot is the one line above — not even the seven-digit rounding that
    costs C-12 its head line. Asserted as a total rather than per section so
    that a new divergence anywhere cannot hide behind a passing sibling.
    """
    n_lines = n_diff = 0
    for key, path in CP_FIXTURES.items():
        text = path.read_text()
        for mt, section in all_sections(path).items():
            source = [line[:66] for line in data_lines(text, 6, mt)]
            got = [line[:66] for line in data_lines(str(section), 6, mt)]
            assert len(source) == len(got), f"{key} MT{mt}: line count changed"
            n_lines += len(source)
            n_diff += sum(1 for a, b in zip(source, got) if a != b)

    assert n_lines == 1187, f"fixture line count drifted: {n_lines}"
    assert n_diff == 1, f"{n_diff} lines differ, expected only the padding one"


def test_a_targeted_mf6_parse_finds_the_mat_and_zaid():
    """``read_endf(path, mf_numbers=[6])`` must agree with a full parse.

    ``test_targeted_parse_identity`` asks this of MF1/3/4 against the Fe-56
    fixture, which carries no MF6; the same question for MF6 belongs here, next
    to the fixtures that do.
    """
    for key, path in FIXTURES.items():
        full = read_endf(str(path))
        targeted = read_endf(str(path), mf_numbers=[6])
        assert targeted.mat == full.mat, key
        assert targeted.zaid == full.zaid, key


# ---------------------------------------------------------------------------
# The path a user actually takes
# ---------------------------------------------------------------------------

def test_splicing_every_section_back_into_a_tape_changes_only_its_send_records(
        tmp_path):
    """``str(section)`` is half the write path; this is the other half.

    ``write_mf_section_to_file`` finds the section's boundaries in the source
    file and splices the new text between them, so it can go wrong in ways the
    section's own ``__str__`` cannot — wrong boundaries, a lost FEND, a changed
    record width. Every MF6 section of a fixture is written back in turn and the
    result compared with the original.

    **The only difference is the SEND record**, and it is not MF6's. Li-6 leaves
    columns 1-66 of its SEND blank; ``format_endf_send_record`` zero-fills them.
    One line per section, no data on it, shared with every other MF, and already
    recorded in ``docs/mf7_tsl_notes.md``. ENDF/B-VIII.1's O-16 zero-fills its
    SENDs and splices back **byte-identical**, which is what says the rest of
    the path is exact.
    """
    import shutil
    from kika.endf.writers._section_writer import write_mf_section_to_file

    source = FIXTURES["li6"]
    current = tmp_path / source.name
    shutil.copy(source, current)

    sections = all_sections(source)
    for mt, section in sorted(sections.items()):
        out = tmp_path / f"spliced_{mt}.endf"
        write_mf_section_to_file(str(current), section, str(out),
                                 update_directory=False, match_source_width=True)
        current = out

    before = source.read_text().splitlines()
    after = current.read_text().splitlines()
    assert len(before) == len(after), "the splice changed the tape's length"

    changed = [(a, b) for a, b in zip(before, after) if a != b]
    assert len(changed) == len(sections), (
        f"{len(changed)} lines changed for {len(sections)} sections; only the "
        f"SEND of each should differ"
    )
    for a, b in changed:
        assert a[70:75] == b[70:75] == " 6  0", "a non-SEND record changed"
        assert not a[:66].strip(), "the source SEND was not blank after all"
        assert b[:66].strip(), "kika's SEND was not zero-filled after all"


# ---------------------------------------------------------------------------
# The full tapes the fixtures were cut from
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tape,mts", [
    ("be9_b81", (16, 600, 650, 700, 701, 800)),
    ("li6_b81", (32, 41, 52, 103, 105)),
])
def test_full_tape_roundtrip(tape, mts, request):
    """The fixtures are verbatim cuts, so this mostly re-asserts the gate —
    except for the sections the cut dropped, which nothing else covers."""
    path = request.getfixturevalue(f"{tape}_tape")
    for mt in mts:
        assert not roundtrip(path, mt), f"{tape} MT{mt}"


#: The five charged-particle sublibraries of ENDF/B-VIII.0, and how many tapes
#: each holds. Named so the sweep below fails loudly if a sublibrary goes
#: missing, rather than quietly sweeping four of five and reporting a census.
CP_SUBLIBRARIES = {"protons": 49, "deuterons": 5, "tritons": 5,
                   "helium3s": 3, "alphas": 1}


@pytest.mark.slow
def test_the_whole_charged_particle_census_holds(request):
    """The numbers the committed fixtures are chosen from, re-measured.

    Four fixtures gate four cells. This gates the claim those four were picked
    *out of* — that the cells are the only cells, that the identity has no
    exception in 4 710 nodes, and that ``LTP=2/14/15`` and ``LANG=11-15`` are
    empty rather than merely unlooked-at. Without it those numbers live only in
    ``docs/mf6_witness_hunt.md``, where nothing re-checks them.

    ~21 s over the share, which is why it is ``slow``-marked and skips when the
    share is not there, like :func:`test_endfb81_o16_roundtrip`.
    """
    root = Path("/share_snc/lib/endf/endfb8")
    if not root.is_dir():
        pytest.skip("ENDF/B-VIII.0 charged-particle sublibraries not reachable")

    tapes = []
    for sub, expected in CP_SUBLIBRARIES.items():
        found = sorted((root / sub).glob("*.endf"))
        assert len(found) == expected, (
            f"{sub}: {len(found)} tapes, census says {expected} — the sweep "
            f"below would report a different library's numbers"
        )
        tapes.extend(found)
    assert len(tapes) == 63

    law5_mts, lidp, ltp_nodes = [], Counter(), Counter()
    law_products, law1_lang, law2_lang = Counter(), Counter(), Counter()
    for path in tapes:
        n_law5 = 0
        for mt, section in all_sections(path).items():
            for product in section.products:
                law_products[product.law] += 1
                body = product.law_data
                if product.law == 1 and hasattr(body, "lang"):
                    law1_lang[body.lang] += 1
                if product.law == 2 and hasattr(body, "lang"):
                    law2_lang.update(body.lang)
                if product.law != 5:
                    continue
                n_law5 += 1
                law5_mts.append(mt)
                lidp[body.lidp] += 1
                ltp_nodes.update(body.ltp)
                for k in range(len(body.ltp)):
                    nl = body.nl(k)
                    if body.ltp[k] <= 2:
                        want = 4 * nl + 3 if body.lidp == 0 else 3 * nl + 3
                    else:
                        want = 2 * nl
                    assert len(body.values[k]) == want, (
                        f"{path.name} MT{mt} node {k}: LTP={body.ltp[k]} "
                        f"LIDP={body.lidp} NL={nl} wants {want}, "
                        f"got {len(body.values[k])}"
                    )
        assert n_law5 == 1, f"{path.name}: {n_law5} LAW=5 products, expected 1"

    assert set(law5_mts) == {2}, f"LAW=5 outside MT2: {sorted(set(law5_mts))}"
    assert dict(lidp) == {0: 58, 1: 5}
    assert dict(ltp_nodes) == {1: 931, 12: 3779}
    assert sum(ltp_nodes.values()) == 4710
    assert dict(law_products) == {1: 2400, 2: 29, 4: 41, 5: 63, 6: 6}

    # LAW=2/LANG=12 has a witness here and nowhere in the neutron sublibrary.
    assert dict(law2_lang) == {0: 1201, 12: 503}
    # LAW=1/LANG=11-15 has none here either. The wide sweep over every library
    # on the share is in ``myworkspace/mf6_witness/`` in kika-workspace; until
    # it logs "=== DONE", zero hits there means "not looked at yet".
    assert dict(law1_lang) == {1: 2204, 2: 196}


@pytest.mark.slow
def test_endfb81_o16_roundtrip(request):
    """O-16 is the tape that found the interpolation-padding defect.

    Its MF6 zero-fills the unused fields of an interpolation record, where every
    tape the TSL work measured blank-fills them — 52 records of MT5 alone were
    written back wrongly by an emitter with no way to say which convention it
    had read. Kept as a gate because the fixtures are all blank-padded and so
    cannot see it.
    """
    path = Path("/share_snc/lib/endf/endfb81/n-008_O_016.endf")
    if not path.exists():
        pytest.skip("ENDF/B-VIII.1 O-16 not reachable")
    for mt in (5, 16, 22):
        assert not roundtrip(path, mt), f"O-16 MT{mt}"
