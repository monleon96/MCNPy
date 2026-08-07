"""``metadata`` is an untyped dict, and it is public API. This pins it.

**Why.** Every flat class carries ``metadata: Dict`` — "format-specific extras
preserved for lossless round-trips". Four packages reach into it by string key
and nothing declares what the keys are. ``CrossSection.to_endf`` fails without
``qm``/``qi``/``lr``; ``kika/processing/reconstruct.py`` reads ``awr`` and
``mat``. An untyped dict with load-bearing keys is API whether it is documented
or not.

Phase 3c replaces this dict with a typed ``Provenance``. The conversion is
lossy the moment a key is forgotten, and nothing today would notice — the
round-trip tests would still pass, because a missing ``qm`` only surfaces when
someone writes ENDF back out. So the key sets get written down first, here,
while the dict is still the only implementation.

**On value types.** ENDF's fixed-format floats are parsed to ``int`` whenever
the value happens to be integral, so ``emax`` comes back as ``150000000`` and
``abundance`` as ``1``, both ``int``, on this tape — while on another tape the
same fields would be ``float``. That is why float-valued ENDF fields are
checked as ``NUMBER`` (int or float) and not as ``float``. A typed
``Provenance`` that declares ``emax: float`` will be wrong on the first tape it
meets; ``test_endf_float_fields_are_not_reliably_float`` records that trap
rather than leaving phase 3c to rediscover it.
"""
from __future__ import annotations

import pytest

from kika.endf.read_endf import read_endf
from kika.nuclear_data import (
    AngularDistribution,
    CrossSection,
    NuclideInfo,
    ResonanceParameters,
)

#: Value-type categories. NUMBER exists because of the int/float trap above.
STR, INT, NUMBER, LIST = "str", "int", "number", "list"

_CHECK = {
    STR: lambda v: isinstance(v, str),
    INT: lambda v: isinstance(v, int) and not isinstance(v, bool),
    NUMBER: lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    LIST: lambda v: isinstance(v, list),
}

#: ENDF-sourced ``metadata``, harvested from the committed micro-tape.
ENDF_METADATA: dict[str, dict[str, str]] = {
    "CrossSection": {
        "mat": INT,
        "awr": NUMBER,
        "qm": NUMBER,
        "qi": NUMBER,
        "lr": INT,
        "interpolation_regions": LIST,
    },
    "AngularDistribution": {
        "mat": INT,
        "awr": NUMBER,
        "ltt": INT,
        "li": INT,
        "lct": INT,
        "energy_interpolation": LIST,
        "tab_interpolation": LIST,
        # Added by phase 3d, deliberately, and the only two keys this contract
        # has ever gained. They are what lets `to_endf` write an LTT=3 section
        # back byte for byte -- `docs/library-gaps.md` D2. `nm` is the
        # evaluation's declared highest Legendre order, which had nowhere to
        # live; `legendre_orders` is each energy's own NL, without which a
        # trailing zero the evaluator wrote is indistinguishable from the
        # padding `coefficients` adds, and was trimmed on the way out.
        "nm": INT,
        "legendre_orders": LIST,
    },
    "ResonanceParameters": {
        "mat": INT,
        "awr": NUMBER,
        "lrf": INT,
        "nlsc": INT,
        "abundance": NUMBER,
    },
    "NuclideInfo": {
        # Added when `docs/library-gaps.md` M2 was fixed, and the two keys are
        # the whole of what made `to_endf` possible. MF1/451's NWD descriptive
        # records -- up to 700 lines of an evaluator's comment block, parsed
        # into nothing else -- and its NXC directory had no home in the fields
        # or here, so the section could be read and not written. NWD and NXC
        # themselves are deliberately absent: they are these lists' lengths,
        # and a stored count that can disagree with what it counts is the shape
        # of the MF34 defect 98d7d23 fixed.
        "text": LIST,
        "directory": LIST,
        "mat": INT,
        "lrp": INT,
        "lfi": INT,
        "nlib": INT,
        "nmod": INT,
        "elis": NUMBER,
        "sta": INT,
        "lis": INT,
        "liso": INT,
        "nfor": INT,
        "awi": NUMBER,
        "emax": NUMBER,
        "lrel": INT,
        "nsub": INT,
        "nver": INT,
        "ldrv": INT,
    },
}

#: ``NuclideInfo.evaluation_info`` is a second untyped dict, all strings.
ENDF_EVALUATION_INFO = {
    "laboratory", "authors", "eval_date", "reference",
    "dist_date", "revision_date", "material_id",
}

#: ACE-sourced ``metadata``. A different namespace in the same dict — which is
#: the structural problem ``Provenance`` exists to fix.
ACE_METADATA: dict[str, dict[str, str]] = {
    "CrossSection": {
        "source_format": STR,
        "ace_zaid": INT,
        "ace_extension": STR,
        "awr": NUMBER,
        "ace_comment": STR,
        "ace_date": STR,
        # Gained when `docs/library-gaps.md` D4 was fixed. ACE's LQR block
        # carries one QI per reaction and `from_ace` had never read it, so
        # callers were passing `to_endf(qi=...)` by hand for a number that was
        # in the file. Present only for an MT that has an LQR entry — the
        # fixture below asks for MT 2, whose Q is zero by definition. A
        # composite has none, and that absence is data.
        "qi": NUMBER,
    },
    "AngularDistribution": {
        "source_format": STR,
        "ace_zaid": INT,
        "ace_distribution_type": STR,
    },
    "NuclideInfo": {
        "source_format": STR,
        "ace_extension": STR,
        "ace_matid": INT,
        "ace_format_version": STR,
    },
}

ACE_EVALUATION_INFO = {"date", "source", "reference"}


def _assert_contract(label: str, actual: dict, expected: dict[str, str]) -> None:
    assert set(actual) == set(expected), (
        f"{label}: metadata keys drifted\n"
        f"  lost:  {sorted(set(expected) - set(actual))}\n"
        f"  new:   {sorted(set(actual) - set(expected))}"
    )
    wrong = {
        k: (kind, type(actual[k]).__name__)
        for k, kind in expected.items()
        if not _CHECK[kind](actual[k])
    }
    assert not wrong, "\n".join(
        f"{label}.metadata[{k!r}] should be {kind}, is {got}"
        for k, (kind, got) in wrong.items()
    )


# ---------------------------------------------------------------------------
# ENDF side — fast lane, committed micro-tape
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def endf_objects(micro_tape):
    """One of each flat class, decoded from the committed Fe-56 slice."""
    endf = read_endf(str(micro_tape))
    resonances = ResonanceParameters.from_endf(endf.mf[2].mt[151])
    return {
        "CrossSection": CrossSection.from_endf(endf.mf[3].mt[2]),
        "AngularDistribution": AngularDistribution.from_endf(endf.mf[4].mt[2]),
        "ResonanceParameters": resonances[0],
        "NuclideInfo": NuclideInfo.from_endf(endf.mf[1].mt[451]),
    }


@pytest.mark.parametrize("cls_name", sorted(ENDF_METADATA))
def test_endf_metadata_keys_are_unchanged(endf_objects, cls_name):
    _assert_contract(cls_name, endf_objects[cls_name].metadata, ENDF_METADATA[cls_name])


def test_endf_evaluation_info_keys_are_unchanged(endf_objects):
    assert set(endf_objects["NuclideInfo"].evaluation_info) == ENDF_EVALUATION_INFO
    for key, value in endf_objects["NuclideInfo"].evaluation_info.items():
        assert isinstance(value, str), f"evaluation_info[{key!r}] is {type(value).__name__}"


def test_to_endf_consumes_exactly_the_keys_this_file_pins(endf_objects):
    """The concrete reason the ``CrossSection`` key set is load-bearing.

    ``to_endf`` reads ``qm``/``qi``/``lr`` out of ``metadata`` and raises when
    they are absent — that is the phase 1 fix for silently writing Q = 0. Drop
    one of those keys in phase 3c and this is what breaks.
    """
    xs = endf_objects["CrossSection"]
    assert {"qm", "qi", "lr"} <= set(xs.metadata)

    stripped = CrossSection(
        energies=xs.energies, values=xs.values, reaction=xs.reaction,
        nuclide_id=xs.nuclide_id, metadata={"mat": xs.metadata["mat"]},
    )
    with pytest.raises(ValueError):
        stripped.to_endf()


def test_endf_float_fields_are_not_reliably_float(endf_objects):
    """Records the int/float trap for phase 3c's typed ``Provenance``.

    ENDF writes these in its fixed-format float notation, but the parser hands
    back ``int`` when the value is integral. On this tape ``emax`` is
    ``150000000`` and ``abundance`` is ``1`` — both ``int``. A ``Provenance``
    field annotated ``float`` would be a lie here, and annotating it ``int``
    would be a lie on the next tape. It has to accept both, or the decoder has
    to coerce at the boundary and say so.
    """
    assert isinstance(endf_objects["NuclideInfo"].metadata["emax"], int)
    assert isinstance(endf_objects["ResonanceParameters"].metadata["abundance"], int)


# ---------------------------------------------------------------------------
# ACE side — needs the shared tree, auto-marked `tape`
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ace_objects(fe56_ace):
    from kika.ace.parsers import read_ace

    ace = read_ace(str(fe56_ace))
    return {
        "CrossSection": CrossSection.from_ace(ace, 2),
        "AngularDistribution": AngularDistribution.from_ace(ace, 2),
        "NuclideInfo": NuclideInfo.from_ace(ace),
    }


@pytest.mark.parametrize("cls_name", sorted(ACE_METADATA))
def test_ace_metadata_keys_are_unchanged(ace_objects, cls_name):
    _assert_contract(cls_name, ace_objects[cls_name].metadata, ACE_METADATA[cls_name])


def test_ace_evaluation_info_keys_are_unchanged(ace_objects):
    assert set(ace_objects["NuclideInfo"].evaluation_info) == ACE_EVALUATION_INFO


def test_ace_carries_qi_but_not_qm_or_lr(ace_objects):
    """The structural half of the phase 1 Q = 0 fix, with D4's correction.

    ``to_endf`` on an ACE-sourced section must still refuse rather than
    silently write Q = 0 — but the reason is narrower than this test used to
    assert. ACE **does** carry QI, one per reaction in the LQR block, and
    ``from_ace`` reads it. QM and LR are the two ENDF header fields ACE has no
    counterpart for, so those are what the refusal is about.

    The distinction is the whole of ``docs/library-gaps.md`` D4: "ACE has no Q
    values" was written down, believed, and made callers hand-supply a number
    the file already held.
    """
    xs = ace_objects["CrossSection"]
    assert "qi" in xs.metadata, "the LQR block is not reaching the flat class"
    assert not ({"qm", "lr"} & set(xs.metadata))

    with pytest.raises(ValueError) as excinfo:
        xs.to_endf()
    message = str(excinfo.value)
    assert "qm" in message and "lr" in message
    assert "carries no qm/lr" in message, (
        f"the refusal should name only the fields ACE really lacks: {message}"
    )
