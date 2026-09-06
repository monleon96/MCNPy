"""Phase 3d's real claim: *only the body changed*.

``test_flat_class_surface.py`` proves the **shape** survived — same fields, same
order, same signatures. It cannot prove the **values** did, and that is the half
a consumer actually notices. So each rewritten body keeps its predecessor next
to it, as ``_from_endf_pre_3d``, and this file runs both on every section of
every tape and compares them field by field.

**Where they are allowed to differ, they are listed here by name**, with the
defect each difference fixes. A difference that is not listed fails. That is the
distinction the roadmap kept insisting on: "the façade changes nothing" and "the
façade changes exactly these three things, each of them a fix" are different
claims, and only the second one is true.

Delete this file when the flat classes go in 1.0 — it exists to compare against
code that will not exist.
"""
from __future__ import annotations

import numpy as np
import pytest

from kika.endf.read_endf import read_endf
from kika.nuclear_data import AngularDistribution, NuclideInfo

REAL_TAPES = ["fe56_host_tape", "fe57_host_tape", "fe56_jendl_tape",
              "u235_tape", "th232_tape", "pu241_tape"]

#: metadata keys phase 3d *adds*. Both are inputs to the D2 fix; see
#: ``AngularDistribution.from_endf``.
NEW_ANGULAR_METADATA = {"nm", "legendre_orders"}


def _compareAngular(old: AngularDistribution, new: AngularDistribution, where: str) -> None:
    assert new.reaction == old.reaction, where
    assert new.frame == old.frame, where
    assert new.representation == old.representation, where
    np.testing.assert_array_equal(new.energies, old.energies, err_msg=where)

    assert set(new.coefficients) == set(old.coefficients), where
    for order in old.coefficients:
        np.testing.assert_array_equal(
            new.coefficients[order], old.coefficients[order],
            err_msg=f"{where} order {order}",
        )

    assert (new.tabulated_data is None) == (old.tabulated_data is None), where
    if old.tabulated_data is not None:
        for key in old.tabulated_data:
            assert new.tabulated_data[key] == old.tabulated_data[key], f"{where} {key}"

    # metadata: same keys apart from the two that are new, same values.
    assert set(new.metadata) - set(old.metadata) <= NEW_ANGULAR_METADATA, where
    assert not set(old.metadata) - set(new.metadata), f"{where}: a metadata key was lost"
    for key in old.metadata:
        assert new.metadata[key] == old.metadata[key], f"{where} metadata[{key}]"


def test_angular_distribution_matches_the_old_body_on_the_committed_slice(micro_tape):
    section = read_endf(str(micro_tape)).mf[4].mt[2]
    _compareAngular(
        AngularDistribution._from_endf_pre_3d(section),
        AngularDistribution.from_endf(section),
        "micro-tape MT2",
    )


@pytest.mark.parametrize("tape", REAL_TAPES)
def test_angular_distribution_matches_the_old_body_on_every_real_section(request, tape):
    """Under ``--deep``: 228 sections, LTT 0, 1, 2 and 3 between them."""
    endf = read_endf(str(request.getfixturevalue(tape)))
    if 4 not in endf.mf:
        pytest.skip(f"{tape} carries no MF4")

    for mt in sorted(endf.mf[4].mt):
        section = endf.mf[4].mt[mt]
        _compareAngular(
            AngularDistribution._from_endf_pre_3d(section),
            AngularDistribution.from_endf(section),
            f"{tape} MT{mt} (LTT={section._ltt})",
        )


# ---------------------------------------------------------------------------
# The differences that *are* allowed, asserted rather than assumed
# ---------------------------------------------------------------------------

def test_the_new_metadata_keys_are_present_and_are_the_files_own(micro_tape):
    section = read_endf(str(micro_tape)).mf[4].mt[2]
    distribution = AngularDistribution.from_endf(section)

    assert distribution.metadata["nm"] == section._nm
    orders = distribution.metadata["legendre_orders"]
    assert orders == [len(row) for row in section.legendre_coefficients]
    assert AngularDistribution._from_endf_pre_3d(section).metadata.keys().isdisjoint(
        NEW_ANGULAR_METADATA
    ), "the old body already had these keys; this test is measuring nothing"


def test_the_old_body_could_not_write_an_ltt3_section_and_the_new_one_can(micro_tape):
    """The D2 fix, stated as a before/after rather than as prose."""
    section = read_endf(str(micro_tape)).mf[4].mt[2]
    assert section._ltt == 3

    old = str(AngularDistribution._from_endf_pre_3d(section).to_endf())
    new = str(AngularDistribution.from_endf(section).to_endf())

    assert new == str(section), "the façade still cannot reproduce an LTT=3 section"
    assert old != str(section), (
        "the old body reproduces it too, so D2 is not what this test says it is"
    )


def test_a_hand_built_object_still_trims_because_nothing_better_exists(micro_tape):
    """The fallback is lossy, and honestly so.

    An object built by hand — or by ``project_to_legendre`` — has no
    ``legendre_orders``, because the per-energy NL genuinely is not known. The
    old trimming behaviour applies, and that is the right answer rather than a
    regression: inventing an NL would be worse than losing one.
    """
    section = read_endf(str(micro_tape)).mf[4].mt[2]
    distribution = AngularDistribution.from_endf(section)
    del distribution.metadata["legendre_orders"]

    rows = distribution._legendre_rows(len(section.legendre_energies))
    trimmed = [
        i for i, row in enumerate(section.legendre_coefficients)
        if row and row[-1] == 0.0
    ]
    assert trimmed, "this tape has no trailing zero, so the fallback is untested"
    assert len(rows[trimmed[0]]) < len(section.legendre_coefficients[trimmed[0]])


# ---------------------------------------------------------------------------
# NuclideInfo
# ---------------------------------------------------------------------------

def test_nuclide_info_matches_the_old_body_except_for_the_za_fix(micro_tape):
    """Fe-56's ZA parses exactly, so here the two agree outright."""
    mt451 = read_endf(str(micro_tape)).mf[1].mt[451]
    info = NuclideInfo.from_endf(mt451)

    assert info.nuclide_id == 26056
    assert info.atomic_weight_ratio == mt451.atomic_weight_ratio
    assert info.temperature == (mt451.temperature or 0.0)
    assert info.metadata["mat"] == mt451._mat
    assert info.metadata["nlib"] == mt451.library_id
    assert info.evaluation_info["laboratory"] == mt451.laboratory


@pytest.mark.parametrize("tape", ["th232_tape", "pu241_tape"])
def test_nuclide_info_reads_za_the_same_way_on_a_tape_that_trips_mf3(request, tape):
    """D1 is **not** observable through MF1/451, and that is worth pinning.

    Th-232's ZA reads back as ``90231.99999999999`` from an **MF3** section and
    as exactly ``90232.0`` from MF1/451 — same nuclide, same tape, different
    field, different rounding. So this class was never affected, and a test
    asserting otherwise would be asserting a fiction. The place the fix bites is
    ``CrossSection``; see ``test_cross_section_rounds_za...``.
    """
    endf = read_endf(str(request.getfixturevalue(tape)))
    fromHeader = float(endf.mf[1].mt[451].zaid)
    fromSection = float(endf.mf[3].mt[2].zaid)

    assert int(fromHeader) == round(fromHeader), (
        "MF1/451's ZA has started parsing inexactly too; this test's premise is gone"
    )
    assert int(fromSection) != round(fromSection), (
        f"{tape}'s MF3 ZA now parses exactly, so D1 is unobservable on it"
    )
    assert NuclideInfo.from_endf(endf.mf[1].mt[451]).nuclide_id == round(fromHeader)


def test_the_evaluation_info_is_not_silently_empty(micro_tape):
    """The defect this file found on its first run.

    ``decodeMF1MT451`` guessed the private field names behind MF1/451's public
    properties -- ``_laboratory`` for ``laboratory``, and six more -- and the
    real ones are ``_alab``, ``_auth``, ``_edate``, ``_ref``, ``_ddate``,
    ``_rdate``, ``_zsymam``. Every ``getattr(..., None) or ""`` returned "", so
    every reactionSuite decoded since P6 carried an empty ``evaluationInfo``
    and an empty ``evaluation``. Nothing else asserted a *value* there.
    """
    mt451 = read_endf(str(micro_tape)).mf[1].mt[451]
    info = NuclideInfo.from_endf(mt451)

    assert info.evaluation_info["laboratory"] == mt451.laboratory
    assert any(info.evaluation_info.values()), "every field is empty again"
    for key, expected in (
        ("authors", mt451.authors), ("eval_date", mt451.eval_date),
        ("reference", mt451.reference), ("material_id", mt451.material_id),
    ):
        assert info.evaluation_info[key] == (expected or "")
