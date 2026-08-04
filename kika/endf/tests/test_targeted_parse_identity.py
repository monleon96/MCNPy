"""A targeted parse must report the same tape identity as a full one.

``read_endf(f, mf_numbers=[...])`` builds a bare ``ENDF`` and fills in only the
requested files. ``mat`` was set solely by ``parse_endf_file``, the full-parse
path, so a targeted parse produced ``mat = None`` — and ``ENDF.zaid`` derives
from ``mat``, so that came back None too.

MAT and ZAID are properties of the tape, not of the sections asked for. Two
parse paths gave two answers for the same file.

``kika/sampling/endf_perturbation.py`` built its output directory from exactly
that call, so every perturbed sample landed in ``endf/unknown/`` while the
pairing stage looked under ``endf/<zaid>/`` and found nothing. That end-to-end
failure is covered by ``test_combined_perturbation``, which needs a tape *and*
NJOY and runs ACER twice; these run on the committed micro-tape in under a
second, so the invariant stays covered in CI.
"""
from __future__ import annotations

import warnings

import pytest

from kika.endf import read_endf


@pytest.fixture(scope="module")
def full(micro_tape):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return read_endf(str(micro_tape))


@pytest.mark.parametrize("mf_numbers", [[1], [3], [4], [1, 3], 2])
def test_a_targeted_parse_finds_the_mat(micro_tape, full, mf_numbers):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        targeted = read_endf(str(micro_tape), mf_numbers=mf_numbers)

    assert targeted.mat == full.mat, (
        f"mf_numbers={mf_numbers!r} gave mat={targeted.mat}, "
        f"a full parse gives {full.mat}"
    )


@pytest.mark.parametrize("mf_numbers", [[1], [3], [4]])
def test_a_targeted_parse_finds_the_zaid(micro_tape, full, mf_numbers):
    """The property kika/sampling actually reads."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        targeted = read_endf(str(micro_tape), mf_numbers=mf_numbers)

    assert targeted.zaid is not None, (
        f"mf_numbers={mf_numbers!r} lost the ZAID; this is what put perturbed "
        f"samples in endf/unknown/"
    )
    assert targeted.zaid == full.zaid


def test_the_fe56_micro_tape_is_mat_2631_zaid_26056(full):
    """Anchor the expected values, so a wrong-but-consistent answer still fails."""
    assert full.mat == 2631
    assert full.zaid == 26056


def test_asking_for_an_absent_mf_still_gives_the_identity(micro_tape, full):
    """MAT comes from the tape, not from the sections that happened to parse."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        targeted = read_endf(str(micro_tape), mf_numbers=[31])

    assert targeted.mat == full.mat
    assert 31 not in targeted.files
