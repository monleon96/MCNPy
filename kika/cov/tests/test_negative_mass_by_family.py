"""Piece B of ``docs/library/autofix_in_the_model.md``: where the negativity lives.

``autofix='medium'`` scored which pair of reaction blocks carried the negative
eigenmodes and then removed it. The scoring survives here as a diagnostic on
the ``definiteness`` finding, so a person reads "91 % of the negative mass is
in MT4 × MT16" and decides what to do about the *evaluation* -- leave MT16 out
of the request, or fix the file -- rather than a repair deciding for them.
"""
from __future__ import annotations

import numpy as np
import pytest

from kika.cov.conditioning import inspect_blocks, negative_mass_by_family


def _psd(n, seed):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    return a @ a.T / n


@pytest.fixture
def poisoned():
    """Two PSD reaction blocks whose cross block is scaled until the joint is not."""
    a, b = _psd(6, 1), _psd(6, 2)
    cross = 1.2 * np.linalg.cholesky(a) @ np.linalg.cholesky(b).T  # 'correlation' > 1
    joint = np.block([[a, cross], [cross.T, b]])
    assert np.linalg.eigvalsh(joint).min() < 0, "the fixture must be indefinite"
    return joint, ["MT4"] * 6 + ["MT16"] * 6


def test_the_split_is_exact(poisoned):
    joint, families = poisoned
    pairs = negative_mass_by_family(joint, families)
    assert {tuple(p["pair"]) for p in pairs} == {("MT4", "MT4"), ("MT4", "MT16"),
                                                ("MT16", "MT16")}
    values = np.linalg.eigvalsh(joint)
    assert np.isclose(sum(p["mass"] for p in pairs), values[values < 0].sum())
    assert np.isclose(sum(p["share"] for p in pairs), 1.0)


def test_the_cross_block_is_named_as_the_culprit(poisoned):
    joint, families = poisoned
    top = negative_mass_by_family(joint, families)[0]
    assert top["pair"] == ["MT4", "MT16"]
    assert top["share"] > 1.0, "the diagonal blocks pull the other way"


def test_a_psd_matrix_has_no_negative_mass():
    assert negative_mass_by_family(_psd(8, 3), ["a"] * 4 + ["b"] * 4) == []


def test_the_finding_says_where_when_told_the_families(poisoned):
    joint, families = poisoned
    told = inspect_blocks({"j": joint}, families={"j": families})
    (finding,) = told.findings("definiteness")
    assert "of the negative mass sits in MT4 × MT16" in finding.summary
    attribution = finding.evidence["negative_mass_by_family"]
    assert attribution[0]["pair"] == ["MT4", "MT16"]

    untold = inspect_blocks({"j": joint})
    (finding,) = untold.findings("definiteness")
    assert "negative mass" not in finding.summary
    assert finding.evidence["negative_mass_by_family"] == []


def test_the_attribution_is_gated_like_the_predictions(poisoned):
    joint, families = poisoned
    cheap = inspect_blocks({"j": joint}, families={"j": families}, predict=False)
    (finding,) = cheap.findings("definiteness")
    assert finding.evidence["negative_mass_by_family"] == []


def test_mismatched_families_are_refused(poisoned):
    joint, _families = poisoned
    with pytest.raises(ValueError, match="labels for a 12-row block"):
        negative_mass_by_family(joint, ["x"] * 5)
