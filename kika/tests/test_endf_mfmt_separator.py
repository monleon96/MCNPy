"""``ENDF_MFMT`` is spelled two ways, and every reader of it takes both.

`kika/tests/` holds invariants of the repository as a whole. This is the third
of them, next to ``test_layering.py`` and ``test_duck_typed_consumers.py``.

**The trap this closes, stated once.** GNDS §25.2.3 (p. 363) defines a
``covarianceSection``'s ``ENDF_MFMT`` as a *comma*-separated pair, and all 270
covariance files of ENDF/B-VIII.1-GNDS write ``"33,2"``. kika's **ENDF** adapter
writes ``"33/2"`` instead. Nine sites across ``kika/sampling``, ``kika/cov`` and
``kika/endf`` used to parse the slash and only the slash, so a ``CovarianceSuite``
that arrived through :func:`kika.gnds.decode.readCovarianceSuite` met filters
that matched **nothing** — and, being filters rather than parsers, did not
raise. The caller assembled zero blocks and carried on. Full account:
``docs/library/gnds_endf_conflicts.md`` §3.1 and §7.1.

**Why this is a repository-wide test rather than four module-local ones.** The
four modules deliberately do *not* share a helper — ``kika/sampling`` may not
import ``kika.endf.model_adapter`` (``test_nothing_imports_the_adapter``), and
``kika/cov`` duck-types the suite so it does not depend on the model at all. The
duplication is the right call and it is also exactly why the defect existed in
four places at once. So the invariant lives where the duplication is visible.

**What is *not* asserted here.** That the adapter should write a comma. It
still writes a slash, on purpose — two of these modules are the deployed thesis
pipeline, and the separator at the *source* is a gated change. What is asserted
is that no reader cares, which is the half that is safe today and the half that
makes the source change a one-liner later.
"""
from __future__ import annotations

import pytest

from kika.nuclear_data.model import (CovarianceMatrix, CovarianceSection,
                                     CovarianceSuite, DataLink)

import numpy as np

#: The same section, said both ways. Nothing else about them differs.
SPELLINGS = ("33/2", "33,2")


def _suite(mfmt: str, mf: int = 33, mt: int = 2) -> CovarianceSuite:
    """A one-section suite whose row link carries *mfmt* verbatim."""
    grid = np.array([1e3, 1e5, 2e7])
    return CovarianceSuite(covarianceSections=[CovarianceSection(
        label=f"MF{mf}-MT{mt}",
        rowData=DataLink(href=f"/reactionSuite/reactions/reaction[@label='MT{mt}']"
                              f"/crossSection", ENDF_MFMT=mfmt),
        form=CovarianceMatrix(matrix=np.eye(2) * 0.01, rowGrid=grid,
                              columnGrid=grid, isRelative=True),
    )])


# ----------------------------------------------------------------------
# the model's own accessor
# ----------------------------------------------------------------------

@pytest.mark.parametrize("mfmt", SPELLINGS)
def test_the_model_accessor_reads_both(mfmt):
    link = DataLink(href="/x", ENDF_MFMT=mfmt)
    assert (link.ENDF_MF, link.ENDF_MT) == (33, 2)


@pytest.mark.parametrize("mfmt", [None, "", "33", "thirty-three,2", "33,2,1x"])
def test_an_unreadable_mfmt_is_none_and_not_a_crash(mfmt):
    """``None`` is a answer the callers can act on; ``ValueError`` from inside
    a list comprehension three frames down is not."""
    link = DataLink(href="/x", ENDF_MFMT=mfmt)
    assert link.ENDF_MF is None or link.ENDF_MT is None


# ----------------------------------------------------------------------
# the four duplicated readers
# ----------------------------------------------------------------------

@pytest.mark.parametrize("mfmt", SPELLINGS)
def test_sampling_model_blocks_selects_either(mfmt):
    from kika.sampling.model_blocks import _endf_mt, _is_endf_mf

    link = DataLink(href="/x", ENDF_MFMT=mfmt)
    assert _is_endf_mf(link, 33) is True
    assert _is_endf_mf(link, 34) is False
    assert _endf_mt(link) == 2


@pytest.mark.parametrize("mfmt", SPELLINGS)
def test_cov_legendre_selects_either(mfmt):
    from kika.cov.legendre_covariance import _endf_mt_of, _is_endf_mf

    link = DataLink(href="/x", ENDF_MFMT=mfmt.replace("33", "34"))
    assert _is_endf_mf(link, 34) is True
    assert _endf_mt_of(link) == 2


@pytest.mark.parametrize("mfmt", SPELLINGS)
def test_cross_section_covariance_takes_the_mt_from_either(mfmt):
    from kika.cov.cross_section_covariance import CrossSectionCovariance

    section = _suite(mfmt).covarianceSections[0]
    covariance = CrossSectionCovariance.from_covariance_section(
        section, nuclide=26056
    )
    assert covariance.reaction_rows == [2]


@pytest.mark.parametrize("mfmt", SPELLINGS)
def test_the_endf_encoder_selects_either(mfmt):
    """The encoder is the direction that would put the defect in a *file*.

    A GNDS-read suite handed to ``encodeMF33MT`` used to raise "no MF33
    covariance sections for MT2" at a suite that had one.
    """
    from kika.endf.model_adapter.covariances import _covarianceSections

    selected = _covarianceSections(_suite(mfmt), 33, 2)
    assert len(selected) == 1
    assert _covarianceSections(_suite(mfmt), 33, 2)[0].label == "MF33-MT2"

    with pytest.raises(ValueError, match="no MF34 covariance sections"):
        _covarianceSections(_suite(mfmt), 34, 2)


# ----------------------------------------------------------------------
# the ratchet
# ----------------------------------------------------------------------

def test_no_module_parses_the_separator_by_hand_again():
    """A grep, as a test, because the next copy of this defect is a paste.

    The failure mode is not a wrong number — it is a filter that matches
    nothing — so nothing else in the suite would notice a regression here.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    offenders = []
    # `ENDF_MFMT` followed by a slash-only parse or a slash-terminated prefix.
    pattern = re.compile(
        r'ENDF_MFMT[^\n]*?(?:\.split\(["\']/["\']\)|startswith\(["\']\d\d/)'
    )
    for path in root.rglob("*.py"):
        if "tests" in path.parts or path.name == __file__.rsplit("/", 1)[-1]:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            if line.lstrip().startswith(("#", "*", '"')) or "``" in line:
                continue                      # prose about the defect, not code
            if pattern.search(line):
                offenders.append(f"{path.relative_to(root)}: {line.strip()}")

    assert not offenders, (
        "these parse ENDF_MFMT on the slash alone, so they select nothing from "
        "a GNDS-decoded suite and do not raise. Use the (ENDF_MF, ENDF_MT) "
        "properties or the module's separator-agnostic helper:\n  "
        + "\n  ".join(offenders)
    )
