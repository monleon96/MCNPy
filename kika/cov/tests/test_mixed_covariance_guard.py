"""§7.4: a section that states several matrices is not a section with one.

**The defect this pins was an asymmetry, not an omission.** Both entry points
met a §25.2 ``mixed`` the same way and answered differently:
``CrossSectionCovariance.from_covariance_section`` raised — with
``"covariance section carries no matrix"``, which is not what happened, the
section carries two or three — and ``LegendreCovariance.from_covariance_suite``
did ``continue``, dropping the section and handing back a carrier that looks
complete. The silent one was on the MF34 path, which is the thesis path.

**Two ways in, on purpose.** The MF33 case runs on a published file: F-19 is the
smallest evaluation in ENDF/B-VIII.1-GNDS carrying a ``mixed``, it is committed,
and it needs no shared tree. The MF34 case has to be hand-built, because *no*
committed fixture puts a ``mixed`` on a ``34,x`` section — La-139 has thirteen
and every one is ``33,x``. Building it here is the convention the first half of
``kika/gnds/tests/test_covariances.py`` already uses, and it is better than
minting a fixture to reach one branch.
"""
from __future__ import annotations

import numpy as np
import pytest

from kika.cov.cross_section_covariance import CrossSectionCovariance
from kika.cov.legendre_covariance import LegendreCovariance
from kika.gnds.covariances import readCovarianceSuite
from kika.gnds.xpath import Document
from kika.nuclear_data.model import CovarianceMatrix, Mixed


class _RowData:
    """The §25.2.3 link, as little of it as the two readers ask for."""

    def __init__(self, ENDF_MFMT, legendreOrder=None):
        self.ENDF_MFMT = ENDF_MFMT
        self.legendreOrder = legendreOrder


class _Section:
    def __init__(self, form, ENDF_MFMT, legendreOrder=None, label="eval"):
        self.form = form
        self.label = label
        self.rowData = _RowData(ENDF_MFMT, legendreOrder)
        self.columnData = None
        self.provenance = None


def _mixed(*shapes):
    """A ``Mixed`` of square matrices of the given orders, on matching grids."""
    return Mixed(
        label="eval",
        components=[
            CovarianceMatrix(
                matrix=np.eye(n),
                rowGrid=np.linspace(1e3, 2e7, n + 1),
                isRelative=True,
            )
            for n in shapes
        ],
    )


# ---------------------------------------------------------------------------
# MF33: the loud one, whose message was wrong
# ---------------------------------------------------------------------------

def test_the_message_says_how_many_components_there_are():
    section = _Section(_mixed(3, 628), "33/1")
    with pytest.raises(ValueError) as raised:
        CrossSectionCovariance.from_covariance_section(section, nuclide=26056)

    message = str(raised.value)
    assert "<mixed> of 2 components" in message
    assert "'eval'" in message, "the section is not named"
    assert "shortRangeSelfScalingVariance" in message, "the reason is not given"
    assert "docs/gnds_endf_conflicts.md" in message, "nowhere to go"


def test_a_sum_is_refused_as_a_sum_and_not_as_a_mixed():
    """The other matrix-less form. Its components are elsewhere, not merged."""
    from kika.nuclear_data.model import Sum, Summand

    section = _Section(Sum(summands=[Summand(href="#a"), Summand(href="#b")]),
                       "33/1")
    with pytest.raises(ValueError, match=r"<sum> of 2 summands"):
        CrossSectionCovariance.from_covariance_section(section, nuclide=26056)


def test_f19s_own_mixed_sections_raise_rather_than_answering(gnds_data_dir):
    """The published case, end to end through the GNDS reader."""
    suite, _ = readCovarianceSuite(
        Document.parse(gnds_data_dir / "Covariances/n-009_F_019.endf.gnds-covar.xml")
    )
    mixed = [s for s in suite.covarianceSections if isinstance(s.form, Mixed)]
    assert mixed, "F-19 lost its mixed sections"

    for section in mixed:
        with pytest.raises(ValueError, match="components, not as one matrix"):
            CrossSectionCovariance.from_covariance_section(section, nuclide=9019)


def test_a_section_with_no_form_at_all_says_to_read_the_report():
    section = _Section(None, "33/1")
    with pytest.raises(ValueError, match="carries no covariance form at all"):
        CrossSectionCovariance.from_covariance_section(section, nuclide=26056)


# ---------------------------------------------------------------------------
# MF34: the silent one
# ---------------------------------------------------------------------------

def test_an_mf34_mixed_no_longer_vanishes_from_the_carrier():
    """This is the whole point of the increment.

    Before the guard the assembled ``LegendreCovariance`` had zero matrices and
    said nothing, so a caller comparing it against JEFF would be comparing an
    empty carrier.
    """
    suite = [_Section(_mixed(3, 40), "34/2", legendreOrder=1)]
    with pytest.raises(ValueError, match=r"MF34 covariance section 'eval'"):
        LegendreCovariance.from_covariance_suite(suite, nuclide=26056)


def test_the_mf34_filter_still_runs_first():
    """A ``mixed`` on an MF33 section in the same suite is not this reader's.

    The guard sits *after* the ``34,x`` filter, so widening it into "raise on
    every mixed in the suite" would break every Fe-56 file — MT1, MT2 and MT102
    are all mixed there, and all MF33.
    """
    suite = [_Section(_mixed(3, 628), "33/1")]
    assembled = LegendreCovariance.from_covariance_suite(suite, nuclide=26056)
    assert assembled.num_matrices == 0


def test_a_plain_mf34_matrix_still_assembles():
    """The guard returns the form, so the ordinary path is unchanged."""
    section = _Section(
        CovarianceMatrix(matrix=np.eye(3), rowGrid=np.linspace(1e3, 2e7, 4),
                         isRelative=True),
        "34/2", legendreOrder=1,
    )
    assembled = LegendreCovariance.from_covariance_suite([section], nuclide=26056)
    assert assembled.num_matrices == 1
    np.testing.assert_array_equal(assembled.matrices[0], np.eye(3))
