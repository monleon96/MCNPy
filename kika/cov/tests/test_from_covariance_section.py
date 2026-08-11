"""The model as one more source format for ``CrossSectionCovariance``.

MF35 is why this exists. Its bands reach the model directly and never pass
through ``kika/cov``, so until now there was no way to hand one to
``plot_covariance_heatmap`` — the covariance was readable, samplable and
unplottable. The adapter is deliberately generic: it takes a §25.2
``covarianceSection``, so MF31, MF33 and MF34 sections work through it too.

The second half of this file is about the defect that made the first half
useless when it was written. Every MF35 outgoing-energy grid starts at exactly
0 eV, and ``to_heatmap_data`` used to clamp a non-positive edge to ``1e-300``
before taking ``log10``. That is arithmetically safe and visually fatal: the
first bin came out 295 decades wide, covered 98 % of the figure, and the plot
rendered as one flat block of colour — which reads as "this covariance has no
structure" rather than "this axis is broken". The correlations underneath run
from -0.999 to +0.999.
"""
from __future__ import annotations

import numpy as np
import pytest

from kika.cov.cross_section_covariance import CrossSectionCovariance, _log_edges


class _Form:
    def __init__(self, matrix, rowGrid, isRelative=False):
        self.matrix = np.asarray(matrix, dtype=float)
        self.rowGrid = None if rowGrid is None else np.asarray(rowGrid, dtype=float)
        self.isRelative = isRelative


class _RowData:
    def __init__(self, ENDF_MFMT):
        self.ENDF_MFMT = ENDF_MFMT


class _Section:
    def __init__(self, matrix, rowGrid, ENDF_MFMT="35/18", isRelative=False):
        self.form = _Form(matrix, rowGrid, isRelative)
        self.rowData = None if ENDF_MFMT is None else _RowData(ENDF_MFMT)


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------

def test_a_section_becomes_a_single_matrix_covariance():
    section = _Section([[4.0, 1.0], [1.0, 9.0]], [1.0, 2.0, 3.0])
    covariance = CrossSectionCovariance.from_covariance_section(
        section, nuclide=98252)

    assert len(covariance.matrices) == 1
    np.testing.assert_array_equal(covariance.matrices[0],
                                  [[4.0, 1.0], [1.0, 9.0]])
    assert covariance.isotope_rows == [98252] and covariance.reaction_rows == [18]
    assert covariance.isotope_cols == [98252] and covariance.reaction_cols == [18]
    assert covariance.energy_grid == [1.0, 2.0, 3.0]
    assert covariance.energy_grids == [[1.0, 2.0, 3.0]]
    assert covariance.is_relative == [False]


def test_the_mt_comes_from_endf_mfmt_when_not_given():
    assert CrossSectionCovariance.from_covariance_section(
        _Section([[1.0]], [1.0, 2.0], ENDF_MFMT="33/102")).reaction_rows == [102]


def test_an_explicit_mt_wins_over_the_header():
    assert CrossSectionCovariance.from_covariance_section(
        _Section([[1.0]], [1.0, 2.0], ENDF_MFMT="35/18"), mt=452
    ).reaction_rows == [452]


def test_relative_sections_stay_relative():
    section = _Section([[0.01]], [1.0, 2.0], isRelative=True)
    assert CrossSectionCovariance.from_covariance_section(section).is_relative == [True]


def test_a_section_with_no_grid_is_refused_by_name():
    """§25.3 parameter covariances have no grid, and must not land here.

    Silently accepting one would put a matrix indexed by resonance parameters
    onto an energy axis — the exact confusion the two containers exist to keep
    apart — so the message names where it belongs instead.
    """
    with pytest.raises(ValueError, match="ParameterCovarianceMatrix"):
        CrossSectionCovariance.from_covariance_section(_Section([[1.0]], None))


def test_a_grid_that_does_not_match_the_matrix_is_refused():
    with pytest.raises(ValueError, match="one more than the rows"):
        CrossSectionCovariance.from_covariance_section(
            _Section([[1.0, 0.0], [0.0, 1.0]], [1.0, 2.0]))


def test_no_mt_anywhere_is_refused():
    with pytest.raises(ValueError, match="ENDF_MFMT"):
        CrossSectionCovariance.from_covariance_section(
            _Section([[1.0]], [1.0, 2.0], ENDF_MFMT=None))


# ---------------------------------------------------------------------------
# The log-axis fix
# ---------------------------------------------------------------------------

def test_a_zero_lower_edge_does_not_swallow_the_axis():
    """The regression that made an MF35 heatmap a solid rectangle.

    The assertion is on the *proportion* rather than on the substituted value:
    what matters is that the first bin is a small share of the axis, however
    the floor is chosen.
    """
    edges = np.array([0.0, 1e-5, 1e-3, 1e-1, 1e1, 1e3, 1e5, 2e7])
    transformed = _log_edges(edges)

    span = transformed[-1] - transformed[0]
    firstBin = transformed[1] - transformed[0]
    assert firstBin / span < 0.1, (
        f"the first bin takes {firstBin / span:.1%} of the axis; with the old "
        f"1e-300 floor it took 98%"
    )
    assert np.all(np.diff(transformed) > 0)


def test_the_substituted_edge_is_one_decade_below_the_data():
    edges = np.array([0.0, 1e-5, 1e-3])
    np.testing.assert_allclose(_log_edges(edges)[0], np.log10(1e-6))


def test_a_grid_with_no_zero_is_untouched():
    """MF33 grids start positive, so the fix must be inert for them."""
    edges = np.array([1e-5, 1e-3, 1e-1, 1e1])
    np.testing.assert_allclose(_log_edges(edges), np.log10(edges))


def test_an_all_zero_grid_does_not_raise():
    """Degenerate, but it must not take down a plot with a divide or a nan."""
    assert np.isfinite(_log_edges(np.zeros(4))).all()


# ---------------------------------------------------------------------------
# End to end, on the committed PFNS tape
# ---------------------------------------------------------------------------

def test_an_mf35_band_plots_with_its_structure_intact(micro_pfns_tape):
    """The deliverable: decode → wrap → heatmap, on real data.

    The check is that the rendered correlation matrix spans both signs. A
    single-block plot — the failure this whole path had — has one value
    everywhere, so a range test catches it where a "did it return a figure"
    test would not.
    """
    from kika.endf.model_adapter import decodeCovarianceSuite
    from kika.endf.read_endf import read_endf

    endf = read_endf(str(micro_pfns_tape))
    suite, _ = decodeCovarianceSuite(endf)
    sections = [s for s in suite.covarianceSections
                if s.rowData is not None and s.rowData.ENDF_MFMT == "35/18"]
    assert sections, "the PFNS fixture carries no MF35 section"

    band = CrossSectionCovariance.from_covariance_section(
        sections[0], nuclide=98252)
    data = band.to_heatmap_data(nuclide=98252, mt=18, matrix_type="corr",
                                scale="log")

    matrix = np.asarray(data.matrix_data)
    finite = matrix[np.isfinite(matrix)]
    assert finite.min() < -0.5 and finite.max() > 0.5, (
        "the correlation spans both signs in the file; a plot showing one "
        "value everywhere means the energy axis collapsed"
    )

    # And the axis itself: the first bin must not be most of the figure.
    edges = np.asarray(data.x_edges, dtype=float)
    span = edges[-1] - edges[0]
    assert (edges[1] - edges[0]) / span < 0.2
