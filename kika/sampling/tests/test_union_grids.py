"""Union energy grids for an MF34 whose blocks are stated on different grids.

**These three tests asserted nothing until 2026-09-06.** The file was a demo
script that printed its findings and returned ``True``, so all three passed
whatever the code did -- including
``print(f"Matrix is symmetric: {np.allclose(cov, cov.T)}")``, which reports a
failure as cheerfully as a success and to a stream nobody reads. Three green
tests guarding nothing are worse than none: they answer "is this covered?" with
a yes.

What they now assert is what the demo was demonstrating. An MF34 section may
state its ``L=1`` and ``L=2`` families on different energy grids, and a block
that couples them has a third; ``get_union_energy_grids`` is what gives each
Legendre order the common refinement of every grid that mentions it, which is
the only grid on which a piecewise-constant lift of all of them is exact.

The fixture is deliberately asymmetric -- 5 points for L=1, 7 for L=2, and the
off-diagonal block on the shorter one -- because a fixture whose grids agree
cannot tell a union from a copy.
"""
from __future__ import annotations

import numpy as np
import pytest

from kika.cov.legendre_covariance import LegendreCovariance

# A bare name and not a relative import: this directory has no ``__init__.py``,
# so pytest inserts it into ``sys.path`` and there is no package to be
# relative to. Changing that would change collection for every test here.
from union_grids_validation import run_comprehensive_validation

ISOTOPE = 92235
MT = 2

L1_GRID = [1e-5, 1e-3, 1e-1, 1.0, 20.0]
L2_GRID = [1e-5, 5e-4, 1e-3, 1e-1, 0.5, 1.0, 20.0]


@pytest.fixture
def mf34() -> LegendreCovariance:
    """Two diagonal blocks on different grids, and a cross block on the shorter."""
    cov = LegendreCovariance()
    for l_row, l_col, grid, matrix in (
        (1, 1, L1_GRID, np.eye(len(L1_GRID) - 1) * 0.01),
        (2, 2, L2_GRID, np.eye(len(L2_GRID) - 1) * 0.005),
        (1, 2, L1_GRID, np.ones((len(L1_GRID) - 1,) * 2) * 0.002),
    ):
        cov.add_matrix(
            isotope_row=ISOTOPE, reaction_row=MT, l_row=l_row,
            isotope_col=ISOTOPE, reaction_col=MT, l_col=l_col,
            matrix=matrix, energy_grid=grid,
            is_relative=True, frame="LAB",
        )
    return cov


# ---------------------------------------------------------------------------
# The union
# ---------------------------------------------------------------------------

def test_every_legendre_order_gets_a_grid(mf34):
    grids = mf34.get_union_energy_grids()
    assert sorted(grids) == [(ISOTOPE, MT, 1), (ISOTOPE, MT, 2)]


def test_an_order_whose_grids_agree_keeps_that_grid(mf34):
    """L=1 is mentioned twice, on the same grid both times. A union of a grid
    with itself is that grid, and any extra point would be invented."""
    grid = mf34.get_union_energy_grids()[(ISOTOPE, MT, 1)]
    np.testing.assert_allclose(grid, L1_GRID)


def test_an_order_mentioned_on_two_grids_gets_the_common_refinement(mf34):
    """L=2 is stated on its own 7-point grid and reached by a cross block on the
    5-point one. The union has to contain both, and here L1 is a subset of L2 --
    so the answer is L2's grid and *not* something coarser."""
    grid = mf34.get_union_energy_grids()[(ISOTOPE, MT, 2)]
    np.testing.assert_allclose(grid, L2_GRID)
    assert set(L1_GRID) <= set(grid)


def test_the_built_in_validation_accepts_this_section(mf34):
    assert mf34.validate_union_grids(verbose=False) is True


def test_the_comprehensive_validation_accepts_it_too(mf34):
    assert run_comprehensive_validation(mf34, sample_mt_numbers=[MT]) is True


def test_grids_that_differ_are_reported_as_non_uniform(mf34):
    """The property the whole union machinery exists for. If this said ``True``
    the section would need no union at all."""
    assert mf34.has_uniform_energy_grid() is False


# ---------------------------------------------------------------------------
# The assembled matrix
# ---------------------------------------------------------------------------

def test_the_matrix_is_one_uniform_stride_per_order(mf34):
    """Two Legendre orders at the width of the widest, not at their own widths.

    L=1 owns 4 bins and L=2 owns 6, so a layout at each order's own width would
    be 10 x 10. It is 12 x 12: every order occupies the widest stride, and the
    orders that do not fill it are padded. Measured on the diagonal, where L=1
    reads ``[0.01, 0.01, 0.01, 0.01, 0, 0]``.

    **The two zeros are padding, not a statement that the covariance vanishes
    above 1 MeV.** Which rows are real is what ``get_union_energy_grids`` says
    per order, exactly as ``widths`` does for the MF33 index -- a uniform stride
    is a layout convention, and reading the tail as data is the mistake it
    invites. Pinned here because the shape alone does not say it.
    """
    matrix = mf34.covariance_matrix
    widest = max(len(L1_GRID), len(L2_GRID)) - 1
    orders = len(mf34.legendre_indices)

    assert matrix.shape == (orders * widest, orders * widest)
    diagonal = np.diag(matrix)
    np.testing.assert_allclose(diagonal[:len(L1_GRID) - 1], 0.01)
    np.testing.assert_allclose(diagonal[len(L1_GRID) - 1:widest], 0.0)
    np.testing.assert_allclose(diagonal[widest:], 0.005)

    own = mf34.get_union_energy_grids()[(ISOTOPE, MT, 1)]
    assert len(own) - 1 == len(L1_GRID) - 1 < widest


def test_the_matrix_is_symmetric_and_finite(mf34):
    """What the demo printed and never checked."""
    matrix = mf34.covariance_matrix
    assert np.isfinite(matrix).all()
    np.testing.assert_allclose(matrix, matrix.T)


def test_the_section_reports_what_it_holds(mf34):
    assert mf34.num_matrices == 3
    assert mf34.isotopes == [ISOTOPE]
    assert mf34.reactions == [MT]
    assert mf34.legendre_indices == [1, 2]
