"""What ``CrossSectionCovariance.to_plot_data`` answers, and what it refuses.

The method had no test. It is the only way a cross-section covariance becomes
an uncertainty band, it is what ``kika.endf.plotting`` calls for the MF3 half
of its job, and on every ENDF-derived covariance it has never once returned a
band — which is D13 in ``docs/library-gaps.md``.

This file is the freeze that comes before that fix, and it is written from the
two shapes the class actually carries, because the shape is the whole story:

**Multigroup** — what ``read_njoy_covmat`` builds from a GENDF file. One shared
``energy_grid``, ``num_groups`` set, ``cross_sections`` populated, matrices
added with no per-matrix grid. This path works, and the tests that pin it are
the ones that must not move when D13 is fixed.

**Pointwise** — what ``MF33MT.to_xs_covmat`` builds from an ENDF tape.
``num_groups`` is 0, ``energy_grid`` is ``None``, every matrix carries its own
boundaries in ``energy_grids``, and ``cross_sections`` is empty because
``add_matrix`` is the only thing that ever writes to the object and
``add_matrix`` does not take cross sections. This path raises, always, and the
tests below say so as an assertion so that the fix has to come back and change
them.

The fixtures are hand-built rather than read off a tape on purpose: the defect
is about which fields are populated, so populating them explicitly is the
clearest statement of the case, and it keeps the test off the tape lane.
"""
from __future__ import annotations

import numpy as np
import pytest

from kika.cov.cross_section_covariance import CrossSectionCovariance


GRID = [1.0e3, 1.0e4, 1.0e5, 1.0e6]          # 4 boundaries, 3 groups
REL_DIAG = np.diag([0.04, 0.01, 0.0025])      # relative variances → 20 %, 10 %, 5 %
XS = np.array([3.0, 2.0, 1.0])


def _multigroup() -> CrossSectionCovariance:
    """The GENDF shape: shared grid, num_groups set, cross sections present."""
    cov = CrossSectionCovariance(num_groups=3, energy_grid=list(GRID))
    cov.add_matrix(26056, 2, 26056, 2, REL_DIAG.copy())
    cov.cross_sections[(26056, 2)] = XS.copy()
    return cov


def _pointwise(is_relative: bool = True) -> CrossSectionCovariance:
    """The MF33 shape: per-matrix grid, no num_groups, no cross sections."""
    cov = CrossSectionCovariance()
    cov.add_matrix(26056, 2, 26056, 2, REL_DIAG.copy(),
                   energy_grid=list(GRID), is_relative=is_relative)
    return cov


# ---------------------------------------------------------------------------
# The multigroup path — this is the one that works
# ---------------------------------------------------------------------------

def test_multigroup_returns_the_section_and_its_band():
    """Both halves, on the shared grid, with the last bin repeated for the step."""
    xs_data, band = _multigroup().to_plot_data(nuclide=26056, mt=2)

    assert xs_data is not None and band is not None

    # Four boundaries, four y-values: matplotlib's step(where='post') needs the
    # last level repeated to draw the final bin.
    np.testing.assert_allclose(xs_data.x, GRID)
    np.testing.assert_allclose(xs_data.y, [3.0, 2.0, 1.0, 1.0])
    np.testing.assert_allclose(band.x, GRID)
    np.testing.assert_allclose(band.y, [20.0, 10.0, 5.0, 5.0])

    assert band.uncertainty_type == 'relative'
    assert band.step_where == 'post'


def test_the_two_labels_name_the_same_reaction_differently():
    xs_data, band = _multigroup().to_plot_data(nuclide=26056, mt=2)

    assert xs_data.label == "Fe56 MT=2 (n,el) (±1σ)"
    assert band.label == "Fe56 (n,el) Uncertainty (1σ)"


def test_sigma_scales_the_band_and_says_so_in_both_labels():
    xs_data, band = _multigroup().to_plot_data(nuclide=26056, mt=2, sigma=2.0)

    np.testing.assert_allclose(band.y, [40.0, 20.0, 10.0, 10.0])
    assert xs_data.label == "Fe56 MT=2 (n,el) (±2.0σ)"
    assert band.label == "Fe56 (n,el) Uncertainty (2.0σ)"


def test_the_nuclide_may_be_named_instead_of_numbered():
    by_zaid = _multigroup().to_plot_data(nuclide=26056, mt=2)
    by_name = _multigroup().to_plot_data(nuclide='Fe56', mt=2)

    np.testing.assert_allclose(by_zaid[1].y, by_name[1].y)
    assert by_zaid[1].label == by_name[1].label


def test_a_pair_that_is_in_neither_dictionary_is_a_valueerror():
    with pytest.raises(ValueError, match="No data found for ZAID=26056, MT=102"):
        _multigroup().to_plot_data(nuclide=26056, mt=102)


# ---------------------------------------------------------------------------
# The pointwise path — D13, frozen as the defect it is
# ---------------------------------------------------------------------------

def test_a_relative_matrix_with_no_cross_sections_refuses_to_answer():
    """The defect, stated as an assertion.

    ``sqrt(diag)`` of a *relative* covariance already is the fractional
    uncertainty — the multigroup branch above proves it, because it computes
    exactly that and never looks at the cross-section vector it just fetched.
    The vector is read only to decide whether to proceed. So on every ENDF
    covariance, where nothing populates ``cross_sections``, the band is
    refused although it was always computable.

    Recorded as D13 in ``docs/library-gaps.md``. The fix deletes this test.
    """
    with pytest.raises(ValueError, match="No data found for ZAID=26056, MT=2"):
        _pointwise().to_plot_data(nuclide=26056, mt=2)


def test_the_per_matrix_energy_grid_is_never_consulted():
    """And even ungated, the band would land on the wrong axis.

    A pointwise covariance has ``energy_grid is None`` and one grid per matrix
    in ``energy_grids``. Only the shared grid is read, so the fallback puts the
    band at group indices 0…3 under a cross section that lives at keV — wrong
    rather than absent. Both halves of the defect have to go together.
    """
    cov = _pointwise()
    cov.cross_sections[(26056, 2)] = XS.copy()   # force past the gate above

    # The xs half now trips over ``num_groups`` being 0 before the band is
    # ever reached — a third face of the same "multigroup was assumed".
    with pytest.raises(ValueError, match="does not match num_groups"):
        cov.to_plot_data(nuclide=26056, mt=2)


def test_an_absolute_matrix_is_not_distinguished_from_a_relative_one():
    """``is_relative`` is not read here at all.

    ``get_relative_uncertainty`` does read it, and divides by the cross
    section when the matrix is absolute. This method does not, so an absolute
    covariance would be labelled a percentage while holding barns — latent
    only because the gate refuses everything first.
    """
    absolute = _pointwise(is_relative=False)
    assert absolute.is_relative == [False]

    with pytest.raises(ValueError, match="No data found"):
        absolute.to_plot_data(nuclide=26056, mt=2)
