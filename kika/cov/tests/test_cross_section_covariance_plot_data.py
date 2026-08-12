"""What ``CrossSectionCovariance.to_plot_data`` answers, and what it refuses.

The method had no test. It is the only way a cross-section covariance becomes
an uncertainty band, it is what ``kika.endf.plotting`` calls for the MF3 half
of its job, and on every ENDF-derived covariance it has never once returned a
band — which is D13 in ``docs/library-gaps.md``.

The file went in first as the freeze, with the pointwise tests asserting the
refusal; the fix in the commit on top flipped them. It is organised by the two
shapes the class carries, because the shape is the whole story:

**Multigroup** — what ``read_njoy_covmat`` builds from a GENDF file. One shared
``energy_grid``, ``num_groups`` set, ``cross_sections`` populated, matrices
added with no per-matrix grid. This path always worked, and the five tests
that pin it are the ones that did not move across the fix.

**Pointwise** — what ``MF33MT.to_xs_covmat`` builds from an ENDF tape.
``num_groups`` is 0, ``energy_grid`` is ``None``, every matrix carries its own
boundaries in ``energy_grids``, and ``cross_sections`` is empty because
``add_matrix`` is the only thing that ever writes to the object and
``add_matrix`` does not take cross sections. This path raised, always. It now
answers, and the last test below marks the one part of it that still does not:
the *nominal* half is still multigroup-only, which is a separate increment.

The fixtures are hand-built rather than read off a tape on purpose: the defect
is about which fields are populated, so populating them explicitly is the
clearest statement of the case, and it keeps the test off the tape lane.
"""
from __future__ import annotations

import logging

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
# The pointwise path — D13, now answered
# ---------------------------------------------------------------------------

def test_a_relative_matrix_carries_its_own_answer():
    """No cross section is needed, and none is asked for.

    ``sqrt(diag)`` of a *relative* covariance already is the fractional
    uncertainty — the multigroup branch above proves it, because it computes
    exactly that and never looks at the cross-section vector it fetched. So
    the band is produced here with ``cross_sections`` empty, which is the
    state every ENDF-derived covariance is in.
    """
    xs_data, band = _pointwise().to_plot_data(nuclide=26056, mt=2)

    assert xs_data is None      # nothing populated cross_sections, correctly
    assert band is not None
    np.testing.assert_allclose(band.y, [20.0, 10.0, 5.0, 5.0])


def test_the_band_lands_on_the_matrix_s_own_boundaries():
    """The grid comes from ``energy_grids[i]``, not from the absent shared one.

    A pointwise covariance leaves ``energy_grid`` None and gives each matrix
    its own boundaries. Falling back to group indices would put the band at
    0…3 eV under a cross section that lives at keV — wrong rather than absent,
    which is why both halves of D13 had to be fixed together.
    """
    _, band = _pointwise().to_plot_data(nuclide=26056, mt=2)

    np.testing.assert_allclose(band.x, GRID)
    np.testing.assert_allclose(band.energy_bins, GRID)


def test_an_absolute_matrix_with_no_cross_section_says_why_it_declines(caplog):
    """``sqrt(diag)`` of an absolute covariance is in barns, not per cent.

    Relativising it needs σ(E), and nothing on this object carries σ(E).
    Declining is the right answer; declining *silently* is what made D13 take
    eight months, so it is logged.
    """
    with caplog.at_level(logging.WARNING,
                         logger='kika.cov.cross_section_covariance'):
        with pytest.raises(ValueError, match="No data found"):
            _pointwise(is_relative=False).to_plot_data(nuclide=26056, mt=2)

    assert "covariance is absolute" in caplog.text


def test_an_absolute_matrix_is_divided_by_the_cross_section_it_has():
    """The conversion ``get_relative_uncertainty`` already does at line 1786."""
    cov = _pointwise(is_relative=False)
    cov.cross_sections[(26056, 2)] = XS.copy()
    cov.num_groups = 3          # so the xs half agrees with itself

    _, band = cov.to_plot_data(nuclide=26056, mt=2)

    # sqrt([0.04, 0.01, 0.0025]) = [0.2, 0.1, 0.05], / [3, 2, 1], × 100
    np.testing.assert_allclose(band.y, [20.0 / 3.0, 5.0, 5.0, 5.0])


def test_a_matrix_added_without_the_flag_is_read_as_relative():
    """``add_matrix`` may be called without ``is_relative``, and GENDF is.

    The default has to stay True or every multigroup band silently changes
    meaning — the same convention ``get_relative_uncertainty`` uses.
    """
    cov = CrossSectionCovariance()
    cov.add_matrix(26056, 2, 26056, 2, REL_DIAG.copy(), energy_grid=list(GRID))
    assert cov.is_relative == []

    _, band = cov.to_plot_data(nuclide=26056, mt=2)
    np.testing.assert_allclose(band.y, [20.0, 10.0, 5.0, 5.0])


def test_the_cross_section_half_is_still_multigroup_only():
    """Not fixed here, and pinned so it is not mistaken for fixed.

    ``_extract_xs_data`` sizes itself from ``num_groups``, which a pointwise
    covariance leaves at 0. Nothing produces this state — ``to_xs_covmat``
    never writes ``cross_sections`` — but a caller who populates it by hand
    gets an exception from the *nominal* half before the band is reached. D13
    was about the band; the nominal is a separate increment.
    """
    cov = _pointwise()
    cov.cross_sections[(26056, 2)] = XS.copy()

    with pytest.raises(ValueError, match="does not match num_groups"):
        cov.to_plot_data(nuclide=26056, mt=2)
