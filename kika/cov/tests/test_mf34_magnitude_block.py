"""The MF33 magnitude block that makes MF34's L=0 row readable.

ENDF-6 Sec. 34.1 lets a section state the covariance between the cross section
magnitude and the Legendre coefficients through a_0, and Sec. 34.3 says the
(0, 0) self-block belongs to MF33 and must not be repeated in MF34.  A
conforming LTT=3 section therefore writes (0, 0) null -- and that null makes
``correlation_matrix`` divide by zero, so the whole L=0 row and column, cross
terms included, comes back NaN.

:meth:`LegendreCovariance.attach_magnitude_covariance` supplies the missing
self-block from MF33.  These tests pin the arithmetic on a hand-computable
case, the guards that stop the two files being double counted or mixed in
different units, and -- on a real evaluation -- the fact that the cross block
stops being NaN.
"""
from __future__ import annotations

import numpy as np
import pytest

from kika.cov.cross_section_covariance import CrossSectionCovariance
from kika.cov.legendre_covariance import LegendreCovariance

ISO, MT = 92238, 2

# Two deliberately interleaved grids, so the union is a strict refinement of
# both and neither leg can be read off without lifting.
GRID_34 = [1.0, 2.0, 4.0]
GRID_33 = [1.0, 3.0, 4.0]

C11 = np.array([[4.0e-4, 1.0e-4], [1.0e-4, 9.0e-4]])
C01 = np.array([[2.0e-4, 0.0], [0.0, 3.0e-4]])
C00 = np.array([[1.0e-2, 0.0], [0.0, 4.0e-2]])


def _mf34(self_block=None, with_cross=True) -> LegendreCovariance:
    """An LTT=3-shaped carrier: null (0, 0), a (0, 1) cross, a (1, 1) self."""
    lc = LegendreCovariance(energy_unit="eV")
    block = np.zeros((2, 2)) if self_block is None else self_block
    lc.add_matrix(ISO, MT, 0, ISO, MT, 0, block, GRID_34,
                  is_relative=True, frame="LAB")
    if with_cross:
        lc.add_matrix(ISO, MT, 0, ISO, MT, 1, C01, GRID_34,
                      is_relative=True, frame="LAB")
    lc.add_matrix(ISO, MT, 1, ISO, MT, 1, C11, GRID_34,
                  is_relative=True, frame="LAB")
    return lc


def _mf33(matrix=C00, mt=MT, is_relative=True) -> CrossSectionCovariance:
    xs = CrossSectionCovariance(energy_unit="eV")
    xs.add_matrix(ISO, mt, ISO, mt, matrix, GRID_33, is_relative=is_relative)
    return xs


def _l0_rows(cov: LegendreCovariance):
    """Index range of the L=0 block in the assembled matrix."""
    unions = cov.compute_union_energy_grids()
    triplets = cov._get_param_triplets()
    max_g = max(len(unions[t]) - 1 for t in triplets)
    t0 = next(t for t in triplets if t[2] == 0)
    start = triplets.index(t0) * max_g
    return slice(start, start + len(unions[t0]) - 1)


# ---------------------------------------------------------------------------
# The defect being fixed
# ---------------------------------------------------------------------------

def test_null_self_block_nans_the_whole_l0_row():
    """Without MF33 the cross term is unreadable, though it is right there."""
    lc = _mf34()
    corr = lc.correlation_matrix
    rows = _l0_rows(lc)

    off_diagonal = corr[rows]
    off_diagonal = off_diagonal[~np.eye(corr.shape[0], dtype=bool)[rows]]
    assert np.isnan(off_diagonal).all(), (
        "the L=0 row is expected to be entirely NaN before the fix; that is "
        "the whole reason attach_magnitude_covariance exists"
    )
    # ... while the data itself was parsed perfectly well.
    assert lc.magnitude_cross_summary(ISO, MT)["cross_max_abs"] == pytest.approx(3.0e-4)


# ---------------------------------------------------------------------------
# The arithmetic
# ---------------------------------------------------------------------------

def test_attached_block_lands_on_the_common_refinement():
    joint = _mf34().attach_magnitude_covariance(_mf33(), isotope=ISO, mt=MT)
    unions = joint.compute_union_energy_grids()

    l0 = next(g for t, g in unions.items() if t[2] == 0)
    l1 = next(g for t, g in unions.items() if t[2] == 1)
    # union([1, 2, 4], [1, 3, 4]) -- neither grid contains the other.
    np.testing.assert_allclose(l0, [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(l1, GRID_34)


def test_cross_correlation_matches_hand_computation():
    """corr(a0, a1) = C01 / sqrt(var0 * var1), both legs lifted to the union.

    var0 on [1,2,3,4] lifts MF33's [1e-2, 4e-2] over [1,3,4] to
    [1e-2, 1e-2, 4e-2]; C01's rows lift from [1,2,4] to [1,2,3,4] as
    [row0, row1, row1].  The three non-zero cells follow.
    """
    joint = _mf34().attach_magnitude_covariance(_mf33(), isotope=ISO, mt=MT)
    hd = joint.to_heatmap_data(nuclide=ISO, mt=MT, legendre_coeffs=(0, 1),
                               matrix_type="corr")
    corr = np.asarray(hd.matrix_data, dtype=float)

    assert corr.shape == (3, 2)
    assert np.isfinite(corr).all(), "no cell of the cross block may be NaN"
    np.testing.assert_allclose(
        corr,
        [[0.10, 0.00],
         [0.00, 0.10],
         [0.00, 0.05]],
        rtol=1e-12, atol=1e-12,
    )


def test_diagonal_block_is_the_mf33_covariance():
    joint = _mf34().attach_magnitude_covariance(_mf33(), isotope=ISO, mt=MT)
    hd = joint.to_heatmap_data(nuclide=ISO, mt=MT, legendre_coeffs=[0],
                               matrix_type="cov")
    cov = np.asarray(hd.matrix_data, dtype=float)

    # MF33's two bins, lifted onto the three union bins.
    np.testing.assert_allclose(np.diag(cov), [1.0e-2, 1.0e-2, 4.0e-2])


def test_l_ge_1_blocks_are_untouched():
    """The fix must not move the orders the app already plots."""
    lc = _mf34()
    before = lc.to_heatmap_data(nuclide=ISO, mt=MT, legendre_coeffs=[1],
                                matrix_type="cov")
    joint = lc.attach_magnitude_covariance(_mf33(), isotope=ISO, mt=MT)
    after = joint.to_heatmap_data(nuclide=ISO, mt=MT, legendre_coeffs=[1],
                                  matrix_type="cov")
    np.testing.assert_array_equal(before.matrix_data, after.matrix_data)


def test_receiver_is_not_mutated():
    """``to_ang_covmat`` caches what it returns; mutating it would leak."""
    lc = _mf34()
    n_blocks = len(lc.matrices)
    lc.attach_magnitude_covariance(_mf33(), isotope=ISO, mt=MT)

    assert len(lc.matrices) == n_blocks
    assert lc.magnitude_cross_summary(ISO, MT)["self_is_null"] is True
    assert "magnitude_block_source" not in lc.metadata


def test_no_duplicate_self_block():
    """The null placeholder is replaced, not added beside."""
    joint = _mf34().attach_magnitude_covariance(_mf33(), isotope=ISO, mt=MT)
    n_self = sum(
        1 for i in range(len(joint.matrices))
        if int(joint.l_rows[i]) == 0 and int(joint.l_cols[i]) == 0
    )
    assert n_self == 1


def test_provenance_is_recorded():
    joint = _mf34().attach_magnitude_covariance(_mf33(), isotope=ISO, mt=MT)
    assert joint.metadata["magnitude_block_source"] == {
        "mf": 33, "isotope": ISO, "mt": MT, "ne": len(GRID_33),
    }


# ---------------------------------------------------------------------------
# The guards
# ---------------------------------------------------------------------------

def test_refuses_a_section_without_a0_blocks():
    lc = LegendreCovariance(energy_unit="eV")
    lc.add_matrix(ISO, MT, 1, ISO, MT, 1, C11, GRID_34, is_relative=True, frame="LAB")
    with pytest.raises(ValueError, match="no L=0 blocks"):
        lc.attach_magnitude_covariance(_mf33(), isotope=ISO, mt=MT)


def test_refuses_a_non_null_self_block():
    """A file that already states (0, 0) would be double counted."""
    lc = _mf34(self_block=np.array([[5.0e-3, 0.0], [0.0, 5.0e-3]]))
    with pytest.raises(ValueError, match="double count"):
        lc.attach_magnitude_covariance(_mf33(), isotope=ISO, mt=MT)


def test_refuses_when_mf33_lacks_the_reaction():
    with pytest.raises(ValueError, match="no self-covariance"):
        _mf34().attach_magnitude_covariance(_mf33(mt=102), isotope=ISO, mt=MT)


def test_refuses_an_absolute_magnitude_block():
    """The a_0 family is relative; mixing units would rescale every term."""
    with pytest.raises(ValueError, match="absolute"):
        _mf34().attach_magnitude_covariance(
            _mf33(is_relative=False), isotope=ISO, mt=MT
        )


def test_magnitude_mt_can_be_overridden():
    xs = _mf33(mt=102)
    joint = _mf34().attach_magnitude_covariance(
        xs, isotope=ISO, mt=MT, magnitude_mt=102
    )
    assert joint.metadata["magnitude_block_source"]["mt"] == 102


# ---------------------------------------------------------------------------
# Telling a real cross term from a declared-but-empty one
# ---------------------------------------------------------------------------

def test_summary_reports_a_real_cross_term():
    s = _mf34().magnitude_cross_summary(ISO, MT)
    assert s["present"] and s["self_present"] and s["self_is_null"]
    assert s["cross_orders"] == [1]
    assert s["cross_is_null"] is False


def test_summary_flags_a_declared_but_null_cross_term():
    """ENDF/B-VIII.1's shape: LTT=3 kept, values zeroed to round-off."""
    lc = LegendreCovariance(energy_unit="eV")
    lc.add_matrix(ISO, MT, 0, ISO, MT, 0, np.zeros((2, 2)), GRID_34,
                  is_relative=True, frame="LAB")
    lc.add_matrix(ISO, MT, 0, ISO, MT, 1, np.full((2, 2), 2.2e-19), GRID_34,
                  is_relative=True, frame="LAB")
    lc.add_matrix(ISO, MT, 1, ISO, MT, 1, C11, GRID_34,
                  is_relative=True, frame="LAB")

    s = lc.magnitude_cross_summary(ISO, MT)
    assert s["present"] is True
    assert s["cross_orders"] == [1]
    assert s["cross_is_null"] is True, (
        "1e-19 against a 1e-3 diagonal is round-off, not a cross term"
    )


def test_summary_on_a_section_with_no_a0_at_all():
    lc = LegendreCovariance(energy_unit="eV")
    lc.add_matrix(ISO, MT, 1, ISO, MT, 1, C11, GRID_34, is_relative=True, frame="LAB")
    s = lc.magnitude_cross_summary(ISO, MT)
    assert s == {
        "present": False, "self_present": False, "self_is_null": False,
        "cross_orders": [], "cross_max_abs": 0.0, "cross_is_null": True,
    }


# ---------------------------------------------------------------------------
# A real evaluation
# ---------------------------------------------------------------------------

@pytest.mark.tape
def test_u238_b80_cross_block_becomes_readable(u238_b80_tape):
    """U-238 ENDF/B-VIII.0 MT2 is the carrier: LTT=3 with non-zero a_0 terms.

    JEFF-4.0 has no MF34 for U-235 at all and LTT=1 for U-238, and VIII.1
    zeroed its a_0 values, so VIII.0 is the only evaluation on hand where the
    block holds physics.
    """
    from kika.endf import read_endf

    endf = read_endf(str(u238_b80_tape), mf_numbers=[33, 34])
    lc = endf.get_file(34).sections[2].to_ang_covmat()

    summary = lc.magnitude_cross_summary(92238, 2)
    assert summary["cross_orders"] == [1, 2]
    assert summary["self_is_null"] is True
    assert summary["cross_is_null"] is False

    before = lc.to_heatmap_data(nuclide=92238, mt=2, legendre_coeffs=(0, 1),
                                matrix_type="corr")
    assert np.isnan(np.asarray(before.matrix_data, float)).all()

    joint = lc.attach_magnitude_covariance(
        endf.get_file(33).sections[2].to_xs_covmat(), isotope=92238, mt=2
    )
    after = np.asarray(
        joint.to_heatmap_data(nuclide=92238, mt=2, legendre_coeffs=(0, 1),
                              matrix_type="corr").matrix_data,
        dtype=float,
    )
    assert np.isfinite(after).all()
    assert np.abs(after).max() > 1e-3, "the cross block must carry signal"


@pytest.mark.tape
def test_u238_b80_extremes_are_not_introduced_by_the_attachment(u238_b80_tape):
    """The a_0 blocks of this evaluation are not Cauchy-Schwarz consistent.

    Measured, not assumed: MT2's worst native |corr(a0, a1)| is 3.96 on the
    file's own grids, with no lifting involved, and the assembled joint matrix
    reaches exactly the same 4.69 as the L>=1 sub-block does on its own.  So
    the attachment adds no error of its own, and the clipping
    ``clipped_correlation_matrix`` already applies to L>=1 is the right
    treatment for L=0 too.  This is why VIII.1 zeroed these blocks.
    """
    from kika.endf import read_endf

    endf = read_endf(str(u238_b80_tape), mf_numbers=[33, 34])
    lc = endf.get_file(34).sections[2].to_ang_covmat()
    joint = lc.attach_magnitude_covariance(
        endf.get_file(33).sections[2].to_xs_covmat(), isotope=92238, mt=2
    )

    def worst(cov):
        d = np.diag(cov)
        with np.errstate(divide="ignore", invalid="ignore"):
            c = cov / np.outer(np.sqrt(d), np.sqrt(d))
        finite = c[np.isfinite(c)]
        return float(np.abs(finite).max())

    assert worst(joint.covariance_matrix) == pytest.approx(worst(lc.covariance_matrix))
