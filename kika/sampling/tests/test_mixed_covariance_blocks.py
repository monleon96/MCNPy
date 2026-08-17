"""§7.4 from the sampling side: what a ``mixed`` section did to the seam.

The three entry points in :mod:`kika.sampling.model_blocks` all reached for
``section.form.matrix`` — or, worse, for ``section.form.isRelative`` one line
earlier — and a §25.2 ``mixed`` has neither. The failure was a bare
``AttributeError: 'Mixed' object has no attribute 'isRelative'`` raised from
inside a *filter*, which is loud enough not to be mistaken for a data defect
and useless enough that the caller has to come and read this module to find out
what the file actually said.

**Duck-typed stand-ins, not model objects**, following ``test_legendre_blocks``:
``model_blocks`` does not import :mod:`kika.nuclear_data.model` and neither does
this file. The seam's contract is what it asks the suite for, and a test that
imported the model would stop measuring that. The one exception is that the
stand-in is a real class named ``Mixed`` rather than a ``SimpleNamespace``,
because the guard names the node from the class name and a namespace would let
that go wrong quietly.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from kika.sampling.model_blocks import (covariance_suite_blocks,
                                        cross_section_covariance_blocks,
                                        legendre_covariance_blocks)


class Mixed:
    """§25.2's several-forms-in-one-section, with the surface it really has.

    No ``matrix``, no ``rowGrid``, no ``isRelative`` — which is the point.
    """

    def __init__(self, components):
        self.components = components


def _matrix(order: int, grid=None) -> SimpleNamespace:
    grid = (np.linspace(1e3, 2e7, order + 1) if grid is None
            else np.asarray(grid, dtype=float))
    return SimpleNamespace(matrix=np.eye(order), rowGrid=grid,
                           columnGrid=None, isRelative=True)


def _link(mfmt: str, order: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        ENDF_MFMT=mfmt,
        href=f"#{mfmt}/L{order}",
        slices=SimpleNamespace(slices=(SimpleNamespace(domainValue=order),)),
    )


def _section(form, mfmt: str, order: int = 1) -> SimpleNamespace:
    link = _link(mfmt, order)
    return SimpleNamespace(label="eval", rowData=link, columnData=link,
                           provenance=SimpleNamespace(za=26056), form=form)


def _mixedSection(mfmt: str, order: int = 1) -> SimpleNamespace:
    """Fe-56's shape: a 3×3 short-range block and the 628×628 beside it."""
    return _section(Mixed([_matrix(3), _matrix(628)]), mfmt, order)


def test_covariance_suite_blocks_names_the_section_and_the_node():
    with pytest.raises(ValueError) as raised:
        covariance_suite_blocks([_mixedSection("35/18")], isotope=26056, mt=18)
    assert "<mixed> of 2 components" in str(raised.value)


def test_the_mf34_blocks_refuse_before_reading_isRelative():
    """``relative=`` used to decide the section's fate one line too early, so
    the ``AttributeError`` came out of a filter and read as a bad argument."""
    with pytest.raises(ValueError, match=r"MF34 covariance section 'eval'"):
        legendre_covariance_blocks([_mixedSection("34/2")], isotope=26056,
                                   relative=True)


def test_the_mf33_blocks_refuse_the_same_way():
    with pytest.raises(ValueError, match=r"MF33 covariance section 'eval'"):
        cross_section_covariance_blocks([_mixedSection("33/2")], isotope=26056,
                                        relative=True)


def test_a_section_of_another_mf_is_still_filtered_out_first():
    """The guard is not a suite-wide veto. Fe-56's MF33 sections are all mixed,
    and asking that file for its MF34 blocks has to keep working."""
    suite = [_mixedSection("33/2"), _section(_matrix(3), "34/2")]
    assert len(legendre_covariance_blocks(suite, isotope=26056)) == 1


def test_the_ordinary_path_is_unchanged():
    blocks = cross_section_covariance_blocks([_section(_matrix(3), "33/2")],
                                             isotope=26056)
    assert len(blocks) == 1
    np.testing.assert_array_equal(blocks[0][1], np.eye(3))
