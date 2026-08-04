"""What the ENDF field-format constants actually do.

``ENDF_FORMAT_INT_ZERO`` was documented as "integer with zero rendered as 0
(not blank)", which reads as a contrast with ``ENDF_FORMAT_INT``. There has
never been one: ``format_endf_data_line`` gave the two constants two identical
branches. A name that promises a behaviour it does not have is worse than no
name, because call sites get chosen on the strength of the promise.

These tests state what each constant renders, so the next person reads it from
here rather than from the comment.
"""
from __future__ import annotations

import pytest

from kika.endf.utils import (
    ENDF_FORMAT_BLANK,
    ENDF_FORMAT_FLOAT,
    ENDF_FORMAT_INT,
    ENDF_FORMAT_INT_ZERO,
    format_endf_data_line,
    format_endf_number,
)

MAT, MF, MT = 2631, 3, 2


def _data_part(values, formats) -> str:
    """Columns 1-66 of a rendered line."""
    return format_endf_data_line(values, MAT, MF, MT, 0, formats=formats)[:66]


def test_int_zero_is_the_same_constant_as_int():
    assert ENDF_FORMAT_INT_ZERO is ENDF_FORMAT_INT


@pytest.mark.parametrize("value", [0, 1, -1, 99999, 26056])
def test_the_two_int_constants_render_identically(value):
    values = [value] * 6
    assert _data_part(values, [ENDF_FORMAT_INT] * 6) == _data_part(
        values, [ENDF_FORMAT_INT_ZERO] * 6
    )


def test_an_integer_zero_is_written_as_a_right_aligned_zero_not_a_blank():
    """Neither int constant has ever blanked a zero. Blanking is BLANK's job."""
    rendered = _data_part([0] * 6, [ENDF_FORMAT_INT] * 6)
    assert rendered == "          0" * 6
    assert rendered.strip() != ""


def test_blank_is_what_produces_an_empty_field():
    assert _data_part([0] * 6, [ENDF_FORMAT_BLANK] * 6) == " " * 66


def test_a_float_zero_is_written_in_endf_float_notation():
    """The distinction that actually matters: field *type*, not zero-ness."""
    assert _data_part([0.0] * 6, [ENDF_FORMAT_FLOAT] * 6) == " 0.000000+0" * 6


def test_the_float_helper_itself_gets_zero_right():
    """format_endf_number has an explicit zero case and always has."""
    assert format_endf_number(0) == " 0.000000+0"
    assert format_endf_number(0.0) == " 0.000000+0"


def test_a_line_is_eighty_characters_with_its_identification():
    line = format_endf_data_line([0] * 6, MAT, MF, MT, 42, formats=[ENDF_FORMAT_INT] * 6)
    assert len(line) == 80
    assert line[66:] == f"{MAT:4d}{MF:2d}{MT:3d}{42:5d}"
