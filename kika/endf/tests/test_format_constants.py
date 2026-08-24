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
    parse_number,
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


def test_a_three_digit_exponent_is_written_rather_than_flushed_to_zero():
    """It used to return ``" 0.000000+0"`` for anything below 1e-100.

    Silently, and at the right field width, so nothing downstream could see it.
    ``tsl-ortho-H.endf`` tabulates S(α, β) down to 1.5963e-100 on 1 403 records
    and every one of them was written back as zero.

    The mantissa gives up a decimal to pay for the extra exponent digit, which
    is what keeps the field at 11 characters — the same trade the two-digit case
    already made.
    """
    assert format_endf_number(1.5963e-100) == " 1.5963-100"
    assert format_endf_number(-1.5963e-100) == "-1.5963-100"
    assert format_endf_number(1.2345e100) == " 1.2345+100"
    assert format_endf_number(1e-308) == " 1.0000-308"


@pytest.mark.parametrize("source,canonical", [
    (" 0.12640975", " 1.264097-1"),
    (" .061916773", " 6.191677-2"),
    (" 1680612.75", " 1.680613+6"),
    ("  583632.75", " 5.836328+5"),
    (" 6012.00000", " 6.012000+3"),
])
def test_a_field_written_without_an_exponent_can_hold_more_digits_than_we_write(
        source, canonical):
    """kika's writer keeps **seven significant digits**, and some tapes write more.

    ENDF's 11-character field can be spent two ways. kika always spends it the
    same way — sign, ``M.MMMMMM``, exponent sign, exponent — which leaves seven
    significant digits. An evaluator who writes ``0.12640975`` instead spends
    nothing on an exponent and gets nine. Reading is unaffected
    (``parse_number`` takes either); **writing the value back loses the extra
    digits**, with relative error bounded by half of the seventh digit, ~5e-7.

    Found by the MF6 round-trip sweep over ENDF/B-VIII.1 and measured there: of
    12 388 MF6 sections, the ones that do not come back byte-identical differ
    *only* in fields of this kind, and the largest value change over the tapes
    checked was 4.8e-7 relative. **The JEFF-4.0 Fe-56 host tape the thesis
    pipeline writes has 0 such fields in 1 967 724**, so nothing that pipeline
    produces is affected — which is why this is pinned as a measured property of
    the writer rather than fixed under an MF6 branch. Fixing it means teaching
    ``format_endf_number`` to prefer whichever spelling preserves more digits,
    and that changes MF3, MF4, MF5 and MF7 output too.
    """
    value = parse_number(source)
    assert format_endf_number(value) == canonical
    reread = parse_number(canonical)
    assert reread == pytest.approx(value, rel=5e-7)


def test_the_seven_digit_limit_is_where_the_loss_starts():
    """Seven digits survive exactly; the eighth is where it begins."""
    assert parse_number(format_endf_number(0.1264097)) == 0.1264097
    assert parse_number(format_endf_number(0.12640975)) != 0.12640975


@pytest.mark.parametrize("field, decimal", [
    ("2.427894+7", "24278940.0"),
    ("2.892336+7", "28923360.0"),
    ("5.658661+7", "56586610.0"),
    ("1.100000+8", "110000000.0"),
    ("4.907887-1", "0.4907887"),
    ("2.559080-1", "0.255908"),
    ("2.530000-2", "0.0253"),
    ("1.000000-5", "0.00001"),
])
def test_the_two_spellings_of_one_value_read_as_the_same_double(field, decimal):
    """ENDF's exponent form and a plain decimal must give the *same* double.

    Both spellings are legal in column 1-11 and the same tapes carry both:
    C-12's ENDF/B-VIII.1 MF3/MT5 writes its energy grid as ``24278940.0`` and
    kika writes it back as ``2.427894+7``. If the two decoded to different
    doubles, a tape read, written and read again would not be its own fixed
    point -- which is exactly how this was found, as
    ``test_a_tape_with_mf6_comes_back_with_all_of_it[c12]``.

    They agree only because ``parse_number`` reassembles the field into one
    decimal string and converts it *once*. ``mantissa * 10 ** exponent`` rounds
    twice and lands a unit in the last place out: ``2.427894 * 10**7`` is
    24278940.000000004.
    """
    assert parse_number(field) == float(decimal)


@pytest.mark.parametrize("value", [
    1.5963e-100, 5.1183e-100, 1.96567e-17, 8.89108e-18, 1.234567e5,
    -3.14159e-1, 1.0e10, 9.999999e9, 2.0e7, 1e-308, 4009.0,
])
def test_every_written_number_is_eleven_columns_and_reads_back(value):
    """Width and value survive a write-read cycle across the exponent ladder.

    Eleven columns is not cosmetic: a twelfth character shifts every field to
    its right, so a formatter that overflows on one value corrupts the whole
    record rather than just that number.
    """
    written = format_endf_number(value)
    assert len(written) == 11
    assert parse_number(written) == pytest.approx(value, rel=1e-4)


def test_a_line_is_eighty_characters_with_its_identification():
    line = format_endf_data_line([0] * 6, MAT, MF, MT, 42, formats=[ENDF_FORMAT_INT] * 6)
    assert len(line) == 80
    assert line[66:] == f"{MAT:4d}{MF:2d}{MT:3d}{42:5d}"
