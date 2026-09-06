"""MF1 nu-bar evaluated at an energy, under the section's own law.

The three nu-bar sections carry a TAB1 (or a polynomial) and until now nothing
read it back: everything downstream took ``energies`` and ``nubar_values`` and
interpolated for itself, which puts a second copy of the ENDF interpolation
rule wherever it happens. :func:`evaluate_nubar` is the one copy.

What is worth pinning here is not the arithmetic -- ``interpolate_1d`` has its
own tests -- but the two things a caller gets wrong on its own: the sum rule
that makes prompt nu-bar recoverable from a file that does not state it, and
the out-of-range convention, which is *not* MF3's.
"""
from pathlib import Path

import numpy as np
import pytest

from kika import read_endf
from kika.endf.classes.mf1.mf1mt452 import evaluate_nubar

DATA = Path(__file__).resolve().parent / "data"
NUBAR = DATA / "micro_u235_nubar.endf"

THERMAL = 2.53e-2


@pytest.fixture(scope="module")
def mf1():
    return read_endf(str(NUBAR)).mf[1]


def test_reads_the_tabulated_value_the_evaluator_wrote(mf1):
    """A tabulated point comes back as itself, not as an interpolation of it."""
    section = mf1.mt[456]
    energy = section.energies[3]
    assert section.get_nubar(energy) == pytest.approx(
        section.nubar_values[3], rel=1e-12)


def test_prompt_is_total_minus_delayed(mf1):
    """The identity a file without MT456 has to be read through.

    MT452 is *defined* as MT455 + MT456, so a file stating only the total and
    the delayed part still states the prompt one exactly. ENDF/B-VIII.1 U-235
    gives all three, which is what makes it the fixture that can check it.
    """
    total = mf1.mt[452].get_nubar(THERMAL)
    delayed = mf1.mt[455].get_nubar(THERMAL)
    prompt = mf1.mt[456].get_nubar(THERMAL)
    assert total - delayed == pytest.approx(prompt, abs=1e-8)


def test_holds_rather_than_zeroing_outside_the_table(mf1):
    """The convention that separates nu-bar from a cross section.

    MF3 returns zero below its first point, and it is right to: a reaction
    really does not happen below threshold. Nu-bar below the first tabulated
    energy is the thermal value, and zero there would say a fission releases no
    neutrons -- so ``'hold'`` is the default here and the difference is worth a
    test rather than a comment.
    """
    section = mf1.mt[456]
    first, last = section.energies[0], section.energies[-1]
    assert section.get_nubar(first * 1e-3) == pytest.approx(
        section.nubar_values[0])
    assert section.get_nubar(last * 10) == pytest.approx(
        section.nubar_values[-1])
    assert section.get_nubar(first * 1e-3, out_of_range="zero") == 0.0


def test_evaluates_the_polynomial_form():
    """LNU=1, which no ENDF/B-VIII.1 actinide uses and older tapes do."""
    coefficients = [2.4, 1.5e-7, -2.0e-15]
    energy = 1.0e6
    expected = sum(c * energy ** n for n, c in enumerate(coefficients))
    assert evaluate_nubar(1, coefficients, [], [], [], energy) == pytest.approx(
        expected)


def test_takes_an_array_as_readily_as_a_scalar(mf1):
    """The spectrum panels ask for one energy; a plot asks for a grid."""
    section = mf1.mt[452]
    grid = np.array([THERMAL, 1.0e6, 1.4e7])
    values = section.get_nubar(grid)
    assert values.shape == grid.shape
    assert values[0] == pytest.approx(section.get_nubar(THERMAL))
    # nu-bar rises with incident energy; a file where it did not would be the
    # first thing to look at, and a reading where it did not would be this bug.
    assert np.all(np.diff(values) > 0)


def test_refuses_a_representation_it_does_not_know():
    with pytest.raises(ValueError, match="LNU=3"):
        evaluate_nubar(3, [], [], [], [], 1.0)
