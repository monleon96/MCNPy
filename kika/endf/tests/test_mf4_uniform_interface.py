"""Every MF4 class answers ``extract_legendre_coefficients`` the same way.

A caller holding an MF4 section does not know, and should not have to know,
which of the four ENDF representations it turned out to be -- that is the
whole point of the shared method on ``MF4MT``.  The keywords drifted anyway:
``trim`` existed only on the mixed class, so the app's off-grid evaluation
(which passes ``trim=False``) raised ``TypeError`` the moment it was pointed
at a tabulated section, i.e. at any JEFF-4.0 U-235 angular distribution.

Testing the signature rather than a value is deliberate: the failure was not a
wrong number, it was a method that could not be called.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from kika.endf.classes.mf4.isotropic import MF4MTIsotropic
from kika.endf.classes.mf4.mixed import MF4MTMixed
from kika.endf.classes.mf4.polynomial import MF4MTLegendre
from kika.endf.classes.mf4.tabulated import MF4MTTabulated

MF4_CLASSES = [MF4MTIsotropic, MF4MTLegendre, MF4MTTabulated, MF4MTMixed]


@pytest.mark.parametrize("cls", MF4_CLASSES, ids=lambda c: c.__name__)
def test_every_class_accepts_the_shared_keywords(cls) -> None:
    params = inspect.signature(cls.extract_legendre_coefficients).parameters
    for name in ("energy", "max_legendre_order", "trim", "trim_tol", "out_of_range"):
        assert name in params, f"{cls.__name__} cannot be called with {name}"


@pytest.mark.parametrize("cls", MF4_CLASSES, ids=lambda c: c.__name__)
def test_the_shared_keywords_are_keyword_only_past_the_order(cls) -> None:
    """So a positional call cannot silently land a flag in the wrong slot."""
    params = inspect.signature(cls.extract_legendre_coefficients).parameters
    for name in ("trim", "trim_tol", "out_of_range"):
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY


def test_an_isotropic_section_is_callable_with_trim() -> None:
    """The one class where trimming has to be inert: a_l>0 are exactly zero,
    and dropping them would remove orders the caller asked for by name."""
    section = MF4MTIsotropic(number=18)
    coeffs = section.extract_legendre_coefficients(
        np.array([1.0e6]), max_legendre_order=4, trim=True
    )
    assert sorted(coeffs) == [0, 1, 2, 3, 4]
    assert float(np.atleast_1d(coeffs[0])[0]) == pytest.approx(1.0)


# --- the distribution itself, not a reconstruction of it ---------------------


def _peaked_tabulated() -> MF4MTTabulated:
    """A section whose table is sharply forward-peaked at both grid energies.

    Peaked on purpose: a truncated Legendre expansion is indistinguishable from
    the table on a flat distribution, so a test built on one would pass either
    way and prove nothing.
    """
    mu = [-1.0, 0.0, 0.9, 1.0]
    return MF4MTTabulated(
        number=2,
        _energies=[1.0e6, 2.0e6],
        _cosines=[mu, mu],
        _probabilities=[
            [0.01, 0.02, 1.0, 8.0],
            [0.01, 0.02, 2.0, 16.0],
        ],
        _angular_interpolation=[[(4, 2)], [(4, 2)]],
        _interpolation=[(2, 2)],
    )


@pytest.mark.parametrize("cls", MF4_CLASSES, ids=lambda c: c.__name__)
def test_every_class_can_be_asked_for_the_distribution(cls) -> None:
    params = inspect.signature(cls.evaluate_angular_pdf).parameters
    assert {"mu", "energy", "out_of_range"} <= set(params)


def test_a_tabulated_section_returns_its_own_stored_values() -> None:
    """At a grid energy and a stored cosine, the answer is the file's number."""
    section = _peaked_tabulated()
    got = section.evaluate_angular_pdf([-1.0, 0.0, 0.9, 1.0], 1.0e6)[0]
    assert got == pytest.approx([0.01, 0.02, 1.0, 8.0])


def test_a_tabulated_section_is_read_and_not_projected() -> None:
    """The peak survives. Projecting to 10 orders and summing back does not
    reproduce it, which is the whole reason this path exists."""
    section = _peaked_tabulated()
    exact = section.evaluate_angular_pdf([1.0], 1.0e6)[0][0]
    coeffs = section.extract_legendre_coefficients(1.0e6, max_legendre_order=10)
    projected = float(
        np.polynomial.legendre.legval(
            1.0,
            [0.5 * (2 * l + 1) * (1.0 if l == 0 else coeffs[l]) for l in range(11)],
        )
    )
    assert exact == pytest.approx(8.0)
    assert abs(projected - exact) > 0.1


def test_it_interpolates_between_energies_under_the_endf_law() -> None:
    """Lin-lin in energy: halfway between the two tables is their average."""
    section = _peaked_tabulated()
    got = section.evaluate_angular_pdf([1.0], 1.5e6)[0][0]
    assert got == pytest.approx(12.0)


def test_outside_the_energy_grid_follows_out_of_range() -> None:
    section = _peaked_tabulated()
    assert section.evaluate_angular_pdf([1.0], 5.0e6)[0][0] == pytest.approx(0.0)
    held = section.evaluate_angular_pdf([1.0], 5.0e6, out_of_range="hold")[0][0]
    assert held == pytest.approx(16.0)


def test_an_isotropic_section_is_one_half_everywhere() -> None:
    section = MF4MTIsotropic(number=18)
    got = section.evaluate_angular_pdf([-1.0, 0.0, 1.0], [1.0e3, 1.0e6])
    assert got.shape == (2, 3)
    assert got == pytest.approx(0.5)


def test_the_shape_is_energies_by_cosines() -> None:
    section = _peaked_tabulated()
    assert section.evaluate_angular_pdf([-1.0, 0.0, 1.0], [1.0e6, 1.5e6, 2.0e6]).shape == (3, 3)
