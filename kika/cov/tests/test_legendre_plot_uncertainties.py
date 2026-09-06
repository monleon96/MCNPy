"""``LegendreCovariance.plot_uncertainties`` draws something.

The method raised ``ModuleNotFoundError`` on every call from December 2025 until
this test existed: it delegated to ``kika.cov.mf34cov_heatmap``, a shim to the
never-committed ``kika.cov.legacy``. The delegation now goes to
``kika.plotting.covariance.plot_mf34_uncertainties``, which is the same place
every other plotting method on this class already goes.

The old implementation is **not** recoverable verbatim — it was built on
``kika/_plot_settings.py``, deleted by the same commit — so the replacement is
written on ``to_plot_data`` + ``PlotBuilder`` and these tests pin its contract
rather than a pixel comparison against something that no longer exists.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

from kika.endf import read_endf

FIXTURE = Path(__file__).resolve().parents[2] / "endf" / "tests" / "data" / "micro_fe56_cov.endf"


@pytest.fixture(scope="module")
def legendreCovariance():
    """MF34 for Fe-56 MT2 off the committed synthetic micro-tape.

    Two Legendre orders, L=1 and L=2, which is exactly enough to exercise the
    single-order, multi-order and unavailable-order branches.
    """
    endf = read_endf(str(FIXTURE))
    return endf.mf[34].mt[2].to_ang_covmat(mf4_data=endf.mf.get(4))


def test_a_single_order_draws(legendreCovariance):
    figure = legendreCovariance.plot_uncertainties(isotope=26056, mt=2, legendre_coeffs=1)
    assert figure.axes
    axis = figure.axes[0]
    assert axis.get_ylabel() == "Relative Uncertainty (%)"
    # The energy unit comes off the covariance rather than being hardcoded; the
    # sibling function for cross sections hardcodes MeV and is wrong for MF34.
    assert axis.get_xlabel() == f"Energy ({legendreCovariance.energy_unit})"
    assert len(axis.lines) >= 1


def test_several_orders_draw_several_curves(legendreCovariance):
    one = legendreCovariance.plot_uncertainties(isotope=26056, mt=2, legendre_coeffs=1)
    two = legendreCovariance.plot_uncertainties(isotope=26056, mt=2, legendre_coeffs=[1, 2])
    assert len(two.axes[0].lines) > len(one.axes[0].lines)


def test_an_empty_sequence_means_every_available_order(legendreCovariance):
    everything = legendreCovariance.plot_uncertainties(isotope=26056, mt=2, legendre_coeffs=[])
    explicit = legendreCovariance.plot_uncertainties(isotope=26056, mt=2, legendre_coeffs=[1, 2])
    assert len(everything.axes[0].lines) == len(explicit.axes[0].lines)


def test_absolute_uncertainties_relabel_the_axis(legendreCovariance):
    figure = legendreCovariance.plot_uncertainties(
        isotope=26056, mt=2, legendre_coeffs=1, uncertainty_type="absolute"
    )
    assert figure.axes[0].get_ylabel() == "Absolute Uncertainty"


def test_the_methods_own_default_style_works(legendreCovariance):
    """``style="default"`` is the signature's default and PlotBuilder rejects it.

    Mapping it to ``light`` is what keeps the method callable with no arguments
    beyond the required ones. Without the mapping this raises ``ValueError``,
    which would be a second broken method wearing a better exception.
    """
    figure = legendreCovariance.plot_uncertainties(
        isotope=26056, mt=2, legendre_coeffs=1, style="default"
    )
    assert figure.axes


def test_a_symbol_names_the_isotope_too(legendreCovariance):
    figure = legendreCovariance.plot_uncertainties(isotope="Fe56", mt=2, legendre_coeffs=1)
    assert figure.axes


def test_an_unavailable_order_says_what_is_available(legendreCovariance):
    with pytest.raises(ValueError, match=r"Available: \[1, 2\]"):
        legendreCovariance.plot_uncertainties(isotope=26056, mt=2, legendre_coeffs=9)


def test_an_unknown_isotope_mt_pair_raises(legendreCovariance):
    with pytest.raises(ValueError, match="No Legendre coefficients found"):
        legendreCovariance.plot_uncertainties(isotope=26056, mt=102, legendre_coeffs=1)


def test_an_unrecognised_uncertainty_type_raises(legendreCovariance):
    with pytest.raises(ValueError, match="relative"):
        legendreCovariance.plot_uncertainties(
            isotope=26056, mt=2, legendre_coeffs=1, uncertainty_type="fractional"
        )


def test_the_method_no_longer_reaches_a_dead_module():
    """The regression itself, stated as a property rather than as a call.

    ``kika.cov.mf34cov_heatmap`` is gone; this asserts nothing quietly
    reintroduces a delegation to it or to any other ``kika.cov.legacy`` shim.
    """
    source = (Path(__file__).resolve().parents[1] / "legendre_covariance.py").read_text()
    assert "kika.cov.legacy" not in source
    assert "mf34cov_heatmap" not in source
