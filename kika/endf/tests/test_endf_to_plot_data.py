"""What the ``ENDF`` container adds on top of its MF files, frozen.

``ENDF.to_plot_data`` is 292 of the class's 413 lines and it had no test at
all. Phase 4's P3 moves those lines off the dataclass, so they are pinned here
first — every branch, on the two committed micro-tapes, before anything moves.
The point is not that plotting works; it is that **the association does not
change**, because the association is the only thing the container contributes.

**What the container actually does.** An MF file can turn *itself* into plot
data; only the container knows that MF34 is MF4's covariance and MF33 is MF3's.
That pairing — "which nominal goes with which uncertainty" — is re-derived by
hand in two places, once per pair, and it is what the GNDS model answers with a
``DataLink.href`` instead. Until it does, these tests are the contract.

**Freezing it first found two defects**, and the tests say where they are. A
characterization test that quietly rounds off what it finds buys nothing: the
move carried both warts across unchanged, and then the colour collision was
fixed in the commit on top, which is what these tests were for. The MF33 band
is still unreachable and its test still asserts the failure.

The fixtures are chosen for what they carry, not for the isotope:
``micro_fe56_structural`` has MF3 and MF4 and MF34 but no MF33;
``micro_fe56_cov`` has MF33 and MF34 but its MF3/MT2 is a bare header. Neither
committed tape has MF3 and MF33 together, so the MF3↔MF33 test grafts one onto
the other — legitimate here precisely because ``ENDF`` is a container and the
code under test reads nothing but ``self.files``.
"""
from __future__ import annotations

import numpy as np
import pytest

from kika.endf import read_endf


@pytest.fixture(scope="module")
def structural(micro_tape):
    """MF1/451, MF2/151, MF3, MF4, MF34 — the nominal side, with MF34."""
    return read_endf(str(micro_tape))


@pytest.fixture(scope="module")
def cov(micro_cov_tape):
    """MF3 (header only), MF33, MF34 — the covariance side."""
    return read_endf(str(micro_cov_tape))


@pytest.fixture
def mf3_with_mf33(micro_tape, micro_cov_tape):
    """One container holding a real MF3 and a real MF33.

    No committed tape has both, and the pair is what ``_mf3_plot_data`` needs.
    Grafting is sound because the container is a dictionary of MF files and
    both sides are the same nuclide, MAT 2631.
    """
    endf = read_endf(str(micro_tape))
    endf.files[33] = read_endf(str(micro_cov_tape)).files[33]
    return endf


# ---------------------------------------------------------------------------
# MF4 ↔ MF34 — the association at endf.py:185-195
# ---------------------------------------------------------------------------

def test_mf4_comes_back_paired_with_its_mf34_band(structural):
    """The default for MF4 is a tuple, and the nominal half is untouched.

    Everything about the first element comes from ``MF4``'s own
    ``to_plot_data``; the container's whole contribution is the second element
    and a suffix on the label. Asserting equality against the MF file's own
    answer is what makes "the container adds nothing else" checkable.
    """
    plot_data, band = structural.to_plot_data(mf=4, mt=2, order=1)

    bare = structural.files[4].to_plot_data(mt=2, order=1)
    np.testing.assert_array_equal(np.asarray(plot_data.x), np.asarray(bare.x))
    np.testing.assert_array_equal(np.asarray(plot_data.y), np.asarray(bare.y))
    assert bare.label == "26056 Mixed L=1"
    assert plot_data.label == "26056 Mixed L=1 (±1σ)"

    assert band is not None
    assert type(band).__name__ == "LegendreUncertaintyPlotData"
    assert band.plot_type == "step"
    assert band.label == "Fe56 MT=2 L=1 (σ %)"
    # The native MF34 grid, not the nominal's 3960 points: the band is handed
    # back sparse on purpose, for PlotBuilder to step.
    assert np.asarray(band.x).shape == (43,)
    assert np.asarray(band.y).shape == (43,)
    assert float(band.y[0]) == pytest.approx(30.11976095522672, rel=0, abs=1e-12)
    assert float(band.y[-1]) == pytest.approx(3.000000166666662, rel=0, abs=1e-12)


def test_styling_the_curve_does_not_cost_you_the_band(structural, capsys):
    """``color`` used to be passed twice, and the band paid for it.

    ``**kwargs`` was forwarded to the covariance after stripping ``order``,
    ``mt``, ``nuclide`` and ``uncertainty_type`` — not ``color`` — and then
    ``color=plot_data.color`` was supplied a second time. So a caller who named
    a colour got ``TypeError`` inside the ``except`` and an empty second
    element, which is also what a file with no covariance returns. Both
    functions had their own copy of the collision.

    Nothing in kika-app hit it: all five callers style the trace afterwards,
    from the ``PlotData`` they get back. The colour still reaches the band —
    via the nominal, which took it from these same kwargs — so this asserts
    both halves.
    """
    plot_data, band = structural.to_plot_data(
        mf=4, mt=2, order=1, color="#123456"
    )
    assert plot_data.color == "#123456"
    assert band is not None
    assert band.color == "#123456"
    assert capsys.readouterr().out == ""

    # The numbers are the ones the uncoloured call gives; only the colour moved.
    _, plain = structural.to_plot_data(mf=4, mt=2, order=1)
    np.testing.assert_array_equal(np.asarray(band.y), np.asarray(plain.y))


def test_sigma_scales_the_mf34_band_and_both_labels(structural):
    """σ multiplies the band and is written into both labels, not just one."""
    _, one_sigma = structural.to_plot_data(mf=4, mt=2, order=1)
    plot_data, band = structural.to_plot_data(mf=4, mt=2, order=1, sigma=2.0)

    np.testing.assert_allclose(
        np.asarray(band.y, dtype=float),
        2.0 * np.asarray(one_sigma.y, dtype=float),
        rtol=0, atol=0,
    )
    assert band.label == "Fe56 MT=2 L=1 (2.0σ %)"
    assert plot_data.label == "26056 Mixed L=1 (±2.0σ)"


def test_uncertainty_false_returns_one_object_not_a_tuple(structural):
    """The app branches on ``isinstance(result, tuple)``; this is that switch."""
    result = structural.to_plot_data(mf=4, mt=2, order=1, uncertainty=False)
    assert not isinstance(result, tuple)
    assert result.label == "26056 Mixed L=1"


def test_mf4_without_an_order_is_the_mf_file_s_own_refusal(structural):
    """``order`` is required by MF4 itself, so the container never sees it.

    Worth pinning because the container *also* has an "'order' is required"
    message, at endf.py:190, and it is unreachable: ``self.files[4]
    .to_plot_data`` is called first and raises before the MF34 block runs.
    """
    with pytest.raises(TypeError, match="order"):
        structural.to_plot_data(mf=4, mt=2)


def test_mf34_asked_for_directly_is_pure_delegation(cov):
    """``mf=34`` is how the app draws an uncertainty on its own.

    ``uncertainty`` defaults to False for MF34, so none of the association code
    runs — the (None, band) tuple is MF34's own shape, handed straight through.
    """
    nothing, band = cov.to_plot_data(mf=34, mt=2, order=1)
    assert nothing is None
    assert band.label == "Fe56 MT=2 L=1 (σ %)"
    assert np.asarray(band.x).shape == (4,)
    assert float(band.y[0]) == pytest.approx(10.2469507659596, rel=0, abs=1e-12)


# ---------------------------------------------------------------------------
# MF3 ↔ MF33 — the association at endf.py:292-325
# ---------------------------------------------------------------------------

def test_mf3_with_no_mf33_is_the_raw_section_and_a_none(structural):
    """MF3 defaults to ``uncertainty=True`` too, so the tuple comes back empty."""
    plot_data, band = structural.to_plot_data(mf=3, mt=102)
    bare = structural.files[3].to_plot_data(mt=102)

    np.testing.assert_array_equal(np.asarray(plot_data.y), np.asarray(bare.y))
    assert plot_data.data_source == "endf"
    assert band is None


def test_the_mf3_uncertainty_band_cannot_be_produced_at_all(mf3_with_mf33, capsys):
    """A defect, frozen as one — the MF33 half of this method is unreachable.

    ``MF33MT.to_xs_covmat`` builds a ``CrossSectionCovariance`` by calling
    ``add_matrix`` only, and ``add_matrix`` never populates ``cross_sections``.
    ``CrossSectionCovariance.to_plot_data`` needs ``cross_sections[(zaid, mt)]``
    for *both* halves of its answer and raises ``ValueError`` when it has
    neither. So this branch always raises internally, and endf.py:354 catches
    every exception, prints, and returns ``None``.

    It is structural rather than a property of these fixtures — no tape
    populates a dictionary nothing writes to — and it is invisible in
    production because the caller gets a plot with no band, which is also what
    a file with no covariance looks like.

    Recorded in ``docs/library-gaps.md``. Pinned here so the move carries it
    across unchanged and so the fix, when it comes, has to delete this test.
    """
    plot_data, band = mf3_with_mf33.to_plot_data(mf=3, mt=2)

    assert plot_data is not None
    assert band is None
    assert "Could not create MF33 uncertainty band" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# The pendf overlay
# ---------------------------------------------------------------------------

def test_reconstructed_without_pendf_warns_and_gives_the_background(structural):
    """No silent substitution: the caller asked for σ they did not supply.

    ``reconstructed=True`` used to run an in-Python reconstructor whose own
    docstring called its numbers wrong. Nothing populates ``pendf`` on its own
    any more, so the honest answer is the raw MF3 background plus a warning.
    """
    with pytest.warns(UserWarning, match="endf.pendf is not set"):
        plot_data, _ = structural.to_plot_data(mf=3, mt=2, reconstructed=True)

    bare = structural.files[3].to_plot_data(mt=2)
    np.testing.assert_array_equal(np.asarray(plot_data.y), np.asarray(bare.y))
    assert plot_data.data_source == "endf"
    assert plot_data.label == "26056 MT=2"


def test_pendf_is_read_and_the_curve_says_so(micro_tape):
    """The overlay changes the source and the label, and only those.

    ``pendf`` is fed the raw MF3 section here rather than a reconstruction:
    this test is about *which section the container reaches for*, and using
    the same numbers on both sides is what makes the label and ``data_source``
    the only differences it can report.
    """
    endf = read_endf(str(micro_tape))
    endf.pendf = {2: endf.files[3].mt[2]}

    plot_data, _ = endf.to_plot_data(mf=3, mt=2, reconstructed=True)

    assert plot_data.data_source == "pendf"
    assert plot_data.label == "26056 MT=2 (reconstructed)"


def test_a_custom_label_is_not_annotated(micro_tape):
    """"(reconstructed)" is appended only to a label the container invented."""
    endf = read_endf(str(micro_tape))
    endf.pendf = {2: endf.files[3].mt[2]}

    plot_data, _ = endf.to_plot_data(
        mf=3, mt=2, reconstructed=True, label="mine"
    )
    assert plot_data.label == "mine"
    assert plot_data.data_source == "pendf"


def test_an_mt_missing_from_pendf_falls_back_and_names_what_is_there(micro_tape):
    """Partial pendf is the normal case — NJOY writes the MTs it was asked for."""
    endf = read_endf(str(micro_tape))
    endf.pendf = {2: endf.files[3].mt[2]}

    with pytest.warns(UserWarning, match=r"MT102 not available.*\[2\]"):
        plot_data, _ = endf.to_plot_data(mf=3, mt=102, reconstructed=True)

    assert plot_data.data_source == "endf"


# ---------------------------------------------------------------------------
# The refusals
# ---------------------------------------------------------------------------

def test_a_missing_mf_is_a_keyerror_before_anything_else(structural):
    assert 5 not in structural.files
    with pytest.raises(KeyError, match="MF file 5 not found"):
        structural.to_plot_data(mf=5, mt=18)


def test_reconstruction_is_refused_for_every_mf_but_three(structural):
    with pytest.raises(ValueError, match="only supported for MF3"):
        structural.to_plot_data(mf=4, mt=2, order=1, reconstructed=True)


def test_uncertainty_is_refused_for_every_mf_but_three_and_four(structural):
    with pytest.raises(ValueError, match="only supported for MF3 and MF4"):
        structural.to_plot_data(mf=1, mt=451, uncertainty=True)


def test_a_covariance_file_is_never_asked_for_its_own_uncertainty(cov):
    """``uncertainty=True`` on MF33/MF34 is downgraded, not refused.

    Without the downgrade at endf.py:160 the next check would reject MF34 as
    "not MF3 or MF4" — so this is what lets the app's ``mf=34`` call survive a
    caller that passes ``uncertainty=True`` along with it.
    """
    forced = cov.to_plot_data(mf=34, mt=2, order=1, uncertainty=True)
    default = cov.to_plot_data(mf=34, mt=2, order=1)
    assert forced[0] is None and default[0] is None
    np.testing.assert_array_equal(
        np.asarray(forced[1].y), np.asarray(default[1].y)
    )
