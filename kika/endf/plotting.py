"""Which nominal goes with which covariance — the ENDF plotting adapter.

An MF file can turn *itself* into plot data. What it cannot know is that MF34
is MF4's covariance and MF33 is MF3's: that pairing is a property of the tape
as a whole, so it lived on the ``ENDF`` container, and it grew until 292 of
that class's 413 lines were plotting orchestration. Phase 4's P3 moves it here
and leaves the dataclass a syntactic container: a dictionary of MF files, a
MAT, and the derived names.

**Why this module sits in ``kika/endf`` and not in ``kika/plotting``.** It is
not generic plotting — it never draws anything, and it produces the same
``PlotData`` objects the MF files produce. What it *is* is a piece of ENDF
knowledge: MF33 covaries MF3, MF34 covaries MF4, MF34 needs MF4's Legendre
coefficients handed to it to be interpretable. That vocabulary is the format's,
so the code belongs on the format's side of the arrow, above the container and
below ``kika.plotting``. ``kika/plotting`` today imports no format package at
all, and this would have been its first.

**The two associations are re-derived here, once per call, by hand.** They are
the thing GNDS answers with a ``DataLink.href``: a covariance node points at
the data it covaries, so nothing has to guess from an MF/MT pair. When
``kika/cov`` reads and writes through ``CovarianceSuite`` these two functions
stop asking the question and start following the link. That is the shape of
the work, not this increment.

**Freezing the 292 lines before moving them found two defects, and only one of
them is small enough to fix here.**

*Fixed, in the commit after the move.* Naming a colour cost you the band.
``**kwargs`` was forwarded to the covariance after stripping a short list of
keys that did not include ``color``, and then ``color=`` was supplied a second
time explicitly — so ``TypeError``, caught, band ``None``. Both functions had
their own copy of it. "Draw this in my colour" and "draw this with its
uncertainty" were mutually exclusive, and the symptom was indistinguishable
from a file that carries no covariance.

*Fixed afterwards, in ``kika/cov``: the MF33 band could not be produced at
all.* ``MF33MT.to_xs_covmat`` builds its ``CrossSectionCovariance`` with
``add_matrix`` alone, and ``add_matrix`` does not take cross sections — which
``CrossSectionCovariance.to_plot_data`` gated its whole answer on, and then
did not use, because ``sqrt(diag)`` of a relative covariance already is the
fractional uncertainty. So the MF3 branch below raised internally, was caught,
and returned ``None``, on every tape for eight months, and the symptom was
again indistinguishable from a file that carries no covariance. Recorded as
D13 in ``docs/library/library-gaps.md``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - vocabulary only, no runtime coupling
    from .classes.endf import ENDF


def endf_to_plot_data(endf: 'ENDF', mf: int, mt: int, uncertainty: bool = None,
                      sigma: float = 1.0, reconstructed: bool = False,
                      tolerance: float = 1e-3, **kwargs):
    """Plot data for one MF/MT section, paired with its covariance if there is one.

    The implementation behind :meth:`kika.endf.classes.endf.ENDF.to_plot_data`,
    which is the documented entry point and where the parameters are described.
    Everything here reads ``endf.files`` and ``endf.zaid`` and nothing else, so
    it works on any container that carries those two.
    """
    if mf not in endf.files:
        raise KeyError(f"MF file {mf} not found in ENDF")

    # Validate reconstructed is only used with MF3
    if reconstructed and mf != 3:
        raise ValueError(
            f"reconstructed is only supported for MF3 (cross sections), not MF{mf}."
        )

    # Auto-set uncertainty based on MF file type
    if uncertainty is None:
        uncertainty = (mf in (3, 4))

    # Force uncertainty=False for covariance files themselves. Without this the
    # next check would reject MF33/MF34 as "not MF3 or MF4", and mf=34 is how
    # the app draws an uncertainty on its own.
    if mf in (33, 34) and uncertainty:
        uncertainty = False

    # Validate that uncertainty is only used with MF3 or MF4
    if uncertainty and mf not in (3, 4):
        raise ValueError(
            f"uncertainty is only supported for MF3 and MF4, not MF{mf}."
        )

    # --- MF3: cross sections with optional reconstruction and uncertainty ---
    if mf == 3 and (reconstructed or uncertainty):
        return _mf3_plot_data(endf, mt, uncertainty, sigma, reconstructed,
                              tolerance, **kwargs)

    # Get the main plot data
    plot_data = endf.files[mf].to_plot_data(mt=mt, **kwargs)

    # If uncertainties are not requested, return just the plot data
    if not uncertainty:
        return plot_data

    # --- MF4: angular distributions with MF34 uncertainty ---
    uncertainty_band = None

    # Check if MF34 exists and has the requested MT section
    if 34 in endf.files and mt in endf.files[34].mt:
        try:
            # Extract the order parameter (required for MF4). Unreachable in
            # practice: MF4's own to_plot_data above already raises for a
            # missing order, so this never gets to speak.
            order = kwargs.get('order')
            if order is None:
                raise ValueError("'order' parameter is required when uncertainty=True for MF4")

            # Get MF34 covariance data (pass MF4 to populate Legendre coefficients)
            mf34_mt = endf.files[34].mt[mt]
            mf4 = endf.files.get(4)
            mf34_covmat = mf34_mt.to_ang_covmat(mf4_data=mf4)

            # Get isotope ID (ZAID)
            isotope_id = endf.zaid if endf.zaid is not None else int(mf34_mt._za)

            # Prepare kwargs for LegendreCovariance.to_plot_data - remove
            # parameters we're setting explicitly. ``color`` is one of them:
            # the band is coloured to match the nominal below, and the nominal
            # already took its colour from these same kwargs.
            styling_kwargs = {k: v for k, v in kwargs.items()
                              if k not in ['order', 'mt', 'nuclide',
                                           'uncertainty_type', 'color']}

            # Get uncertainty data from MF34 (returns tuple: (None, unc_data))
            _, unc_native = mf34_covmat.to_plot_data(
                nuclide=isotope_id,
                mt=mt,
                order=order,
                uncertainty_type='relative',
                color=plot_data.color,  # Match the main plot color
                **styling_kwargs
            )

            if unc_native is not None:
                # Return the native MF34 uncertainty data directly (on sparse grid)
                # This allows it to be:
                # 1. Plotted as a step plot (preserves native structure)
                # 2. Used as uncertainty band (PlotBuilder will handle interpolation)

                # Apply sigma multiplier if needed
                if sigma != 1.0:
                    import numpy as np
                    unc_native.y = np.array(unc_native.y) * sigma
                    # Update label to reflect sigma level
                    if unc_native.label:
                        unc_native.label = unc_native.label.replace('(σ %)', f'({sigma}σ %)')

                # Match color to nominal data
                unc_native.color = plot_data.color

                # Use the native uncertainty data directly
                uncertainty_band = unc_native

                # Append uncertainty info to the main plot label
                if plot_data.label is not None:
                    sigma_suffix = f" (±{sigma}σ)" if sigma != 1.0 else " (±1σ)"
                    plot_data.label = plot_data.label + sigma_suffix
            else:
                uncertainty_band = None

        except Exception as e:
            # If anything goes wrong, just skip the uncertainty band
            print(f"Warning: Could not create uncertainty band: {e}")
            uncertainty_band = None

    return plot_data, uncertainty_band


def _mf3_plot_data(endf: 'ENDF', mt: int, uncertainty: bool, sigma: float,
                   reconstructed: bool, tolerance: float, **kwargs):
    """Handle MF3 plot data with optional reconstruction and uncertainty.

    Returns PlotData if uncertainty=False, or (PlotData, unc_band) if True.
    """
    import warnings

    # --- Nominal data: reconstructed or raw ---
    plot_data = None

    if reconstructed:
        # Consume what the caller supplied; never reconstruct behind their
        # back. This used to call reconstruct_xs(), which produced numbers
        # its own docstring said were wrong, and labelled the curve
        # "(reconstructed)" without qualification.
        pendf = endf.pendf

        if pendf is None:
            warnings.warn(
                "reconstructed=True but endf.pendf is not set — returning "
                "the raw MF3 background. Populate it first, e.g. "
                "endf.pendf = kika.processing.njoy_reconstruct(path, "
                "njoy_executable=...)."
            )

        if pendf is not None and mt in pendf:
            plot_data = pendf[mt].to_plot_data(**kwargs)
            # Mark as reconstructed if user didn't supply a custom label
            if 'label' not in kwargs and plot_data.label is not None:
                plot_data.label += " (reconstructed)"
            plot_data.data_source = "pendf"
        elif pendf is not None:
            warnings.warn(
                f"MT{mt} not available in reconstructed data "
                f"(available: {sorted(pendf.keys())}). "
                f"Falling back to raw MF3."
            )

    # Fall back to raw MF3 if not reconstructed or reconstruction unavailable
    if plot_data is None:
        plot_data = endf.files[3].to_plot_data(mt=mt, **kwargs)

    # --- Uncertainty from MF33 ---
    if not uncertainty:
        return plot_data

    uncertainty_band = None

    if 33 in endf.files and mt in endf.files[33].mt:
        try:
            mf33_mt = endf.files[33].mt[mt]

            # Build sibling and MF3 section maps for NC LTY=0 resolution
            mf33_siblings = {
                int(k): v for k, v in endf.files[33].sections.items()
                if int(k) != mt
            } if hasattr(endf.files[33], 'sections') else None

            # No auto-reconstruction: if the caller set endf.pendf it is
            # overlaid below, otherwise the raw MF3 background is used.
            # This used to call reconstruct_xs() inside a bare
            # `except Exception: pass`, so a failure of a reconstructor
            # already known to be wrong was invisible twice over.

            # Build cross-section map: start with raw MF3, then overlay
            # reconstructed (PENDF) data for MTs that need resonance
            # contributions (MT1, MT2, MT102, ...).
            mf3_secs = None
            if 3 in endf.files and hasattr(endf.files[3], 'sections'):
                mf3_secs = {int(k): v for k, v in
                            endf.files[3].sections.items()}
            if endf.pendf:
                if mf3_secs is None:
                    mf3_secs = {}
                for k, v in endf.pendf.items():
                    mf3_secs[int(k)] = v

            mf33_covmat = mf33_mt.to_xs_covmat(
                sibling_sections=mf33_siblings,
                mf3_sections=mf3_secs,
            )

            isotope_id = endf.zaid if endf.zaid is not None else int(mf33_mt._za)

            # Prepare styling kwargs (remove MF3-specific params, and the ones
            # supplied explicitly just below — ``color`` above all)
            styling_kwargs = {k: v for k, v in kwargs.items()
                              if k not in ['mt', 'nuclide',
                                           'uncertainty_type', 'color']}

            _, unc_native = mf33_covmat.to_plot_data(
                nuclide=isotope_id,
                mt=mt,
                sigma=sigma,
                color=plot_data.color,
                **styling_kwargs
            )

            if unc_native is not None:
                if sigma != 1.0 and unc_native.label:
                    unc_native.label = unc_native.label.replace(
                        '(σ %)', f'({sigma}σ %)')

                unc_native.color = plot_data.color
                uncertainty_band = unc_native

                if plot_data.label is not None:
                    sigma_suffix = f" (±{sigma}σ)" if sigma != 1.0 else " (±1σ)"
                    plot_data.label += sigma_suffix

        except Exception as e:
            print(f"Warning: Could not create MF33 uncertainty band: {e}")
            uncertainty_band = None

    return plot_data, uncertainty_band
