"""
Plot benchmark sensitivity profiles through the shared kika plotting stack.

Rather than hand-rolling matplotlib, this routes the stored per-group vectors
through :func:`kika.sensitivities.sensitivity_to_plot_data` and
:class:`kika.plotting.PlotBuilder`, the same path used for SDF objects.
"""

from typing import List, Optional, Sequence, Tuple, Union

from kika.benchmarks.database import BenchmarksDatabase
from kika.plotting import PlotBuilder
from kika.sensitivities import sensitivity_to_plot_data

# Colorblind-safe cycle (kika light palette) used when the caller plots several
# reactions on one axis.
_PALETTE = [
    "#0173B2", "#DE8F05", "#029E73", "#D55E00", "#CC78BC",
    "#CA9161", "#FBAFE4", "#949494", "#ECE133", "#56B4E9",
]


def plot_profile(
    source: Union[str, dict],
    reactions: Optional[Sequence[Tuple[int, int]]] = None,
    per_lethargy: bool = True,
    uncertainty: bool = True,
    profile_id: Optional[int] = None,
    db_path: Optional[str] = None,
    style: str = "light",
    figsize: Tuple[float, float] = (8, 6),
    title: Optional[str] = None,
    show: bool = False,
):
    """
    Plot the sensitivity profile(s) of a benchmark.

    Parameters
    ----------
    source : str or dict
        A benchmark id (its preferred profile is used) or a profile-vector dict as
        returned by :meth:`BenchmarksDatabase.get_profile_vector`.
    reactions : sequence of (zaid, mt), optional
        Which reactions to draw. Defaults to every reaction in the vector.
    per_lethargy : bool
        Plot sensitivity per unit lethargy (default True).
    uncertainty : bool
        Draw per-group error bars where error vectors are available.
    profile_id : int, optional
        Plot this specific profile instead of the benchmark's preferred one
        (ignored when ``source`` is a dict).
    db_path : str, optional
        Explicit database path (otherwise resolved from config/env/default).
    style, figsize, title, show
        Forwarded to / used with :class:`~kika.plotting.PlotBuilder`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if isinstance(source, dict):
        vector = source
        default_title = title
    else:
        with BenchmarksDatabase(db_path) as db:
            if profile_id is None:
                profile_id = db.get_preferred_profile(source)["profile_id"]
            vector = db.get_profile_vector(profile_id)
        default_title = title or source

    pert_energies = vector["pert_energies"]
    available = vector["reactions"]
    if reactions is not None:
        wanted = {(int(z), int(m)) for z, m in reactions}
        available = [r for r in available if (r["zaid"], r["mt"]) in wanted]
    if not available:
        raise ValueError("No matching reactions to plot for this benchmark.")

    builder = PlotBuilder(style=style, figsize=figsize)
    for i, r in enumerate(available):
        pd = sensitivity_to_plot_data(
            pert_energies,
            r["sensitivity"],
            error=r.get("error"),
            zaid=r["zaid"],
            mt=r["mt"],
            nuclide=r.get("nuclide"),
            reaction_name=r.get("reaction_name"),
            per_lethargy=per_lethargy,
            uncertainty=uncertainty,
            color=_PALETTE[i % len(_PALETTE)],
        )
        builder.add_data(pd)

    y_label = "Sensitivity per unit lethargy" if per_lethargy else "Sensitivity"
    builder.set_labels(
        title=default_title, x_label="Energy (MeV)", y_label=y_label
    )
    builder.set_scales(log_x=True, log_y=False)
    return builder.build(show=show)
