"""Query result container for weight window queries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class QueryResult:
    """Structured result from a weight window spatial/energy/time query.

    Parameters
    ----------
    particle_types : list[int]
        Particle indices included in the result.
    ww_values : list[np.ndarray]
        One array per particle, shape ``(nt, ne, nfz, nfy, nfx)`` (sliced).
    energy_intervals : list[tuple[np.ndarray, np.ndarray]]
        ``(starts, ends)`` for energy bins per particle.
    time_intervals : list[tuple[np.ndarray, np.ndarray]]
        ``(starts, ends)`` for time bins per particle.
    spatial_intervals : dict[str, tuple[np.ndarray, np.ndarray]]
        ``{label: (starts, ends)}`` for each spatial axis.
    """

    particle_types: list[int]
    ww_values: list[np.ndarray]
    energy_intervals: list[tuple[np.ndarray, np.ndarray]]
    time_intervals: list[tuple[np.ndarray, np.ndarray]]
    spatial_intervals: dict[str, tuple[np.ndarray, np.ndarray]]

    def to_dataframe(self) -> "pd.DataFrame":
        """Flatten the query result into a pandas DataFrame."""
        import pandas as pd  # lazy import

        rows: list[dict] = []
        axis_labels = list(self.spatial_intervals.keys())

        for p_idx, p_type in enumerate(self.particle_types):
            e_starts, e_ends = self.energy_intervals[p_idx]
            t_starts, t_ends = self.time_intervals[p_idx]
            ww = self.ww_values[p_idx]

            if t_starts.size == 0:
                t_starts = np.array([0.0])
                t_ends = np.array([np.inf])

            # spatial intervals per axis
            s_starts, s_ends = {}, {}
            for label in axis_labels:
                s_starts[label], s_ends[label] = self.spatial_intervals[label]

            a1, a2, a3 = axis_labels

            for t_i, (ts, te) in enumerate(zip(t_starts, t_ends)):
                for e_i, (es, ee) in enumerate(zip(e_starts, e_ends)):
                    for k, (k_s, k_e) in enumerate(
                        zip(s_starts[a3], s_ends[a3])
                    ):
                        for j, (j_s, j_e) in enumerate(
                            zip(s_starts[a2], s_ends[a2])
                        ):
                            for i, (i_s, i_e) in enumerate(
                                zip(s_starts[a1], s_ends[a1])
                            ):
                                val = float(ww[t_i, e_i, k, j, i])
                                rows.append(
                                    {
                                        "particle_type": p_type,
                                        "time_start": ts,
                                        "time_end": te,
                                        "energy_start": es,
                                        "energy_end": ee,
                                        f"{a1}_start": i_s,
                                        f"{a1}_end": i_e,
                                        f"{a2}_start": j_s,
                                        f"{a2}_end": j_e,
                                        f"{a3}_start": k_s,
                                        f"{a3}_end": k_e,
                                        "ww_value": val,
                                    }
                                )

        return pd.DataFrame(rows)
