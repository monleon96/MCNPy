"""Data models for parsed Serpent detector output files (_det.m).

Serpent writes detector results in MATLAB format with arrays like::

    DET<name> = [
      idx Eb Ub Cb Mb Lb Rb Zb Yb Xb value rel_err
      ...
    ];
    DET<name>E = [Emin Emax Emid; ...];   % energy grid
    DET<name>T = [Tmin Tmax Tmid; ...];   % time grid (optional)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# Column names for the standard 12-column DET format (no time binning)
DET_COLUMNS = [
    "idx", "Eb", "Ub", "Cb", "Mb", "Lb", "Rb", "Zb", "Yb", "Xb",
    "value", "rel_error",
]
# With time binning: 13 columns (Tb inserted after idx)
DET_COLUMNS_TIME = [
    "idx", "Tb", "Eb", "Ub", "Cb", "Mb", "Lb", "Rb", "Zb", "Yb", "Xb",
    "value", "rel_error",
]


@dataclass
class DetectorResult:
    """A single detector's results from a ``_det.m`` file."""
    name: str
    data: np.ndarray             # raw matrix, shape (N, 12) or (N, 13)
    energy_grid: Optional[np.ndarray] = None  # from DET<name>E, shape (M, 3)
    time_grid: Optional[np.ndarray] = None    # from DET<name>T, shape (K, 3)

    @property
    def has_time_bins(self) -> bool:
        return self.data.shape[1] == 13

    @property
    def values(self) -> np.ndarray:
        """Mean tally values."""
        return self.data[:, -2]

    @property
    def errors(self) -> np.ndarray:
        """Relative statistical errors."""
        return self.data[:, -1]

    @property
    def n_bins(self) -> int:
        return self.data.shape[0]

    @property
    def n_energy_bins(self) -> int:
        if self.energy_grid is not None:
            return self.energy_grid.shape[0]
        return 0

    @property
    def n_time_bins(self) -> int:
        if self.time_grid is not None:
            return self.time_grid.shape[0]
        return 0

    @property
    def energy_mid(self) -> Optional[np.ndarray]:
        """Mid-point energies (MeV), if energy grid is available."""
        if self.energy_grid is not None:
            return self.energy_grid[:, 2]
        return None

    @property
    def energy_edges(self) -> Optional[np.ndarray]:
        """Energy bin edges [Emin, Emax] pairs, if energy grid is available."""
        if self.energy_grid is not None:
            return self.energy_grid[:, :2]
        return None

    @property
    def absolute_errors(self) -> np.ndarray:
        """Absolute statistical errors (value * relative_error)."""
        return self.values * self.errors

    @property
    def bin_dimensions(self) -> Dict[str, int]:
        """Active binning dimensions (those with more than one unique value).

        Returns a dict mapping dimension name to unique bin count,
        e.g. ``{"Eb": 40, "Mb": 3}``.
        """
        cols = DET_COLUMNS_TIME if self.has_time_bins else DET_COLUMNS
        result: Dict[str, int] = {}
        # Skip idx (col 0) and value + rel_error (last two cols)
        for name in cols[1:-2]:
            col_idx = cols.index(name)
            n_unique = len(np.unique(self.data[:, col_idx].astype(int)))
            if n_unique > 1:
                result[name] = n_unique
        return result

    def spectrum_data(self, time_index: Optional[int] = None) -> Dict[str, Any]:
        """Plot-ready energy spectrum data.

        Parameters
        ----------
        time_index : int, optional
            0-based time bin index.  Required when the detector has time
            bins; ignored otherwise.

        Returns
        -------
        dict with keys ``energy_mid``, ``energy_low``, ``energy_high``,
        ``values``, ``errors`` (all 1-D arrays), or empty dict if no
        energy grid is available.
        """
        if self.energy_grid is None:
            return {}

        data = self.data
        if self.has_time_bins and self.time_grid is not None:
            tb_col = DET_COLUMNS_TIME.index("Tb")
            if time_index is not None:
                # Filter rows for the requested time bin (1-based in data)
                mask = data[:, tb_col].astype(int) == (time_index + 1)
                data = data[mask]
            else:
                # Default to first time bin
                mask = data[:, tb_col].astype(int) == 1
                data = data[mask]

        n_energy = self.energy_grid.shape[0]
        # If data has more rows than energy bins (extra spatial/material bins),
        # aggregate by energy bin index.
        eb_col = (DET_COLUMNS_TIME if self.has_time_bins else DET_COLUMNS).index("Eb")
        eb_indices = data[:, eb_col].astype(int)  # 1-based

        values = np.zeros(n_energy)
        errors = np.zeros(n_energy)
        for ei in range(n_energy):
            mask = eb_indices == (ei + 1)
            if mask.any():
                vals = data[mask, -2]
                errs = data[mask, -1]
                # Sum values across other bins, propagate errors in quadrature
                values[ei] = np.sum(vals)
                abs_errs = vals * errs
                errors[ei] = np.sqrt(np.sum(abs_errs ** 2)) / values[ei] if values[ei] != 0 else 0.0

        return {
            "energy_mid": self.energy_grid[:, 2],
            "energy_low": self.energy_grid[:, 0],
            "energy_high": self.energy_grid[:, 1],
            "values": values,
            "errors": errors,
        }

    def time_series_data(self, energy_index: Optional[int] = None) -> Dict[str, Any]:
        """Plot-ready time-series data.

        Parameters
        ----------
        energy_index : int, optional
            0-based energy bin index.  If ``None``, sums over all energy
            bins.

        Returns
        -------
        dict with keys ``time_mid``, ``time_low``, ``time_high``,
        ``values``, ``errors`` (all 1-D arrays), or empty dict if no
        time grid is available.
        """
        if self.time_grid is None or not self.has_time_bins:
            return {}

        cols = DET_COLUMNS_TIME
        tb_col = cols.index("Tb")
        eb_col = cols.index("Eb")
        n_time = self.time_grid.shape[0]

        values = np.zeros(n_time)
        errors = np.zeros(n_time)

        for ti in range(n_time):
            mask = self.data[:, tb_col].astype(int) == (ti + 1)
            if energy_index is not None:
                mask &= self.data[:, eb_col].astype(int) == (energy_index + 1)
            if mask.any():
                vals = self.data[mask, -2]
                errs = self.data[mask, -1]
                values[ti] = np.sum(vals)
                abs_errs = vals * errs
                errors[ti] = np.sqrt(np.sum(abs_errs ** 2)) / values[ti] if values[ti] != 0 else 0.0

        return {
            "time_mid": self.time_grid[:, 2],
            "time_low": self.time_grid[:, 0],
            "time_high": self.time_grid[:, 1],
            "values": values,
            "errors": errors,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to a pandas DataFrame with named columns."""
        cols = DET_COLUMNS_TIME if self.has_time_bins else DET_COLUMNS
        return pd.DataFrame(self.data, columns=cols)

    def __repr__(self) -> str:
        parts = [f"DetectorResult('{self.name}', bins={self.n_bins}"]
        if self.n_energy_bins:
            parts.append(f"ene_bins={self.n_energy_bins}")
        if self.n_time_bins:
            parts.append(f"time_bins={self.n_time_bins}")
        return ", ".join(parts) + ")"


@dataclass
class DetectorFile:
    """Container for all detectors parsed from a ``_det.m`` file."""
    detectors: Dict[str, DetectorResult] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.detectors)

    def __getitem__(self, name: str) -> DetectorResult:
        return self.detectors[name]

    def __contains__(self, name: str) -> bool:
        return name in self.detectors

    def __iter__(self):
        return iter(self.detectors.values())

    @property
    def names(self) -> List[str]:
        return list(self.detectors.keys())

    def summary(self) -> str:
        """Compact text summary."""
        lines = [f"DetectorFile with {len(self.detectors)} detectors:"]
        for name, det in self.detectors.items():
            parts = [f"  {name}: {det.n_bins} bins"]
            if det.n_energy_bins:
                parts.append(f"{det.n_energy_bins} energy bins")
            if det.n_time_bins:
                parts.append(f"{det.n_time_bins} time bins")
            lines.append(", ".join(parts))
        return "\n".join(lines)

    def __repr__(self) -> str:
        names = ", ".join(f"'{n}'" for n in list(self.detectors.keys())[:5])
        if len(self.detectors) > 5:
            names += f", ... ({len(self.detectors)} total)"
        return f"DetectorFile([{names}])"
