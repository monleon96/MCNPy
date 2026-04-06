"""Data models for parsed Serpent results output files (_res.m).

Serpent writes run statistics in MATLAB format with indexed variables::

    ANA_KEFF                  (idx, [1:   6]) = [ ... ];
    IMP_KEFF                  (idx, [1:   2]) = [ ... ];
    RUNNING_TIME              (idx, 1)        = [ ... ];

Values are stored as ``[mean, rel_error]`` pairs.  For burnup runs each
``idx`` corresponds to a burnup step, so variables grow row-by-row.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class ResultsFile:
    """All variables from a Serpent ``_res.m`` file.

    Parameters
    ----------
    variables : dict
        Mapping of variable name to numpy array of shape
        ``(n_burnup_steps, n_values)``.
    n_burnup_steps : int
        Number of burnup steps (rows per variable).  1 for non-burnup runs.
    """
    variables: Dict[str, np.ndarray] = field(default_factory=dict)
    n_burnup_steps: int = 1

    # ── Named accessors for common variables ────────────────────────────

    @property
    def ana_keff(self) -> Optional[np.ndarray]:
        """Analog k-eff, shape ``(n_steps, 6)`` — 3 ``[mean, relerr]`` pairs."""
        return self.variables.get("ANA_KEFF")

    @property
    def imp_keff(self) -> Optional[np.ndarray]:
        """Implicit k-eff, shape ``(n_steps, 2)`` — ``[mean, relerr]``."""
        return self.variables.get("IMP_KEFF")

    @property
    def col_keff(self) -> Optional[np.ndarray]:
        """Collision k-eff, shape ``(n_steps, 2)`` — ``[mean, relerr]``."""
        return self.variables.get("COL_KEFF")

    @property
    def running_time(self) -> Optional[np.ndarray]:
        """Wall-clock time in seconds, shape ``(n_steps, 1)``."""
        return self.variables.get("RUNNING_TIME")

    @property
    def cycle_idx(self) -> Optional[np.ndarray]:
        """Cycle count per step, shape ``(n_steps, 1)``."""
        return self.variables.get("CYCLE_IDX")

    @property
    def variable_names(self) -> List[str]:
        """Sorted list of all variable names."""
        return sorted(self.variables.keys())

    @property
    def n_variables(self) -> int:
        return len(self.variables)

    # ── Utility methods ─────────────────────────────────────────────────

    def get_variable(self, name: str) -> Optional[np.ndarray]:
        """Get a variable by name, or ``None`` if absent."""
        return self.variables.get(name)

    def get_mean_error_pairs(
        self, name: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Split interleaved ``[mean, err, mean, err, ...]`` columns.

        Returns
        -------
        (means, errors) : tuple of ndarray
            ``means`` has shape ``(n_steps, n_values/2)``,
            ``errors`` has shape ``(n_steps, n_values/2)``.
            Returns ``None`` if the variable is absent or has odd column count.
        """
        arr = self.variables.get(name)
        if arr is None:
            return None
        ncols = arr.shape[1] if arr.ndim == 2 else 1
        if ncols % 2 != 0:
            return None
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        means = arr[:, 0::2]
        errors = arr[:, 1::2]
        return means, errors

    def get_last_step(self, name: str) -> Optional[np.ndarray]:
        """Get the values from the last burnup step for a variable."""
        arr = self.variables.get(name)
        if arr is None:
            return None
        return arr[-1] if arr.ndim == 2 else arr

    def __repr__(self) -> str:
        return (
            f"ResultsFile({self.n_variables} variables, "
            f"{self.n_burnup_steps} burnup steps)"
        )
