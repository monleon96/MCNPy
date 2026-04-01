"""Header dataclass for WWINP files."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Header:
    """Header information for a WWINP file.

    Parameters
    ----------
    if_ : int
        File type indicator (always 1).
    iv : int
        Time-dependence flag (1 = no, 2 = yes).
    ni : int
        Number of particle types.
    nr : int
        Mesh format (10 = rectangular, 16 = cylindrical/spherical).
    probid : str
        Problem identification string.
    nt : list[int]
        Number of time bins per particle type.
    ne : list[int]
        Number of energy bins per particle type.
    nfx, nfy, nfz : int
        Total fine mesh counts along each axis.
    x0, y0, z0 : float
        Origin coordinates.
    ncx, ncy, ncz : int
        Number of coarse mesh segments along each axis.
    nwg : int
        Geometry type (1 = cartesian, 2 = cylindrical, 3 = spherical).
    ref_vec : np.ndarray or None
        First reference vector (x1, y1, z1) for nr=16 geometries.
    ref_dir : np.ndarray or None
        Second reference vector (x2, y2, z2) for nr=16 geometries.
    """

    if_: int = 1
    iv: int = 1
    ni: int = 1
    nr: int = 10
    probid: str = ""
    nt: list[int] = field(default_factory=list)
    ne: list[int] = field(default_factory=list)
    nfx: int = 0
    nfy: int = 0
    nfz: int = 0
    x0: float = 0.0
    y0: float = 0.0
    z0: float = 0.0
    ncx: int = 0
    ncy: int = 0
    ncz: int = 0
    nwg: int = 1
    ref_vec: Optional[np.ndarray] = field(default=None, repr=False)
    ref_dir: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def has_time_dependency(self) -> bool:
        return self.iv == 2

    @property
    def n_spatial(self) -> int:
        return self.nfx * self.nfy * self.nfz

    @property
    def mesh_type(self) -> str:
        return {1: "cartesian", 2: "cylindrical", 3: "spherical"}.get(
            self.nwg, "unknown"
        )
