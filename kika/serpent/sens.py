# ──────────────────────────────────────────────────────────────────────────────
# File: serpent_sens/models.py
# Core data structures for SERPENT sensitivity outputs (no plotting)
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
import numpy as np
import xarray as xr
from kika.energy_grids.utils import _identify_energy_grid
from kika._utils import zaid_to_symbol



class PertCategory(str, Enum):
    MT_XS = "mt_xs"
    LEGENDRE_MOMENT = "legendre_moment"
    NUBAR = "nubar"
    CHI = "chi"
    OTHER = "other"


@dataclass(frozen=True)
class Material:
    index: int
    name: str


@dataclass(frozen=True)
class Nuclide:
    index: int
    zai: int

    @property
    def label(self) -> str:
        # Return just the nuclide number without "ZAI" prefix
        return str(self.zai)


@dataclass(frozen=True)
class Perturbation:
    index: int
    raw_label: str
    category: PertCategory
    mt: Optional[int] = None
    channel: Optional[str] = None
    moment: Optional[int] = None
    short_label: Optional[str] = None


@dataclass(frozen=True)
class Response:
    base_name: str               # e.g. "sens_ratio"
    bin_index: Optional[int]     # None or 0..N-1
    full_name: str               # e.g. "sens_ratio_BIN_0"
    label: Optional[str] = None


class SensitivitySet:
    """Holds arrays for one *base* response across all response bins.

    Shapes:
      energy_dependent: (n_bins, M, Z, P, E, 2)
      integrated:       (n_bins, M, Z, P, 2)
    where last-axis 2 = [value, relative_error]
    """

    def __init__(
        self,
        responses: List[Response],
        energy_dependent: np.ndarray,
        integrated: np.ndarray,
    ) -> None:
        self.responses = responses
        self.energy_dependent = np.asarray(energy_dependent, dtype=float)
        self.integrated = np.asarray(integrated, dtype=float)
        self._validate_shapes()

    # ------------------------------
    # Validation
    # ------------------------------
    def _validate_shapes(self) -> None:
        ed = self.energy_dependent
        it = self.integrated
        if ed.ndim != 6 or it.ndim != 5:
            raise ValueError(
                "Unexpected shapes: ED should be 6D (bins,M,Z,P,E,2), INT 5D (bins,M,Z,P,2)."
            )
        if ed.shape[-1] != 2 or it.shape[-1] != 2:
            raise ValueError("Last axis must be size 2 (value, rel_err).")
        if ed.shape[0] != it.shape[0]:
            raise ValueError("ED and INT must have same n_bins along axis 0.")
        if list(ed.shape[1:4]) != list(it.shape[1:4]):
            raise ValueError("ED and INT must align on (M,Z,P).")
        if len(self.responses) != ed.shape[0]:
            raise ValueError("Responses list length must equal n_bins.")

    # ------------------------------
    # Accessors by index (index-only; user-friendly wrappers live on SensitivityFile)
    # ------------------------------
    def get_ed_by_index(
        self,
        bin_index: int,
        mat: Optional[Union[int, Sequence[int]]] = None,
        zai: Optional[Union[int, Sequence[int]]] = None,
        pert: Optional[Union[int, Sequence[int]]] = None,
        energy: Optional[Union[int, Sequence[int]]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        ed = self.energy_dependent
        # Build slices for (bins, M, Z, P, E, stat)
        sl = [bin_index, slice(None), slice(None), slice(None), slice(None), slice(None)]
        if mat is not None:
            sl[1] = mat if isinstance(mat, (list, tuple, np.ndarray)) else int(mat)
        if zai is not None:
            sl[2] = zai if isinstance(zai, (list, tuple, np.ndarray)) else int(zai)
        if pert is not None:
            sl[3] = pert if isinstance(pert, (list, tuple, np.ndarray)) else int(pert)
        if energy is not None:
            sl[4] = energy if isinstance(energy, (list, tuple, np.ndarray)) else int(energy)
        arr = ed[tuple(sl)]
        # Separate value and relative error along last axis
        return arr[..., 0], arr[..., 1]

    def get_int_by_index(
        self,
        bin_index: int,
        mat: Optional[Union[int, Sequence[int]]] = None,
        zai: Optional[Union[int, Sequence[int]]] = None,
        pert: Optional[Union[int, Sequence[int]]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        it = self.integrated
        sl = [bin_index, slice(None), slice(None), slice(None), slice(None)]
        if mat is not None:
            sl[1] = mat if isinstance(mat, (list, tuple, np.ndarray)) else int(mat)
        if zai is not None:
            sl[2] = zai if isinstance(zai, (list, tuple, np.ndarray)) else int(zai)
        if pert is not None:
            sl[3] = pert if isinstance(pert, (list, tuple, np.ndarray)) else int(pert)
        arr = it[tuple(sl)]
        return arr[..., 0], arr[..., 1]


class SensitivityFile:
    """Top-level container for a SERPENT .sens file (adjoint perturbation sensitivities).

    Holds descriptors (materials, nuclides, perturbations), energy grid, and a
    mapping from base response names to SensitivitySet.
    """

    # Parameters
    n_materials: int
    n_nuclides: int
    n_perturbations: int
    n_energy_bins: int
    n_mu_bins: Optional[int]

    # Descriptors
    materials: List[Material]
    nuclides: List[Nuclide]
    perturbations_raw: List[str]
    perturbations: List[Perturbation]

    # Energy grid
    energy_grid: np.ndarray
    lethargy_widths: np.ndarray

    # Data
    data: Dict[str, SensitivitySet]  # base response name -> set

    # Misc
    meta: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        n_materials: int,
        n_nuclides: int,
        n_perturbations: int,
        n_energy_bins: int,
        n_mu_bins: Optional[int],
        materials: List[str],
        nuclides: List[int],
        perturbations_raw: List[str],
        perturbations: List[Perturbation],
        energy_grid: np.ndarray,
        lethargy_widths: np.ndarray,
        data: Dict[str, SensitivitySet],
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.n_materials = int(n_materials)
        self.n_nuclides = int(n_nuclides)
        self.n_perturbations = int(n_perturbations)
        self.n_energy_bins = int(n_energy_bins)
        self.n_mu_bins = int(n_mu_bins) if n_mu_bins is not None else None

        self.materials = [Material(i, m) for i, m in enumerate(materials)]
        # Remove trailing zero from nuclide ZAI values (e.g., 260560 -> 26056)
        self.nuclides = [Nuclide(i, int(z) // 10 if int(z) % 10 == 0 else int(z)) for i, z in enumerate(nuclides)]
        self.perturbations_raw = list(perturbations_raw)
        self.perturbations = perturbations

        self.energy_grid = np.asarray(energy_grid, dtype=float)
        self.lethargy_widths = np.asarray(lethargy_widths, dtype=float)
        self.data = data
        self.meta = dict(meta or {})

        self.validate()

    # ------------------------------
    # Convenience
    # ------------------------------
    @property
    def energy_bin_count(self) -> int:
        return self.n_energy_bins

    @property
    def responses(self) -> List[str]:
        out: List[str] = []
        for base, sset in self.data.items():
            for r in sset.responses:
                out.append(r.full_name)
        return out

    @property
    def reactions(self) -> List[int]:
        """List of unique MT reaction numbers available in the perturbations."""
        mt_numbers = set()
        for p in self.perturbations:
            if p.mt is not None:
                mt_numbers.add(p.mt)
        return sorted(list(mt_numbers))

    # Filtering helpers for perturbations
    def by_mt(self, mt: int) -> List[Perturbation]:
        return [p for p in self.perturbations if p.mt == mt]

    def by_legendre(
        self, channel: Optional[str] = None, moment: Optional[int] = None
    ) -> List[Perturbation]:
        def ok(p: Perturbation) -> bool:
            if p.category != PertCategory.LEGENDRE_MOMENT:
                return False
            if channel is not None and (p.channel or "").lower() != channel.lower():
                return False
            if moment is not None and p.moment != moment:
                return False
            return True

        return [p for p in self.perturbations if ok(p)]

    # Index resolvers
    def _material_index(self, m: Union[int, str]) -> int:
        if isinstance(m, int):
            return m
        # resolve by name (case-insensitive)
        key = m.strip().lower()
        for mat in self.materials:
            if mat.name.lower() == key:
                return mat.index
        raise KeyError(f"Unknown material: {m}")

    def _nuclide_index(self, n: Union[int, str]) -> int:
        if isinstance(n, int):
            # Could be ZAI or index; prefer index if within range
            if 0 <= n < self.n_nuclides:
                return n
            # else, try match by ZAI value (handle both original and cleaned formats)
            # Original format: 260560, cleaned format: 26056
            search_zai = n // 10 if n % 10 == 0 else n  # Remove trailing zero if present
            for nu in self.nuclides:
                if nu.zai == search_zai or nu.zai == n:  # Match either format
                    return nu.index
            raise KeyError(f"Unknown nuclide (index or ZAI): {n}")
        # String label like "ZAI260560" or just "260560"
        s = n.strip().lower()
        if s.startswith("zai") and s[3:].isdigit():
            target = int(s[3:])
        elif s.isdigit():
            target = int(s)
        else:
            raise KeyError(f"Unknown nuclide format: {n}")
            
        # Handle both original and cleaned formats
        search_zai = target // 10 if target % 10 == 0 else target
        for nu in self.nuclides:
            if nu.zai == search_zai or nu.zai == target:
                return nu.index
        raise KeyError(f"Unknown nuclide: {n}")

    def _pert_indices_from_selector(
        self,
        selector: Optional[Union[int, Sequence[int], Mapping[str, Any]]],
    ) -> Union[int, List[int], slice]:
        if selector is None:
            return slice(None)
        if isinstance(selector, (list, tuple, np.ndarray, int)):
            return selector  # pass-through
        # dict-based selector, e.g., {"mt": 2} or {"category": "LEGENDRE_MOMENT", "channel": "ela"}
        def ok(p: Perturbation) -> bool:
            for k, v in selector.items():
                pv = getattr(p, k, None)
                if isinstance(pv, str) and isinstance(v, str):
                    if pv.lower() != v.lower():
                        return False
                else:
                    if pv != v:
                        return False
            return True

        return [p.index for p in self.perturbations if ok(p)]

    def _collect_perturbations(
        self,
        mt: Optional[Union[int, Sequence[int]]] = None,
        leg: Optional[Union[int, Sequence[int]]] = None,
        leg_channel: Optional[str] = None,
    ) -> List[int]:
        """
        Build perturbation index list from MT numbers and/or Legendre orders.
        - mt: int or list[int] of MT values (e.g., 1, 2, 102, ...). 
          Note: MT 500X are automatically recognized as Legendre moments.
        - leg: int or list[int] of Legendre orders (L), e.g., 1, 2, 3
          These are converted to MT 500X format internally.
        - leg_channel: deprecated (Legendre moments are elastic only), kept for backward compatibility
        If both mt and leg are None, returns ALL perturbations.
        """
        def _to_set(x):
            if x is None:
                return None
            if isinstance(x, (list, tuple, np.ndarray)):
                return set(int(v) for v in x)
            return {int(x)}

        mt_set = _to_set(mt)
        leg_set = _to_set(leg)

        # Convert Legendre orders to MT 500X format and merge with mt_set
        if leg_set is not None:
            leg_as_mt = {5000 + L for L in leg_set}  # L=1 -> MT=5001, L=2 -> MT=5002, etc.
            if mt_set is not None:
                mt_set = mt_set | leg_as_mt
            else:
                mt_set = leg_as_mt

        if mt_set is None:
            return list(range(self.n_perturbations))

        # Find perturbations matching the MT numbers (including 500X for Legendre)
        idxs: List[int] = []
        for p in self.perturbations:
            if p.mt is not None and p.mt in mt_set:
                idxs.append(p.index)
        
        return idxs

    # Public data access (user-friendly)
    def _locate_set_and_bin(self, response_full: str) -> Tuple[SensitivitySet, int]:
        # Find which base it belongs to
        for base, sset in self.data.items():
            for i, r in enumerate(sset.responses):
                if r.full_name == response_full:
                    return sset, i
        raise KeyError(f"Unknown response: {response_full}")

    def get_energy_dependent(
        self,
        response_full: str,
        mat: Optional[Union[int, str, Sequence[int]]] = None,
        zai: Optional[Union[int, str, Sequence[int]]] = None,
        mt: Optional[Union[int, Sequence[int]]] = None,
        leg: Optional[Union[int, Sequence[int]]] = None,
        leg_channel: Optional[str] = None,
        energy: Optional[Union[int, Sequence[int]]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get energy-dependent sensitivity data.
        
        Parameters
        ----------
        response_full : str
            Full response name (e.g., 'sens_ratio_BIN_0')
        mat : int, str, or sequence, optional
            Material selector(s)
        zai : int, str, or sequence, optional
            Nuclide ZAI selector(s)
        mt : int or sequence, optional
            MT number(s) to select (e.g., 1, 2, 102, ...)
        leg : int or sequence, optional
            Legendre order(s) to select (e.g., 1, 2, 3)
        leg_channel : str, optional
            Legendre channel (deprecated, kept for compatibility)
        energy : int or sequence, optional
            Energy group selector(s)
            
        Returns
        -------
        values : ndarray
            Sensitivity values
        rel_errors : ndarray
            Relative errors
        """
        sset, b = self._locate_set_and_bin(response_full)
        m_idx = (
            [self._material_index(x) for x in mat] if isinstance(mat, (list, tuple)) else
            (self._material_index(mat) if mat is not None and not isinstance(mat, (slice, np.ndarray)) else mat)
        )
        z_idx = (
            [self._nuclide_index(x) for x in zai] if isinstance(zai, (list, tuple)) else
            (self._nuclide_index(zai) if zai is not None and not isinstance(zai, (slice, np.ndarray)) else zai)
        )
        
        # Use clean parameter structure
        p_idx = self._collect_perturbations(mt=mt, leg=leg, leg_channel=leg_channel)
            
        return sset.get_ed_by_index(b, m_idx, z_idx, p_idx, energy)

    def get_integrated(
        self,
        response_full: str,
        mat: Optional[Union[int, str, Sequence[int]]] = None,
        zai: Optional[Union[int, str, Sequence[int]]] = None,
        mt: Optional[Union[int, Sequence[int]]] = None,
        leg: Optional[Union[int, Sequence[int]]] = None,
        leg_channel: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get integrated sensitivity data.
        
        Parameters
        ----------
        response_full : str
            Full response name (e.g., 'sens_ratio_BIN_0')
        mat : int, str, or sequence, optional
            Material selector(s)
        zai : int, str, or sequence, optional
            Nuclide ZAI selector(s)
        mt : int or sequence, optional
            MT number(s) to select (e.g., 1, 2, 102, ...)
        leg : int or sequence, optional
            Legendre order(s) to select (e.g., 1, 2, 3)
        leg_channel : str, optional
            Legendre channel (deprecated, kept for compatibility)
            
        Returns
        -------
        values : ndarray
            Sensitivity values
        rel_errors : ndarray
            Relative errors
        """
        sset, b = self._locate_set_and_bin(response_full)
        m_idx = (
            [self._material_index(x) for x in mat] if isinstance(mat, (list, tuple)) else
            (self._material_index(mat) if mat is not None and not isinstance(mat, (slice, np.ndarray)) else mat)
        )
        z_idx = (
            [self._nuclide_index(x) for x in zai] if isinstance(zai, (list, tuple)) else
            (self._nuclide_index(zai) if zai is not None and not isinstance(zai, (slice, np.ndarray)) else zai)
        )
        
        # Use clean parameter structure
        p_idx = self._collect_perturbations(mt=mt, leg=leg, leg_channel=leg_channel)
            
        return sset.get_int_by_index(b, m_idx, z_idx, p_idx)

    # DataFrame view
    def to_dataframe(self, response_full: Optional[str] = None, include_edges: bool = True):
        try:
            import pandas as pd
        except Exception as e:
            raise RuntimeError(
                "pandas is required for to_dataframe(); please `pip install pandas`."
            ) from e

        records = []

        def _dump_one(resp_name: str, sset: SensitivitySet, bin_idx: int):
            ed = sset.energy_dependent[bin_idx]  # (M,Z,P,E,2)
            M, Z, P, E, _ = ed.shape
            for mi in range(M):
                mname = self.materials[mi].name
                for zi in range(Z):
                    zlab = self.nuclides[zi].label
                    for pi in range(P):
                        plab = self.perturbations[pi].short_label or self.perturbations[pi].raw_label
                        vals = ed[mi, zi, pi, :, 0]
                        rels = ed[mi, zi, pi, :, 1]
                        for ei in range(E):
                            rec = {
                                "response": resp_name,
                                "material": mname,
                                "nuclide": zlab,
                                "pert_index": pi,
                                "pert_label": plab,
                                "energy_group": ei,
                                "value": float(vals[ei]),
                                "rel_err": float(rels[ei]),
                            }
                            if include_edges:
                                rec["E_low"] = float(self.energy_grid[ei])
                                rec["E_high"] = float(self.energy_grid[ei + 1])
                            records.append(rec)

        if response_full is None:
            # Include ALL responses by default
            for base, sset in self.data.items():
                for bi, r in enumerate(sset.responses):
                    _dump_one(r.full_name, sset, bi)
        else:
            sset, b = self._locate_set_and_bin(response_full)
            _dump_one(response_full, sset, b)

        return pd.DataFrame.from_records(records)

    # xarray view
    def to_xarray(self, response_full: Optional[str] = None, include_edges: bool = True):
        # Build selection of responses
        selected: list[tuple[str, SensitivitySet, int]] = []
        if response_full is not None:
            sset, b = self._locate_set_and_bin(response_full)
            selected.append((response_full, sset, b))
        else:
            for _, sset in self.data.items():
                for bi, r in enumerate(sset.responses):
                    selected.append((r.full_name, sset, bi))

        R = len(selected)
        M, Z, P, E = self.n_materials, self.n_nuclides, self.n_perturbations, self.n_energy_bins

        ed_val = np.empty((R, M, Z, P, E), dtype=float)
        ed_rel = np.empty((R, M, Z, P, E), dtype=float)
        it_val = np.empty((R, M, Z, P), dtype=float)
        it_rel = np.empty((R, M, Z, P), dtype=float)
        rnames: list[str] = []

        for ri, (rname, sset, bi) in enumerate(selected):
            arr_ed = sset.energy_dependent[bi]  # (M,Z,P,E,2)
            arr_it = sset.integrated[bi]        # (M,Z,P,2)
            ed_val[ri] = arr_ed[..., 0]
            ed_rel[ri] = arr_ed[..., 1]
            it_val[ri] = arr_it[..., 0]
            it_rel[ri] = arr_it[..., 1]
            rnames.append(rname)

        coords = {
            "response": rnames,
            "material": [m.name for m in self.materials],
            "nuclide": [n.label for n in self.nuclides],
            "perturbation": [p.short_label or p.raw_label for p in self.perturbations],
            "energy": np.arange(E, dtype=int),
        }
        ds = xr.Dataset(
            data_vars={
                "ed_value": (("response", "material", "nuclide", "perturbation", "energy"), ed_val),
                "ed_rel_err": (("response", "material", "nuclide", "perturbation", "energy"), ed_rel),
                "int_value": (("response", "material", "nuclide", "perturbation"), it_val),
                "int_rel_err": (("response", "material", "nuclide", "perturbation"), it_rel),
            },
            coords=coords,
        )
        if include_edges:
            ds = ds.assign_coords(
                E_low=("energy", self.energy_grid[:-1]),
                E_high=("energy", self.energy_grid[1:]),
            )
        return ds

    def plot_energy_sensitivity(self, *args, **kwargs):
        from kika.serpent.plotting import plot_energy_sensitivity as _plot
        return _plot(self, *args, **kwargs)


    # Validation and summary
    def validate(self) -> None:
        # Basic sizes
        if len(self.materials) != self.n_materials:
            raise ValueError("materials length != n_materials")
        if len(self.nuclides) != self.n_nuclides:
            raise ValueError("nuclides length != n_nuclides")
        if len(self.perturbations) != self.n_perturbations:
            raise ValueError("perturbations length != n_perturbations")
        if self.energy_grid.shape[0] != self.n_energy_bins + 1:
            raise ValueError("energy_grid length must be n_energy_bins + 1")
        if self.lethargy_widths.shape[0] != self.n_energy_bins:
            raise ValueError("lethargy_widths length must be n_energy_bins")
        # Data shapes
        for base, sset in self.data.items():
            ed = sset.energy_dependent
            it = sset.integrated
            if ed.shape[1] != self.n_materials or it.shape[1] != self.n_materials:
                raise ValueError("M dimension mismatch in data arrays")
            if ed.shape[2] != self.n_nuclides or it.shape[2] != self.n_nuclides:
                raise ValueError("Z dimension mismatch in data arrays")
            if ed.shape[3] != self.n_perturbations or it.shape[3] != self.n_perturbations:
                raise ValueError("P dimension mismatch in data arrays")
            if ed.shape[4] != self.n_energy_bins:
                raise ValueError("E dimension mismatch in ED array")

    def summary(self) -> str:
        lines = []
        lines.append("SERPENT Sensitivity File Summary")
        lines.append("- Materials ({}): {}".format(
            self.n_materials, ", ".join(m.name for m in self.materials)))
        lines.append("- Nuclides ({}): {}".format(
            self.n_nuclides, ", ".join(n.label for n in self.nuclides)))
        lines.append("- Perturbations ({}): e.g., {}".format(
            self.n_perturbations,
            ", ".join((self.perturbations[i].short_label or self.perturbations[i].raw_label)
                      for i in range(min(5, self.n_perturbations))) + ("..." if self.n_perturbations>5 else "")
        ))
        resp_list = self.responses
        lines.append(f"- Responses ({len(resp_list)}): " + ", ".join(resp_list[:5]) + ("..." if len(resp_list)>5 else ""))
        lines.append(f"- Energy bins: {self.n_energy_bins}")
        if self.n_mu_bins is not None:
            lines.append(f"- Mu bins (undocumented): {self.n_mu_bins}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Returns a detailed formatted string representation of the sensitivity file.
        
        This method is called when the object is evaluated in interactive environments
        like Jupyter notebooks or the Python interpreter.
        
        :return: Formatted string representation of the sensitivity file
        :rtype: str
        """
        # Create a visually appealing header with a border
        header_width = 70
        header = "=" * header_width + "\n"
        header += f"{'SERPENT Sensitivity File':^{header_width}}\n"
        header += "=" * header_width + "\n\n"
        
        # Create aligned key-value pairs with consistent width
        label_width = 25  # Width for labels
        
        # Basic information section
        info_lines = []
        info_lines.append(f"{'Materials:':{label_width}} {self.n_materials}")
        info_lines.append(f"{'Nuclides:':{label_width}} {self.n_nuclides}")
        
        resp_list = self.responses

        info_lines.append(f"{'Perturbations:':{label_width}} {self.n_perturbations}")  
        
        # Enhanced energy bins line with grid identification
        try:
            # Handle potential numpy array issues by converting to list
            energy_list = list(self.energy_grid) if hasattr(self.energy_grid, '__iter__') else self.energy_grid
            if len(energy_list) >= 2:
                grid_name = _identify_energy_grid(energy_list)
                if grid_name:
                    energy_info = f"{self.n_energy_bins} ({grid_name.lower()})"
                else:
                    energy_info = str(self.n_energy_bins)
            else:
                energy_info = str(self.n_energy_bins)
        except:
            energy_info = str(self.n_energy_bins)
        info_lines.append(f"{'Energy bins:':{label_width}} {energy_info}")
        info_lines.append(f"{'Responses:':{label_width}} {len(resp_list)}")
        
        if self.n_mu_bins is not None:
            info_lines.append(f"{'Mu bins:':{label_width}} {self.n_mu_bins}")
        
        stats = "\n".join(info_lines)
        
        # Energy grid preview
        energy_preview = "\n\nEnergy grid:"
        edges = self.energy_grid
        if len(edges) > 6:
            energy_grid = "  " + ", ".join(f"{e:.6e}" for e in edges[:3])
            energy_grid += ", ... , " 
            energy_grid += ", ".join(f"{e:.6e}" for e in edges[-3:])
        else:
            energy_grid = "  " + ", ".join(f"{e:.6e}" for e in edges)
            
        energy_preview += "\n" + energy_grid
        
        # Materials preview
        materials_preview = "\n\nMaterials:\n"
        materials_list = [m.name for m in self.materials[:5]]
        if self.n_materials > 5:
            materials_list.append(f"... ({self.n_materials - 5} more)")
        materials_preview += "  " + ", ".join(materials_list)
        
        # Nuclides preview with atomic symbols
        nuclides_preview = "\n\nNuclides:\n"
        nuclides_list = []
        for n in self.nuclides[:8]:
            try:
                symbol = zaid_to_symbol(n.zai)
                formatted_symbol = f"{symbol} ({n.zai})"
                nuclides_list.append(formatted_symbol)
            except Exception:
                nuclides_list.append(str(n.zai))
        if self.n_nuclides > 8:
            nuclides_list.append(f"... ({self.n_nuclides - 8} more)")
        nuclides_preview += "  " + ", ".join(nuclides_list)
        
        # Perturbation Access mapping
        pert_summary = "\n\nPerturbation Access:\n"
        
        # Show perturbations with clear MT to index mapping
        for i, p in enumerate(self.perturbations[:15]):
            if p.mt is not None:
                pert_summary += f"  MT={p.mt:<3}      .perturbations[{i}]\n"
            else:
                # For non-MT perturbations, show the category/type
                if p.category == PertCategory.LEGENDRE_MOMENT:
                    label = f"L={p.moment}" if p.moment is not None else "Legendre"
                else:
                    label = p.short_label or p.raw_label[:10]
                pert_summary += f"  {label:<8}   .perturbations[{i}]\n"
        
        if self.n_perturbations > 15:
            pert_summary += f"  ... ({self.n_perturbations - 15} more perturbations)\n"
        
        # Responses summary
        responses_summary = "\n\nResponses (with access names):\n"
        for i, resp in enumerate(resp_list):
            responses_summary += f"  '{resp}'\n"
        
        # Usage examples in two-column format
        if resp_list:
            first_resp = resp_list[0]
            sample_mat = self.materials[0].name if self.materials else "material_0"
            sample_nuc = self.nuclides[0].zai if self.nuclides else 26056
            
            # Find a suitable MT number for example
            sample_mt = 2  # Default to elastic
            for p in self.perturbations:
                if p.mt is not None and p.mt < 100:
                    sample_mt = p.mt
                    break
            
            usage_examples = f"\n\nUsage Examples:\n"
            usage_examples += f"{'Action':<35} {'Method Call':<35}\n"
            usage_examples += f"{'-'*35} {'-'*35}\n"
            usage_examples += f"{'Get energy-dependent data':<35} .get_energy_dependent(...)\n"
            usage_examples += f"{'Get integrated data':<35} .get_integrated(...)\n"
            usage_examples += f"{'Convert to DataFrame':<35} .to_dataframe(...)\n"
            usage_examples += f"{'Convert to xarray Dataset':<35} .to_xarray(...)\n"
            usage_examples += f"{'Get compact summary':<35} .summary()\n"
        else:
            usage_examples = "\n\nNo responses available for usage examples."
        
        # Footer with available methods
        footer = "\n\nAvailable methods:\n"
        footer += "- .get_energy_dependent() - Get energy-dependent sensitivity data\n"
        footer += "- .get_integrated() - Get integrated sensitivity data\n"
        footer += "- .to_dataframe() - Convert to pandas DataFrame\n"
        footer += "- .to_xarray() - Convert to xarray Dataset\n"
        footer += "- .summary() - Get compact summary\n"
        
        return header + stats + energy_preview + materials_preview + nuclides_preview + pert_summary + responses_summary + usage_examples + footer