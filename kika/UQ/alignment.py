"""Strict, format-neutral alignment of sensitivities and covariance blocks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np

from kika.cov.cross_section_covariance import CrossSectionCovariance
from kika.cov.multigroup.mg_legendre_covariance import MultigroupLegendreCovariance
from kika.sensitivities.profile import SensitivityProfile
from kika.sensitivities.sdf import SDFData


AliasPolicy = Literal["exact", "tsurfer"]
MissingPolicy = Literal["error", "drop"]


@dataclass(frozen=True, order=True)
class ParameterKey:
    """A nuclear-data parameter independent of its source file format."""

    kind: Literal["xs", "legendre"]
    zaid: int
    mt: int
    order: Optional[int] = None

    @classmethod
    def from_sensitivity(cls, zaid: int, mt: int) -> "ParameterKey":
        if mt >= 4000:
            return cls("legendre", int(zaid), 2, int(mt) - 4000)
        return cls("xs", int(zaid), int(mt), None)

    @property
    def label(self) -> str:
        suffix = f" P{self.order}" if self.kind == "legendre" else ""
        return f"ZAID={self.zaid} MT={self.mt}{suffix}"


@dataclass(frozen=True)
class ParameterIndex:
    """Meaning of one element in an aligned flat vector."""

    key: ParameterKey
    group: int
    energy_low: float
    energy_high: float
    energy_unit: str = "MeV"


@dataclass(frozen=True)
class AliasMapping:
    profile: int
    source: ParameterKey
    target: ParameterKey
    reason: str


@dataclass
class AlignmentReport:
    """Complete record of non-exact decisions made while aligning data."""

    aliases: List[AliasMapping] = field(default_factory=list)
    zeros_inserted: Dict[int, List[ParameterKey]] = field(default_factory=dict)
    zero_covariance_blocks: List[Tuple[ParameterKey, ParameterKey]] = field(default_factory=list)
    missing_covariance: List[ParameterKey] = field(default_factory=list)
    dropped: List[ParameterKey] = field(default_factory=list)
    policy_exclusions: Dict[int, List[ParameterKey]] = field(default_factory=dict)
    assumptions: List[str] = field(default_factory=list)
    parameter_coverage: List[float] = field(default_factory=list)
    sensitivity_coverage: List[float] = field(default_factory=list)


class AlignmentError(ValueError):
    """Base error for a sensitivity/covariance alignment failure."""

    def __init__(self, message: str, report: Optional[AlignmentReport] = None):
        super().__init__(message)
        self.report = report


class MissingCovarianceError(AlignmentError):
    """Raised when sensitivity parameters lack covariance data."""


@dataclass
class AlignmentResult:
    """Aligned sensitivity vectors, absolute sigmas, relative covariance and index."""

    sensitivity_vectors: np.ndarray
    sensitivity_uncertainties: np.ndarray
    covariance: np.ndarray
    index: Tuple[ParameterIndex, ...]
    parameter_keys: Tuple[ParameterKey, ...]
    reaction_spans: Dict[int, Tuple[int, int]]
    profiles: Tuple[SensitivityProfile, ...]
    report: AlignmentReport


@dataclass
class PreparedCovariance:
    """Normalized covariance blocks reusable for profiles on one exact grid."""

    energy_grid_mev: np.ndarray
    blocks: Dict[Tuple[ParameterKey, ParameterKey], np.ndarray]
    assumptions: Tuple[str, ...] = ()


CovarianceInput = Union[CrossSectionCovariance, Sequence[CrossSectionCovariance]]
LegendreCovarianceInput = Union[
    MultigroupLegendreCovariance, Sequence[MultigroupLegendreCovariance]
]


def _as_list(value):
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _as_profile(value: Union[SensitivityProfile, SDFData]) -> SensitivityProfile:
    if isinstance(value, SensitivityProfile):
        return value
    if isinstance(value, SDFData):
        return value.to_sensitivity_profile()
    raise TypeError("profiles must contain SensitivityProfile or SDFData objects")


def _grid_to_mev(grid: Iterable[float], unit: str) -> np.ndarray:
    values = np.asarray(grid, dtype=float)
    if unit == "MeV":
        return values
    if unit == "eV":
        return values * 1.0e-6
    raise AlignmentError(f"unsupported energy unit {unit!r}; expected 'eV' or 'MeV'")


def _require_grid_match(
    expected: np.ndarray,
    actual: Iterable[float],
    unit: str,
    *,
    rtol: float,
    source: str,
) -> None:
    candidate = _grid_to_mev(actual, unit)
    if candidate.ndim != 1 or np.any(np.diff(candidate) <= 0.0):
        raise AlignmentError(f"{source} energy grid must be one-dimensional and strictly increasing")
    if candidate.shape != expected.shape or not np.allclose(
        candidate, expected, rtol=rtol, atol=0.0
    ):
        raise AlignmentError(
            f"{source} energy grid does not exactly match the sensitivity grid after unit "
            "conversion. Partial alignment is not performed; explicitly condense the "
            "sensitivity profile or covariance grid first (roadmap B4)."
        )


def _store_block(
    blocks: Dict[Tuple[ParameterKey, ParameterKey], np.ndarray],
    row: ParameterKey,
    col: ParameterKey,
    matrix: np.ndarray,
    source: str,
) -> None:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise AlignmentError(f"non-square covariance block {row.label} x {col.label} in {source}")
    if not np.all(np.isfinite(matrix)):
        raise AlignmentError(f"non-finite covariance block {row.label} x {col.label} in {source}")

    for key, value in (((row, col), matrix), ((col, row), matrix.T)):
        previous = blocks.get(key)
        if previous is not None and not np.allclose(previous, value, rtol=1e-12, atol=1e-15):
            raise AlignmentError(
                f"incompatible duplicate covariance block {key[0].label} x {key[1].label}"
            )
        blocks[key] = value.copy()


def _relative_xs_blocks(
    covariance: Sequence[CrossSectionCovariance],
    expected_grid: np.ndarray,
    report: AlignmentReport,
    energy_rtol: float,
) -> Dict[Tuple[ParameterKey, ParameterKey], np.ndarray]:
    blocks: Dict[Tuple[ParameterKey, ParameterKey], np.ndarray] = {}
    for source_index, cov in enumerate(covariance):
        source = f"cross-section covariance[{source_index}]"
        if cov.energy_unit != "MeV":
            report.assumptions.append(
                f"{source} energy grid converted from {cov.energy_unit} to MeV"
            )
        if cov.energy_grid is None:
            raise AlignmentError(f"{source} has no shared multigroup energy grid")
        _require_grid_match(
            expected_grid, cov.energy_grid, cov.energy_unit, rtol=energy_rtol, source=source
        )
        if cov.is_relative and len(cov.is_relative) != len(cov.matrices):
            raise AlignmentError(f"{source} has an incomplete is_relative block index")
        unmarked = not cov.is_relative
        if unmarked:
            note = f"{source} has no is_relative flags; blocks were treated as relative"
            if note not in report.assumptions:
                report.assumptions.append(note)

        for block_index, (zaid_r, mt_r, zaid_c, mt_c, raw) in enumerate(
            zip(
                cov.isotope_rows,
                cov.reaction_rows,
                cov.isotope_cols,
                cov.reaction_cols,
                cov.matrices,
            )
        ):
            row = ParameterKey("xs", int(zaid_r), int(mt_r))
            col = ParameterKey("xs", int(zaid_c), int(mt_c))
            matrix = np.asarray(raw, dtype=float)
            if matrix.shape != (len(expected_grid) - 1, len(expected_grid) - 1):
                raise AlignmentError(
                    f"{source} block {block_index} shape {matrix.shape} does not match the energy grid"
                )
            is_relative = True if unmarked else bool(cov.is_relative[block_index])
            if not is_relative:
                xs_row = cov.cross_sections.get((row.zaid, row.mt))
                xs_col = cov.cross_sections.get((col.zaid, col.mt))
                if xs_row is None or xs_col is None:
                    raise AlignmentError(
                        f"absolute covariance block {row.label} x {col.label} requires "
                        "row and column cross sections for conversion to relative covariance"
                    )
                xs_row = np.asarray(xs_row, dtype=float)
                xs_col = np.asarray(xs_col, dtype=float)
                if xs_row.shape != (len(expected_grid) - 1,) or xs_col.shape != xs_row.shape:
                    raise AlignmentError("cross-section vectors do not match the covariance grid")
                scale = np.outer(xs_row, xs_col)
                if np.any(scale == 0.0):
                    raise AlignmentError(
                        f"absolute covariance block {row.label} x {col.label} cannot be "
                        "converted because a required cross section is zero"
                    )
                matrix = matrix / scale
                report.assumptions.append(
                    f"{source} block {row.label} x {col.label} converted from absolute "
                    "to relative covariance using its row and column cross sections"
                )
            _store_block(blocks, row, col, matrix, source)
    return blocks


def _relative_legendre_blocks(
    covariance: Sequence[MultigroupLegendreCovariance],
    expected_grid: np.ndarray,
    report: AlignmentReport,
    energy_rtol: float,
) -> Dict[Tuple[ParameterKey, ParameterKey], np.ndarray]:
    blocks: Dict[Tuple[ParameterKey, ParameterKey], np.ndarray] = {}
    for source_index, cov in enumerate(covariance):
        source = f"Legendre covariance[{source_index}]"
        if cov.energy_unit != "MeV":
            report.assumptions.append(
                f"{source} energy grid converted from {cov.energy_unit} to MeV"
            )
        _require_grid_match(
            expected_grid, cov.energy_grid, cov.energy_unit, rtol=energy_rtol, source=source
        )
        lengths = {
            len(cov.isotope_rows), len(cov.reaction_rows), len(cov.l_rows),
            len(cov.isotope_cols), len(cov.reaction_cols), len(cov.l_cols),
            len(cov.relative_matrices),
        }
        if len(lengths) != 1:
            raise AlignmentError(f"{source} has inconsistent block metadata lengths")
        for values in zip(
            cov.isotope_rows, cov.reaction_rows, cov.l_rows,
            cov.isotope_cols, cov.reaction_cols, cov.l_cols,
            cov.relative_matrices,
        ):
            zaid_r, mt_r, order_r, zaid_c, mt_c, order_c, matrix = values
            row = ParameterKey("legendre", int(zaid_r), int(mt_r), int(order_r))
            col = ParameterKey("legendre", int(zaid_c), int(mt_c), int(order_c))
            if np.asarray(matrix).shape != (len(expected_grid) - 1, len(expected_grid) - 1):
                raise AlignmentError(f"{source} block shape does not match the energy grid")
            _store_block(blocks, row, col, matrix, source)
    return blocks


_ZAID_ALIASES = {4309: 4009, 4509: 4009, 1801: 1001, 1901: 1001, 6312: 6000}


def _alias_candidates(key: ParameterKey) -> List[Tuple[ParameterKey, str]]:
    candidates: List[Tuple[ParameterKey, str]] = []
    mapped_zaid = _ZAID_ALIASES.get(key.zaid)
    if mapped_zaid is not None:
        candidates.append((ParameterKey(key.kind, mapped_zaid, key.mt, key.order), "bound/natural ZAID"))
    if key.kind == "xs" and key.mt in (101, 102):
        other = 102 if key.mt == 101 else 101
        candidates.append((ParameterKey("xs", key.zaid, other), "MT 101/102 equivalence"))
        if mapped_zaid is not None:
            candidates.append((ParameterKey("xs", mapped_zaid, other), "ZAID and MT 101/102 equivalence"))
    if key.kind == "xs" and key.mt == -2:
        candidates.append((ParameterKey("xs", key.zaid, 101), "MCNP MT -2 to capture"))
        if mapped_zaid is not None:
            candidates.append((ParameterKey("xs", mapped_zaid, 101), "MCNP MT -2 and ZAID alias"))
    return candidates


def _resolve_key(
    source: ParameterKey,
    available: set,
    alias_policy: AliasPolicy,
) -> Tuple[Optional[ParameterKey], Optional[str]]:
    if source in available:
        return source, None
    if source.mt < 0 and source.mt != -2:
        raise AlignmentError(f"unsupported negative reaction identifier: {source.label}")
    if alias_policy == "tsurfer":
        matches = [(candidate, reason) for candidate, reason in _alias_candidates(source) if candidate in available]
        unique = {candidate for candidate, _ in matches}
        if len(unique) > 1:
            raise AlignmentError(
                f"ambiguous TSURFER alias for {source.label}: "
                + ", ".join(candidate.label for candidate in sorted(unique))
            )
        if matches:
            return matches[0]
    return None, None


def prepare_covariance(
    profile: Union[SensitivityProfile, SDFData],
    covariance: Optional[CovarianceInput] = None,
    legendre_covariance: Optional[LegendreCovarianceInput] = None,
    *,
    energy_rtol: float = 1.0e-8,
) -> PreparedCovariance:
    """Normalize covariance blocks once for repeated calculations on one grid."""
    neutral = _as_profile(profile)
    expected_grid = neutral.grid_in("MeV")
    report = AlignmentReport()
    blocks = _relative_xs_blocks(_as_list(covariance), expected_grid, report, energy_rtol)
    legendre_blocks = _relative_legendre_blocks(
        _as_list(legendre_covariance), expected_grid, report, energy_rtol
    )
    for key, matrix in legendre_blocks.items():
        if key in blocks:
            raise AlignmentError(f"duplicate covariance block across covariance types: {key}")
        blocks[key] = matrix
    if not blocks:
        raise AlignmentError("at least one covariance block is required")
    return PreparedCovariance(expected_grid.copy(), blocks, tuple(report.assumptions))

def align_sensitivity_covariance(
    profiles: Sequence[Union[SensitivityProfile, SDFData]],
    covariance: Optional[CovarianceInput] = None,
    legendre_covariance: Optional[LegendreCovarianceInput] = None,
    *,
    alias_policy: AliasPolicy = "exact",
    missing: MissingPolicy = "error",
    energy_rtol: float = 1.0e-8,
    prepared_covariance: Optional[PreparedCovariance] = None,
) -> AlignmentResult:
    """Align one or more sensitivity profiles with relative covariance blocks.

    Profile keys are combined by union. A missing key in one profile is an
    explicit zero sensitivity; a key missing from covariance is an error unless
    ``missing="drop"`` is requested.
    """
    if alias_policy not in ("exact", "tsurfer"):
        raise ValueError("alias_policy must be 'exact' or 'tsurfer'")
    if missing not in ("error", "drop"):
        raise ValueError("missing must be 'error' or 'drop'")
    if not profiles:
        raise ValueError("at least one sensitivity profile is required")

    neutral = tuple(_as_profile(profile) for profile in profiles)
    expected_grid = neutral[0].grid_in("MeV")
    profile_energy_conversions = [
        f"sensitivity profile[{index}] energy grid converted from {profile.energy_unit} to MeV"
        for index, profile in enumerate(neutral)
        if profile.energy_unit != "MeV"
    ]
    for index, profile in enumerate(neutral[1:], 1):
        _require_grid_match(
            expected_grid,
            profile.energy_grid,
            profile.energy_unit,
            rtol=energy_rtol,
            source=f"sensitivity profile[{index}]",
        )

    report = AlignmentReport()
    report.assumptions.extend(profile_energy_conversions)
    if prepared_covariance is not None:
        if covariance is not None or legendre_covariance is not None:
            raise ValueError("pass covariance inputs or prepared_covariance, not both")
        _require_grid_match(
            expected_grid,
            prepared_covariance.energy_grid_mev,
            "MeV",
            rtol=energy_rtol,
            source="prepared covariance",
        )
        blocks = prepared_covariance.blocks
        report.assumptions.extend(prepared_covariance.assumptions)
    else:
        prepared_covariance = prepare_covariance(
            neutral[0], covariance, legendre_covariance, energy_rtol=energy_rtol
        )
        blocks = prepared_covariance.blocks
        report.assumptions.extend(prepared_covariance.assumptions)

    available = {row for row, col in blocks if row == col}
    resolved_maps: List[Dict[ParameterKey, ParameterKey]] = []
    missing_keys = set()
    source_maps: List[Dict[ParameterKey, object]] = []
    for profile_index, profile in enumerate(neutral):
        reaction_map = {
            ParameterKey.from_sensitivity(reaction.zaid, reaction.mt): reaction
            for reaction in profile.reactions
        }
        source_maps.append(reaction_map)
        resolved: Dict[ParameterKey, ParameterKey] = {}
        targets = set()
        for source in sorted(reaction_map):
            target, reason = _resolve_key(source, available, alias_policy)
            if target is None:
                missing_keys.add(source)
                continue
            if target in targets:
                raise AlignmentError(
                    f"alias collision in sensitivity profile[{profile_index}]: multiple "
                    f"source parameters map to {target.label}"
                )
            targets.add(target)
            resolved[source] = target
            if reason is not None:
                report.aliases.append(AliasMapping(profile_index, source, target, reason))
        resolved_maps.append(resolved)

    report.missing_covariance = sorted(missing_keys)
    if missing_keys and missing == "error":
        labels = ", ".join(key.label for key in sorted(missing_keys))
        raise MissingCovarianceError(
            "No covariance data are available for: " + labels + ". "
            "To continue explicitly with the covered subset, repeat the call with "
            'missing="drop". The returned AlignmentReport will record the loss.',
            report,
        )
    report.dropped = sorted(missing_keys)

    selected_keys = sorted({target for resolved in resolved_maps for target in resolved.values()})
    if not selected_keys:
        raise AlignmentError("no sensitivity parameters remain after covariance alignment", report)

    n_profiles = len(neutral)
    n_groups = len(expected_grid) - 1
    n_parameters = len(selected_keys) * n_groups
    vectors = np.zeros((n_profiles, n_parameters))
    uncertainties = np.zeros_like(vectors)
    spans: Dict[int, Tuple[int, int]] = {}

    for key_index, key in enumerate(selected_keys):
        spans[key_index] = (key_index * n_groups, n_groups)
    for profile_index, (profile, reaction_map, resolved) in enumerate(
        zip(neutral, source_maps, resolved_maps)
    ):
        reverse = {target: source for source, target in resolved.items()}
        zeros = []
        for key_index, key in enumerate(selected_keys):
            start = key_index * n_groups
            source = reverse.get(key)
            if source is None:
                zeros.append(key)
                continue
            reaction = reaction_map[source]
            vectors[profile_index, start:start + n_groups] = reaction.sensitivity
            if reaction.uncertainty is None:
                uncertainties[profile_index, start:start + n_groups] = np.nan
            else:
                uncertainties[profile_index, start:start + n_groups] = reaction.uncertainty
        if zeros:
            report.zeros_inserted[profile_index] = zeros

        original_count = len(reaction_map)
        kept_sources = len(resolved)
        report.parameter_coverage.append(
            kept_sources / original_count if original_count else 1.0
        )
        total_abs = sum(float(np.sum(np.abs(r.sensitivity))) for r in reaction_map.values())
        kept_abs = sum(
            float(np.sum(np.abs(reaction_map[source].sensitivity))) for source in resolved
        )
        report.sensitivity_coverage.append(kept_abs / total_abs if total_abs else 1.0)

    full_covariance = np.zeros((n_parameters, n_parameters))
    for row_index, row in enumerate(selected_keys):
        for col_index, col in enumerate(selected_keys):
            block = blocks.get((row, col))
            if block is None:
                if row_index <= col_index:
                    report.zero_covariance_blocks.append((row, col))
                continue
            r0 = row_index * n_groups
            c0 = col_index * n_groups
            full_covariance[r0:r0 + n_groups, c0:c0 + n_groups] = block
    if not np.allclose(full_covariance, full_covariance.T, rtol=1e-12, atol=1e-15):
        raise AlignmentError("assembled covariance matrix is not symmetric", report)
    full_covariance = 0.5 * (full_covariance + full_covariance.T)

    flat_index = tuple(
        ParameterIndex(
            key=key,
            group=group,
            energy_low=float(expected_grid[group]),
            energy_high=float(expected_grid[group + 1]),
        )
        for key in selected_keys
        for group in range(n_groups)
    )
    return AlignmentResult(
        sensitivity_vectors=vectors,
        sensitivity_uncertainties=uncertainties,
        covariance=full_covariance,
        index=flat_index,
        parameter_keys=tuple(selected_keys),
        reaction_spans=spans,
        profiles=neutral,
        report=report,
    )
