"""Parser for Serpent input files.

Extracts materials, detectors, energy grids, time grids, and sensitivity
configuration from a Serpent input file.  Material parsing is delegated to
the existing ``kika.materials.parse_serpent`` module.

Only core detector options are parsed (de, di, dv, dm, dc, dr).
Include statements are not resolved — single-file inputs only.
"""

from __future__ import annotations

import re
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from kika.serpent.input import (
    DetectorDefinition,
    EnergyGrid,
    SensitivityConfig,
    SensPerturbation,
    SensResponse,
    SerpentInput,
    TimeGrid,
)
from kika.materials.parse_serpent import read_serpent_materials


# ── Helpers ──────────────────────────────────────────────────────────────────

def _strip_comment(line: str) -> str:
    """Remove Serpent ``%`` comments."""
    idx = line.find("%")
    return line[:idx] if idx >= 0 else line


def _tokenize(lines: List[str]) -> List[str]:
    """Flatten lines into a single token stream, stripping comments."""
    tokens: List[str] = []
    for line in lines:
        cleaned = _strip_comment(line).strip()
        if cleaned:
            tokens.extend(cleaned.split())
    return tokens


# Card keywords that start a new top-level card
_CARD_KEYWORDS = frozenset({
    "mat", "surf", "cell", "det", "src", "set", "ene", "tme", "sens",
    "plot", "mesh", "pin", "lat", "nest", "pbed", "umsh", "trans",
    "utrans", "strans", "ftrans", "dtrans", "ifc", "div", "dep",
    "include", "therm", "thermstoch", "branch", "coef", "hisv",
    "mflow", "wwin", "wwgen", "sample", "datamesh", "dataifc",
    "casematrix", "phb", "resmat", "voro", "particle", "disp",
    "fun", "rep", "finbin", "finrod", "solid", "ria",
})


def _collect_card_lines(all_lines: List[str], start: int) -> Tuple[List[str], int]:
    """Collect lines belonging to a single card starting at *start*.

    Returns the card lines and the index of the first line after the card.
    """
    card_lines = [all_lines[start]]
    i = start + 1
    while i < len(all_lines):
        stripped = _strip_comment(all_lines[i]).strip()
        if not stripped:
            i += 1
            continue
        first_token = stripped.split()[0].lower()
        if first_token in _CARD_KEYWORDS:
            break
        card_lines.append(all_lines[i])
        i += 1
    return card_lines, i


# ── Detector parser ──────────────────────────────────────────────────────────

def _parse_detector(card_lines: List[str]) -> DetectorDefinition:
    """Parse a ``det`` card from its collected lines."""
    tokens = _tokenize(card_lines)
    # tokens[0] == "det", tokens[1] == name
    name = tokens[1]
    det = DetectorDefinition(name=name)

    i = 2
    while i < len(tokens):
        tok = tokens[i].lower()

        # Particle type
        if tok in ("n", "neutron"):
            det.particle = "n"
            i += 1
            continue
        if tok in ("p", "photon", "gamma", "g"):
            det.particle = "p"
            i += 1
            continue

        # Energy grid reference
        if tok == "de" and i + 1 < len(tokens):
            det.energy_grid = tokens[i + 1]
            i += 2
            continue

        # Time grid reference
        if tok == "di" and i + 1 < len(tokens):
            det.time_grid = tokens[i + 1]
            i += 2
            continue

        # Volume
        if tok == "dv" and i + 1 < len(tokens):
            try:
                det.volume = float(tokens[i + 1])
            except ValueError:
                pass
            i += 2
            continue

        # Material filter
        if tok == "dm" and i + 1 < len(tokens):
            det.materials.append(tokens[i + 1])
            i += 2
            continue

        # Cell filter
        if tok == "dc" and i + 1 < len(tokens):
            try:
                det.cells.append(int(tokens[i + 1]))
            except ValueError:
                pass
            i += 2
            continue

        # Response function (dr: mt_number material_name)
        if tok == "dr" and i + 2 < len(tokens):
            try:
                mt = int(tokens[i + 1])
                det.response_mt.append(mt)
            except ValueError:
                pass
            i += 3  # dr <mt> <material>
            continue

        # Skip unknown options — store them for future use
        i += 1

    return det


# ── Energy grid parser ───────────────────────────────────────────────────────

def _parse_energy_grid(card_lines: List[str]) -> EnergyGrid:
    """Parse an ``ene`` card."""
    tokens = _tokenize(card_lines)
    # tokens: ene <name> <type> [params...]
    name = tokens[1]
    grid_type = int(tokens[2])

    if grid_type == 1:
        # Arbitrary: remaining tokens are energy boundaries
        boundaries = np.array([float(t) for t in tokens[3:]])
    elif grid_type == 2:
        # Uniform linear: N Emin Emax
        n = int(tokens[3])
        emin = float(tokens[4])
        emax = float(tokens[5])
        boundaries = np.linspace(emin, emax, n + 1)
    elif grid_type == 3:
        # Uniform logarithmic: N Emin Emax
        n = int(tokens[3])
        emin = float(tokens[4])
        emax = float(tokens[5])
        boundaries = np.geomspace(emin, emax, n + 1)
    elif grid_type == 4:
        # Predefined: just store name reference
        boundaries = np.array([])
    else:
        boundaries = np.array([float(t) for t in tokens[3:]])

    return EnergyGrid(name=name, grid_type=grid_type, boundaries=boundaries)


# ── Time grid parser ─────────────────────────────────────────────────────────

def _parse_time_grid(card_lines: List[str]) -> TimeGrid:
    """Parse a ``tme`` card."""
    tokens = _tokenize(card_lines)
    name = tokens[1]
    grid_type = int(tokens[2])

    if grid_type == 1:
        boundaries = np.array([float(t) for t in tokens[3:]])
    elif grid_type == 2:
        n = int(tokens[3])
        tmin = float(tokens[4])
        tmax = float(tokens[5])
        boundaries = np.linspace(tmin, tmax, n + 1)
    elif grid_type == 3:
        n = int(tokens[3])
        tmin = float(tokens[4])
        tmax = float(tokens[5])
        boundaries = np.geomspace(tmin, tmax, n + 1)
    else:
        boundaries = np.array([float(t) for t in tokens[3:]])

    return TimeGrid(name=name, grid_type=grid_type, boundaries=boundaries)


# ── Sensitivity parser ───────────────────────────────────────────────────────

def _parse_sens_card(card_lines: List[str], config: SensitivityConfig) -> None:
    """Parse a ``sens`` card and accumulate into *config*.

    Serpent allows multiple ``sens`` cards; each adds to the configuration.
    """
    tokens = _tokenize(card_lines)
    # tokens[0] == "sens", tokens[1] == sub-type
    if len(tokens) < 2:
        return

    sub_type = tokens[1].lower()

    if sub_type == "resp":
        _parse_sens_resp(tokens[2:], config)
    elif sub_type == "pert":
        _parse_sens_pert(tokens[2:], config)
    elif sub_type == "opt":
        _parse_sens_opt(tokens[2:], config)


def _parse_sens_resp(tokens: List[str], config: SensitivityConfig) -> None:
    """Parse ``sens resp <type> [args]``."""
    if not tokens:
        return
    resp_type = tokens[0].lower()

    if resp_type in ("keff", "leff"):
        config.responses.append(SensResponse(response_type=resp_type))
    elif resp_type == "beff":
        config.responses.append(SensResponse(response_type="beff"))
    elif resp_type == "lambda":
        config.responses.append(SensResponse(response_type="lambda"))
    elif resp_type == "void":
        mat_name = tokens[1] if len(tokens) > 1 else None
        config.responses.append(SensResponse(response_type="void", material=mat_name))
    elif resp_type in ("ratio", "detratio"):
        # ratio/detratio <name> <det1> <det2> [p1 p2]
        name = tokens[1] if len(tokens) > 1 else None
        det1 = tokens[2] if len(tokens) > 2 else None
        det2 = tokens[3] if len(tokens) > 3 else None
        config.responses.append(SensResponse(
            response_type="ratio", name=name, det1=det1, det2=det2,
        ))


def _parse_sens_pert(tokens: List[str], config: SensitivityConfig) -> None:
    """Parse ``sens pert <type> [args]``."""
    if not tokens:
        return
    pert_type = tokens[0].lower()

    if pert_type == "xs":
        mode = tokens[1].lower() if len(tokens) > 1 else "all"
        nuclides: List[str] = []
        # Remaining tokens after mode are nuclide names
        for t in tokens[2:]:
            if t.lower() in _CARD_KEYWORDS:
                break
            nuclides.append(t)
        config.perturbations.append(SensPerturbation(
            pert_type="xs", mode=mode, nuclides=nuclides,
        ))
    elif pert_type == "eleg":
        order = int(tokens[1]) if len(tokens) > 1 else 0
        config.perturbations.append(SensPerturbation(
            pert_type="eleg", legendre_order=order,
        ))
    elif pert_type in ("chi", "nubar", "temperature", "elamu", "inlmu"):
        config.perturbations.append(SensPerturbation(pert_type=pert_type))


def _parse_sens_opt(tokens: List[str], config: SensitivityConfig) -> None:
    """Parse ``sens opt <key> <value>``."""
    if len(tokens) >= 2:
        key = tokens[0].lower()
        if key == "egrid":
            config.energy_grid = tokens[1]


# ── Main entry point ─────────────────────────────────────────────────────────

def read_serpent_input(source: Union[str, Path]) -> SerpentInput:
    """Parse a Serpent input file for materials, detectors, energy grids,
    time grids, and sensitivity configuration.

    Parameters
    ----------
    source : str or Path
        File path (if it exists on disk) or raw text content.

    Returns
    -------
    SerpentInput
    """
    # Determine if source is a file path or text content
    path = Path(source) if not isinstance(source, Path) else source
    if path.exists():
        text = path.read_text(encoding="utf-8")
    else:
        text = str(source)

    lines = text.splitlines(keepends=True)

    # Parse materials using existing parser
    materials = read_serpent_materials(lines)

    detectors: Dict[str, DetectorDefinition] = {}
    energy_grids: Dict[str, EnergyGrid] = {}
    time_grids: Dict[str, TimeGrid] = {}
    sens_config = SensitivityConfig()

    i = 0
    while i < len(lines):
        stripped = _strip_comment(lines[i]).strip()
        if not stripped:
            i += 1
            continue

        first_token = stripped.split()[0].lower()

        if first_token == "det":
            card_lines, i = _collect_card_lines(lines, i)
            try:
                det = _parse_detector(card_lines)
                detectors[det.name] = det
            except (IndexError, ValueError):
                pass
            continue

        if first_token == "ene":
            card_lines, i = _collect_card_lines(lines, i)
            try:
                eg = _parse_energy_grid(card_lines)
                energy_grids[eg.name] = eg
            except (IndexError, ValueError):
                pass
            continue

        if first_token == "tme":
            card_lines, i = _collect_card_lines(lines, i)
            try:
                tg = _parse_time_grid(card_lines)
                time_grids[tg.name] = tg
            except (IndexError, ValueError):
                pass
            continue

        if first_token == "sens":
            card_lines, i = _collect_card_lines(lines, i)
            try:
                _parse_sens_card(card_lines, sens_config)
            except (IndexError, ValueError):
                pass
            continue

        i += 1

    return SerpentInput(
        materials=materials,
        detectors=detectors,
        energy_grids=energy_grids,
        time_grids=time_grids,
        sensitivity=sens_config if (sens_config.responses or sens_config.perturbations) else None,
    )
