"""NJOY Input Deck reader — parse existing decks into structured data.

Public API
----------
- ``read_deck(text)``      — parse a string
- ``read_deck_file(path)`` — parse a file
- ``ParsedDeck``           — result container
- ``ParsedModule``         — single-module container
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

# Comprehensive set of all NJOY module names (lowercase).
NJOY_MODULE_NAMES: set[str] = {
    "moder", "reconr", "broadr", "unresr", "heatr", "thermr",
    "groupr", "gaminr", "errorr", "covr", "dtfr", "ccccr",
    "matxsr", "resxsr", "acer", "powr", "wimsr", "plotr",
    "viewr", "mixr", "purr", "leapr", "gaspr",
}


@dataclass
class ParsedModule:
    """A single parsed NJOY module from an input deck."""

    name: str              # Lowercase module name (e.g. "moder")
    raw_lines: list[str]   # Original lines (comments + name + cards)
    params: dict | None = None  # Extracted parameters; None for unknown modules


@dataclass
class ParsedDeck:
    """Complete parsed NJOY input deck."""

    preamble: list[str] = field(default_factory=list)
    modules: list[ParsedModule] = field(default_factory=list)
    stop_line: str | None = None

    def render(self) -> str:
        """Reconstruct the exact original text from raw_lines."""
        all_lines = list(self.preamble)
        for mod in self.modules:
            all_lines.extend(mod.raw_lines)
        if self.stop_line is not None:
            all_lines.append(self.stop_line)
        return "\n".join(all_lines)

    def __getitem__(self, name: str) -> ParsedModule:
        """Get first module by name (case-insensitive)."""
        name_lower = name.lower()
        for mod in self.modules:
            if mod.name == name_lower:
                return mod
        raise KeyError(f"No module named {name!r}")

    def get_all(self, name: str) -> list[ParsedModule]:
        """Get all modules matching name (case-insensitive)."""
        name_lower = name.lower()
        return [mod for mod in self.modules if mod.name == name_lower]


def read_deck(text: str) -> ParsedDeck:
    """Parse an NJOY input deck string into structured data.

    Unknown/unimplemented modules are preserved (with ``params=None``)
    and a warning is emitted.
    """
    from .modules import parse_module

    lines = text.split("\n")

    # Locate module-name lines and the stop line
    module_indices: list[tuple[int, str]] = []
    stop_idx: int | None = None

    for i, line in enumerate(lines):
        stripped = line.strip().lower()
        first_word = stripped.split()[0] if stripped else ""
        if first_word in NJOY_MODULE_NAMES:
            module_indices.append((i, first_word))
        elif first_word == "stop":
            stop_idx = i

    if not module_indices:
        return ParsedDeck(
            preamble=lines[: stop_idx] if stop_idx is not None else lines[:],
            modules=[],
            stop_line=lines[stop_idx] if stop_idx is not None else None,
        )

    # Walk backwards from each module name to attach preceding -- comments
    block_starts: list[tuple[int, int, str]] = []  # (block_start, name_idx, name)
    for line_idx, name in module_indices:
        start = line_idx
        while start > 0 and lines[start - 1].strip().startswith("--"):
            start -= 1
        block_starts.append((start, line_idx, name))

    # Determine end index for each block
    blocks: list[tuple[int, int, int, str]] = []  # (start, end, name_idx, name)
    for i, (start, name_idx, name) in enumerate(block_starts):
        if i + 1 < len(block_starts):
            end = block_starts[i + 1][0]
        elif stop_idx is not None:
            end = stop_idx
        else:
            end = len(lines)
        blocks.append((start, end, name_idx, name))

    preamble = lines[: blocks[0][0]]
    stop_line = lines[stop_idx] if stop_idx is not None else None

    modules: list[ParsedModule] = []
    for start, end, name_idx, name in blocks:
        raw_lines = lines[start:end]

        # Extract data card lines (skip comments and the module-name line)
        card_lines: list[str] = []
        for ln in raw_lines:
            stripped = ln.strip()
            if stripped.startswith("--"):
                continue
            first_word = stripped.lower().split()[0] if stripped else ""
            if first_word == name:
                continue
            card_lines.append(ln)

        params = parse_module(name, card_lines)
        modules.append(ParsedModule(name=name, raw_lines=raw_lines, params=params))

    return ParsedDeck(preamble=preamble, modules=modules, stop_line=stop_line)


def read_deck_file(path: str | Path) -> ParsedDeck:
    """Read and parse an NJOY input deck from a file."""
    return read_deck(Path(path).read_text(encoding="utf-8"))
