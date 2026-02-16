"""MODER — convert between ENDF binary/ASCII formats."""

from __future__ import annotations

from ._base import Lines, parse_card_values


def generate(p: dict) -> Lines:
    nin = p.get("nin", "")
    nout = p.get("nout", "")
    return [
        "moder",
        f"{nin} {nout}",
    ]


def parse(card_lines: list[str]) -> dict:
    """Parse MODER card lines into a parameter dict."""
    vals = parse_card_values(card_lines[0])
    return {"nin": int(vals[0]), "nout": int(vals[1])}
