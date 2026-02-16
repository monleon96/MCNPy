"""GASPR — calculate gas production cross sections."""

from __future__ import annotations

from ._base import Lines, parse_card_values


def generate(p: dict) -> Lines:
    nendf = p.get("nendf", "")
    nin = p.get("nin", "")
    nout = p.get("nout", "")
    return [
        "-- calculate gas production",
        "gaspr",
        f"{nendf} {nin} {nout}",
    ]


def parse(card_lines: list[str]) -> dict:
    """Parse GASPR card lines into a parameter dict."""
    vals = parse_card_values(card_lines[0])
    return {"nendf": int(vals[0]), "nin": int(vals[1]), "nout": int(vals[2])}
