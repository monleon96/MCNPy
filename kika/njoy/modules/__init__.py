"""NJOY module generators and parsers — registry and dispatchers."""

from __future__ import annotations

import warnings
from typing import Any, Callable

from . import moder, reconr, broadr, heatr, thermr, purr, unresr, gaspr, groupr, gaminr, errorr, acer, viewr, covr, leapr, resxsr, ccccr, matxsr, wimsr, dtfr, powr, plotr

GENERATORS: dict[str, Callable[[dict], list[str]]] = {
    "MODER": moder.generate,
    "RECONR": reconr.generate,
    "BROADR": broadr.generate,
    "HEATR": heatr.generate,
    "THERMR": thermr.generate,
    "PURR": purr.generate,
    "UNRESR": unresr.generate,
    "GASPR": gaspr.generate,
    "GROUPR": groupr.generate,
    "GAMINR": gaminr.generate,
    "ERRORR": errorr.generate,
    "ACER": acer.generate,
    "VIEWR": viewr.generate,
    "COVR": covr.generate,
    "LEAPR": leapr.generate,
    "RESXSR": resxsr.generate,
    "CCCCR": ccccr.generate,
    "MATXSR": matxsr.generate,
    "WIMSR": wimsr.generate,
    "DTFR": dtfr.generate,
    "POWR": powr.generate,
    "PLOTR": plotr.generate,
}

PARSERS: dict[str, Callable[[list[str]], dict]] = {
    "moder": moder.parse,
    "reconr": reconr.parse,
    "broadr": broadr.parse,
    "heatr": heatr.parse,
    "thermr": thermr.parse,
    "purr": purr.parse,
    "unresr": unresr.parse,
    "gaspr": gaspr.parse,
    "groupr": groupr.parse,
    "gaminr": gaminr.parse,
    "errorr": errorr.parse,
    "acer": acer.parse,
    "viewr": viewr.parse,
    "covr": covr.parse,
    "leapr": leapr.parse,
    "resxsr": resxsr.parse,
    "ccccr": ccccr.parse,
    "matxsr": matxsr.parse,
    "wimsr": wimsr.parse,
    "dtfr": dtfr.parse,
    "powr": powr.parse,
    "plotr": plotr.parse,
}


def generate_module(name: str, params: dict[str, Any]) -> list[str]:
    """Generate input lines for a single NJOY module by name."""
    gen = GENERATORS.get(name.upper())
    if gen is None:
        raise ValueError(f"Unknown NJOY module: {name!r}")
    return gen(params)


def parse_module(name: str, card_lines: list[str]) -> dict | None:
    """Parse card lines for a module, returning params dict or None.

    Returns *None* (with a warning) for unknown or unparseable modules.
    """
    parser = PARSERS.get(name.lower())
    if parser is None:
        warnings.warn(
            f"No parser for module {name!r} — parameters not extracted",
            stacklevel=3,
        )
        return None
    try:
        return parser(card_lines)
    except Exception as exc:
        warnings.warn(
            f"Failed to parse module {name!r}: {exc}",
            stacklevel=3,
        )
        return None
