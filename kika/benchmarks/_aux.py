"""
Parsers for the DICE auxiliary data that sits alongside ``sensitivity/``:

* ``keff/NRG_VanDerMark_Tables.xlsx`` — experimental (benchmark) k-eff ± σ.
* ``spectra/*_SPEC.gz``              — 299-group by-group flux spectrum.
* ``balance/*_BAL.gz``              — 1-group reaction-rate balance.
* ``input/*_INP.gz``               — KENO/MCNP input decks (stored verbatim, no parse).

The xlsx is a zipped XML workbook parsed with the standard library (``zipfile`` +
``xml.etree``) so no spreadsheet dependency (openpyxl/pandas) is introduced. The
spectrum is a clean numeric table; the balance is irregular free text, so we keep
it verbatim and only extract the scalar summary (k-eff, leakage, zone count).
"""

import re
import xml.etree.ElementTree as ET
import zipfile
from typing import Dict, List, Optional, Tuple

_XL_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


# ---------------------------------------------------------------------------
# Experimental k-eff workbook
# ---------------------------------------------------------------------------
def _col_letters(cell_ref: str) -> str:
    """Return the column letters of an A1-style cell reference (``'F12' -> 'F'``)."""
    return re.match(r"[A-Z]+", cell_ref).group()


def parse_keff_workbook(path: str) -> Dict[str, Tuple[float, Optional[float]]]:
    """
    Parse the NRG k-eff workbook into ``{dice_abbrev: (keff, uncertainty)}``.

    Only the experimental value is returned (the rows whose label column reads
    ``"Benchmark"``; the workbook also lists per-library calculated k-eff which we
    ignore). Keys are the DICE abbreviations used in column A (e.g. ``"hmm5-1"``);
    map a benchmark id to them with
    :func:`kika.benchmarks._naming.benchmark_id_to_keff_abbrevs`.
    """
    with zipfile.ZipFile(path) as z:
        shared = [
            "".join(t.text or "" for t in si.iter(f"{_XL_NS}t"))
            for si in ET.fromstring(z.read("xl/sharedStrings.xml")).iter(f"{_XL_NS}si")
        ]
        sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))

    def cell_value(c) -> Optional[str]:
        v = c.find(f"{_XL_NS}v")
        if v is None:
            return None
        return shared[int(v.text)] if c.get("t") == "s" else v.text

    result: Dict[str, Tuple[float, Optional[float]]] = {}
    for row in sheet.iter(f"{_XL_NS}row"):
        cells = {_col_letters(c.get("r")): cell_value(c) for c in row.iter(f"{_XL_NS}c")}
        abbrev, label = cells.get("A"), cells.get("E")
        if not abbrev or label != "Benchmark":
            continue
        try:
            keff = float(cells["F"])
        except (KeyError, TypeError, ValueError):
            continue
        try:
            unc = float(cells.get("G")) if cells.get("G") else None
        except ValueError:
            unc = None
        # First occurrence wins (a benchmark appears once as "Benchmark").
        result.setdefault(abbrev, (keff, unc))
    return result


# ---------------------------------------------------------------------------
# Flux spectrum (*_SPEC.gz)
# ---------------------------------------------------------------------------
# A data row is "<grp int> <E-up> <flux> <capture> <fission> <(n,2n)> <product>".
_SPEC_ROW_RE = re.compile(
    r"^\s*\d+\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)"
)


def parse_spectrum(text: str) -> Optional[Tuple[List[float], List[float]]]:
    """
    Parse a ``*_SPEC`` by-group balance into ``(energies_upper, flux)``.

    ``energies_upper`` are the per-group upper energy edges (MeV, as written —
    descending), ``flux`` the corresponding group flux. Returns ``None`` if no
    numeric rows are found.
    """
    energies: List[float] = []
    flux: List[float] = []
    for line in text.splitlines():
        if line.lstrip().upper().startswith("TOTAL"):
            break
        m = _SPEC_ROW_RE.match(line)
        if not m:
            continue
        try:
            energies.append(float(m.group(1)))
            flux.append(float(m.group(2)))
        except ValueError:
            continue
    if not energies:
        return None
    return energies, flux


# ---------------------------------------------------------------------------
# Reaction-rate balance (*_BAL.gz) — scalar summary only
# ---------------------------------------------------------------------------
_BAL_KEFF_RE = re.compile(r"K-EFF\s*=\s*([-+0-9.eE]+)")
_BAL_LEAK_RE = re.compile(r"LEAKAGE\s*=\s*([-+0-9.eE]+)")
_BAL_ZONES_RE = re.compile(r"NUMBER OF ZONES IN THE CORE:\s*(\d+)")


def parse_balance_summary(text: str) -> Dict[str, float]:
    """Extract the scalar summary (``keff``, ``leakage``, ``n_zones``) from a balance file."""
    summary: Dict[str, float] = {}
    m = _BAL_KEFF_RE.search(text)
    if m:
        try:
            summary["keff"] = float(m.group(1))
        except ValueError:
            pass
    m = _BAL_LEAK_RE.search(text)
    if m:
        try:
            summary["leakage"] = float(m.group(1))
        except ValueError:
            pass
    m = _BAL_ZONES_RE.search(text)
    if m:
        summary["n_zones"] = int(m.group(1))
    return summary
