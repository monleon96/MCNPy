"""
Parse DICE sensitivity filenames into structured benchmark metadata.

DICE sensitivity files are named:

    <BENCHMARK-ID>_<CODE>_<LIBRARY>---<GROUPS>-Group_SENS[.<n>].gz     (multigroup)
    <BENCHMARK-ID>_<CODE>_<LIBRARY>-Continuous_SENS[.<n>].gz           (continuous)

for example::

    HEU-COMP-THERM-021-079_KENO_ENDF-B-VII.0---238-Group_SENS.gz
    HEU-COMP-THERM-007-003_KENO_ENDF-B-VII.0-Continuous_SENS.1.gz

The benchmark id itself is usually five hyphen-separated fields
``CATEGORY-TYPE-SPECTRUM-SERIES-CASE`` (e.g. ``HEU-COMP-THERM-021-079``), but some
ids carry extra prefixes (``SUB-...``); those are ingested with the raw id and
NULL structured fields rather than dropped.
"""

import re
from dataclasses import dataclass
from typing import Optional

# Group form uses a triple-hyphen separator before "<n>-Group"; the library part
# may itself contain single hyphens (ENDF-B-VII.0), hence the non-greedy capture.
_FILE_GROUP_RE = re.compile(
    r"^(?P<benchmark_id>.+?)_(?P<code>[A-Za-z0-9]+)_"
    r"(?P<library>.+?)---(?P<groups>\d+)-Group_SENS(?:\.(?P<dup>\d+))?$"
)
# Continuous-energy form: "<LIBRARY>-Continuous".
_FILE_CONT_RE = re.compile(
    r"^(?P<benchmark_id>.+?)_(?P<code>[A-Za-z0-9]+)_"
    r"(?P<library>.+?)-Continuous_SENS(?:\.(?P<dup>\d+))?$"
)

_BENCH_ID_RE = re.compile(
    r"^(?P<category>[A-Za-z0-9]+)-(?P<btype>[A-Za-z]+)-(?P<spectrum>[A-Za-z]+)-"
    r"(?P<series>\d+)-(?P<case>\d+)$"
)

# Category -> DICE k-eff-table abbreviation letter. The NRG workbook keys rows by
# a short DICE code (e.g. "hmm5-1" = HEU-MET-MIXED series 5 case 1): category
# letter + type initial + spectrum initial + int(series) + "-" + int(case). MIX
# additionally appears with a doubled "mm" prefix for some series, so we try both.
_KEFF_CATEGORY_ABBR = {
    "HEU": "h", "IEU": "i", "LEU": "l", "MIX": "m", "PU": "p", "SPEC": "s", "U233": "u",
}


def benchmark_id_to_keff_abbrevs(benchmark_id: str) -> list:
    """
    Return candidate DICE k-eff-table abbreviations for a benchmark id.

    Empty list if the id does not follow ``CATEGORY-TYPE-SPECTRUM-SERIES-CASE`` or
    the category is unknown. Match any candidate against the keys returned by
    :func:`kika.benchmarks._aux.parse_keff_workbook`.
    """
    parts = benchmark_id.split("-")
    if len(parts) < 5:
        return []
    category, btype, spectrum, series, case = parts[0], parts[1], parts[2], parts[3], parts[4]
    letter = _KEFF_CATEGORY_ABBR.get(category)
    if not letter:
        return []
    try:
        stem = f"{btype[0].lower()}{spectrum[0].lower()}{int(series)}-{int(case)}"
    except (ValueError, IndexError):
        return []
    candidates = [f"{letter}{stem}"]
    if category == "MIX":
        candidates.append(f"mm{stem}")
    return candidates


@dataclass
class ProfileMeta:
    """Structured metadata parsed from a DICE sensitivity filename."""

    benchmark_id: str
    category: Optional[str]
    btype: Optional[str]
    spectrum: Optional[str]
    series: Optional[int]
    case_number: Optional[int]
    code: Optional[str]
    library: Optional[str]
    group_structure: Optional[str]  # "238-Group" / "299-Group" / "continuous"
    ngroups: Optional[int]
    variant_index: int  # the ".<n>" duplicate suffix, 0 if absent
    matched: bool  # False when the filename did not match the expected pattern


def _strip_gz_stem(filename: str) -> str:
    """Return the filename without a trailing ``.gz`` (keeps any ``.<n>`` suffix)."""
    return filename[:-3] if filename.endswith(".gz") else filename


def parse_filename(filename: str, folder_category: Optional[str] = None) -> ProfileMeta:
    """
    Parse a DICE sensitivity filename into a :class:`ProfileMeta`.

    Parameters
    ----------
    filename : str
        The file name (with or without ``.gz``), e.g.
        ``HEU-COMP-THERM-021-079_KENO_ENDF-B-VII.0---238-Group_SENS.gz``.
    folder_category : str, optional
        Category inferred from the containing folder (HEU/IEU/...). Used as a
        fallback when the benchmark id cannot be parsed.
    """
    stem = _strip_gz_stem(filename)

    m = _FILE_GROUP_RE.match(stem)
    if m:
        group_structure = f"{m.group('groups')}-Group"
        ngroups = int(m.group("groups"))
    else:
        m = _FILE_CONT_RE.match(stem)
        if m:
            group_structure = "continuous"
            ngroups = None

    if not m:
        # Unrecognized name: ingest anyway with minimal info.
        return ProfileMeta(
            benchmark_id=stem,
            category=folder_category,
            btype=None,
            spectrum=None,
            series=None,
            case_number=None,
            code=None,
            library=None,
            group_structure=None,
            ngroups=None,
            variant_index=0,
            matched=False,
        )

    benchmark_id = m.group("benchmark_id")
    dup = m.group("dup")
    variant_index = int(dup) if dup else 0

    bm = _BENCH_ID_RE.match(benchmark_id)
    if bm:
        category = bm.group("category")
        btype = bm.group("btype")
        spectrum = bm.group("spectrum")
        series = int(bm.group("series"))
        case_number = int(bm.group("case"))
    else:
        category = folder_category
        btype = spectrum = None
        series = case_number = None

    return ProfileMeta(
        benchmark_id=benchmark_id,
        category=category,
        btype=btype,
        spectrum=spectrum,
        series=series,
        case_number=case_number,
        code=m.group("code"),
        library=m.group("library"),
        group_structure=group_structure,
        ngroups=ngroups,
        variant_index=variant_index,
        matched=True,
    )
