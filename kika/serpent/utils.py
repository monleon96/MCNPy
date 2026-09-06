# ──────────────────────────────────────────────────────────────────────────────
# File: serpent_sens/utils.py
# Small helpers for parsing and label normalization.
# ──────────────────────────────────────────────────────────────────────────────
import re
from typing import Iterable, List, Tuple

from kika.serpent.sens import Perturbation, PertCategory
from kika._constants import MT_TO_REACTION


_NUM_RE = re.compile(r"[+-]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[Ee][+-]?\d+)?")


def extract_numeric_list(block: str) -> List[float]:
    return [float(x) for x in _NUM_RE.findall(block)]


def extract_int_list(block: str) -> List[int]:
    return [int(float(x)) for x in _NUM_RE.findall(block)]


def extract_string_list(block: str) -> List[str]:
    # Matches lines like 'SS304       '\n 'Another     '
    items = []
    for m in re.finditer(r"'([^']*)'", block):
        items.append(m.group(1).strip())
    return items


def _mt_label(mt: int) -> str:
    """Build a human-readable label for an MT number, e.g. 'MT 102 (n,γ)'."""
    name = MT_TO_REACTION.get(mt)
    if name:
        return f"MT {mt} {name}"
    return f"MT {mt}"


def parse_perturbation_label(raw: str, index: int) -> Perturbation:
    s = raw.strip().lower()
    # total xs
    if re.fullmatch(r"total\s+xs", s):
        return Perturbation(
            index=index,
            raw_label=raw,
            category=PertCategory.MT_XS,
            mt=1,
            short_label=_mt_label(1),
        )
    # mt N xs
    m = re.fullmatch(r"mt\s+(\d+)\s+xs", s)
    if m:
        mt = int(m.group(1))
        # Check if this is a Legendre moment encoded as MT 400X
        if 4001 <= mt <= 4099:
            moment = mt - 4000  # Convert MT 4001 -> L=1, MT 4002 -> L=2, etc.
            return Perturbation(
                index=index,
                raw_label=raw,
                category=PertCategory.LEGENDRE_MOMENT,
                mt=mt,  # Keep original MT number for reference
                channel="ela",  # Only elastic channel possible for Legendre moments
                moment=moment,
                short_label=f"ela L={moment}",
            )
        else:
            # Regular cross-section
            return Perturbation(
                index=index,
                raw_label=raw,
                category=PertCategory.MT_XS,
                mt=mt,
                short_label=_mt_label(mt),
            )
    # inelastic scattering (SERPENT sometimes uses labels like 'inl scatt xs')
    # map these to MT=4 so downstream code treats them correctly
    if re.fullmatch(r"(?:inl|inel(?:astic)?)\s+scatt(?:er)?\s+xs", s):
        return Perturbation(
            index=index,
            raw_label=raw,
            category=PertCategory.MT_XS,
            mt=4,
            short_label=_mt_label(4),
        )
    # nubar total | nubar prompt | nubar delayed → ENDF MT 452/456/455
    m = re.fullmatch(r"nubar\s+(total|prompt|delayed)", s)
    if m:
        nubar_mt = {"total": 452, "prompt": 456, "delayed": 455}[m.group(1)]
        return Perturbation(
            index=index,
            raw_label=raw,
            category=PertCategory.NUBAR,
            mt=nubar_mt,
            short_label=_mt_label(nubar_mt),
        )
    # chi total | chi prompt | chi delayed → MT 1018 (ENDF) / 1019 / 1020 (Serpent extension)
    m = re.fullmatch(r"chi\s+(total|prompt|delayed)", s)
    if m:
        chi_mt = {"total": 1018, "prompt": 1019, "delayed": 1020}[m.group(1)]
        return Perturbation(
            index=index,
            raw_label=raw,
            category=PertCategory.CHI,
            mt=chi_mt,
            short_label=_mt_label(chi_mt),
        )
    # <channel> leg mom N (current format for Legendre moments)
    m = re.fullmatch(r"(\w+)\s+leg\s+mom\s+(\d+)", s)
    if m:
        channel = m.group(1)
        moment = int(m.group(2))
        # Map to MT 400X format for consistency
        mt = 4000 + moment  # L=1 -> MT=4001, L=2 -> MT=4002, etc.
        return Perturbation(
            index=index,
            raw_label=raw,
            category=PertCategory.LEGENDRE_MOMENT,
            mt=mt,  # Use MT 400X format
            channel=channel,
            moment=moment,
            short_label=f"{channel} L={moment}",
        )
    # Fallback
    return Perturbation(
        index=index,
        raw_label=raw,
        category=PertCategory.OTHER,
        short_label=raw.strip(),
    )