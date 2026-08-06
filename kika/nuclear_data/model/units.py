"""GNDS-2.1 §3.5 units: the prefix and symbol tables, and the combination rules.

**Why kika has its own registry rather than pint.** GNDS defines its own unit
syntax — a prefix table (Table 3.4), symbol tables for SI (3.6), derived SI
(3.7) and fundamental constants (3.5), and combination by ``*``, ``/`` and
``**``. Adopting pint would mean translating on every read and every write, and
the translation is where unit bugs live. The specification is explicit that it
is modelled on ``Scientific/Physics/PhysicalQuantities.py``.

**Where units live.** Never per array element. §2.3.3 puts a unit on a scalar
``physicalQuantity``; §5.1.2 puts one on each ``axis`` of a function. So arrays
stay plain numpy and the unit rides on the axis — full traceability, no cost in
the hot path, and free at serialisation time.

**The standing prohibition (§3.5, verbatim).** *"Note that the common nuclear
physics convention of expressing masses in 'MeV' (rather than 'MeV/c**2')
should not be allowed."* :func:`check_mass_unit` enforces it; nothing calls it
implicitly, because deciding that a quantity *is* a mass is the caller's job.

Spaces are insignificant: ``'kg*m**2/s**2'`` and ``'kg * m**2 / s**2'`` are the
same unit.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, Iterable, Mapping

__all__ = [
    "PREFIXES",
    "SI_UNITS",
    "DERIVED_SI_UNITS",
    "FUNDAMENTAL_CONSTANTS",
    "Unit",
    "UnitError",
    "parse_unit",
    "conversion_factor",
    "check_mass_unit",
]


class UnitError(ValueError):
    """A unit string is not spellable in the GNDS §3.5 grammar."""


#: Table 3.4. One prefix may be appended to the beginning of any base unit, and
#: only one: the spec says use ``Mg``, never ``kkg``. Note ``mu`` for micro (not
#: ``u`` and not ``µ``) and ``da`` for deka — both are two characters, which is
#: why the parser tries the longest prefix first.
PREFIXES: Dict[str, float] = {
    "Y": 1e24, "Z": 1e21, "E": 1e18, "P": 1e15, "T": 1e12, "G": 1e9,
    "M": 1e6, "k": 1e3, "h": 1e2, "da": 1e1,
    "d": 1e-1, "c": 1e-2, "m": 1e-3, "mu": 1e-6, "n": 1e-9, "p": 1e-12,
    "f": 1e-15, "a": 1e-18, "z": 1e-21, "y": 1e-24,
}

#: Table 3.6 — the seven SI base units.
SI_UNITS: Dict[str, str] = {
    "m": "length",
    "kg": "mass",
    "s": "time",
    "A": "electrical current",
    "K": "thermodynamic temperature",
    "mol": "amount of substance",
    "cd": "luminous intensity",
}

#: Table 3.7 — derived SI units, with the admixture the spec gives for each.
DERIVED_SI_UNITS: Dict[str, str] = {
    "Hz": "1/s",
    "N": "m*kg/s**2",
    "Pa": "N/m**2",
    "J": "N*m",
    "W": "J/s",
    "C": "s*A",
    "V": "W/A",
    "F": "C/V",
    "ohm": "V/A",
    "S": "A/V",
    "Wb": "V*s",
    "T": "Wb/m**2",
    "H": "Wb/A",
    "lm": "cd*sr",
    "lx": "lm/m**2",
    "Bq": "1/s",
    "Gy": "J/kg",
    "Sv": "J/kg",
}

#: Table 3.5 — fundamental constants, by symbol. Values are not needed to *parse*
#: a unit; ``c`` matters because ``MeV/c**2`` is the mandated spelling for mass.
FUNDAMENTAL_CONSTANTS: Dict[str, str] = {
    "c": "speed of light",
    "mu0": "permeability of vacuum",
    "eps0": "permittivity of vacuum",
    "Grav": "gravitational constant",
    "hplanck": "Planck constant",
    "hbar": "Planck constant / 2pi",
    "e": "elementary charge",
    "Nav": "Avogadro number",
    "k": "Boltzmann constant",
}

#: Units in daily use in this field that are not SI and not derived SI. Kept
#: separate so it is obvious which symbols the specification names and which
#: this library adds.
EXTRA_UNITS: Dict[str, str] = {
    "eV": "energy",
    "b": "cross section (barn)",
    "sr": "solid angle (steradian)",
    "rad": "plane angle",
    "amu": "atomic mass unit",
    "g": "mass",
}

_KNOWN = set(SI_UNITS) | set(DERIVED_SI_UNITS) | set(FUNDAMENTAL_CONSTANTS) | set(EXTRA_UNITS)

#: Symbols that measure the same thing as another symbol, and the factor between
#: them. Without this, ``g`` and ``kg`` would carry *different* dimension keys
#: and ``conversion_factor('g', 'kg')`` would refuse a conversion that is
#: obviously fine. The awkwardness is the specification's: Table 3.6 makes ``kg``
#: the SI base symbol even though the prefix belongs to the gramme.
_ALIASES: Dict[str, tuple[float, str]] = {
    "g": (1e-3, "kg"),
}

_SYMBOL = re.compile(r"[A-Za-z][A-Za-z0-9]*")


@dataclass(frozen=True)
class Unit:
    """A parsed GNDS unit: a scale factor and integer powers of base symbols.

    ``dimensions`` maps an *unprefixed* symbol to its exponent, so ``MeV/c**2``
    is ``factor=1e6`` with ``{'eV': 1, 'c': -2}``. Two units are compatible when
    their dimensions match; converting between them is then a single
    multiplication by the ratio of their factors.
    """

    text: str
    factor: float
    dimensions: Mapping[str, Fraction]

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.text

    @property
    def isDimensionless(self) -> bool:
        return not self.dimensions

    def isCompatibleWith(self, other: "Unit") -> bool:
        return dict(self.dimensions) == dict(other.dimensions)


def _split_symbol(token: str) -> tuple[float, str]:
    """``('MeV')`` → ``(1e6, 'eV')``. Longest prefix wins, and a bare known
    symbol always beats a prefix reading of it.

    ``m`` is metre, not milli-nothing; ``cd`` is candela, not centi-day; ``T``
    is tesla, not tera-nothing. Checking the whole token against the symbol
    tables *first* is what keeps those right.
    """
    if token in _KNOWN:
        scale, canonical = _ALIASES.get(token, (1.0, token))
        return scale, canonical
    for length in (2, 1):  # 'da' and 'mu' are two characters
        prefix, rest = token[:length], token[length:]
        if prefix in PREFIXES and rest in _KNOWN:
            scale, canonical = _ALIASES.get(rest, (1.0, rest))
            return PREFIXES[prefix] * scale, canonical
    raise UnitError(
        f"unknown unit symbol {token!r}; not an SI unit (§3.6), a derived SI "
        f"unit (§3.7), a fundamental constant (§3.5) or a prefixed form of one"
    )


def parse_unit(text: str) -> Unit:
    """Parse a GNDS §3.5 unit string into a :class:`Unit`.

    Grammar: symbols combined with ``*``, ``/`` and ``**``, spaces ignored. An
    empty string is dimensionless, which §2.3.3 makes the default for a
    ``physicalQuantity`` with no ``unit`` attribute.
    """
    original = text
    text = text.replace(" ", "")
    if not text:
        return Unit("", 1.0, {})

    factor = 1.0
    dims: Dict[str, Fraction] = {}
    sign = 1
    position = 0

    while position < len(text):
        match = _SYMBOL.match(text, position)
        if not match:
            raise UnitError(f"expected a unit symbol at {position} in {original!r}")
        scale, symbol = _split_symbol(match.group())
        position = match.end()

        exponent = Fraction(1)
        if text.startswith("**", position):
            position += 2
            exp_match = re.match(r"[+-]?\d+(?:\.\d+)?", text[position:])
            if not exp_match:
                raise UnitError(f"expected an exponent after '**' in {original!r}")
            exponent = Fraction(exp_match.group()).limit_denominator(1000)
            position += exp_match.end()

        signed = exponent * sign
        factor *= scale ** float(signed)
        dims[symbol] = dims.get(symbol, Fraction(0)) + signed

        if position == len(text):
            break
        operator = text[position]
        if operator in "*/":
            sign = 1 if operator == "*" else -1
            position += 1
            if position == len(text):
                raise UnitError(
                    f"{original!r} ends with {operator!r}; an operator needs a "
                    f"symbol after it"
                )
        else:
            raise UnitError(
                f"expected '*' or '/' at {position} in {original!r}, got {operator!r}"
            )

    return Unit(original, factor, {s: e for s, e in dims.items() if e != 0})


def conversion_factor(source: str | Unit, target: str | Unit) -> float:
    """Multiply a value in ``source`` by this to express it in ``target``.

    Raises when the two are not the same physical quantity, which is the whole
    point — the ``* 1e6  # MeV -> eV`` comments scattered through the flat
    classes are the bug class this replaces.
    """
    a = source if isinstance(source, Unit) else parse_unit(source)
    b = target if isinstance(target, Unit) else parse_unit(target)
    if not a.isCompatibleWith(b):
        raise UnitError(
            f"cannot convert {a.text!r} to {b.text!r}: dimensions "
            f"{dict(a.dimensions)} and {dict(b.dimensions)} differ"
        )
    return a.factor / b.factor


def check_mass_unit(text: str) -> None:
    """Enforce the §3.5 prohibition on spelling a mass as an energy.

    *"Note that the common nuclear physics convention of expressing masses in
    'MeV' (rather than 'MeV/c**2') should not be allowed."* Not called
    implicitly: whether a number is a mass is the caller's knowledge, not the
    unit string's.
    """
    unit = parse_unit(text)
    if dict(unit.dimensions) == {"eV": Fraction(1)}:
        raise UnitError(
            f"{text!r} is an energy, not a mass. GNDS §3.5 forbids the "
            f"nuclear-physics shorthand of writing a mass in eV; use "
            f"{text}/c**2, or 'amu', or 'kg'."
        )
