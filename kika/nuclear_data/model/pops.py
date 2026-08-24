"""A minimal GNDS §12 ``PoPs`` — just enough particle data for phase 3.

The full properties-of-particles database is its own document (NEA, 2016b) and
its own project. ``reactionSuite`` *requires* a ``PoPs`` child, so this has to
exist; what phase 3 actually needs from it is the projectile, the target, the
reaction products, and their masses and spins.

Masses go through :func:`~kika.nuclear_data.model.units.check_mass_unit`, which
enforces the §3.5 prohibition on writing a mass in eV. That is the one place in
the model where the rule bites, and it bites here on purpose.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, Optional, Union

from ..._constants import ATOMIC_NUMBER_TO_SYMBOL, SYMBOL_TO_ATOMIC_NUMBER
from .quantities import PhysicalQuantity
from .units import check_mass_unit

__all__ = ["Particle", "Nuclide", "PoPs", "pidFromZA", "zaFromPid"]


@dataclass
class Particle:
    """A particle with an id and, optionally, a mass and a spin."""

    id: str
    mass: Optional[PhysicalQuantity] = None
    spin: Optional[PhysicalQuantity] = None
    parity: Optional[int] = None
    charge: Optional[int] = None
    #: §12's ``halflife``, which GNDS writes **two ways**: a ``<double>`` with a
    #: time unit, or a ``<string value="stable">``. Both are representable here
    #: — a :class:`~kika.nuclear_data.model.quantities.PhysicalQuantity` for the
    #: first, the literal string for the second — because collapsing "stable"
    #: onto infinity would lose the difference between an evaluator saying a
    #: nuclide does not decay and one saying its halflife is unmeasurably long.
    #:
    #: It is modelled because the schema **requires** it on every ``baryon`` and
    #: ``gaugeBoson``: a PoPs written without it does not validate, and this is
    #: the one §12 property whose absence a writer cannot report its way out of.
    halflife: Optional[Union[PhysicalQuantity, str]] = None

    def __post_init__(self) -> None:
        if self.mass is not None:
            check_mass_unit(self.mass.unit)


@dataclass
class Nuclide(Particle):
    """A nuclide, which additionally knows its Z and A."""

    Z: Optional[int] = None
    A: Optional[int] = None
    nuclearLevel: int = 0

    @property
    def ZA(self) -> Optional[int]:
        """``1000*Z + A`` — ENDF's identifier, derived rather than stored."""
        if self.Z is None or self.A is None:
            return None
        return 1000 * self.Z + self.A


@dataclass
class PoPs:
    """§12. The particle database for one evaluation."""

    particles: Dict[str, Particle] = field(default_factory=dict)
    name: Optional[str] = None
    version: Optional[str] = None

    def __len__(self) -> int:
        return len(self.particles)

    def __iter__(self) -> Iterator[str]:
        return iter(self.particles)

    def __bool__(self) -> bool:
        # A declared slot is *present* even when empty; only `len()` speaks to
        # content. Without this, `if suite.styles:` reads an evaluation that
        # models styles and has none yet as one that does not model them.
        return True

    def __contains__(self, pid: str) -> bool:
        return pid in self.particles

    def __getitem__(self, pid: str) -> Particle:
        try:
            return self.particles[pid]
        except KeyError:
            raise KeyError(f"no particle {pid!r} in PoPs; have {sorted(self.particles)}") from None

    def add(self, particle: Particle) -> None:
        self.particles[particle.id] = particle

    def __repr__(self) -> str:
        return f"PoPs(n={len(self.particles)}, {sorted(self.particles)})"


# ---------------------------------------------------------------------------
# ENDF's ZA, and the §12 id it names
# ---------------------------------------------------------------------------

#: ``ZA`` values that are not nuclides. ENDF spends the same field on them, and
#: §12 gives them names of their own rather than a Z/A pair: a photon is a
#: gauge boson and a neutron is a baryon, and neither has an element symbol.
_SPECIAL_ZA = {0: "photon", 1: "n", 11: "e-"}
_SPECIAL_PID = {pid: za for za, pid in _SPECIAL_ZA.items()}


def pidFromZA(za: int, lip: int = 0) -> str:
    """ENDF's ``ZA`` (and ``LIP``) → the GNDS particle id that names it.

    ``0 → "photon"``, ``1 → "n"``, ``1001 → "H1"``, ``26056 → "Fe56"``. This is
    the spelling every distributed GNDS file uses and the one
    :mod:`kika.gnds.decode` reads back, so it is what makes a suite decoded from
    a tape and the same suite decoded from XML name the same particle.

    **The ``_e`` suffix is only for nuclides, and that is not fussiness.**
    ENDF's ``LIP`` is the excited-state number for a heavy product — Li-6's
    MT52 recoil is ``Li6_e2`` — but for a photon or a neutron it is a *line
    index* instead: ENDF/B-VIII.1's U-235 MT18 lists forty products with
    ``ZAP=0, LIP=1..40`` and fourteen with ``ZAP=1, LIP=1..14``, all of them
    deferring their distribution to MF15 and MF5. Spelling those ``photon_e37``
    would invent a state nobody declared. They are excluded here by ``za <
    1000``, and separately by the MF6 adapter, which does not give a product to
    a law that defers.

    ``lip`` is a *label* either way. The number itself is kept verbatim in the
    ENDF provenance, so nothing about the round trip depends on this function.
    """
    za = int(za)
    if za in _SPECIAL_ZA:
        return _SPECIAL_ZA[za]

    z, a = divmod(za, 1000)
    symbol = ATOMIC_NUMBER_TO_SYMBOL.get(z)
    if symbol is None:
        # Not a refusal: a ZA outside the table is still a thing the file
        # names, and losing it would be worse than spelling it oddly. The
        # caller's report is where this gets said out loud.
        return f"ZA{za}"

    # A=0 is ENDF's natural-element ZA (26000 is natural iron). GNDS spells it
    # with the symbol alone; there is no A to write.
    pid = f"{symbol}{a}" if a else symbol
    return f"{pid}_e{int(lip)}" if lip and za >= 1000 else pid


def zaFromPid(pid: str) -> int:
    """The inverse of :func:`pidFromZA`, ignoring any ``_e`` suffix.

    Returns ``ZA`` alone: the excited-state number is *not* recovered, because
    an encoder that needs ``LIP`` reads it from the provenance where it was
    kept, and inferring it here would give two sources for one field.
    """
    name = pid.split("_e")[0]
    if name in _SPECIAL_PID:
        return _SPECIAL_PID[name]
    if name.startswith("ZA") and name[2:].isdigit():
        return int(name[2:])

    symbol = name.rstrip("0123456789")
    digits = name[len(symbol):]
    z = SYMBOL_TO_ATOMIC_NUMBER.get(symbol)
    if z is None:
        raise ValueError(
            f"{pid!r} is not a particle id this can spell as an ENDF ZA: "
            f"{symbol!r} is not an element symbol, and it is neither "
            f"{sorted(_SPECIAL_PID)} nor a ZA<n> fallback from pidFromZA"
        )
    return 1000 * z + (int(digits) if digits else 0)
