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
from typing import Dict, Iterator, Optional

from .quantities import PhysicalQuantity
from .units import check_mass_unit

__all__ = ["Particle", "Nuclide", "PoPs"]


@dataclass
class Particle:
    """A particle with an id and, optionally, a mass and a spin."""

    id: str
    mass: Optional[PhysicalQuantity] = None
    spin: Optional[PhysicalQuantity] = None
    parity: Optional[int] = None
    charge: Optional[int] = None

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
