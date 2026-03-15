"""General-purpose materials module for multi-code support.

This module provides classes for defining, manipulating, and exporting
material compositions for various Monte Carlo codes including MCNP and Serpent.

Classes
-------
Nuclide
    Lightweight nuclide entry for a material with ZAID, fraction, and libraries.
NuclideAccessor
    Dictionary-like accessor that allows retrieving nuclides by ZAID or symbol.
Material
    General-purpose material representation with nuclide composition.
MaterialCollection
    Container class for multiple materials.

Functions
---------
read_material
    Parse a material card from MCNP input lines.

Examples
--------
>>> from kika.materials import Material, MaterialCollection

>>> # Create a simple material
>>> mat = Material(id=1)
>>> mat.add_nuclide('H', 2.0, 'ao')
>>> mat.add_nuclide('O', 1.0, 'ao')
>>> mat.normalize()
>>> mat.set_density(1.0, 'g/cc')

>>> # Export to different codes
>>> print(mat.to_mcnp())
>>> print(mat.to_serpent())
"""

from .material import Nuclide, NuclideAccessor, Material, MaterialCollection
from .parse_mcnp import read_material
from .parse_serpent import read_serpent_materials

__all__ = [
    'Nuclide',
    'NuclideAccessor',
    'Material',
    'MaterialCollection',
    'read_material',
    'read_serpent_materials',
]
