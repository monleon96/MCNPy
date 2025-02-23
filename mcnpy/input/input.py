from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple

@dataclass
class Input:
    """Main class for storing MCNP input data.

    :ivar pert: Container for all perturbation cards in the input
    :type pert: Pert
    """
    pert: 'Pert' = None

@dataclass
class Pert:
    """Container class for MCNP perturbation cards.

    :ivar perturbation: Dictionary mapping perturbation IDs to Perturbation objects
    :type perturbation: Dict[int, Perturbation]
    """
    perturbation: Dict[int, 'Perturbation'] = field(default_factory=dict)
    
    @property
    def reactions(self) -> List[Optional[int]]:
        """Get unique reaction numbers from all perturbations.

        :returns: Sorted list of unique reaction numbers across all perturbations
        :rtype: List[Optional[int]]
        """
        return sorted(list({pert.reaction for pert in self.perturbation.values()}))
    
    @property
    def pert_energies(self) -> List[float]:
        """Get unique energy values from all perturbation energy ranges.

        :returns: Sorted list of unique energy values from all perturbation ranges
        :rtype: List[float]
        """
        energy_values = set()
        for pert in self.perturbation.values():
            if pert.energy:
                energy_values.add(pert.energy[0])
                energy_values.add(pert.energy[1])
        return sorted(list(energy_values))
    
    def group_perts_by_reaction(self, method: int) -> Dict[Optional[int], List[int]]:
        """Groups perturbation IDs by their reaction numbers for a given method.

        :param method: The perturbation method to filter by
        :type method: int
        :returns: Dictionary mapping reaction numbers to lists of perturbation IDs
        :rtype: Dict[Optional[int], List[int]]
        :raises ValueError: If no perturbations are defined
        """
        if not self.perturbation:
            raise ValueError("No perturbations defined")
            
        # Filter perturbations by method
        filtered = {id: pert for id, pert in self.perturbation.items() if pert.method == method}
        if not filtered:
            return {}
            
        groups = {}
        for id, pert in filtered.items():
            reaction = pert.reaction
            if reaction not in groups:
                groups[reaction] = []
            groups[reaction].append(id)
        return groups

@dataclass
class Perturbation:
    """Represents a single MCNP perturbation card.

    :ivar id: Perturbation identifier number
    :type id: int
    :ivar particle: Particle type (e.g., 'n' for neutron)
    :type particle: str
    :ivar cell: List of cell numbers affected by the perturbation
    :type cell: Optional[List[int]]
    :ivar material: Material number for the perturbation
    :type material: Optional[int]
    :ivar rho: Density value for the perturbation
    :type rho: Optional[float]
    :ivar method: Method number for the perturbation calculation
    :type method: Optional[int]
    :ivar reaction: Reaction number for the perturbation
    :type reaction: Optional[int]
    :ivar energy: Energy range (min, max) for the perturbation
    :type energy: Optional[Tuple[float, float]]
    """
    id: int
    particle: str
    cell: Optional[List[int]] = None
    material: Optional[int] = None
    rho: Optional[float] = None
    method: Optional[int] = None
    reaction: Optional[int] = None
    energy: Optional[Tuple[float, float]] = None

