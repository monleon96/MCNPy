from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple

@dataclass
class Input:
    pert: 'Pert' = None

@dataclass
class Pert:
    perturbation: Dict[int, 'Perturbation'] = field(default_factory=dict)
    
    @property
    def reactions(self) -> List[Optional[int]]:
        """Get unique reaction numbers from all perturbations."""
        return sorted(list({pert.reaction for pert in self.perturbation.values()}))
    
    @property
    def pert_energies(self) -> List[float]:
        """Get unique energy values from all perturbation energy ranges."""
        energy_values = set()
        for pert in self.perturbation.values():
            if pert.energy:
                energy_values.add(pert.energy[0])
                energy_values.add(pert.energy[1])
        return sorted(list(energy_values))
    
    def group_perts_by_reaction(self, method: int) -> Dict[Optional[int], List[int]]:
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
    id: int
    particle: str
    cell: Optional[List[int]] = None
    material: Optional[int] = None
    rho: Optional[float] = None
    method: Optional[int] = None
    reaction: Optional[int] = None
    energy: Optional[Tuple[float, float]] = None

