from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
import pandas as pd
from kika.energy_grids.utils import _identify_energy_grid

class PertCollection(dict):
    """A collection class for perturbation cards that provides a nice summary representation.
    
    This class extends the standard dictionary with a custom __repr__ method
    to provide a formatted summary of the perturbation data.
    """
    
    def __repr__(self):
        """Returns a condensed formatted summary of the perturbation cards.
        
        :return: Formatted string showing perturbation summary
        :rtype: str
        """
        if not self:
            return "No perturbation cards available"
            
        # Get sorted perturbation numbers
        pert_nums = sorted(self.keys())
        
        # Find ranges of consecutive perturbation numbers
        ranges = []
        start = pert_nums[0]
        prev = pert_nums[0]
        
        for i in range(1, len(pert_nums)):
            # If there's a gap in sequence
            if pert_nums[i] > prev + 1:
                # Add the completed range
                if start == prev:
                    ranges.append(str(start))
                else:
                    ranges.append(f"{start}-{prev}")
                # Start a new range
                start = pert_nums[i]
            prev = pert_nums[i]
        
        # Add the last range
        if start == prev:
            ranges.append(str(start))
        else:
            ranges.append(f"{start}-{prev}")
        
        # Build the compact summary
        width = 60
        header = f"{'=' * width}\n"
        header += f"{'Perturbation Cards Collection':^{width}}\n"
        header += f"{'=' * width}\n\n"
        
        # Basic information
        label_width = 25  # Width for labels
        info_lines = []
        
        # Count perturbations
        info_lines.append(f"{'Number of perturbations:':{label_width}} {len(self)}")
        info_lines.append(f"{'Perturbation numbers:':{label_width}} {', '.join(ranges)}")
        
        # Get particle types, reactions, and methods
        particles = set(pert.particle for pert in self.values() if pert.particle)
        if particles:
            info_lines.append(f"{'Particle types:':{label_width}} {', '.join(particles)}")
        
        # Get unique reactions
        reactions = set(pert.reaction for pert in self.values() if pert.reaction is not None)
        if reactions:
            info_lines.append(f"{'Reactions available:':{label_width}} {', '.join(map(str, sorted(reactions)))}")
        
        # Get unique methods
        methods = set(pert.method for pert in self.values() if pert.method is not None)
        if methods:
            info_lines.append(f"{'Methods available:':{label_width}} {', '.join(map(str, sorted(methods)))}")

        # Get unique ZAIDs
        zaids = sorted({pert.zaid for pert in self.values() if pert.zaid is not None})
        if zaids:
            info_lines.append(f"{'ZAIDs detected:':{label_width}} {', '.join(map(str, zaids))}")

        # Get energy ranges if available
        energy_values = set()
        for pert in self.values():
            if pert.energy:
                energy_values.add(pert.energy[0])
                energy_values.add(pert.energy[1])
                
        if energy_values:
            info_lines.append(f"{'Energy range:':{label_width}} {min(energy_values):.2e} - {max(energy_values):.2e} MeV")
            info_lines.append(f"{'Number of energy bins:':{label_width}} {len(energy_values)-1}")
            
            # Try to identify energy structure if we have a sorted list
            energy_list = sorted(list(energy_values))
            grid_name = _identify_energy_grid(energy_list)
            if grid_name:
                info_lines.append(f"{'Energy structure:':{label_width}} {grid_name}")
        
        content = "\n".join(info_lines)
        
        # Add examples of accessing data
        examples = "\n\n\nExamples of accessing data:\n"
        examples += "- [perturbation_number] - Access a specific perturbation\n"
        
        return header + content + examples


@dataclass
class Perturbation:
    """Container class for MCNP perturbation cards.

    :ivar pert: Dictionary mapping perturbation IDs to Pert objects
    :type pert: Dict[int, Pert]
    """
    pert: PertCollection = field(default_factory=PertCollection)
    
    def __post_init__(self):
        # Convert existing dictionary to PertCollection if needed
        if self.pert is not None and not isinstance(self.pert, PertCollection):
            self.pert = PertCollection(self.pert)
    
    @property
    def reactions(self) -> List[Optional[int]]:
        """Get unique reaction numbers from all perturbations.

        :returns: Sorted list of unique reaction numbers across all perturbations
        :rtype: List[Optional[int]]
        """
        return sorted(list({pert.reaction for pert in self.pert.values()}))
    
    @property
    def zaids(self) -> List[int]:
        """Get unique ZAIDs from all perturbations.

        :returns: Sorted list of unique ZAIDs across all perturbations
        :rtype: List[int]
        """
        return sorted({p.zaid for p in self.pert.values() if p.zaid is not None})

    @property
    def pert_energies(self) -> List[float]:
        """Get unique energy values from all perturbation energy ranges.

        :returns: Sorted list of unique energy values from all perturbation ranges
        :rtype: List[float]
        """
        energy_values = set()
        for pert in self.pert.values():
            if pert.energy:
                energy_values.add(pert.energy[0])
                energy_values.add(pert.energy[1])
        return sorted(list(energy_values))
    
    def _group_perts_by_reaction(self, method: int) -> Dict[Optional[int], List[int]]:
        """Groups perturbation IDs by their reaction numbers for a given method.

        :param method: The perturbation method to filter by
        :type method: int
        :returns: Dictionary mapping reaction numbers to lists of perturbation IDs
        :rtype: Dict[Optional[int], List[int]]
        :raises ValueError: If no perturbations are defined
        """
        if not self.pert:
            raise ValueError("No perturbations defined")
            
        # Filter perturbations by method
        filtered = {id: pert for id, pert in self.pert.items() if pert.method == method}
        if not filtered:
            return {}
            
        groups = {}
        for id, pert in filtered.items():
            reaction = pert.reaction
            if reaction not in groups:
                groups[reaction] = []
            groups[reaction].append(id)
        return groups
    
    def to_dataframe(self) -> 'pd.DataFrame':
        """Convert perturbation data to a pandas DataFrame.

        This method creates a structured DataFrame containing all perturbation information
        with perturbation IDs as the index.

        :return: DataFrame containing perturbation data
        :rtype: pd.DataFrame
        :raises ValueError: If no perturbations are defined
        """
        if not self.pert:
            raise ValueError("No perturbations defined")
            
        data = []
        for pert_id, pert in sorted(self.pert.items()):
            # Convert cell list to string representation if it exists
            cells = ', '.join(map(str, pert.cell)) if pert.cell else None
            
            # Extract energy range if it exists
            e_min = pert.energy[0] if pert.energy else None
            e_max = pert.energy[1] if pert.energy else None
            
            # Create a row for each perturbation
            data.append({
                'id': pert_id,
                'particle': pert.particle,
                'cell': cells,
                'material': pert.material,
                'rho': pert.rho,
                'method': pert.method,
                'reaction': pert.reaction,
                'e_min': e_min,
                'e_max': e_max,
                'zaid': pert.zaid,
            })
        
        # Create DataFrame with perturbation ID as index
        df = pd.DataFrame(data)
        if not df.empty:
            df.set_index('id', inplace=True)
        return df
    
    def __repr__(self):
        """Returns a formatted string representation of the perturbation data.
        
        This method is called when the object is evaluated in interactive environments
        like Jupyter notebooks or the Python interpreter.
        
        :return: Formatted string representation of the perturbation data
        :rtype: str
        """
        header_width = 60
        header = "=" * header_width + "\n"
        header += f"{'MCNP Perturbation Data':^{header_width}}\n"
        header += "=" * header_width + "\n\n"
        
        label_width = 25  # Width for labels
        
        # Basic information
        info_lines = []
        num_perts = len(self.pert)
        info_lines.append(f"{'Number of perturbations:':{label_width}} {num_perts}")
        
        # Get sorted perturbation numbers and display them in ranges
        if num_perts > 0:
            pert_nums = sorted(self.pert.keys())
            
            # Find ranges of consecutive perturbation numbers
            ranges = []
            start = pert_nums[0]
            prev = pert_nums[0]
            
            for i in range(1, len(pert_nums)):
                # If there's a gap in sequence
                if pert_nums[i] > prev + 1:
                    # Add the completed range
                    if start == prev:
                        ranges.append(str(start))
                    else:
                        ranges.append(f"{start}-{prev}")
                    # Start a new range
                    start = pert_nums[i]
                prev = pert_nums[i]
            
            # Add the last range
            if start == prev:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{prev}")
                
            info_lines.append(f"{'Perturbation numbers:':{label_width}} {', '.join(ranges)}")
        
        # If we have perturbations, show more detailed summary
        if num_perts > 0:
            # Show unique particle types
            particles = set(pert.particle for pert in self.pert.values() if pert.particle)
            if particles:
                info_lines.append(f"{'Particle types:':{label_width}} {', '.join(particles)}")
            
            # Show unique reactions
            reactions = self.reactions
            if reactions:
                info_lines.append(f"{'Reactions available:':{label_width}} {', '.join(map(str, reactions))}")
            
            # Show unique methods
            methods = set(pert.method for pert in self.pert.values() if pert.method is not None)
            if methods:
                info_lines.append(f"{'Methods available:':{label_width}} {', '.join(map(str, sorted(methods)))}")

            # Show detected ZAIDs
            zaids = self.zaids
            if zaids:
                info_lines.append(f"{'ZAIDs detected:':{label_width}} {', '.join(map(str, zaids))}")

            # Show energy ranges if available
            energies = self.pert_energies
            if energies and len(energies) > 0:
                info_lines.append(f"{'Energy range:':{label_width}} {min(energies):.2e} - {max(energies):.2e} MeV")
                info_lines.append(f"{'Number of energy bins:':{label_width}} {len(energies)-1}")
                
                # Add grid structure identification
                grid_name = _identify_energy_grid(energies)
                if grid_name:
                    info_lines.append(f"{'Energy structure:':{label_width}} {grid_name}")
        
        content = "\n".join(info_lines)
        
        # Add examples of accessing data
        examples = "\n\n\nExamples of accessing data:\n"
        examples += "- .pert[perturbation_number] - Access a specific perturbation\n"
        
        return header + content + examples


@dataclass
class Pert:
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
    zaid: Optional[int] = None

    def __repr__(self):
        """Returns a formatted string representation of the perturbation.
        
        This method is called when the object is evaluated in interactive environments
        like Jupyter notebooks or the Python interpreter.
        
        :return: Formatted string representation of the perturbation
        :rtype: str
        """
        header_width = 60
        header = "=" * header_width + "\n"
        header += f"{'MCNP Perturbation Card':^{header_width}}\n"
        header += "=" * header_width + "\n\n"
        
        # Define label width for consistent column alignment
        label_width = 16
        
        # Basic information in neat aligned format
        info_lines = []
        info_lines.append(f"{'Perturbation ID:':{label_width}} {self.id}")
        info_lines.append(f"{'Particle type:':{label_width}} {self.particle}")
        
        # Add additional details if available
        if self.cell:
            if len(self.cell) == 1:
                info_lines.append(f"{'Cell:':{label_width}} {self.cell[0]}")
            else:
                info_lines.append(f"{'Cells:':{label_width}} {', '.join(map(str, self.cell))}")
        
        if self.material is not None:
            info_lines.append(f"{'Material:':{label_width}} {self.material}")
        
        if self.rho is not None:
            info_lines.append(f"{'Density:':{label_width}} {self.rho:.6e}")
        
        if self.method is not None:
            info_lines.append(f"{'Method:':{label_width}} {self.method}")
        
        if self.reaction is not None:
            info_lines.append(f"{'Reaction:':{label_width}} {self.reaction}")
        
        if self.energy:
            info_lines.append(f"{'Energy range:':{label_width}} {self.energy[0]:.6e} - {self.energy[1]:.6e} MeV")

        if self.zaid is not None:
            info_lines.append(f"{'ZAID:':{label_width}} {self.zaid}")

        # Join all information lines
        content = "\n".join(info_lines)
        
        return header + content