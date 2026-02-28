from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union
from kika.ace.classes.xss import XssEntry
from kika.ace.classes.cross_section.cross_section_repr import reaction_xs_repr, xs_data_repr
from kika._constants import MT_GROUPS, MT_COMPOSITES, MT_COMPOSITE_ORDER
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

@dataclass
class ReactionCrossSection:
    """Container for a single reaction's cross section data."""
    mt: int = 0  # MT number for this reaction 
    energy_idx: int = 0  # Starting energy grid index
    num_energies: int = 0  # Number of consecutive energy points
    _xs_entries: List[XssEntry] = field(default_factory=list)  # Original XssEntry objects for cross section values
    _energy_entries: List[XssEntry] = field(default_factory=list)  # Original XssEntry objects for energy values
    
    def __post_init__(self):
        """Initialize after creation, ensuring values are properly stored."""
        # Convert XssEntry to value if needed
        if hasattr(self.mt, 'value'):
            self.mt = int(self.mt.value)
    
    @property
    def xs_values(self) -> List[float]:
        """Get cross section values as floats."""
        return [entry.value for entry in self._xs_entries]
    
    @property
    def energies(self) -> List[float]:
        """Get energy values as floats."""
        return [entry.value for entry in self._energy_entries]
    
    def plot(self, ax=None, **kwargs):
        """
        Plot cross section for this reaction.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on, if None a new figure is created
        **kwargs
            Additional keyword arguments passed to plot function
            
        Returns
        -------
        matplotlib.axes.Axes
            The matplotlib axes containing the plot
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))
        
        if not self._xs_entries or not self._energy_entries:
            raise ValueError("No cross section values or energies available for plotting")
        
        # Get the energy points and cross section values as DataFrame
        df = self.to_dataframe()
        
        # Plot data
        ax.plot(df["Energy"], df["Cross Section"], label=f"MT={self.mt}", **kwargs)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Energy (MeV)')
        ax.set_ylabel('Cross Section (barns)')
        ax.set_title(f"Cross Section for MT={self.mt}")
        ax.legend()
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        
        return ax
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert this reaction's cross section data to a pandas DataFrame.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with energy and cross section values
        """
        if not self._xs_entries or not self._energy_entries:
            return pd.DataFrame({"Energy": [], "Cross Section": []})
        
        # Ensure they have the same length
        if len(self._energy_entries) != len(self._xs_entries):
            num_points = min(len(self._energy_entries), len(self._xs_entries))
            energies = [e.value for e in self._energy_entries[:num_points]]
            xs_values = [xs.value for xs in self._xs_entries[:num_points]]
        else:
            energies = [e.value for e in self._energy_entries]
            xs_values = [xs.value for xs in self._xs_entries]
        
        # Create DataFrame
        return pd.DataFrame({
            "Energy": energies,
            "Cross Section": xs_values
        })
    
    def __repr__(self):
        return reaction_xs_repr(self)

@dataclass
class CrossSectionData:
    """Container for all reaction cross sections from the SIG block."""
    reaction: Dict[int, ReactionCrossSection] = field(default_factory=dict)  # MT number -> cross section data
    energy_grid: Optional[List[XssEntry]] = None  # Store energy grid for convenience
    _composite_cache: Dict[int, ReactionCrossSection] = field(default_factory=dict)  # Cache for computed composites
    
    def set_energy_grid(self, energy_grid: List[XssEntry]) -> None:
        """
        Set the energy grid for this cross section data.
        
        Parameters
        ----------
        energy_grid : List[XssEntry]
            The energy grid to use for plotting and interpolation
        """
        self.energy_grid = energy_grid
        
        # Update energy entries for each reaction
        for mt, reaction in self.reaction.items():
            # Calculate the proper energy range for this reaction
            if reaction.energy_idx >= 0 and reaction.energy_idx < len(energy_grid):
                end_idx = min(reaction.energy_idx + reaction.num_energies, len(energy_grid))
                reaction._energy_entries = energy_grid[reaction.energy_idx:end_idx]
    
    def add_standard_xs(self, esz_block):
        """
        Add standard cross sections from ESZ block (MT=1, MT=2, MT=101).
        
        Parameters
        ----------
        esz_block : EszBlock
            The ESZ block containing standard cross sections
        """
        if not esz_block or not esz_block.has_data:
            return
            
        # Add total cross section (MT=1)
        if esz_block.total_xs and len(esz_block.total_xs) > 0:
            # Create a ReactionCrossSection for MT=1 (total)
            total_xs = ReactionCrossSection(
                mt=1,  # Total XS
                energy_idx=0,  # Start from beginning of energy grid
                num_energies=len(esz_block.total_xs),
                _xs_entries=esz_block.total_xs,  # Store original XssEntry objects
                _energy_entries=esz_block.energies  # Store original XssEntry objects
            )
            self.reaction[1] = total_xs
            
        # Add elastic cross section (MT=2)
        if esz_block.elastic_xs and len(esz_block.elastic_xs) > 0:
            # Create a ReactionCrossSection for MT=2 (elastic)
            elastic_xs = ReactionCrossSection(
                mt=2,  # Elastic XS
                energy_idx=0,  # Start from beginning of energy grid
                num_energies=len(esz_block.elastic_xs),
                _xs_entries=esz_block.elastic_xs,  # Store original XssEntry objects
                _energy_entries=esz_block.energies  # Store original XssEntry objects
            )
            self.reaction[2] = elastic_xs
            
        # Add absorption cross section (MT=101)
        if esz_block.absorption_xs and len(esz_block.absorption_xs) > 0:
            # Create a ReactionCrossSection for MT=101 (absorption)
            absorption_xs = ReactionCrossSection(
                mt=101,  # Absorption XS
                energy_idx=0,  # Start from beginning of energy grid
                num_energies=len(esz_block.absorption_xs),
                _xs_entries=esz_block.absorption_xs,  # Store original XssEntry objects
                _energy_entries=esz_block.energies  # Store original XssEntry objects
            )
            self.reaction[101] = absorption_xs
    
    @property
    def has_data(self) -> bool:
        """Check if any reaction cross section data is available."""
        return len(self.reaction) > 0

    @property
    def mt_numbers(self) -> List[int]:
        """Get a list of available MT numbers in ascending order."""
        return sorted(self.reaction.keys())

    def clear_composite_cache(self) -> None:
        """
        Clear the composite reaction cache.

        Call this method if the underlying reaction data changes and you need
        to recompute composite reactions.
        """
        self._composite_cache.clear()

    def get_composite_info(self, mt: int) -> dict:
        """
        Get information about a composite MT reaction.

        Parameters
        ----------
        mt : int
            MT reaction number to query

        Returns
        -------
        dict
            Dictionary with keys:
            - is_composite: bool - True if MT is defined in MT_COMPOSITES
            - description: str - Description of the composite (empty if not composite)
            - components: list - List of component MTs (resolved, no @references)
            - available_components: list - Components that exist in the data
            - missing_components: list - Components not in the data
            - can_compute: bool - True if at least one component exists
            - is_complete: bool - True if all components exist
        """
        if mt not in MT_COMPOSITES:
            return {
                'is_composite': False,
                'description': '',
                'components': [],
                'available_components': [],
                'missing_components': [],
                'can_compute': mt in self.reaction,
                'is_complete': mt in self.reaction
            }

        components_def, description = MT_COMPOSITES[mt]

        # Resolve all component MTs (including @references)
        resolved_components = []
        for comp in components_def:
            if isinstance(comp, str) and comp.startswith('@'):
                # Reference to another composite
                ref_mt = int(comp[1:])
                resolved_components.append(ref_mt)
            elif isinstance(comp, range):
                resolved_components.extend(list(comp))
            else:
                resolved_components.append(comp)

        # Check which components are available
        available = []
        missing = []
        for comp_mt in resolved_components:
            # Check if it's directly available or is itself a computable composite
            if comp_mt in self.reaction:
                available.append(comp_mt)
            elif comp_mt in MT_COMPOSITES:
                # Recursively check if the composite can be computed
                comp_info = self.get_composite_info(comp_mt)
                if comp_info['can_compute']:
                    available.append(comp_mt)
                else:
                    missing.append(comp_mt)
            else:
                missing.append(comp_mt)

        return {
            'is_composite': True,
            'description': description,
            'components': resolved_components,
            'available_components': available,
            'missing_components': missing,
            'can_compute': len(available) > 0,
            'is_complete': len(missing) == 0
        }
    
    def _get_or_compute_reaction(self, mt: int) -> Optional[ReactionCrossSection]:
        """
        Get a reaction by MT number, computing it from composite definitions if necessary.

        This method uses MT_COMPOSITES for full dependency-aware computation of
        composite reactions. It handles:
        - Direct reactions (from self.reaction)
        - Cached composite reactions (from self._composite_cache)
        - Computing new composites using MT_COMPOSITE_ORDER for proper dependency order
        - '@MT' references to other composites

        Parameters
        ----------
        mt : int
            MT reaction number

        Returns
        -------
        ReactionCrossSection or None
            The reaction data, either direct or computed, or None if unavailable
        """
        # If we have it directly, return it
        if mt in self.reaction:
            return self.reaction[mt]

        # Check the composite cache
        if mt in self._composite_cache:
            return self._composite_cache[mt]

        # Check if this is a composite MT that we can compute
        if mt not in MT_COMPOSITES:
            return None

        # Ensure dependencies are computed first by processing in order
        for order_mt in MT_COMPOSITE_ORDER:
            if order_mt == mt:
                break
            # Compute dependencies if needed (recursive call handles caching)
            if order_mt in MT_COMPOSITES and order_mt not in self.reaction and order_mt not in self._composite_cache:
                self._get_or_compute_reaction(order_mt)

        # Now compute this composite
        result = self._compute_composite(mt)
        if result is not None:
            self._composite_cache[mt] = result

        return result

    def _compute_composite(self, mt: int) -> Optional[ReactionCrossSection]:
        """
        Compute a composite reaction by summing its components.

        Parameters
        ----------
        mt : int
            MT reaction number (must be in MT_COMPOSITES)

        Returns
        -------
        ReactionCrossSection or None
            The computed composite reaction, or None if no components available
        """
        if mt not in MT_COMPOSITES:
            return None

        components_def, _ = MT_COMPOSITES[mt]

        # Collect all component reactions
        component_reactions: List[ReactionCrossSection] = []

        for comp in components_def:
            if isinstance(comp, str) and comp.startswith('@'):
                # Reference to another composite
                ref_mt = int(comp[1:])
                ref_reaction = self._get_or_compute_reaction(ref_mt)
                if ref_reaction is not None:
                    component_reactions.append(ref_reaction)
            elif isinstance(comp, range):
                # Range of MTs - add all that exist
                for range_mt in comp:
                    if range_mt in self.reaction:
                        component_reactions.append(self.reaction[range_mt])
            else:
                # Single MT number
                comp_mt = int(comp)
                if comp_mt in self.reaction:
                    component_reactions.append(self.reaction[comp_mt])
                elif comp_mt in self._composite_cache:
                    component_reactions.append(self._composite_cache[comp_mt])

        if not component_reactions:
            return None

        # Use the full energy grid if available, otherwise use first component's grid
        if self.energy_grid is not None:
            energy_entries = self.energy_grid
            num_energies = len(energy_entries)
        else:
            energy_entries = component_reactions[0]._energy_entries
            num_energies = len(energy_entries)

        # Initialize sum array for full energy grid
        xs_sum = np.zeros(num_energies)

        # Sum all component cross sections, respecting their energy indices
        for comp_reaction in component_reactions:
            comp_xs = comp_reaction.xs_values
            start_idx = comp_reaction.energy_idx
            end_idx = start_idx + len(comp_xs)

            # Clamp to valid range
            if start_idx < 0:
                start_idx = 0
            if end_idx > num_energies:
                end_idx = num_energies

            actual_len = end_idx - start_idx
            if actual_len > 0 and actual_len <= len(comp_xs):
                xs_sum[start_idx:end_idx] += np.array(comp_xs[:actual_len])

        # Create XssEntry-like objects for the summed cross sections
        class XssValue:
            def __init__(self, val):
                self.value = val

        xs_entries = [XssValue(val) for val in xs_sum]

        # Create and return the computed reaction
        return ReactionCrossSection(
            mt=mt,
            energy_idx=0,  # Computed composite covers full energy grid
            num_energies=num_energies,
            _xs_entries=xs_entries,
            _energy_entries=energy_entries
        )
    
    def plot(self, mt: Union[int, List[int]], ax=None, **kwargs):
        """
        Plot cross section for one or more reactions.
        
        Parameters
        ----------
        mt : int or List[int]
            MT number(s) of the reaction(s) to plot
        ax : matplotlib.axes.Axes, optional
            Axes to plot on, if None a new figure is created
        **kwargs
            Additional keyword arguments passed to plot function
            
        Returns
        -------
        matplotlib.axes.Axes
            The matplotlib axes containing the plot
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))
        
        # Convert single MT to list for consistent handling
        mt_list = [mt] if isinstance(mt, int) else mt
        
        # Check if we have valid MT numbers
        if not mt_list:
            raise ValueError("No MT numbers provided for plotting")
        
        # Keep track of whether we successfully plotted anything
        plotted = False
        
        # Plot each requested MT
        for mt_num in mt_list:
            reaction = self._get_or_compute_reaction(mt_num)
            if reaction and reaction._xs_entries and reaction._energy_entries:
                # Get data for this reaction
                energies = reaction.energies
                xs_values = reaction.xs_values
                
                # Plot with appropriate label
                ax.plot(energies, xs_values, label=f"MT={mt_num}", **kwargs)
                plotted = True
        
        if not plotted:
            available_mts = ", ".join(str(mt) for mt in self.mt_numbers[:10])
            if len(self.mt_numbers) > 10:
                available_mts += ", ..."
            raise ValueError(f"No valid cross section data available for requested MT numbers. Available: {available_mts}")
        
        # Set plot properties
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Energy (MeV)')
        ax.set_ylabel('Cross Section (barns)')
        ax.legend()
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        
        return ax
    
    def to_dataframe(self, mt_list=None) -> pd.DataFrame:
        """
        Convert cross section data to a pandas DataFrame.
        
        Parameters
        ----------
        mt_list : int or List[int], optional
            MT number(s) to include. If None, all available reactions are included.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with energies and cross sections for requested MT numbers
        """
        if self.energy_grid is None:
            raise ValueError("Energy grid is required but none is available")
        
        # Get the energy values
        energy_values = [e.value for e in self.energy_grid]
        
        # Create DataFrame with energy column
        result = {"Energy": energy_values}
        
        # Use all available MT numbers if none specified
        if mt_list is None:
            mt_list = self.mt_numbers  # This is now sorted
        elif isinstance(mt_list, int):
            mt_list = [mt_list]
        
        # Add columns for each MT number, but only if they exist in our data
        for mt in mt_list:
            # Get reaction (computed if necessary)
            reaction = self._get_or_compute_reaction(mt)
            
            if reaction is not None:
                if reaction._xs_entries:
                    # Create array of zeros for the full energy grid
                    xs_values = np.zeros(len(energy_values))
                    
                    # Determine where to place the values in the full array
                    start_idx = reaction.energy_idx
                    end_idx = start_idx + reaction.num_energies
                    
                    # Ensure bounds are valid
                    if start_idx >= 0 and start_idx < len(energy_values):
                        # Place the values (clipping if necessary)
                        actual_length = min(len(reaction._xs_entries), min(end_idx, len(energy_values)) - start_idx)
                        if actual_length > 0:
                            xs_values[start_idx:start_idx + actual_length] = [xs.value for xs in reaction._xs_entries[:actual_length]]
                    
                    result[f"MT={mt}"] = xs_values
                else:
                    # If reaction has no values, fill with zeros
                    result[f"MT={mt}"] = np.zeros(len(energy_values))
            else:
                # Skip MTs that don't exist and can't be computed
                continue
        
        return pd.DataFrame(result)
    
    def to_plot_data(self, mt: int, **kwargs):
        """
        Extract cross section plot data in a format compatible with PlotBuilder.
        
        This method provides direct access to cross section data without going through
        the parent Ace object. It's equivalent to calling ace.to_plot_data('xs', mt=mt).
        
        Parameters
        ----------
        mt : int
            MT reaction number
        **kwargs
            Additional parameters for styling:
            - label (str): Custom label (default: auto-generated)
            - color (str): Line color
            - linestyle (str): Line style ('-', '--', '-.', ':')
            - linewidth (float): Line width
            - marker (str): Marker style ('o', 's', '^', etc.)
            - markersize (float): Marker size
            
        Returns
        -------
        PlotData
            PlotData object compatible with PlotBuilder
            
        Examples
        --------
        >>> ace = kika.read_ace('fe56.ace')
        >>> 
        >>> # Direct access from cross_section object
        >>> xs_data = ace.cross_section.to_plot_data(mt=2, label='Elastic')
        >>> 
        >>> # Use with PlotBuilder
        >>> from kika.plotting import PlotBuilder
        >>> fig = (PlotBuilder()
        ...        .add_data(xs_data)
        ...        .set_labels(x_label='Energy (MeV)', y_label='XS (barns)')
        ...        .set_scales(log_x=True, log_y=True)
        ...        .build())
        """
        from kika.plotting import CrossSectionPlotData
        import numpy as np

        if not self.has_data:
            raise ValueError("No cross section data available")

        reaction = self._get_or_compute_reaction(mt)
        if reaction is None:
            available_mts = self.mt_numbers
            # Check if it's a grouped MT
            grouped_info = ""
            for grouped_mt, mt_range in MT_GROUPS:
                if mt == grouped_mt:
                    grouped_info = f" (grouped MT from range {list(mt_range)[0]}-{list(mt_range)[-1]})"
                    break
            raise ValueError(f"MT={mt} not found{grouped_info}. Available MT numbers: {available_mts}")
        energies = reaction.energies  # In MeV
        xs_values = reaction.xs_values  # In barns

        # Get label
        label = kwargs.get('label', None)
        if label is None:
            # Create default label from MT number
            label = f"MT={mt}"

        return CrossSectionPlotData(
            x=np.array(energies),
            y=np.array(xs_values),
            label=label,
            color=kwargs.get('color', None),
            linestyle=kwargs.get('linestyle', '-'),
            linewidth=kwargs.get('linewidth', None),
            marker=kwargs.get('marker', None),
            markersize=kwargs.get('markersize', None),
            plot_type='line',
            mt=mt,
            data_source='ace',
        )
    
    def to_bulk_plot_data(self, mt_list: List[int] = None) -> Dict[str, Union[List[float], Dict[int, List[float]], List[int]]]:
        """
        Extract ALL cross sections at once for bulk loading.

        This is optimized for frontend applications that need to cache all
        cross sections and switch between them without additional backend requests.

        The method returns a shared energy grid (sent once) and cross section
        values for each MT. Reactions that start at a higher energy index are
        padded with zeros at the beginning.

        Parameters
        ----------
        mt_list : List[int], optional
            List of MT numbers to include. If None, all available MTs are included.

        Returns
        -------
        dict
            Dictionary with keys:
            - 'energies': List[float] - shared energy grid in MeV
            - 'xs_by_mt': Dict[int, List[float]] - cross sections keyed by MT number
            - 'available_mts': List[int] - list of available MT numbers
            - 'x_unit': str - unit for x-axis
            - 'y_unit': str - unit for y-axis

        Examples
        --------
        >>> ace = kika.read_ace('fe56.ace')
        >>> bulk_data = ace.cross_section.to_bulk_plot_data()
        >>> # Access MT=2 directly - no additional computation needed
        >>> elastic_xs = bulk_data['xs_by_mt'][2]
        >>> energies = bulk_data['energies']
        """
        if not self.has_data:
            raise ValueError("No cross section data available")

        if self.energy_grid is None:
            raise ValueError("Energy grid is required but none is available")

        # Get full energy grid
        energy_values = [e.value for e in self.energy_grid]
        num_energies = len(energy_values)

        # Determine which MTs to include
        if mt_list is None:
            mt_list = self.mt_numbers

        # Build cross sections dictionary
        xs_by_mt: Dict[int, List[float]] = {}
        available_mts: List[int] = []

        for mt in mt_list:
            reaction = self._get_or_compute_reaction(mt)
            if reaction is None:
                continue

            available_mts.append(mt)

            if not reaction._xs_entries:
                # No data - fill with zeros
                xs_by_mt[mt] = [0.0] * num_energies
                continue

            # Create array padded to full energy grid
            xs_values = [0.0] * num_energies

            # Determine where to place values
            start_idx = reaction.energy_idx
            end_idx = min(start_idx + reaction.num_energies, num_energies)

            if start_idx >= 0 and start_idx < num_energies:
                actual_length = min(len(reaction._xs_entries), end_idx - start_idx)
                for i in range(actual_length):
                    xs_values[start_idx + i] = reaction._xs_entries[i].value

            xs_by_mt[mt] = xs_values

        return {
            'energies': energy_values,
            'xs_by_mt': xs_by_mt,
            'available_mts': sorted(available_mts),
            'x_unit': 'Energy (MeV)',
            'y_unit': 'Cross Section (barns)',
        }

    def __repr__(self):
        return xs_data_repr(self)
