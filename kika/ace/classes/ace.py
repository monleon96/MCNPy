from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any
import logging
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import defaultdict
from kika.ace.classes.xss import XssEntry
from kika.ace.classes.header import Header
from kika.ace.classes.nubar.nubar import NuContainer
from kika.ace.classes.delayed_neutron.delayed_neutron import DelayedNeutronData
from kika.ace.classes.mt_reaction.mtr import ReactionMTData
from kika.ace.classes.q_values import QValues
from kika.ace.classes.particle_release.particle_release import ParticleRelease
from kika.ace.classes.cross_section.cross_section_locators import CrossSectionLocators
from kika.ace.classes.cross_section.cross_section_data import CrossSectionData
from kika.ace.classes.angular_distribution.angular_locators import AngularDistributionLocators
from kika.ace.classes.angular_distribution.container import AngularDistributionContainer
from kika.ace.classes.energy_distribution.locators import EnergyDistributionLocators
from kika.ace.classes.energy_distribution.container import EnergyDistributionContainer
from kika.ace.classes.gpd import PhotonProductionData
from kika.ace.classes.photon_production_xs import PhotonProductionCrossSections, ParticleProductionCrossSections
from kika.ace.classes.yield_multipliers import PhotonYieldMultipliers, SecondaryParticleYieldMultipliers
from kika.ace.classes.fission_xs import FissionCrossSection
from kika.ace.classes.unresolved_resonance import UnresolvedResonanceTables
from kika.ace.classes.secondary_particles.secondary_particle_cross_sections import SecondaryParticleCrossSections
from kika.ace.classes.secondary_particles.secondary_particle_data_locators import SecondaryParticleDataLocators
from kika.ace.classes.secondary_particles.secondary_particle_reactions import SecondaryParticleReactions
from kika.ace.classes.secondary_particles.secondary_particles_types import SecondaryParticleTypes
from kika.ace.classes.esz import EszBlock
from kika.ace.classes.ace_repr import ace_repr
from kika._constants import MT_COMPOSITES, MT_COMPOSITE_ORDER


@dataclass
class Ace:
    """
    Class representing ACE format data.
    
    ACE (A Compact ENDF) is a format used in nuclear data libraries.
    """
    # Original filename
    filename: Optional[str] = None
    
    # Header information - loaded immediately
    header: Optional[Header] = None
    
    # XSS data array - loaded immediately
    xss_data: Optional[List[XssEntry]] = None  # Main data array
    
    # Standard ACE data blocks
    esz_block: Optional[EszBlock] = None
    nubar: Optional[NuContainer] = None
    delayed_neutron_data: Optional[DelayedNeutronData] = None
    reaction_mt_data: Optional[ReactionMTData] = None
    q_values: Optional[QValues] = None
    particle_release: Optional[ParticleRelease] = None
    xs_locators: Optional[CrossSectionLocators] = None
    cross_section: Optional[CrossSectionData] = None 
    angular_locators: Optional[AngularDistributionLocators] = None
    angular_distributions: Optional[AngularDistributionContainer] = None
    energy_distribution_locators: Optional[EnergyDistributionLocators] = None
    energy_distributions: Optional[EnergyDistributionContainer] = None
    photon_production_data: Optional[PhotonProductionData] = None
    photon_production_xs: Optional[PhotonProductionCrossSections] = None
    particle_production_xs: Optional[ParticleProductionCrossSections] = None
    photon_yield_multipliers: Optional[PhotonYieldMultipliers] = None
    particle_yield_multipliers: Optional[SecondaryParticleYieldMultipliers] = None
    fission_xs: Optional[FissionCrossSection] = None
    unresolved_resonance: Optional[UnresolvedResonanceTables] = None
    
    # User-friendly attribute names for secondary particle data
    secondary_particle_types: Optional[SecondaryParticleTypes] = None
    secondary_particle_reactions: Optional[SecondaryParticleReactions] = None
    secondary_particle_data_locations: Optional[SecondaryParticleDataLocators] = None
    secondary_particle_cross_sections: Optional[SecondaryParticleCrossSections] = None
    
    # Cache for energy distributions (still lazy-loaded)
    _cache: Dict[str, Any] = field(default_factory=dict, repr=False)
    
    # Debug flag for parsers
    _debug: bool = False
    
    @property
    def zaid(self) -> int:
        """
        Get the ZAID (Z number, A number, isotope identifier) of the ACE file.
        
        :returns: ZAID as an integer
        :rtype: int
        """
        if self.header and self.header.zaid:
            return self.header.zaid
        return None

    @property
    def energies(self):
        """Energy grid - returns list of float values"""
        if self.esz_block and self.esz_block.energies:
            return [e.value for e in self.esz_block.energies]
        return None
    
    @property
    def total_xs(self):
        """Total cross section - returns list of float values"""
        if self.esz_block and self.esz_block.total_xs:
            return [xs.value for xs in self.esz_block.total_xs]
        return None
    
    @property
    def absorption_xs(self):
        """Absorption cross section - returns list of float values"""
        if self.esz_block and self.esz_block.absorption_xs:
            return [xs.value for xs in self.esz_block.absorption_xs]
        return None
    
    @property
    def elastic_xs(self):
        """Elastic cross section - returns list of float values"""
        if self.esz_block and self.esz_block.elastic_xs:
            return [xs.value for xs in self.esz_block.elastic_xs]
        return None
    
    def copy(self) -> 'Ace':
        """
        Create an exact copy of the Ace object.
        
        Returns:
        --------
        Ace
            A new Ace object that is an exact copy of this one
        """
        import copy
        return copy.deepcopy(self)
    
    @property
    def heating_numbers(self):
        """Heating numbers - returns list of float values"""
        if self.esz_block and self.esz_block.heating_numbers:
            return [h.value for h in self.esz_block.heating_numbers]
        return None
    
    @property
    def mt_numbers(self) -> List[int]:
        """
        Get a list of all available MT numbers, including standard ones (total, elastic, absorption).
        
        :returns: List of MT numbers in ascending order
        :rtype: List[int]
        """
        mt_list = [1, 2, 101]  # Standard MT numbers (total, elastic, absorption)
        
        # Add reaction-specific MT numbers if available
        if self.reaction_mt_data and self.reaction_mt_data.has_neutron_mt_data:
            # Extract integer values from XssEntry objects
            mt_list.extend([int(entry.value) for entry in self.reaction_mt_data.incident_neutron])
        
        # Remove duplicates and sort
        return sorted(list(set(mt_list)))
    
    __repr__ = ace_repr

    def get_cross_section(self, reaction: Union[int, List[int], None] = None) -> pd.DataFrame:
        """
        Get cross section data for a specific reaction or list of reactions.
        
        This method provides a unified way to access all cross sections using MT numbers:
        - MT=1: Total cross section
        - MT=2: Elastic scattering cross section
        - MT=101: Absorption cross section
        - Other MT numbers: Reaction-specific cross sections
        
        Examples:
        ---------
        # Get total cross section (MT=1)
        >>> xs = ace.get_cross_section(1)
        
        # Get cross sections for total, elastic, and fission (MT=18)
        >>> xs = ace.get_cross_section([1, 2, 18])
        
        # Get all available cross sections
        >>> all_xs = ace.get_cross_section()
        
        Parameters:
        -----------
        reaction : int or List[int], optional
            MT number(s) of the reaction(s). If None, returns all available cross sections.
                
        Returns:
        --------
        pd.DataFrame
            DataFrame with energies and cross sections
        
        Raises:
        -------
        ValueError
            If the requested reaction is not available in the data
        TypeError
            If reaction is not an integer or list of integers
        """
        
        # If reaction is None, get all available cross sections
        if reaction is None:
            # Get all available MT numbers, including standard ones and reaction-specific ones
            reaction_list = self.mt_numbers
        # Handle single value or list of reactions
        elif isinstance(reaction, int):
            reaction_list = [reaction]
        elif isinstance(reaction, list) and all(isinstance(r, int) for r in reaction):
            reaction_list = reaction
        else:
            raise TypeError("reaction must be an integer (MT number) or a list of integers")
        
        # For full energy range, return DataFrame
        result = {}
        
        # Add energy column
        if not self.esz_block or not self.esz_block.has_data:
            raise ValueError("Energy grid is not available")
        # Extract values from XssEntry objects
        energy_values = [e.value for e in self.esz_block.energies]
        result["Energy"] = energy_values
        
        # Add standard cross sections and reaction-specific cross sections
        for mt in reaction_list:
            try:
                if mt == 1:  # Total
                    if not self.esz_block.total_xs or len(self.esz_block.total_xs) != len(self.esz_block.energies):
                        raise ValueError(f"Cross section data for MT={mt} (Total) is not available")
                    result[f"MT={mt}"] = [xs.value for xs in self.esz_block.total_xs]
                elif mt == 2:  # Elastic
                    if not self.esz_block.elastic_xs or len(self.esz_block.elastic_xs) != len(self.esz_block.energies):
                        raise ValueError(f"Cross section data for MT={mt} (Elastic) is not available")
                    result[f"MT={mt}"] = [xs.value for xs in self.esz_block.elastic_xs]
                elif mt == 101:  # Absorption
                    if not self.esz_block.absorption_xs or len(self.esz_block.absorption_xs) != len(self.esz_block.energies):
                        raise ValueError(f"Cross section data for MT={mt} (Absorption) is not available")
                    result[f"MT={mt}"] = [xs.value for xs in self.esz_block.absorption_xs]
                else:
                    # Handle reaction-specific cross sections
                    if not self.cross_section or not self.cross_section.has_data:
                        raise ValueError(f"Cross section data for MT={mt} is not available")
                        
                    # FIX: Use the reaction dictionary directly instead of calling a non-existent method
                    reaction_xs = self.cross_section.reaction.get(mt)
                    if not reaction_xs:
                        raise ValueError(f"Cross section data for MT={mt} is not available")
                    
                    # Get the energy index and number of energies for this reaction
                    energy_idx = reaction_xs.energy_idx
                    num_energies = reaction_xs.num_energies
                    
                    # Note: energy_idx is stored as a 0-indexed value, but we need
                    # to use it as an index into the energy_values array
                    if energy_idx < 0 or energy_idx >= len(energy_values) or num_energies <= 0:
                        raise ValueError(f"Invalid energy index or number of energies for MT={mt}")
                    
                    # Extract the cross section values from XssEntry objects
                    rx_xs_values = [xs.value for xs in reaction_xs._xs_entries]
                    
                    # Verify the length matches the declared number of energies
                    if len(rx_xs_values) != num_energies:
                        raise ValueError(f"Mismatch in cross section data length for MT={mt}: "
                                        f"declared {num_energies}, found {len(rx_xs_values)}")
                    
                    # Create array with full energy grid length (all zeros)
                    xs_values = np.zeros(len(energy_values))
                    
                    # Check if we can fit the data into the array
                    # energy_idx is 0-indexed, so we don't need to convert it
                    if energy_idx + num_energies > len(energy_values):
                        raise ValueError(f"Cross section data for MT={mt} would extend beyond energy grid: "
                                        f"start={energy_idx}, length={num_energies}, "
                                        f"grid size={len(energy_values)}, sum={energy_idx+num_energies}")
                    
                    # Populate the array with cross section values
                    # Use energy_idx directly since it's already 0-indexed
                    xs_values[energy_idx:energy_idx+num_energies] = rx_xs_values
                    
                    result[f"MT={mt}"] = xs_values
            except ValueError as e:
                if reaction is not None:
                    raise e
                # If getting all available cross sections, ignore those with errors
        
        return pd.DataFrame(result)

    def plot_cross_section(self, reactions=None, energies=None, ax=None, **kwargs):
        """
        Plot cross sections for specified reactions.
        
        Examples:
        ---------
        # Plot total, elastic, and fission cross sections
        >>> ace.plot_cross_section([1, 2, 18])
        
        # Plot with custom styling
        >>> ace.plot_cross_section([1, 2], linewidth=2, color=['blue', 'red'])
        
        # Plot on an existing axis
        >>> fig, ax = plt.subplots()
        >>> ace.plot_cross_section([18], ax=ax)
        
        Parameters:
        -----------
        reactions : int or list of int, optional
            MT number(s) to plot. Default is [1, 2, 101] (total, elastic, absorption)
        energies : list, optional
            Energy range to plot, default is the full energy grid
        ax : matplotlib.axes.Axes, optional
            Axes to plot on, if None a new figure is created
        **kwargs : dict
            Additional keyword arguments passed to plot function
            
        Returns:
        --------
        matplotlib.axes.Axes
            The matplotlib axes containing the plot
        
        Raises:
        -------
        ValueError
            If any of the requested cross sections are not available
        """
        
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))
                
        if reactions is None:
            reactions = [1, 2, 101]  # Default: total, elastic, absorption
        
        # Convert single reaction to list for consistent handling
        if isinstance(reactions, int):
            reactions = [reactions]
                
        if energies is None and self.energies:
            energies = self.energies
        
        # Get data for all reactions in one go
        try:
            xs_data = self.get_cross_section(reactions)
        except ValueError as e:
            # Add debug information
            print(f"Error in plot_cross_section for reactions {reactions}")
            print(f"Available MT numbers: {self.mt_numbers}")
            
            # If we have the specific reaction that's causing issues:
            if isinstance(reactions, (int, list)) and (isinstance(reactions, int) or len(reactions) == 1):
                mt = reactions if isinstance(reactions, int) else reactions[0]
                if self.cross_section and mt in self.cross_section.reaction:
                    rx = self.cross_section.reaction[mt]
                    print(f"MT={mt} energy_idx={rx.energy_idx}, num_energies={rx.num_energies}")
            
            raise ValueError(f"Could not plot: {str(e)}")
            
        # Plot each reaction
        for mt in reactions:
            ax.plot(xs_data["Energy"], xs_data[f"MT={mt}"], label=f"MT={mt}", **kwargs)
                            
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Energy (MeV)')
        ax.set_ylabel('Cross Section (barns)')
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        ax.legend()
            
        return ax
    

    def update_cross_sections(self):
        cs = self.cross_section.reaction

        # ----------------------------------------------------------------------------
        # 1) Map "feeders → total" ordered deepest first
        # ----------------------------------------------------------------------------
        partial_mt_map = {
            103: range(600, 650),
            104: range(650, 700),
            105: range(700, 750),
            106: range(750, 800),
            107: range(800, 850),

            101: range(102, 118),

            18: list(range(19, 22)) + [38],

            4:  range(50, 92),

            3: (
                [4, 5, 11]
                + list(range(16, 19))
                + list(range(22, 27))
                + list(range(28, 38))
                + [41, 42, 44, 45]
            ),
        }

        # ----------------------------------------------------------------------------
        # 2+3) For each total_mt, we sum from its feeders:
        #       - if cs[mt] exists, we use its values
        #       - if it does not exist but we already calculated computed_totals[mt], we use that
        # ----------------------------------------------------------------------------
        computed_totals = {}
        for total_mt, feeders in partial_mt_map.items():
            energy_sums = defaultdict(float)
            has_feeder = False

            for mt in feeders:
                if mt in cs:
                    has_feeder = True
                    rx = cs[mt]
                    for E, x in zip(rx.energies, rx.xs_values):
                        energy_sums[E] += x

                elif mt in computed_totals and computed_totals[mt]:
                    has_feeder = True
                    for E, x in computed_totals[mt].items():
                        energy_sums[E] += x

                # if it's neither in cs nor in computed_totals, we ignore it

            computed_totals[total_mt] = energy_sums

            # Rebuild the redundant total only when at least one partial is
            # present to sum. With no feeders, the total is stored directly
            # (e.g. MT=18 fission without MT 19/20/21/38) and was already
            # perturbed in place — overwriting it here would zero it out.
            if total_mt in cs:
                if has_feeder:
                    total_rx = cs[total_mt]
                    for E, entry in zip(total_rx.energies, total_rx._xs_entries):
                        entry.value = energy_sums.get(E, 0.0)
                else:
                    logging.debug(
                        "update_cross_sections: MT=%d stored directly (no "
                        "partials present); leaving its values unchanged.",
                        total_mt,
                    )

        # ----------------------------------------------------------------------------
        # 4) Rebuild MT=1 = ΣMT: 2 + 3 + 101
        # ----------------------------------------------------------------------------
        energy_mt1 = defaultdict(float)
        mt1_has_feeder = False
        for mt in (2, 3, 101):
            if mt in cs:
                mt1_has_feeder = True
                rx = cs[mt]
                for E, x in zip(rx.energies, rx.xs_values):
                    energy_mt1[E] += x
            elif computed_totals.get(mt):
                mt1_has_feeder = True
                for E, x in computed_totals[mt].items():
                    energy_mt1[E] += x

        rx1 = cs.get(1)
        if not rx1:
            raise RuntimeError("MT=1 not found in cross section data")
        if mt1_has_feeder:
            for E, entry in zip(rx1.energies, rx1._xs_entries):
                entry.value = energy_mt1.get(E, 0.0)
        else:
            logging.debug(
                "update_cross_sections: MT=1 stored directly (no components "
                "present); leaving its values unchanged."
            )
    
    def to_plot_data(self, data_type: str, mt: int, **kwargs):
        """
        Extract plot data from ACE file in a format compatible with PlotBuilder.
        
        Parameters
        ----------
        data_type : str
            Type of data to extract: 'cross_section' (or 'xs') or 'angular' (or 'ang')
        mt : int
            MT reaction number
        **kwargs
            Additional parameters:
            - For angular distributions: 'energy' (incident energy in MeV)
            - For styling: 'label', 'color', 'linestyle', etc.
            
        Returns
        -------
        PlotData
            PlotData object compatible with PlotBuilder
            
        Examples
        --------
        >>> ace = kika.read_ace('fe56.ace')
        >>> 
        >>> # Extract cross section data (full name or alias)
        >>> xs_data = ace.to_plot_data('cross_section', mt=2)
        >>> xs_data = ace.to_plot_data('xs', mt=2)  # Same as above
        >>> 
        >>> # Extract angular distribution data (full name or alias)
        >>> ang_data = ace.to_plot_data('angular', mt=2, energy=5.0)
        >>> ang_data = ace.to_plot_data('ang', mt=2, energy=5.0)  # Same as above
        >>> 
        >>> # Use with PlotBuilder
        >>> from kika.plotting import PlotBuilder
        >>> fig = (PlotBuilder()
        ...        .add_data(xs_data)
        ...        .set_labels(title='Fe-56 Elastic XS', x_label='Energy (MeV)', y_label='Cross Section (barns)')
        ...        .set_scales(log_x=True, log_y=True)
        ...        .build())
        """
        # Normalize data_type (accept aliases)
        data_type_lower = data_type.lower()
        if data_type_lower in ('cross_section', 'xs'):
            return self._to_plot_data_cross_section(mt, **kwargs)
        elif data_type_lower in ('angular', 'ang'):
            return self._to_plot_data_angular(mt, **kwargs)
        else:
            raise ValueError(f"Unknown data_type: {data_type}. Must be 'cross_section'/'xs' or 'angular'/'ang'")
    
    def _to_plot_data_cross_section(self, mt: int, **kwargs):
        """Extract cross section data for plotting."""
        from kika.plotting import PlotData
        import numpy as np
        
        if not self.cross_section or not self.cross_section.has_data:
            raise ValueError("No cross section data available in this ACE file")
        
        if mt not in self.cross_section.reaction:
            available_mts = self.cross_section.mt_numbers
            raise ValueError(f"MT={mt} not found. Available MT numbers: {available_mts}")
        
        reaction = self.cross_section.reaction[mt]
        energies = reaction.energies  # In MeV
        xs_values = reaction.xs_values  # In barns
        
        # Get label
        label = kwargs.get('label', None)
        if label is None:
            # Create default label from isotope and MT using consistent formatting
            from kika._constants import format_plot_label
            zaid = self.header.zaid if self.header else None
            label = format_plot_label(zaid=zaid, mt=mt)

        return PlotData(
            x=np.array(energies),
            y=np.array(xs_values),
            label=label,
            color=kwargs.get('color', None),
            linestyle=kwargs.get('linestyle', '-'),
            linewidth=kwargs.get('linewidth', None),
            marker=kwargs.get('marker', None),
            markersize=kwargs.get('markersize', None),
            plot_type='line'
        )
    
    def _to_plot_data_angular(self, mt: int, **kwargs):
        """Extract angular distribution data for plotting."""
        from kika.plotting import PlotData
        import numpy as np
        
        if not self.angular_distributions:
            raise ValueError("No angular distribution data available in this ACE file")
        
        # Get energy parameter (required for angular distributions)
        energy = kwargs.get('energy', None)
        if energy is None:
            raise ValueError("'energy' parameter is required for angular distributions")
        
        # Additional parameters for to_dataframe
        particle_type = kwargs.get('particle_type', 'neutron')
        particle_idx = kwargs.get('particle_idx', 0)
        num_points = kwargs.get('num_points', 100)
        interpolate = kwargs.get('interpolate', True)
        normalize_to_xs = kwargs.get('normalize_to_xs', False)
        cross_section_unit = kwargs.get('cross_section_unit', 'b')
        energy_folding = kwargs.get('energy_folding', False)
        tof = kwargs.get('tof', (27.037, 5.0))

        # Get the angular distribution data as a DataFrame
        df_kwargs = dict(
            mt=mt,
            energy=energy,
            particle_type=particle_type,
            particle_idx=particle_idx,
            ace=self,
            num_points=num_points,
            interpolate=interpolate,
            normalize_to_xs=normalize_to_xs,
            cross_section_unit=cross_section_unit,
            energy_folding=energy_folding,
            tof=tof,
        )
        df = self.angular_distributions.to_dataframe(**df_kwargs)

        if df is None:
            raise ValueError(f"Could not extract angular distribution for MT={mt} at energy={energy} MeV")

        # Extract cosine (mu) and the correct y-column from the DataFrame
        mu = df['cosine'].values
        if 'dsigma_dOmega' in df.columns:
            pdf = df['dsigma_dOmega'].values
        elif 'dsigma_dmu' in df.columns:
            pdf = df['dsigma_dmu'].values
        else:
            pdf = df['pdf'].values
        
        # Get label
        label = kwargs.get('label', None)
        if label is None:
            from kika._constants import format_plot_label
            zaid = self.header.zaid if self.header else None
            label = format_plot_label(zaid=zaid, mt=mt, energy=energy)

        return PlotData(
            x=np.array(mu),
            y=np.array(pdf),
            label=label,
            color=kwargs.get('color', None),
            linestyle=kwargs.get('linestyle', '-'),
            linewidth=kwargs.get('linewidth', None),
            marker=kwargs.get('marker', None),
            markersize=kwargs.get('markersize', None),
            plot_type='line'
        )