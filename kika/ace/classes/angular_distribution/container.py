from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, Tuple, Any
import pandas as pd
import numpy as np
from kika.ace.classes.xss import XssEntry
from kika.ace.classes.angular_distribution.base import AngularDistribution
from kika.ace.classes.angular_distribution.utils import (
    ErrorMessageDict,
    ErrorMessageList,
)
from kika._utils import create_repr_section
from kika.ace.classes.angular_distribution.distributions.isotropic import IsotropicAngularDistribution
from kika.ace.classes.angular_distribution.distributions.kalbach_mann import KalbachMannAngularDistribution
from kika.ace.classes.angular_distribution.distributions.equiprobable import EquiprobableAngularDistribution
from kika.ace.classes.angular_distribution.distributions.tabulated import TabulatedAngularDistribution
from kika.ace.classes.angular_distribution.utils import Law44DataError


from kika._constants import (
    NEUTRON_MASS_MEV as _NEUTRON_MASS_MEV,
    SPEED_OF_LIGHT_M_NS as _SPEED_OF_LIGHT_M_PER_NS,
)
from kika.utils.energy_folding import tof_energy_resolution
from kika.utils.numerics import gauss_hermite_nodes

# Default TOF parameters (GELINA facility)
_DEFAULT_FLIGHT_PATH_M = 27.037  # meters
_DEFAULT_DELTA_T_NS = 5.0  # nanoseconds — quoted as a FWHM


def _compute_energy_resolution_tof(
    energy_mev: float,
    flight_path_m: float = _DEFAULT_FLIGHT_PATH_M,
    delta_t_ns: float = _DEFAULT_DELTA_T_NS,
    delta_t_is_fwhm: bool = True,
) -> float:
    """
    Calculate TOF (Time-of-Flight) energy resolution σE in MeV.

    In TOF experiments, neutron energy is determined by measuring time-of-flight.
    This introduces energy uncertainty due to finite time resolution.

    Thin adapter over :func:`kika.utils.energy_folding.tof_energy_resolution`,
    which is the single definition of the formula.

    Energy resolution formula:
        FWHM_E = E × 2 × δt / t,   σE = FWHM_E / 2.3548
    where t = L / v = L / (c × sqrt(2E/m_n))

    Parameters
    ----------
    energy_mev : float
        Neutron energy in MeV
    flight_path_m : float, optional
        Flight path distance in meters (default: 27.037 m, GELINA facility)
    delta_t_ns : float, optional
        Timing spread in nanoseconds (default: 5.0 ns); see ``delta_t_is_fwhm``.
    delta_t_is_fwhm : bool, optional
        Whether ``delta_t_ns`` is a FWHM (default, the usual experimental
        convention) or already a sigma.  The two differ by a factor 2.355.

    Returns
    -------
    float
        Energy resolution σE in MeV

    Examples
    --------
    >>> _compute_energy_resolution_tof(1.0, 27.037, 5.0)  # ~0.0023 MeV
    >>> _compute_energy_resolution_tof(10.0, 27.037, 5.0)  # ~0.073 MeV
    """
    return tof_energy_resolution(
        energy_mev,
        flight_path_m=flight_path_m,
        delta_t_ns=delta_t_ns,
        delta_t_is_fwhm=delta_t_is_fwhm,
    )


def _get_xs_at_energy(ace, mt: int, energy: float) -> Optional[float]:
    """
    Get the cross-section value at a specific energy using linear interpolation.

    Parameters
    ----------
    ace : Ace
        ACE object containing cross-section data
    mt : int
        MT reaction number
    energy : float
        Energy in MeV

    Returns
    -------
    float or None
        Cross-section value in barns, or None if not available
    """
    if ace is None:
        return None

    # Handle standard reactions stored in esz_block
    if mt == 1 and hasattr(ace, 'esz_block') and ace.esz_block and ace.esz_block.total_xs:
        energies = [e.value for e in ace.esz_block.energies]
        xs_values = [x.value for x in ace.esz_block.total_xs]
    elif mt == 2 and hasattr(ace, 'esz_block') and ace.esz_block and ace.esz_block.elastic_xs:
        energies = [e.value for e in ace.esz_block.energies]
        xs_values = [x.value for x in ace.esz_block.elastic_xs]
    elif mt == 101 and hasattr(ace, 'esz_block') and ace.esz_block and ace.esz_block.absorption_xs:
        energies = [e.value for e in ace.esz_block.energies]
        xs_values = [x.value for x in ace.esz_block.absorption_xs]
    elif hasattr(ace, 'cross_section') and ace.cross_section and mt in ace.cross_section.reaction:
        reaction = ace.cross_section.reaction[mt]
        energies = reaction.energies
        xs_values = reaction.xs_values
    else:
        return None

    if not energies or not xs_values:
        return None

    # Linear interpolation
    if energy <= energies[0]:
        return xs_values[0]
    elif energy >= energies[-1]:
        return xs_values[-1]
    else:
        return float(np.interp(energy, energies, xs_values))


@dataclass
class AngularDistributionContainer:
    """Container for all angular distributions."""
    elastic: Optional[AngularDistribution] = None  # Angular distribution for elastic scattering
    incident_neutron: Dict[int, AngularDistribution] = field(default_factory=dict)  # MT -> distribution for neutrons
    photon_production: Dict[int, AngularDistribution] = field(default_factory=dict)  # MT -> distribution for photons
    particle_production: List[Dict[int, AngularDistribution]] = field(default_factory=list)  # Particle index -> (MT -> distribution)
    
    def __post_init__(self):
        """Convert standard dictionaries to ErrorMessageDict and lists to ErrorMessageList."""
        # Convert incident_neutron dictionary to ErrorMessageDict
        if isinstance(self.incident_neutron, dict) and not isinstance(self.incident_neutron, ErrorMessageDict):
            self.incident_neutron = ErrorMessageDict(self.incident_neutron, dict_name="incident_neutron distributions")
        
        # Convert photon_production dictionary to ErrorMessageDict
        if isinstance(self.photon_production, dict) and not isinstance(self.photon_production, ErrorMessageDict):
            self.photon_production = ErrorMessageDict(self.photon_production, dict_name="photon_production distributions")
        
        # Convert particle_production list to ErrorMessageList
        if isinstance(self.particle_production, list) and not isinstance(self.particle_production, ErrorMessageList):
            # First convert the list itself
            particle_list = ErrorMessageList(self.particle_production, list_name="particle_production")
            
            # Then convert each dictionary in the list
            for i in range(len(particle_list)):
                if isinstance(particle_list[i], dict) and not isinstance(particle_list[i], ErrorMessageDict):
                    particle_list[i] = ErrorMessageDict(
                        particle_list[i], 
                        dict_name=f"particle_production[{i}] distributions"
                    )
            
            self.particle_production = particle_list
    
    @property
    def has_elastic_data(self) -> bool:
        """Check if elastic scattering angular distribution data is available."""
        return self.elastic is not None and not self.elastic.is_isotropic
    
    @property
    def has_neutron_data(self) -> bool:
        """Check if neutron reaction angular distribution data is available."""
        return len(self.incident_neutron) > 0
    
    @property
    def has_photon_production_data(self) -> bool:
        """Check if photon production angular distribution data is available."""
        return len(self.photon_production) > 0
    
    @property
    def has_particle_production_data(self) -> bool:
        """Check if particle production angular distribution data is available."""
        return len(self.particle_production) > 0 and any(len(p) > 0 for p in self.particle_production)
    
    def get_neutron_reaction_mt_numbers(self) -> List[int]:
        """Get the list of MT numbers for neutron reactions with angular distributions."""
        if isinstance(self.incident_neutron, ErrorMessageDict):
            return sorted(self.incident_neutron.keys_as_int())
        else:
            return sorted(list(self.incident_neutron.keys()))
    
    def get_photon_production_mt_numbers(self) -> List[int]:
        """Get the list of MT numbers for photon production with angular distributions."""
        if isinstance(self.photon_production, ErrorMessageDict):
            return sorted(self.photon_production.keys_as_int())
        else:
            return sorted(list(self.photon_production.keys()))
    
    def get_particle_production_mt_numbers(self, particle_idx: Optional[int] = None) -> Union[Dict[int, List[int]], List[int]]:
        """
        Get the list of MT numbers for particle production with angular distributions.
        
        Parameters
        ----------
        particle_idx : int, optional
            Index of the particle type. If None, returns a dictionary mapping
            particle indices to their MT numbers
            
        Returns
        -------
        Dict[int, List[int]] or List[int]
            If particle_idx is None: Dictionary mapping particle indices to lists of MT numbers
            If particle_idx is given: List of MT numbers for that particle index
        
        Raises
        ------
        IndexError
            If the specified particle index is out of bounds
        """
        # If no particle_idx specified, return dictionary for all particles
        if particle_idx is None:
            result = {}
            for idx in range(len(self.particle_production)):
                particle_data = self.particle_production[idx]
                if isinstance(particle_data, ErrorMessageDict):
                    result[idx] = sorted(particle_data.keys_as_int())
                else:
                    mt_keys = particle_data.keys()
                    if mt_keys and isinstance(next(iter(mt_keys)), XssEntry):
                        result[idx] = sorted([int(mt.value) for mt in mt_keys])
                    else:
                        result[idx] = sorted(list(mt_keys))
            return result
            
        # If particle_idx is specified, return list for that particle
        if particle_idx < 0 or particle_idx >= len(self.particle_production):
            available_indices = list(range(len(self.particle_production)))
            error_message = f"Particle index {particle_idx} is out of bounds."
            
            if len(self.particle_production) == 0:
                error_message += " No particle production data is available."
            else:
                error_message += f" Available particle indices: {available_indices}"
                
                # Add more information about particle counts for each index
                error_message += "\nParticle counts by index:"
                for idx, particle_data in enumerate(self.particle_production):
                    error_message += f"\n  Index {idx}: {len(particle_data)} reactions"
            
            raise IndexError(error_message)
        
        particle_data = self.particle_production[particle_idx]
        
        # Extract the MT values from XssEntry objects before sorting
        if isinstance(particle_data, ErrorMessageDict):
            return sorted(particle_data.keys_as_int())
        else:
            mt_keys = particle_data.keys()
            
            # Check if the keys are XssEntry objects or integers
            if mt_keys and isinstance(next(iter(mt_keys)), XssEntry):
                # If they are XssEntry objects, get their values first
                return sorted([int(mt.value) for mt in mt_keys])
            else:
                # If they are already integers, sort them directly
                return sorted(list(mt_keys))
    
    def get_particle_production_info(self) -> Dict[int, Dict[str, Any]]:
        """
        Get comprehensive information about particle production angular distributions.
        
        This provides a more detailed and user-friendly version of the data compared to
        get_particle_production_mt_numbers().
        
        Returns
        -------
        Dict[int, Dict[str, Any]]
            Dictionary mapping particle indices to dictionaries containing:
            - 'mt_numbers': List of MT numbers
            - 'count': Total number of reactions
            - 'distribution_types': Dictionary counting each distribution type
            - 'description': Text description
            
        Examples
        --------
        >>> info = container.get_particle_production_info()
        >>> for idx, data in info.items():
        ...     print(f"Particle {idx}: {data['count']} reactions, {data['description']}")
        ...     print(f"MT numbers: {data['mt_numbers']}")
        """
        result = {}
        
        for idx in range(len(self.particle_production)):
            particle_data = self.particle_production[idx]
            
            # Get MT numbers for this particle
            if isinstance(particle_data, ErrorMessageDict):
                mt_numbers = sorted(particle_data.keys_as_int())
            else:
                mt_keys = particle_data.keys()
                if mt_keys and isinstance(next(iter(mt_keys)), XssEntry):
                    mt_numbers = sorted([int(mt.value) for mt in mt_keys])
                else:
                    mt_numbers = sorted(list(mt_keys))
            
            # Count distribution types
            distribution_types = {}
            for mt in mt_numbers:
                dist = particle_data[mt]
                dist_type = dist.distribution_type.name
                distribution_types[dist_type] = distribution_types.get(dist_type, 0) + 1
            
            # Create a description
            description = f"{len(mt_numbers)} reactions"
            if distribution_types:
                type_str = ", ".join(f"{count} {dist_type.lower()}" 
                                   for dist_type, count in distribution_types.items())
                description += f" ({type_str})"
            
            # Store all information
            result[idx] = {
                'mt_numbers': mt_numbers,
                'count': len(mt_numbers),
                'distribution_types': distribution_types,
                'description': description
            }
            
        return result

    def _apply_energy_folding(
        self,
        mt: int,
        target_energy: float,
        sigma_E: float,
        ace,
        particle_type: str,
        particle_idx: int,
        num_points: int,
        normalize_to_xs: bool,
        cross_section_unit: str,
        n_sigma: float = 3.0,
        n_samples: int = 21,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Gaussian energy folding to angular distribution.

        This method convolves the angular distribution with a Gaussian kernel
        in energy space to account for TOF energy resolution.

        Parameters
        ----------
        mt : int
            MT reaction number
        target_energy : float
            Target energy in MeV around which to fold
        sigma_E : float
            Energy resolution (standard deviation) in MeV
        ace : Ace
            ACE object containing cross-section and distribution data
        particle_type : str
            Type of particle: 'neutron', 'photon', or 'particle'
        particle_idx : int
            Particle index for particle_type='particle'
        num_points : int
            Number of angular points to generate
        normalize_to_xs : bool
            If True, return differential cross-section instead of PDF
        cross_section_unit : str
            Unit for differential cross-section
        n_sigma : float, optional
            Number of sigma to extend folding window (default: 3.0)
        n_samples : int, optional
            Number of energy samples for numerical integration (default: 21)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (cosine, values) arrays with folded angular distribution
        """
        # Get ACE energy bounds
        if hasattr(ace, 'esz_block') and ace.esz_block and ace.esz_block.energies:
            e_min = ace.esz_block.energies[0].value
            e_max = ace.esz_block.energies[-1].value
        else:
            # Fallback: use a reasonable range
            e_min = 1e-11
            e_max = 20.0

        # Gauss-Hermite quadrature over the resolution kernel — the same nodes
        # kika.utils.numerics.fold_tabulated uses, so the cross-section and
        # angular folding paths share one scheme. This replaces a uniform
        # n_samples-point Riemann sum over +-n_sigma, which needed many more
        # evaluations for the same accuracy.
        sample_energies, weights = gauss_hermite_nodes(
            target_energy, sigma_E, n_nodes=n_samples,
        )
        # Nodes are unbounded, so clamp into the ACE range and renormalise.
        # Only matters within a few sigma of the table edges.
        sample_energies = np.clip(sample_energies, e_min, e_max)
        weights = weights / weights.sum()

        # Initialize accumulators
        cosine_grid = None
        weighted_values = None

        for i, sample_energy in enumerate(sample_energies):
            # Get angular distribution at this energy (always interpolate for consistent grid)
            try:
                df = self.to_dataframe(
                    mt=mt,
                    energy=sample_energy,
                    particle_type=particle_type,
                    particle_idx=particle_idx,
                    ace=ace,
                    num_points=num_points,
                    interpolate=True,
                    normalize_to_xs=normalize_to_xs,
                    cross_section_unit=cross_section_unit,
                    energy_folding=False,  # Prevent recursion
                )
            except Exception:
                # Skip this energy point if distribution is not available
                continue

            if df is None:
                continue

            # Extract values
            cosines = df['cosine'].values
            if 'dsigma_dOmega' in df.columns:
                values = df['dsigma_dOmega'].values
            elif 'dsigma_dmu' in df.columns:
                values = df['dsigma_dmu'].values
            else:
                values = df['pdf'].values

            # Initialize on first valid sample
            if cosine_grid is None:
                cosine_grid = cosines
                weighted_values = np.zeros_like(values)

            # Accumulate weighted values
            weighted_values += weights[i] * values

        if cosine_grid is None or weighted_values is None:
            raise ValueError(
                f"Could not compute energy-folded distribution for MT={mt} "
                f"at energy={target_energy} MeV"
            )

        return cosine_grid, weighted_values

    def to_dataframe(self, mt: int, energy: float, particle_type: str = 'neutron',
                    particle_idx: int = 0, ace=None, num_points: int = 100,
                    interpolate: bool = True, normalize_to_xs: bool = False,
                    cross_section_unit: str = 'b',
                    energy_folding: bool = False,
                    tof: Tuple[float, float] = (27.037, 5.0)) -> Optional[pd.DataFrame]:
        """
        Convert an angular distribution to a pandas DataFrame.

        Parameters
        ----------
        mt : int
            MT number for the reaction
        energy : float
            Incident energy to evaluate the distribution at
        particle_type : str, optional
            Type of particle: 'neutron', 'photon', or 'particle'
        particle_idx : int, optional
            Index of the particle type (used only for particle_type='particle')
        ace : Ace, optional
            ACE object containing the distribution data (required for Kalbach-Mann,
            normalize_to_xs=True, and energy_folding=True)
        num_points : int, optional
            Number of angular points to generate when interpolating, defaults to 100
        interpolate : bool, optional
            Whether to interpolate onto a regular grid (True) or return original points (False)
        normalize_to_xs : bool, optional
            If True, multiply PDF by the cross-section at the given energy to get
            the differential cross-section. Requires ace parameter.
            Default is False (returns normalized PDF).
        cross_section_unit : str, optional
            Unit for differential cross-section when normalize_to_xs=True:
            - 'b' or 'barns': returns dsigma/dmu in barns (default)
            - 'b/sr' or 'barns/sr': returns dsigma/dOmega in barns/steradian
            The conversion is: dsigma/dOmega = dsigma/dmu / (2*pi)
        energy_folding : bool, optional
            If True, apply Gaussian energy folding to account for TOF energy resolution.
            Requires ace parameter. Default is False.
        tof : tuple of (float, float), optional
            TOF parameters as (flight_path_m, delta_t_ns). Default is (27.037, 5.0)
            corresponding to GELINA facility parameters.

        Returns
        -------
        pandas.DataFrame or None
            DataFrame with 'energy', 'cosine', and 'pdf' (or 'dsigma_dmu'/'dsigma_dOmega') columns.
            If normalize_to_xs=True, 'pdf' column is replaced with differential cross-section values.
            Returns None if pandas is not available

        Raises
        ------
        Law44DataError
            If trying to process a Kalbach-Mann distribution without providing ACE data
        KeyError
            If the MT number is not found in the distribution container
        ValueError
            If the particle type is unknown, or if normalize_to_xs=True but ace is None,
            or if energy_folding=True but ace is None
        """
        # Validate cross_section_unit
        valid_units = ('b', 'barns', 'b/sr', 'barns/sr')
        if cross_section_unit not in valid_units:
            raise ValueError(f"Invalid cross_section_unit '{cross_section_unit}'. "
                           f"Valid options: {valid_units}")

        # Handle energy folding if requested
        if energy_folding:
            if ace is None:
                raise ValueError("ACE object required for energy folding. "
                               "Pass ace=ace_object when calling to_dataframe().")

            # Compute energy resolution using TOF parameters
            flight_path_m, delta_t_ns = tof
            sigma_E = _compute_energy_resolution_tof(energy, flight_path_m, delta_t_ns)

            # Apply energy folding
            cosine_grid, folded_values = self._apply_energy_folding(
                mt=mt,
                target_energy=energy,
                sigma_E=sigma_E,
                ace=ace,
                particle_type=particle_type,
                particle_idx=particle_idx,
                num_points=num_points,
                normalize_to_xs=normalize_to_xs,
                cross_section_unit=cross_section_unit,
            )

            # Build DataFrame from folded data
            df = pd.DataFrame({
                'energy': energy,
                'cosine': cosine_grid,
            })

            # Add appropriate value column based on cross_section_unit
            if normalize_to_xs:
                if cross_section_unit in ('b/sr', 'barns/sr'):
                    df['dsigma_dOmega'] = folded_values
                else:
                    df['dsigma_dmu'] = folded_values
            else:
                df['pdf'] = folded_values

            df['particle_type'] = particle_type
            df['mt'] = mt
            if particle_type == 'particle':
                df['particle_idx'] = particle_idx

            return df

        # Special case for elastic scattering (MT=2)
        if particle_type == 'neutron' and mt == 2 and self.elastic:
            df = self.elastic.to_dataframe(energy, num_points, interpolate)
            if df is not None:
                df['particle_type'] = particle_type
                df['mt'] = mt
                # Apply cross-section normalization if requested
                if normalize_to_xs:
                    if ace is None:
                        raise ValueError("ACE object required for cross-section normalization. "
                                       "Pass ace=ace_object when calling to_dataframe().")
                    xs_value = _get_xs_at_energy(ace, mt, energy)
                    if xs_value is not None and xs_value > 0:
                        dsigma_dmu = df['pdf'] * xs_value
                        # Convert to b/sr if requested
                        if cross_section_unit in ('b/sr', 'barns/sr'):
                            df['dsigma_dOmega'] = dsigma_dmu / (2 * np.pi)
                        else:
                            df['dsigma_dmu'] = dsigma_dmu
                        df = df.drop(columns=['pdf'])
                    else:
                        import warnings
                        warnings.warn(f"Could not get cross-section for MT={mt} at E={energy} MeV. "
                                    "Returning normalized PDF instead.")
            return df
        
        # Get the appropriate distribution container
        if particle_type == 'neutron':
            dist_container = self.incident_neutron
        elif particle_type == 'photon':
            dist_container = self.photon_production
        elif particle_type == 'particle':
            if particle_idx < 0 or particle_idx >= len(self.particle_production):
                raise ValueError(f"Particle index {particle_idx} out of bounds")
            dist_container = self.particle_production[particle_idx]
        else:
            raise ValueError(f"Unknown particle type: {particle_type}")
        
        # Get the angular distribution for this MT number
        if mt not in dist_container:
            raise KeyError(f"MT={mt} not found in {particle_type} angular distributions")
        
        # Add information about the particle type and MT number to the dataframe
        distribution = dist_container[mt]
        df = None
        
        # Special handling for Kalbach-Mann distributions
        if isinstance(distribution, KalbachMannAngularDistribution):
            df = distribution.to_dataframe(energy, ace, num_points, interpolate=True)
        else:
            df = distribution.to_dataframe(energy, num_points, interpolate)
            
        if df is not None:
            # Add columns for particle type and MT
            df['particle_type'] = particle_type
            df['mt'] = mt
            if particle_type == 'particle':
                df['particle_idx'] = particle_idx

            # Apply cross-section normalization if requested
            if normalize_to_xs:
                if ace is None:
                    raise ValueError("ACE object required for cross-section normalization. "
                                   "Pass ace=ace_object when calling to_dataframe().")

                xs_value = _get_xs_at_energy(ace, mt, energy)
                if xs_value is not None and xs_value > 0:
                    dsigma_dmu = df['pdf'] * xs_value
                    # Convert to b/sr if requested
                    if cross_section_unit in ('b/sr', 'barns/sr'):
                        df['dsigma_dOmega'] = dsigma_dmu / (2 * np.pi)
                    else:
                        df['dsigma_dmu'] = dsigma_dmu
                    df = df.drop(columns=['pdf'])
                else:
                    import warnings
                    warnings.warn(f"Could not get cross-section for MT={mt} at E={energy} MeV. "
                                "Returning normalized PDF instead.")

            return df
        return None

    def to_plot_data(self, mt: int, energy: float, particle_type: str = 'neutron',
                    particle_idx: int = 0, ace=None, **kwargs):
        """
        Extract angular distribution plot data in a format compatible with PlotBuilder.

        This method provides direct access to angular distribution data without going through
        the parent Ace object. It's equivalent to calling ace.to_plot_data('ang', mt=mt, energy=energy).

        Parameters
        ----------
        mt : int
            MT reaction number
        energy : float
            Incident energy in MeV at which to evaluate the distribution
        particle_type : str, optional
            Type of particle: 'neutron', 'photon', or 'particle' (default: 'neutron')
        particle_idx : int, optional
            Particle index for particle_type='particle' (default: 0)
        ace : Ace, optional
            ACE object (required for Kalbach-Mann distributions and normalize_to_xs=True)
        **kwargs
            Additional parameters for styling and data extraction:
            - label (str): Custom label (default: auto-generated)
            - color (str): Line color
            - linestyle (str): Line style ('-', '--', '-.', ':')
            - linewidth (float): Line width
            - marker (str): Marker style ('o', 's', '^', etc.)
            - markersize (float): Marker size
            - num_points (int): Number of angular points when interpolating (default: 100)
            - interpolate (bool): Whether to interpolate onto regular grid (default: False)
            - normalize_to_xs (bool): If True, return differential cross-section
              instead of normalized PDF. Requires ace parameter. (default: False)
            - cross_section_unit (str): Unit for differential cross-section when
              normalize_to_xs=True. Options: 'b'/'barns' for dsigma/dmu, or
              'b/sr'/'barns/sr' for dsigma/dOmega. (default: 'b')
            - energy_folding (bool): If True, apply Gaussian energy folding to account
              for TOF energy resolution. Requires ace parameter. (default: False)
            - tof (tuple): TOF parameters as (flight_path_m, delta_t_ns).
              (default: (27.037, 5.0) - GELINA facility parameters)

        Returns
        -------
        PlotData
            PlotData object compatible with PlotBuilder

        Examples
        --------
        >>> ace = kika.read_ace('fe56.ace')
        >>>
        >>> # Direct access from angular_distributions object
        >>> ang_data = ace.angular_distributions.to_plot_data(mt=2, energy=5.0, label='5 MeV')
        >>>
        >>> # Use with PlotBuilder
        >>> from kika.plotting import PlotBuilder
        >>> fig = (PlotBuilder()
        ...        .add_data(ang_data)
        ...        .set_labels(x_label='cos(θ)', y_label='Probability Density')
        ...        .build())
        >>>
        >>> # Get differential cross-section in b/sr (for comparison with EXFOR)
        >>> ang_xs = ace.angular_distributions.to_plot_data(
        ...     mt=2, energy=5.0, ace=ace, normalize_to_xs=True,
        ...     cross_section_unit='b/sr', label='5 MeV')
        >>>
        >>> # With energy folding (default TOF parameters from GELINA)
        >>> ace_folded = ace.angular_distributions.to_plot_data(
        ...     mt=2, energy=1.3, ace=ace, normalize_to_xs=True,
        ...     cross_section_unit='b/sr', energy_folding=True, label='ACE (folded)')
        >>>
        >>> # With custom TOF parameters (e.g., ORELA facility)
        >>> ace_orela = ace.angular_distributions.to_plot_data(
        ...     mt=2, energy=1.3, ace=ace, normalize_to_xs=True,
        ...     cross_section_unit='b/sr', energy_folding=True,
        ...     tof=(40.0, 8.0), label='ACE (ORELA folding)')
        """
        from kika.plotting import AngularDistributionPlotData
        import numpy as np

        # Additional parameters for to_dataframe
        num_points = kwargs.pop('num_points', 100)
        interpolate = kwargs.pop('interpolate', False)
        normalize_to_xs = kwargs.pop('normalize_to_xs', False)
        cross_section_unit = kwargs.pop('cross_section_unit', 'b')
        energy_folding = kwargs.pop('energy_folding', False)
        tof = kwargs.pop('tof', (27.037, 5.0))

        # Get the angular distribution data as a DataFrame
        df = self.to_dataframe(
            mt=mt,
            energy=energy,
            particle_type=particle_type,
            particle_idx=particle_idx,
            ace=ace,
            num_points=num_points,
            interpolate=interpolate,
            normalize_to_xs=normalize_to_xs,
            cross_section_unit=cross_section_unit,
            energy_folding=energy_folding,
            tof=tof
        )

        if df is None:
            raise ValueError(f"Could not extract angular distribution for MT={mt} at energy={energy} MeV")

        # Extract cosine (mu) and y values from the DataFrame
        mu = df['cosine'].values
        if 'dsigma_dOmega' in df.columns:
            y_values = df['dsigma_dOmega'].values
        elif 'dsigma_dmu' in df.columns:
            y_values = df['dsigma_dmu'].values
        else:
            y_values = df['pdf'].values

        # Get label
        label = kwargs.get('label', None)
        if label is None:
            # Create default label
            label = f"MT={mt} @ {energy} MeV"

        return AngularDistributionPlotData(
            x=np.array(mu),
            y=np.array(y_values),
            label=label,
            color=kwargs.get('color', None),
            linestyle=kwargs.get('linestyle', '-'),
            linewidth=kwargs.get('linewidth', None),
            marker=kwargs.get('marker', None),
            markersize=kwargs.get('markersize', None),
            plot_type='line',
            energy=energy,
            mt=mt,
        )
    
    def __repr__(self) -> str:
        """
        Returns a user-friendly, formatted string representation of the container.
        
        Returns
        -------
        str
            Formatted string representation
        """
        header_width = 90
        header = "=" * header_width + "\n"
        header += f"{'Angular Distribution Container':^{header_width}}\n"
        header += "=" * header_width + "\n\n"
        
        description = (
            "This container holds angular distributions read directly from the ACE file.\n"
            "Each distribution preserves the original data structure as found in the ACE format.\n\n"
            "Angular distributions describe the probability of scattering as a function of the\n"
            "cosine of the scattering angle (μ), which ranges from -1 (backward scattering) to\n"
            "+1 (forward scattering).\n\n"
        )
        
        # Create a summary table of available data
        property_col_width = 40
        value_col_width = header_width - property_col_width - 3
        
        info_table = "Available Angular Distribution Data:\n"
        info_table += "-" * header_width + "\n"
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Distribution Type", "Status", width1=property_col_width, width2=value_col_width)
        info_table += "-" * header_width + "\n"
        
        # Elastic scattering
        elastic_status = "Available"
        if not self.has_elastic_data:
            elastic_status = "Not available or isotropic"
        else:
            elastic_type = self.elastic.distribution_type.name
            elastic_status = f"Available ({elastic_type})"
        
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Elastic Scattering (MT=2)", elastic_status,
            width1=property_col_width, width2=value_col_width)
        
        # Neutron reaction distributions
        neutron_status = f"Available ({len(self.incident_neutron)} reactions)"
        if not self.has_neutron_data:
            neutron_status = "Not available"
        
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Neutron Reactions", neutron_status,
            width1=property_col_width, width2=value_col_width)
        
        if self.has_neutron_data:
            # Count distribution types
            dist_types = {}
            for mt, dist in self.incident_neutron.items():
                dist_type = dist.distribution_type.name
                dist_types[dist_type] = dist_types.get(dist_type, 0) + 1
            
            # Add distribution type counts
            for dist_type, count in dist_types.items():
                info_table += "{:<{width1}} {:<{width2}}\n".format(
                    f"  {dist_type}", f"{count} reactions",
                    width1=property_col_width, width2=value_col_width)
        
        # Photon production distributions
        photon_status = f"Available ({len(self.photon_production)} reactions)"
        if not self.has_photon_production_data:
            photon_status = "Not available"
        
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Photon Production", photon_status,
            width1=property_col_width, width2=value_col_width)
        
        if self.has_photon_production_data:
            # Count distribution types
            dist_types = {}
            for mt, dist in self.photon_production.items():
                dist_type = dist.distribution_type.name
                dist_types[dist_type] = dist_types.get(dist_type, 0) + 1
            
            # Add distribution type counts
            for dist_type, count in dist_types.items():
                info_table += "{:<{width1}} {:<{width2}}\n".format(
                    f"  {dist_type}", f"{count} reactions",
                    width1=property_col_width, width2=value_col_width)
        
        # Particle production distributions
        if self.has_particle_production_data:
            particle_counts = [len(p) for p in self.particle_production]
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "Particle Production", f"{len(self.particle_production)} types, {sum(particle_counts)} total reactions",
                width1=property_col_width, width2=value_col_width)
            
            # Add details for each particle type
            for i, particle_dict in enumerate(self.particle_production):
                if len(particle_dict) > 0:
                    info_table += "{:<{width1}} {:<{width2}}\n".format(
                        f"  Particle {i}", f"{len(particle_dict)} reactions",
                        width1=property_col_width, width2=value_col_width)
        else:
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "Particle Production", "Not available",
                width1=property_col_width, width2=value_col_width)
        
        info_table += "-" * header_width + "\n\n"
        
        # Add property and method info without examples
        properties_section = (
            "Accessing Angular Distributions:\n"
            f"{'-' * header_width}\n"
            ".elastic                           Elastic scattering distribution (if available)\n"
            ".incident_neutron[mt]              Get neutron reaction distribution by MT number\n"
            ".photon_production[mt]             Get photon production distribution by MT number\n"
            ".particle_production[part_idx][mt] Get particle production distribution by index and MT\n"
            ".get_neutron_reaction_mt_numbers() Get list of available neutron reaction MT numbers\n"
            ".get_photon_production_mt_numbers() Get list of available photon production MT numbers\n\n"
        )
        
        return header + description + info_table + properties_section
