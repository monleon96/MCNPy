from dataclasses import dataclass, field
from kika._constants import ATOMIC_NUMBER_TO_SYMBOL, MT_TO_REACTION
from typing import Dict, Union, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from kika.energy_grids.utils import _identify_energy_grid


@dataclass
class TaylorCoefficients:
    """Container for Taylor series expansion coefficients.
    
    :ivar energy: Energy range string in format "lower_upper" (e.g., "0.00e+00_1.00e-01")
    :type energy: str
    :ivar reaction: Reaction number
    :type reaction: int
    :ivar pert_energies: Perturbation energy boundaries
    :type pert_energies: List[float]
    :ivar c1: First-order Taylor coefficients
    :type c1: List[float]
    :ivar c2: Second-order Taylor coefficients
    :type c2: List[float]
    :ivar ratio: Ratio of c2/c1 for each energy bin
    :type ratio: List[float]
    :ivar c2_errors: Errors of second-order Taylor coefficients
    :type c2_errors: List[float]
    :ivar c1_errors: Errors of first-order Taylor coefficients
    :type c1_errors: List[float]
    """
    energy: str
    reaction: int
    pert_energies: list[float]
    c1: list[float]
    c2: list[float]
    ratio: list[float]
    c2_errors: list[float]
    c1_errors: list[float]
    
    def calculate_nonlinearity(self, p: float) -> float:
        """Calculate the nonlinearity factor at a specific perturbation.
        
        The nonlinearity factor is (c2*p)/c1, which represents the ratio of 
        second-order to first-order term at perturbation magnitude p.
        
        :param p: Perturbation magnitude (0 to 100%)
        :type p: float
        :returns: Average nonlinearity across all energy bins
        :rtype: float
        """
        valid_ratios = [r for r in self.ratio if not np.isnan(r)]
        if not valid_ratios:
            return float('nan')
        # Convert p from percent to fraction (p/100)
        p_fraction = p / 100.0
        return np.mean(valid_ratios) * p_fraction
    
    def calculate_nonlinearity_by_bin(self, p: float) -> list:
        """Calculate the nonlinearity factor for each energy bin at specific perturbation.
        
        The nonlinearity factor is (c2*p)/c1, which represents the ratio of 
        second-order to first-order term at perturbation magnitude p.
        
        :param p: Perturbation magnitude (0 to 100%)
        :type p: float
        :returns: Nonlinearity factor for each energy bin
        :rtype: list
        """
        # Convert p from percent to fraction (p/100)
        p_fraction = p / 100.0
        return [r * p_fraction if not np.isnan(r) else np.nan for r in self.ratio]
    
    def plot(self, ax=None, title=None, top_n=5):
        """Plot the nonlinearity factor vs perturbation magnitude.
        
        The nonlinearity factor is plotted using absolute values for better comparison,
        as the sign of the ratio doesn't provide valuable information in this context.
        
        :param ax: Optional existing axis to plot on
        :type ax: matplotlib.axes.Axes, optional
        :param title: Optional custom title for the plot
        :type title: str, optional
        :param top_n: Number of top absolute ratios to plot (0 means plot all)
        :type top_n: int, optional
        :returns: The axis containing the plot
        :rtype: matplotlib.axes.Axes
        """
        if ax is None:
            # Use smaller figure size to match SensitivityData.plot_ratio()
            fig, ax = plt.subplots(figsize=(8, 4))  # Reduced from (10, 6) to (8, 4)
        
        # Calculate x values (perturbation magnitudes in percent)
        p_values = np.linspace(0, 20, 100)
        
        # Calculate nonlinearity for each energy bin and identify top N ratios by absolute value
        # Include zeros as valid candidates for top_n
        abs_ratios = [abs(r) if not np.isnan(r) else 0 for r in self.ratio]
        sorted_indices = np.argsort(abs_ratios)[::-1]  # Sort in descending order
        
        # If top_n is 0, plot all with labels
        if top_n == 0:
            top_indices = list(range(len(self.ratio)))
        else:
            top_indices = sorted_indices[:min(top_n, len(abs_ratios))]  # Get top N (or fewer if there aren't N)
        
        # Only plot lines for top_n energy bins
        for i in top_indices:
            if np.isnan(self.ratio[i]):
                continue
                
            # Calculate the nonlinearity values for this bin across p_values
            # Convert p_values from percent to fraction for calculation
            # Use absolute values for plotting
            abs_ratio = abs(self.ratio[i])
            y_values = [p/100.0 * abs_ratio * 100 for p in p_values]  # Convert to percentage
            # Include the actual ratio value in the label
            label = f"{self.pert_energies[i]:.2e}-{self.pert_energies[i+1]:.2e} MeV  ({abs_ratio:.2e})"
            
            ax.plot(p_values, y_values, label=label)
        
        # Add a horizontal line at 0
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.2)
        
        # Set plot style
        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title(f"MT {self.reaction} - {self.energy} MeV", fontsize=14)
        
        ax.set_xlabel("Perturbation (%)", fontsize=12)
        ax.set_ylabel("Ratio (c2*p/c1) (%)", fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add legend only if we have any top ratios
        if len(top_indices) > 0:
            legend_title = "All Energy Bins" if top_n == 0 else f"Top {len(top_indices)} Energy Bins"
            ax.legend(title=legend_title, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        return ax
    
    def __repr__(self):
        """Returns a formatted string representation of the Taylor coefficient data.
        
        This method is called when the object is evaluated in interactive environments
        like Jupyter notebooks or the Python interpreter.
        
        :return: Formatted string representation of the Taylor coefficient data
        :rtype: str
        """
        # Create a visually appealing header with a border
        header_width = 70
        header = "=" * header_width + "\n"
        header += f"{'Taylor Coefficient Data - MT ' + str(self.reaction):^{header_width}}\n"
        header += "=" * header_width + "\n\n"
        
        # Create aligned key-value pairs with consistent width
        label_width = 28  # Width for labels
        
        # Basic information
        info_lines = []
        info_lines.append(f"{'Energy Range:':{label_width}} {self.energy}")
        info_lines.append(f"{'Reaction Number (MT):':{label_width}} {self.reaction}")
        info_lines.append(f"{'Number of perturbation bins:':{label_width}} {len(self.ratio)}")
        
        # Add energy grid structure identification
        grid_name = _identify_energy_grid(self.pert_energies)
        if grid_name:
            info_lines.append(f"{'Energy structure:':{label_width}} {grid_name}")
        
        # Calculate average ratio and identify bins with largest absolute ratio
        valid_ratios = [r for r in self.ratio if not np.isnan(r)]
        if valid_ratios:
            avg_ratio = np.mean(valid_ratios)
            max_idx = np.nanargmax(np.abs(self.ratio))
            max_ratio = self.ratio[max_idx]
            info_lines.append(f"{'Average c2/c1 ratio:':{label_width}} {avg_ratio:.6e}")
            info_lines.append(f"{'Max |c2/c1| ratio:':{label_width}} {max_ratio:.6e}")
            info_lines.append(f"{' - at energy bin:':{label_width}} {self.pert_energies[max_idx]:.2e}-{self.pert_energies[max_idx+1]:.2e} MeV")
        
        stats = "\n".join(info_lines)
        
        # Data preview section - show first few and last few values
        n_preview = 3  # Number of values to show at beginning and end
        n_values = len(self.ratio)
        
        data_preview = "\n\nData preview:\n\n"
        
        # Format as a small table
        data_preview += f"{'  Energy Bin':^22} | {'c1':^15} | {'c2':^15} | {'c2/c1':^12}\n"
        data_preview += "-" * 70 + "\n"
        
        for i in range(min(n_preview, n_values)):
            e_low = f"{self.pert_energies[i]:.3e}"
            e_high = f"{self.pert_energies[i+1]:.3e}"
            c1_val = self.c1[i]
            c2_val = self.c2[i]
            ratio_val = self.ratio[i]
            
            data_preview += f"{e_low}-{e_high:^8} | {c1_val:15.6e} | {c2_val:15.6e} | {ratio_val:12.6e}\n"
        
        # Add ellipsis if there are more values than shown
        if n_values > 2 * n_preview:
            data_preview += "..." + " " * 67 + "\n"
            
            # Show last few values
            for i in range(max(n_preview, n_values - n_preview), n_values):
                e_low = f"{self.pert_energies[i]:.3e}"
                e_high = f"{self.pert_energies[i+1]:.3e}"
                c1_val = self.c1[i]
                c2_val = self.c2[i]
                ratio_val = self.ratio[i]
                
                data_preview += f"{e_low}-{e_high:^8} | {c1_val:15.6e} | {c2_val:15.6e} | {ratio_val:12.6e}\n"
        
        # Available methods in table format
        methods_section = "\n\nAvailable Methods:\n"
        methods_section += "-" * header_width + "\n"
        
        # Set column widths for method and description
        method_col_width = 35
        desc_col_width = header_width - method_col_width - 3  # -3 for spacing and formatting
        
        methods_section += "{:<{width1}} {:<{width2}}\n".format(
            "Method", "Description", width1=method_col_width, width2=desc_col_width)
        methods_section += "-" * header_width + "\n"
        
        # Function to add a method and description with proper wrapping
        def add_method(method, description):
            nonlocal methods_section
            if len(description) <= desc_col_width:
                methods_section += "{:<{width1}} {:<{width2}}\n".format(
                    method, description, width1=method_col_width, width2=desc_col_width)
            else:
                # Handle wrapping for long descriptions
                methods_section += "{:<{width1}} {:<{width2}}\n".format(
                    method, description[:desc_col_width], width1=method_col_width, width2=desc_col_width)
                remaining = description[desc_col_width:]
                while remaining:
                    chunk = remaining[:desc_col_width].strip()
                    remaining = remaining[len(chunk):].strip()
                    methods_section += "{:<{width1}} {:<{width2}}\n".format(
                        "", chunk, width1=method_col_width, width2=desc_col_width)
        
        # Add each method with its description
        add_method(".calculate_nonlinearity(p)", "Calculate nonlinearity factor for specified perturbation magnitude")
        add_method(".calculate_nonlinearity_by_bin(p)", "Calculate nonlinearity factor by energy bin")
        add_method(".plot(ax=None, title=None, top_n=5)", "Plot nonlinearity factor vs perturbation magnitude")
        
        methods_section += "-" * header_width + "\n"
        
        # Combine all sections
        return header + stats + data_preview + methods_section


@dataclass
class SensitivityData:
    """Container class for sensitivity analysis data.

    :ivar tally_id: ID of the tally used for sensitivity calculation
    :type tally_id: int
    :ivar pert_energies: List of perturbation energy boundaries
    :type pert_energies: List[float]
    :ivar zaid: ZAID of the nuclide for which sensitivities were calculated
    :type zaid: int
    :ivar label: Label for the sensitivity data set
    :type label: str
    :ivar tally_name: Name of the tally
    :type tally_name: str
    :ivar data: Nested dictionary containing sensitivity coefficients organized by energy and reaction number
    :type data: Dict[str, Dict[int, Coefficients]]
    :ivar coefficients: Dictionary containing Taylor series coefficients organized by energy and reaction number
    :type coefficients: Dict[str, Dict[int, TaylorCoefficients]]
    :ivar lethargy: List of lethargy intervals between perturbation energies
    :type lethargy: List[float]
    :ivar energies: List of energy values used as keys in the data dictionary
    :type energies: List[str]
    :ivar reactions: Sorted list of unique reaction numbers found in the data
    :type reactions: List[int]
    :ivar nuclide: Nuclide symbol for the ZAID
    :type nuclide: str
    """
    tally_id: int
    pert_energies: list[float]
    zaid: int
    label: str
    tally_name: str = None
    data: Dict[str, Dict[int, 'Coefficients']] = None
    coefficients: Dict[str, Dict[int, TaylorCoefficients]] = field(default_factory=dict)
    lethargy: List[float] = field(init=False, repr=False)
    energies: List[str] = field(init=False, repr=False)
    reactions: List[int] = field(init=False, repr=False)
    nuclide: str = field(init=False, repr=False)
    
    def __post_init__(self):
        """Compute attributes once after initialization"""
        # Calculate lethargy intervals
        self.lethargy = [np.log(self.pert_energies[i+1]/self.pert_energies[i]) 
                         for i in range(len(self.pert_energies)-1)]
        
        # Get energy keys
        self.energies = list(self.data.keys()) if self.data else []
        
        # Get unique reaction numbers
        if not self.data:
            self.reactions = []
        else:
            all_reactions = set()
            for energy_data in self.data.values():
                all_reactions.update(energy_data.keys())
            self.reactions = sorted(list(all_reactions))
        
        # Get nuclide symbol
        z = self.zaid // 1000
        a = self.zaid % 1000
        self.nuclide = f"{ATOMIC_NUMBER_TO_SYMBOL[z]}-{a}"

    def plot_sensitivity(self, energy: Union[str, List[str]] = None, 
             reaction: Union[List[int], int] = None, energy_range: tuple = None, xlog: bool = False, ylog: bool = False):
        """Plot sensitivity coefficients for specified energies and reactions.

        Parameters
        ----------
        energy : Union[str, List[str]], optional
            Energy string(s) to plot. If None, plots all energies.
        reaction : Union[List[int], int], optional
            Reaction number(s) to plot. If None, plots all reactions.
        energy_range : tuple, optional
            Optional x-axis limits as (min, max).
        xlog : bool, optional
            Whether to use logarithmic scale for x-axis. Default is False.
        ylog : bool, optional
            Whether to use logarithmic scale for y-axis. Default is False.

        Raises
        ------
        ValueError
            If specified energies are not found in the data.
        """
        # If no energy specified, use all energies
        if energy is None:
            energies = list(self.data.keys())
        else:
            # Ensure energy is always a list
            energies = [energy] if not isinstance(energy, list) else energy
            # Validate all energies exist in data
            invalid_energies = [e for e in energies if e not in self.data]
            if invalid_energies:
                raise ValueError(f"Energies {invalid_energies} not found in sensitivity data.")

        # Ensure reactions is always a list
        if reaction is None:
            # Get unique reactions from all energy data
            reaction = list(set().union(*[d.keys() for d in self.data.values()]))
            # Sort reactions in ascending numerical order
            reaction.sort()
        elif not isinstance(reaction, list):
            reaction = [reaction]

        # Create a separate figure for each energy
        for e in energies:
            coeffs_dict = self.data[e]
            n = len(reaction)
            
            # Use a single Axes if only one reaction
            if n == 1:
                fig, ax = plt.subplots(figsize=(5, 4))
                axes = [ax]
            else:
                cols = 3
                rows = math.ceil(n / cols)
                fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
                # Ensure axes is a flat list of Axes objects
                if hasattr(axes, "flatten"):
                    axes = list(axes.flatten())
                else:
                    axes = [axes]
            
            # Modify title display based on energy string format
            if e == "integral":
                title_text = "Integral Result"
            else:
                # Parse the energy range from the string format
                try:
                    lower, upper = e.split('_')
                    title_text = f"Energy Range: {lower} - {upper} MeV"
                except ValueError:
                    # Fallback if energy doesn't follow expected format
                    title_text = f"Energy = {e}"
            
            # Raise the figure title position to avoid overlap with subplot titles
            fig.suptitle(title_text, y=1.01)
            
            for i, rxn in enumerate(reaction):
                ax = axes[i]
                if rxn not in coeffs_dict:
                    ax.text(0.5, 0.5, f"Reaction {rxn} not found", ha='center', va='center')
                    ax.axis('off')
                else:
                    coef = coeffs_dict[rxn]
                    coef.plot(ax=ax, xlim=energy_range)

            # Hide any extra subplots
            for j in range(n, len(axes)):
                axes[j].axis('off')
            
            # Apply logarithmic scales if requested
            if xlog:
                for ax in axes:
                    ax.set_xscale('log')
            if ylog:
                for ax in axes:
                    ax.set_yscale('log')
            
            plt.tight_layout()
            plt.show()

    def plot_ratio(self, energy: Union[str, List[str]] = None, reaction: Union[List[int], int] = None, top_n: int = 5):
        """Plot ratio of second-order to first-order sensitivity coefficients.
        
        The ratio is calculated as:
            R = (c2 * p) / c1
        
        Where:
            - c2 is the second-order coefficient
            - c1 is the first-order coefficient
            - p is the perturbation fraction

        :param energy: Energy string(s) to plot. If None, plots all energies
        :type energy: Union[str, List[str]], optional
        :param reaction: Reaction number(s) to plot. If None, plots all reactions
        :type reaction: Union[List[int], int], optional
        :param top_n: Number of top absolute ratios to plot with labels (0 means plot all)
        :type top_n: int, optional
        :raises ValueError: If sensitivity data does not contain Taylor coefficients
        :raises ValueError: If specified energies are not found in the data
        """
        if not self.coefficients:
            raise ValueError("Taylor coefficients are required for ratio plots. Please recompute sensitivity with include_second_order=True.")
            
        # If no energy specified, use all energies
        if energy is None:
            energies = list(self.coefficients.keys())
        else:
            # Ensure energy is always a list
            energies = [energy] if not isinstance(energy, list) else energy
            # Validate all energies exist in data
            invalid_energies = [e for e in energies if e not in self.coefficients]
            if invalid_energies:
                raise ValueError(f"Energies {invalid_energies} not found in coefficient data.")

        # Ensure reactions is always a list
        if reaction is None:
            # Get unique reactions from all energy data
            reaction = list(set().union(*[d.keys() for d in self.coefficients.values()]))
            # Sort reactions in ascending numerical order
            reaction.sort()
        elif not isinstance(reaction, list):
            reaction = [reaction]

        # Create a separate figure for each energy
        for e in energies:
            rxn_dict = self.coefficients[e]
            n = len(reaction)
            
            # Use a single Axes if only one reaction
            if n == 1:
                fig, ax = plt.subplots(figsize=(8, 4))  # Reduced height from 6 to 4
                axes = [ax]
            else:
                cols = 1  # Changed from 2 to 1 to have only one figure per row
                rows = n  # Now rows equals the number of reactions
                fig, axes = plt.subplots(rows, cols, figsize=(8, 3 * rows))  # Reduced height per row from 6 to 3
                # Ensure axes is a flat list of Axes objects
                if hasattr(axes, "flatten"):
                    axes = list(axes.flatten())
                else:
                    axes = [axes]
            
            # Energy group title
            if e == "integral":
                title_text = f"Integral Result"
            else:
                # Parse the energy range from the string format
                try:
                    lower, upper = e.split('_')
                    title_text = f"Energy Range: {lower} - {upper} MeV"
                except ValueError:
                    title_text = f"Energy = {e}"
            
            fig.suptitle(title_text, y=1.01, fontsize=14)  # Reduced font size from 16 to 14
            
            for i, rxn in enumerate(reaction):
                ax = axes[i]
                if rxn not in rxn_dict:
                    ax.text(0.5, 0.5, f"Reaction {rxn} not found", ha='center', va='center')
                    ax.axis('off')
                else:
                    # Get the TaylorCoefficients object and plot it
                    coeff_obj = rxn_dict[rxn]
                    coeff_obj.plot(ax=ax, title=f"MT {rxn}", top_n=top_n)

            # Hide any extra subplots
            for j in range(n, len(axes)):
                axes[j].axis('off')
            
            plt.tight_layout()
            plt.show()
            
    def to_dataframe(self) -> pd.DataFrame:
        """Export sensitivity data as a pandas DataFrame for plotting.

        :returns: DataFrame with the following columns:
            - det_energy: Detector energy range string (e.g., "0.00e+00_1.00e-01") or 'integral'
            - energy_lower: Lower energy boundary parsed from det_energy string (None for 'integral')
            - energy_upper: Upper energy boundary parsed from det_energy string (None for 'integral')
            - reaction: Reaction number (MT)
            - e_lower: Lower boundary of perturbation energy bin
            - e_upper: Upper boundary of perturbation energy bin
            - sensitivity: Sensitivity per lethargy value
            - error: Relative error value for the sensitivity
            - label: Sensitivity data label
            - tally_name: Name of the tally
        :rtype: pd.DataFrame
        """
        data_records = []

        for det_energy, rxn_dict in self.data.items():
            # Parse energy bounds from energy string if not "integral"
            energy_lower = None
            energy_upper = None
            if det_energy != "integral":
                try:
                    energy_lower, energy_upper = map(float, det_energy.split('_'))
                except ValueError:
                    # Handle case where energy string doesn't match expected format
                    pass
            
            for rxn, coef in rxn_dict.items():
                energies = coef.pert_energies
                # Calculate values per lethargy
                lp = np.array(coef.values_per_lethargy)
                # Compute error bars from values, errors and lethargy
                leth = np.array(coef.lethargy)
                error_bars = (np.array(coef.values) * np.array(coef.errors) / leth).tolist()
                
                # Create records for each energy bin (using lower and upper boundaries)
                for i in range(len(energies) - 1):
                    data_records.append({
                        'det_energy': det_energy,
                        'energy_lower': energy_lower,
                        'energy_upper': energy_upper,
                        'reaction': rxn,
                        'e_lower': energies[i],
                        'e_upper': energies[i+1],
                        'sensitivity': lp[i],
                        'error': error_bars[i],
                        'label': self.label,
                        'tally_name': self.tally_name
                    })

        return pd.DataFrame(data_records)
        
    def __repr__(self):
        """Returns a formatted string representation of the sensitivity data.
        
        This method is called when the object is evaluated in interactive environments
        like Jupyter notebooks or the Python interpreter.
        
        :return: Formatted string representation of the sensitivity data
        :rtype: str
        """
        # Create a visually appealing header with a border
        header_width = 85
        header = "=" * header_width + "\n"
        header += f"{'Sensitivity Data for ' + self.nuclide:^{header_width}}\n"
        header += "=" * header_width + "\n\n"
        
        # Create aligned key-value pairs with consistent width
        label_width = 32  # Width for labels
        
        # Basic information
        info_lines = []
        info_lines.append(f"{'Label:':{label_width}} {self.label}")
        info_lines.append(f"{'Tally ID:':{label_width}} {self.tally_id}")
        
        if self.tally_name:
            info_lines.append(f"{'Tally Name:':{label_width}} {self.tally_name}")
        
        info_lines.append(f"{'Nuclide (ZAID):':{label_width}} {self.nuclide} ({self.zaid})")
        
        # Data overview
        num_energy_groups = len(self.energies)
        info_lines.append(f"{'Number of detector energy bins:':{label_width}} {num_energy_groups-1 if 'integral' in self.energies else num_energy_groups}")
        num_pert_bins = len(self.pert_energies) - 1
        info_lines.append(f"{'Number of perturbation bins:':{label_width}} {num_pert_bins}")
        
        # Add energy grid structure identification
        grid_name = _identify_energy_grid(self.pert_energies)
        if grid_name:
            info_lines.append(f"{'Energy structure:':{label_width}} {grid_name}")
            
        info_lines.append(f"{'Reactions available:':{label_width}} {', '.join(map(str, self.reactions))}")
        
        # Check if any Taylor coefficients contain valid second-order data (c2)
        has_second_order = False
        if self.coefficients:
            for energy_data in self.coefficients.values():
                for coeff_obj in energy_data.values():
                    if hasattr(coeff_obj, 'c2') and any(c2 != 0 for c2 in coeff_obj.c2 if not np.isnan(c2)):
                        has_second_order = True
                        break
                if has_second_order:
                    break
            
            info_lines.append(f"{'Second Order data available:':{label_width}} {has_second_order}")
        else:
            info_lines.append(f"{'Second Order data available:':{label_width}} No")
        
        stats = "\n".join(info_lines)
        
        # Energy groups summary showing ALL energy-dependent results
        energy_info = "\n\nEnergy group ranges:\n"
        energy_groups = [e for e in self.energies if e != "integral"]
        if energy_groups:
            for e in energy_groups:
                energy_info += f"  - {e}\n"
        else:
            energy_info += "  No energy-dependent results available.\n"
        
        # Add integral entry at the end if it exists
        if "integral" in self.energies:
            energy_info += "  - integral\n"
        
        # Available methods in table format
        methods_section = "\n\nAvailable Methods:\n"
        methods_section += "-" * header_width + "\n"
        
        # Set column widths for method and description
        method_col_width = 38
        desc_col_width = header_width - method_col_width - 3  # -3 for spacing and formatting
        
        methods_section += "{:<{width1}} {:<{width2}}\n".format(
            "Method", "Description", width1=method_col_width, width2=desc_col_width)
        methods_section += "-" * header_width + "\n"
        
        # Function to add a method and description with proper wrapping
        def add_method(method, description):
            nonlocal methods_section
            if len(description) <= desc_col_width:
                methods_section += "{:<{width1}} {:<{width2}}\n".format(
                    method, description, width1=method_col_width, width2=desc_col_width)
            else:
                # Handle wrapping for long descriptions
                methods_section += "{:<{width1}} {:<{width2}}\n".format(
                    method, description[:desc_col_width], width1=method_col_width, width2=desc_col_width)
                remaining = description[desc_col_width:]
                while remaining:
                    chunk = remaining[:desc_col_width].strip()
                    remaining = remaining[len(chunk):].strip()
                    methods_section += "{:<{width1}} {:<{width2}}\n".format(
                        "", chunk, width1=method_col_width, width2=desc_col_width)
        
        # Add each method with its description
        add_method(".plot_sensitivity(...)", "Plot sensitivity profiles")
        add_method(".to_dataframe()", "Get full data as pandas DataFrame")
        
        # Add methods only available with second-order data
        if has_second_order:
            add_method(".plot_ratio(...)", "Plot ratio of 2nd to 1st order coefficients")
            add_method(".plot_perturbed_response(...)", "Plot response comparison of 1st vs 2nd order")
            add_method(".plot_second_order_contribution(...)", "Plot second order term contribution")
        
        methods_section += "-" * header_width + "\n"
        
        # Add section showing how to access data with the same table format
        access_section = "\nAccessing Data:\n"
        access_section += "-" * header_width + "\n"
        
        # Use the same column widths for consistent formatting
        access_section += "{:<{width1}} {:<{width2}}\n".format(
            "Expression", "Description", width1=method_col_width, width2=desc_col_width)
        access_section += "-" * header_width + "\n"
        
        # Function to add an access pattern and description with proper wrapping
        def add_access_pattern(pattern, description):
            nonlocal access_section
            if len(description) <= desc_col_width:
                access_section += "{:<{width1}} {:<{width2}}\n".format(
                    pattern, description, width1=method_col_width, width2=desc_col_width)
            else:
                # Handle wrapping for long descriptions
                access_section += "{:<{width1}} {:<{width2}}\n".format(
                    pattern, description[:desc_col_width], width1=method_col_width, width2=desc_col_width)
                remaining = description[desc_col_width:]
                while remaining:
                    chunk = remaining[:desc_col_width].strip()
                    remaining = remaining[len(chunk):].strip()
                    access_section += "{:<{width1}} {:<{width2}}\n".format(
                        "", chunk, width1=method_col_width, width2=desc_col_width)
        
        # Get the first available energy bin or integral
        first_energy = next((e for e in self.energies if e != "integral"), "integral" if "integral" in self.energies else None)
        
        # Get the first available reaction if any
        first_reaction = self.reactions[0] if self.reactions else None
        
        if first_energy is not None and first_reaction is not None:
            energy_desc = "integral result" if first_energy == "integral" else f"energy bin {first_energy}"
            access_pattern = f".data['{first_energy}'][{first_reaction}]"
            description = f"Get coefficients for {energy_desc}, reaction {first_reaction}"
            add_access_pattern(access_pattern, description)
        
        # If we have integral results and at least two reactions, show second example
        if "integral" in self.energies and len(self.reactions) > 1:
            second_reaction = self.reactions[1]
            access_pattern = f".data['integral'][{second_reaction}]"
            description = f"Get integral coefficients for reaction {second_reaction}"
            add_access_pattern(access_pattern, description)
        
        # Show Taylor coefficients example if available
        if self.coefficients:
            energy_key = next(iter(self.coefficients.keys()))
            rxn_key = next(iter(self.coefficients[energy_key].keys()))
            access_pattern = f".coefficients['{energy_key}'][{rxn_key}]" 
            description = f"Get Taylor coefficient data for energy bin, reaction {rxn_key}"
            add_access_pattern(access_pattern, description)
        
        access_section += "-" * header_width + "\n"
        
        # Combine all sections
        return header + stats + energy_info + methods_section + access_section

    def plot_perturbed_response(self, 
                            energy: Union[str, List[str]] = None, 
                            reaction: Union[List[int], int] = None,
                            p_range: tuple = (-20, 20),
                            n_points: int = 100, 
                            top_n: int = 3,
                            e_bins: List[int] = None,
                            n_sigma: float = 1.0,
                            print_coefficients: bool = False):
        """Plot perturbed response as a function of perturbation magnitude.
        
        Compares first-order approximation R(p) = R0 + c1*p with 
        second-order approximation R(p) = R0 + c1*p + c2*p^2.
        
        :param energy: Energy string(s) to plot. If None, plots all energies
        :type energy: Union[str, List[str]], optional
        :param reaction: Reaction number(s) to plot. If None, plots all reactions
        :type reaction: Union[List[int], int], optional
        :param p_range: Range of perturbation magnitudes as (min%, max%) in percent
        :type p_range: tuple, optional
        :param n_points: Number of points to evaluate for plotting
        :type n_points: int, optional
        :param top_n: Number of perturbation energy bins with highest nonlinearity to show.
                    If 0, shows all bins.
        :type top_n: int, optional
        :param e_bins: Specific indices of perturbation energy bins to plot.
                      Overrides top_n if provided.
        :type e_bins: List[int], optional
        :param n_sigma: Number of standard deviations to show in error bands
        :type n_sigma: float, optional
        :param print_coefficients: Whether to print detailed coefficient info to console instead of on plot
        :type print_coefficients: bool, optional
        :raises ValueError: If sensitivity data does not contain Taylor coefficients
        :raises ValueError: If specified energies are not found in the data
        """
        # Use internal helper method to handle the common parts of plotting
        return self._plot_perturbed_data(energy, reaction, p_range, n_points, top_n, 
                                        'comparative', e_bins, n_sigma, print_coefficients)
    
    def plot_second_order_contribution(self, 
                            energy: Union[str, List[str]] = None, 
                            reaction: Union[List[int], int] = None,
                            p_range: tuple = (-20, 20),
                            n_points: int = 100, 
                            top_n: int = 3,
                            e_bins: List[int] = None,
                            n_sigma: float = 1.0,
                            print_coefficients: bool = False):
        """Plot the contribution of second-order term as percentage of the total response.
        
        This shows how much of the total second-order response comes from the second-order term (c2*p²),
        which indicates the error introduced when using only first-order approximation.
        
        :param energy: Energy string(s) to plot. If None, plots all energies
        :type energy: Union[str, List[str]], optional
        :param reaction: Reaction number(s) to plot. If None, plots all reactions
        :type reaction: Union[List[int], int], optional
        :param p_range: Range of perturbation magnitudes as (min%, max%) in percent
        :type p_range: tuple, optional
        :param n_points: Number of points to evaluate for plotting
        :type n_points: int, optional
        :param top_n: Number of perturbation energy bins with highest second-order contribution to show.
                    If 0, shows all bins.
        :type top_n: int, optional
        :param e_bins: Specific indices of perturbation energy bins to plot.
                      Overrides top_n if provided.
        :type e_bins: List[int], optional
        :param n_sigma: Number of standard deviations to show in error bands
        :type n_sigma: float, optional
        :param print_coefficients: Whether to print detailed coefficient info to console instead of on plot
        :type print_coefficients: bool, optional
        :raises ValueError: If sensitivity data does not contain Taylor coefficients
        :raises ValueError: If specified energies are not found in the data
        """
        # Use internal helper method with 'difference' plot type
        return self._plot_perturbed_data(energy, reaction, p_range, n_points, top_n, 
                                        'difference', e_bins, n_sigma, print_coefficients)
    
    def _plot_perturbed_data(self, 
                            energy: Union[str, List[str]] = None, 
                            reaction: Union[List[int], int] = None,
                            p_range: tuple = (-20, 20),
                            n_points: int = 100, 
                            top_n: int = 3,
                            plot_type: str = 'comparative',
                            e_bins: List[int] = None,
                            n_sigma: float = 1.0,
                            print_coefficients: bool = False):
        """Internal helper method that handles perturbed response plotting.
        
        This method is used by both plot_perturbed_response and plot_second_order_contribution
        to avoid code duplication.
        
        :param plot_type: Type of plot: 'comparative' (1st vs. 1st+2nd order) or 
                         'difference' (second-order term contribution)
        :type plot_type: str
        """
        if not self.coefficients:
            raise ValueError("Taylor coefficients for order 1 and 2 are required for perturbed response plots. ")
            
        # If no energy specified, use all energies
        if energy is None:
            energies = list(self.coefficients.keys())
        else:
            # Ensure energy is always a list
            energies = [energy] if not isinstance(energy, list) else energy
            # Validate all energies exist in data
            invalid_energies = [e for e in energies if e not in self.coefficients]
            if invalid_energies:
                raise ValueError(f"Energies {invalid_energies} not found in coefficient data.")

        # Ensure reactions is always a list
        if reaction is None:
            # Get unique reactions from all energy data
            reaction = list(set().union(*[d.keys() for d in self.coefficients.values()]))
            # Sort reactions in ascending numerical order
            reaction.sort()
        elif not isinstance(reaction, list):
            reaction = [reaction]

        # Validate plot_type
        valid_plot_types = ['comparative', 'difference']
        if plot_type not in valid_plot_types:
            raise ValueError(f"plot_type must be one of {valid_plot_types}")

        # Generate perturbation values (convert from percent to fraction)
        p_values = np.linspace(p_range[0], p_range[1], n_points) / 100.0
        
        # Create a separate figure for each energy
        for e in energies:
            for rxn in reaction:
                if rxn not in self.coefficients[e]:
                    # Skip if reaction not available for this energy
                    continue
                
                # Get Taylor coefficients and their errors
                coeff_obj = self.coefficients[e][rxn]
                c1_values = coeff_obj.c1
                c2_values = coeff_obj.c2
                c1_errors = coeff_obj.c1_errors
                c2_errors = coeff_obj.c2_errors
                r0 = self.data[e][rxn].r0
                e0_relative = self.data[e][rxn].e0  # Unperturbed error (in relative form)
                e0_abs = r0 * e0_relative  # Convert to absolute error
                
                # Get the number of perturbation energy bins
                n_bins = len(c1_values)
                
                if e_bins is not None:
                    # Use specific bins if provided
                    selected_bins = [i for i in e_bins if i < n_bins]
                elif top_n > 0:
                    # Use maximum perturbation for selection
                    max_p = max(abs(p_range[0]), abs(p_range[1])) / 100.0
                    
                    # Calculate the difference metric at max_p for each bin
                    differences = []
                    for i in range(n_bins):
                        c1 = c1_values[i]
                        c2 = c2_values[i]
                        
                        # Calculate first and second order at max perturbation
                        first_order = r0 + c1 * max_p
                        second_order = r0 + c1 * max_p + c2 * (max_p**2)
                        
                        # Calculate percentage contribution of second order term
                        if abs(second_order) > 1e-10:  # Avoid division by zero
                            diff = abs((c2 * max_p**2) / second_order) * 100
                        else:
                            diff = 0
                        
                        differences.append(diff)
                    
                    # Get indices of top_n bins with highest difference
                    # Sort in descending order of difference magnitude
                    selected_bins = np.argsort(differences)[-top_n:][::-1]  # Reverse to get descending order
                else:
                    # Use all bins
                    selected_bins = list(range(n_bins))
                
                # Skip if no bins to plot
                if len(selected_bins) == 0:
                    continue
                
                # Create figure with one row per selected energy bin
                n_selected = len(selected_bins)
                fig, axes = plt.subplots(n_selected, 1, figsize=(10, 4 * n_selected), 
                                       squeeze=False)  # Removed sharex=True
                axes = axes.flatten()
                
                # Set main title
                if e == "integral":
                    title_text = f"MT {rxn} - Integral Result"
                else:
                    try:
                        lower, upper = e.split('_')
                        title_text = f"MT {rxn} - Energy Range: {lower} - {upper} MeV"
                    except ValueError:
                        title_text = f"MT {rxn} - Energy = {e}"
                
                fig.suptitle(title_text, fontsize=16, y=1.02)
                
                # Plot each selected energy bin
                for i, bin_idx in enumerate(selected_bins):
                    ax = axes[i]
                    
                    # Get bin-specific coefficients and errors
                    c1 = c1_values[bin_idx]
                    c2 = c2_values[bin_idx]
                    c1_err_rel = c1_errors[bin_idx]  # Relative errors
                    c2_err_rel = c2_errors[bin_idx]  # Relative errors
                    
                    # Convert relative errors to absolute errors
                    c1_err_abs = c1 * c1_err_rel  # Convert relative to absolute error
                    c2_err_abs = c2 * c2_err_rel  # Convert relative to absolute error
                    
                    # Calculate first and second order responses
                    first_order = r0 + c1 * p_values
                    second_order = r0 + c1 * p_values + c2 * (p_values**2)
                    
                    # Calculate errors for first and second order using error propagation
                    # Now using all absolute errors properly
                    first_order_err = np.sqrt(e0_abs**2 + (c1_err_abs * p_values)**2)
                    second_order_err = np.sqrt(e0_abs**2 + (c1_err_abs * p_values)**2 + 
                                             (c2_err_abs * p_values**2)**2)
                    
                    # Scale errors by n_sigma
                    first_order_err *= n_sigma
                    second_order_err *= n_sigma
                    
                    # Get energy bin boundaries for title
                    e_low = coeff_obj.pert_energies[bin_idx]
                    e_high = coeff_obj.pert_energies[bin_idx+1]
                    
                    # Set subplot title
                    ax.set_title(f"Perturbation Energy Bin: {e_low:.2e} - {e_high:.2e} MeV", fontsize=12)
                    
                    # If using print statements for coefficient details
                    if print_coefficients:
                        print(f"\n--- Energy Bin: {e_low:.2e} - {e_high:.2e} MeV ---")
                        print(f"R₀ = {r0:.4e} ± {e0_relative*100:.2f}% (rel) = {r0:.4e} ± {e0_abs:.4e} (abs)")
                        print(f"c₁ = {c1:.4e} ± {c1_err_rel*100:.2f}% (rel) = {c1:.4e} ± {c1_err_abs:.4e} (abs)")
                        print(f"c₂ = {c2:.4e} ± {c2_err_rel*100:.2f}% (rel) = {c2:.4e} ± {c2_err_abs:.4e} (abs)")
                        print(f"c₂/c₁ = {c2/c1 if c1 != 0 else float('nan'):.4e}")
                        max_p_abs = max(abs(p_range[0]), abs(p_range[1]))
                        nonlin = (c2 * (max_p_abs/100)) / c1 * 100 if c1 != 0 else float('nan')
                        print(f"Ratio (c2*p/c1) at {max_p_abs}%: {nonlin:.2f}%")
                        print("-------------------------------------------")
                    
                    if plot_type == 'comparative':
                        # Plot both approximations with error bands
                        # First order (blue)
                        ax.plot(p_values * 100, first_order, 'b-', linewidth=2, 
                               label=f"First Order: R₀ + c₁·p")
                        ax.fill_between(p_values * 100, 
                                      first_order - first_order_err, 
                                      first_order + first_order_err, 
                                      color='blue', alpha=0.2)
                        
                        # Second order (red) - now using solid line instead of dashed
                        ax.plot(p_values * 100, second_order, 'r-', linewidth=2, 
                               label=f"Second Order: R₀ + c₁·p + c₂·p²")
                        ax.fill_between(p_values * 100, 
                                      second_order - second_order_err, 
                                      second_order + second_order_err, 
                                      color='red', alpha=0.2)
                        
                        # Reference line for unperturbed value (remove error band)
                        ax.axhline(y=r0, color='k', linestyle='-', alpha=0.5, 
                                  label=f"Unperturbed (R₀ = {r0:.4e} ± {e0_relative*100:.2f}%)")
                        
                        # Display coefficient info in a more compact format if not printing
                        if not print_coefficients:
                            max_p_abs = max(abs(p_range[0]), abs(p_range[1]))
                            nonlin = (c2 * (max_p_abs/100)) / c1 * 100 if c1 != 0 else float('nan')
                            
                            # Format the text box with stacked coefficient information
                            coeff_text = (f"R₀ = {r0:.4e} ± {e0_relative*100:.1f}%\n"
                                        f"c₁ = {c1:.4e} ± {c1_err_rel*100:.1f}%\n"
                                        f"c₂ = {c2:.4e} ± {c2_err_rel*100:.1f}%\n"
                                        f"Ratio (c2*p/c1) at {max_p_abs}%: {nonlin:.2f}%")
                            
                            # Position text box in upper left with vertical stacking
                            ax.text(0.02, 0.98, coeff_text, transform=ax.transAxes, fontsize=9,
                                  va='top', ha='left', bbox=dict(facecolor='white', alpha=0.7))
                        
                        ylabel = "Response"
                    
                    else:  # 'difference'
                        # Plot percentage error of using first-order instead of second-order
                        # Calculate the contribution of second-order term relative to full second-order approximation
                        diff = (c2 * p_values**2 / second_order) * 100  # percentage of second-order term
                        
                        # Calculate error in this percentage
                        # Error propagation for (c2*p²)/second_order
                        # We need to consider errors in both c2 and the full second-order approximation
                        second_order_err_rel = second_order_err / second_order  # Relative error in second-order approx
                        # Error propagation for division: rel_error_result² = rel_error_numerator² + rel_error_denominator²
                        diff_err_rel = np.sqrt((c2_err_rel)**2 + (second_order_err_rel)**2)
                        diff_err = np.abs(diff * diff_err_rel)  # Convert to absolute error
                        diff_err *= n_sigma
                        
                        ax.plot(p_values * 100, diff, 'g-', linewidth=2)
                        ax.fill_between(p_values * 100, 
                                      diff - diff_err, 
                                      diff + diff_err, 
                                      color='green', alpha=0.2)
                        
                        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                        
                        # Display coefficient values in compact format if not printing
                        if not print_coefficients:
                            # Format the text box with stacked coefficient information
                            coeff_text = (f"R₀ = {r0:.4e} ± {e0_relative*100:.1f}%\n"
                                        f"c₁ = {c1:.4e} ± {c1_err_rel*100:.1f}%\n"
                                        f"c₂ = {c2:.4e} ± {c2_err_rel*100:.1f}%\n"
                                        f"c₂/c₁ = {c2/c1 if c1 != 0 else float('nan'):.4e}")
                            
                            # Position text box in upper left with vertical stacking
                            ax.text(0.02, 0.98, coeff_text, transform=ax.transAxes, fontsize=9,
                                  va='top', ha='left', bbox=dict(facecolor='white', alpha=0.7))
                        
                        # More descriptive y-axis label
                        ylabel = "Second-order term contribution (%)"
                    
                    ax.set_ylabel(ylabel)
                    ax.grid(True, alpha=0.3)
                    
                    # Add xticks to all subplots (not just the bottom one)
                    # Calculate nice tick positions based on the p_range
                    tick_step = 5  # Default step size of 5%
                    min_tick = math.ceil(p_range[0] / tick_step) * tick_step
                    max_tick = math.floor(p_range[1] / tick_step) * tick_step
                    ticks = np.arange(min_tick, max_tick + tick_step, tick_step)
                    ax.set_xticks(ticks)
                    ax.set_xlabel("Perturbation Magnitude (%)")  # Add xlabel to all plots
                
                plt.tight_layout()
                plt.show()

@dataclass
class Coefficients:
    """Container for sensitivity coefficients for a specific energy and reaction.

    :ivar energy: Energy range string in format "lower_upper" (e.g., "0.00e+00_1.00e-01")
    :type energy: str
    :ivar reaction: Reaction number
    :type reaction: int
    :ivar pert_energies: Perturbation energy boundaries
    :type pert_energies: List[float]
    :ivar values: Sensitivity coefficient values (first-order)
    :type values: List[float]
    :ivar errors: Relative errors for the sensitivity coefficients (first-order)
    :type errors: List[float]
    :ivar r0: Unperturbed tally result
    :type r0: float
    :ivar e0: Unperturbed tally error
    :type e0: float
    :ivar values_second: Second-order sensitivity coefficient values
    :type values_second: List[float], optional
    :ivar errors_second: Relative errors for the second-order sensitivity coefficients
    :type errors_second: List[float], optional
    """
    energy: str
    reaction: int
    pert_energies: list[float]
    values: list[float]
    errors: list[float]
    r0: float = None 
    e0: float = None
    values_second: list[float] = None
    errors_second: list[float] = None

    @property
    def lethargy(self):
        """Calculate lethargy intervals between perturbation energies.

        :returns: List of lethargy intervals
        :rtype: List[float]
        """
        return [np.log(self.pert_energies[i+1]/self.pert_energies[i]) for i in range(len(self.pert_energies)-1)]
    
    @property
    def values_per_lethargy(self):
        """Calculate sensitivity coefficients per unit lethargy.

        :returns: Sensitivity coefficients normalized by lethargy intervals
        :rtype: List[float]
        """
        lethargy_vals = self.lethargy
        return [self.values[i]/lethargy_vals[i] for i in range(len(lethargy_vals))]
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert coefficients data to a pandas DataFrame.
        
        :returns: DataFrame with columns:
            - energy: Energy range string (detector energy)
            - reaction: Reaction number (MT)
            - e_lower: Lower boundary of perturbation energy bin
            - e_upper: Upper boundary of perturbation energy bin
            - sensitivity: Sensitivity coefficient value
            - error: Relative error of the coefficient
        :rtype: pd.DataFrame
        """
        # Create a list to hold the data for each row
        data = []
        
        # Add each perturbation energy bin as a separate row
        for i in range(len(self.values)):
            data.append({
                'energy': self.energy,
                'reaction': self.reaction,
                'e_lower': self.pert_energies[i],
                'e_upper': self.pert_energies[i+1],
                'sensitivity': self.values[i],
                'error': self.errors[i]
            })
        
        # Create and return the DataFrame with the specified column order
        return pd.DataFrame(data, columns=[
            'energy', 'reaction', 'e_lower', 'e_upper', 'sensitivity', 'error'
        ])

    def to_plot_data(
        self,
        per_lethargy: bool = True,
        uncertainty: bool = True,
        sigma: float = 1.0,
        uncertainty_style: str = 'errorbar',
        label: str = None,
        **styling_kwargs,
    ) -> Union['MultigroupXSPlotData', Tuple['MultigroupXSPlotData', 'UncertaintyBand']]:
        """Convert sensitivity coefficients into PlotData objects.

        Returns a :class:`~kika.plotting.MultigroupXSPlotData` (step plot) and,
        optionally, an :class:`~kika.plotting.UncertaintyBand` suitable for
        :class:`~kika.plotting.PlotBuilder`.

        :param per_lethargy: If True, normalise sensitivity values by the lethargy
            width of each bin (matches ``Coefficients.plot()`` behaviour).
        :type per_lethargy: bool
        :param uncertainty: If True, return an ``UncertaintyBand`` alongside the
            nominal data.
        :type uncertainty: bool
        :param sigma: Sigma multiplier for the uncertainty band.
        :type sigma: float
        :param uncertainty_style: Rendering style for the uncertainty band:
            ``'errorbar'`` (default) or ``'band'``.
        :type uncertainty_style: str
        :param label: Legend label. Auto-generated as
            ``"<reaction_name> (MT<reaction>)"`` when *None*.
        :type label: str, optional
        :param styling_kwargs: Forwarded to ``MultigroupXSPlotData``
            (``color``, ``linestyle``, ``linewidth``, etc.).
        :returns: ``MultigroupXSPlotData`` when *uncertainty=False*,
            or ``(MultigroupXSPlotData, UncertaintyBand)`` when *uncertainty=True*.
        :rtype: MultigroupXSPlotData or Tuple[MultigroupXSPlotData, UncertaintyBand]
        """
        from kika.plotting import MultigroupXSPlotData, UncertaintyBand

        energies = np.asarray(self.pert_energies, dtype=float)
        sens = np.asarray(self.values, dtype=float)

        if per_lethargy:
            lethargy = np.log(energies[1:] / energies[:-1])
            y_vals = sens / lethargy
        else:
            y_vals = sens.copy()

        # Step plot: n+1 x-points, repeat last y value
        x = energies
        y = np.append(y_vals, y_vals[-1])

        if label is None:
            reaction_name = MT_TO_REACTION.get(self.reaction, f"MT{self.reaction}")
            label = f"{reaction_name} (MT{self.reaction})"

        plot_data = MultigroupXSPlotData(
            x=x,
            y=y,
            label=label,
            plot_type='step',
            step_where='post',
            energy_bins=energies,
            **styling_kwargs,
        )

        if not uncertainty:
            return plot_data

        # Build UncertaintyBand from relative errors.
        # self.errors stores relative errors (fractional) on the raw sensitivity.
        # Per-lethargy divides by a constant per bin so the relative error is unchanged.
        rel_err = np.asarray(self.errors, dtype=float)
        rel_err_extended = np.append(rel_err, rel_err[-1])

        band = UncertaintyBand(
            x=x,
            relative_uncertainty=rel_err_extended,
            sigma=sigma,
            label=f"{label} ({sigma}\u03c3)" if sigma != 1.0 else None,
            style=uncertainty_style,
        )

        return plot_data, band

    def __repr__(self):
        """Returns a formatted string representation of the coefficients.
        
        This method provides an informative overview of the coefficient data and available methods.
        
        :return: Formatted string representation of the coefficients
        :rtype: str
        """
        # Create a visually appealing header
        header_width = 85
        header = "=" * header_width + "\n"
        header += f"{'Sensitivity Coefficients':^{header_width}}\n"
        header += "=" * header_width + "\n\n"
        
        # Basic information section
        info_lines = []
        info_lines.append(f"Energy: {self.energy}")
        info_lines.append(f"Reaction Number (MT): {self.reaction}")
        info_lines.append(f"Number of perturbation bins: {len(self.pert_energies) - 1}")
        
        if self.r0 is not None:
            # Display the unperturbed result with relative error, showing it's relative
            # by using percentage notation
            info_lines.append(f"Unperturbed result (R₀): {self.r0:.6e} ± {self.e0*100:.4f}% (relative)")
        
        info = "\n".join(info_lines)
        
        # Data preview section - show first few and last few values
        n_preview = 3  # Number of values to show at beginning and end
        n_values = len(self.values)
        
        data_preview = "\n\nData preview (values and relative errors):\n\n"
        
        # Format as a small table
        data_preview += f"{'  Energy Bin':^19} | {'Value':^15} | {'  Rel. Error':^12}\n"
        data_preview += "-" * 50 + "\n"
        
        for i in range(min(n_preview, n_values)):
            e_low = f"{self.pert_energies[i]:.3e}"
            e_high = f"{self.pert_energies[i+1]:.3e}"
            data_preview += f"{e_low}-{e_high:^6} | {self.values[i]:15.6e} | {self.errors[i]:12.6f}\n"
        
        # Add ellipsis if there are more values than shown
        if n_values > 2 * n_preview:
            data_preview += "..." + " " * 47 + "\n"
            
            # Show last few values
            for i in range(max(n_preview, n_values - n_preview), n_values):
                e_low = f"{self.pert_energies[i]:.3e}"
                e_high = f"{self.pert_energies[i+1]:.3e}"
                data_preview += f"{e_low}-{e_high:^6} | {self.values[i]:15.6e} | {self.errors[i]:12.6f}\n"
        
        # Available methods in table format
        methods_section = "\n\nAvailable Methods:\n"
        methods_section += "-" * header_width + "\n"
        
        # Set column widths for method and description
        method_col_width = 27
        desc_col_width = header_width - method_col_width - 3  # -3 for spacing and formatting
        
        methods_section += "{:<{width1}} {:<{width2}}\n".format(
            "Method", "Description", width1=method_col_width, width2=desc_col_width)
        methods_section += "-" * header_width + "\n"
        
        # Function to add a method and description with proper wrapping
        def add_method(method, description):
            nonlocal methods_section
            if len(description) <= desc_col_width:
                methods_section += "{:<{width1}} {:<{width2}}\n".format(
                    method, description, width1=method_col_width, width2=desc_col_width)
            else:
                # Handle wrapping for long descriptions
                methods_section += "{:<{width1}} {:<{width2}}\n".format(
                    method, description[:desc_col_width], width1=method_col_width, width2=desc_col_width)
                remaining = description[desc_col_width:]
                while remaining:
                    chunk = remaining[:desc_col_width].strip()
                    remaining = remaining[len(chunk):].strip()
                    methods_section += "{:<{width1}} {:<{width2}}\n".format(
                        "", chunk, width1=method_col_width, width2=desc_col_width)
        
        # Add each method with its description
        add_method(".lethargy", "Get lethargy intervals as property")
        add_method(".values_per_lethargy", "Get sensitivity per lethargy as property")
        add_method(".plot(...)", "Plot sensitivity coefficients")
        add_method(".to_plot_data(...)", "Get PlotData for use with PlotBuilder")
        add_method(".to_dataframe()", "Export data as pandas DataFrame")
        
        methods_section += "-" * header_width + "\n"
        
        # Combine all sections
        return header + info + data_preview + methods_section
        
    # New helper method to plot onto a provided axis
    def _plot_on_ax(self, ax, xlim=None, xlog=False):
        """Plot sensitivity coefficients on a given matplotlib axis.

        :param ax: The axis to plot on
        :type ax: matplotlib.axes.Axes
        :param xlim: Optional x-axis limits as (min, max). If None, smart limits are applied.
        :type xlim: tuple, optional
        :param xlog: Whether to use logarithmic scale for x-axis
        :type xlog: bool, optional
        """
        # Compute values per lethargy and error ratios
        lp = np.array(self.values_per_lethargy)
        leth = np.array(self.lethargy)
        error_bars = np.array(self.values) * np.array(self.errors) / leth
        x = np.array(self.pert_energies)
        y = np.append(lp, lp[-1])
        color = 'blue'
        ax.step(x, y, where='post', color=color, linewidth=2)
        x_mid = (x[:-1] + x[1:]) / 2.0
        ax.errorbar(x_mid, lp, yerr=np.abs(error_bars), fmt=' ', elinewidth=1.5, ecolor=color, capsize=2.5)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"MT = {self.reaction}")
        ax.set_xlabel("Energy (MeV)")
        ax.set_ylabel("Sensitivity per lethargy")
        
        # Set x-axis scale
        if xlog:
            ax.set_xscale('log')
        
        # Set x-axis limits
        if xlim is not None:
            ax.set_xlim(xlim)
        else:
            # Apply smart limits: exclude consecutive zero bins from edges
            # but keep one zero bin on each side if possible
            n_bins = len(lp)
            
            # Find first non-zero bin from the left
            first_nonzero = 0
            for i in range(n_bins):
                if lp[i] != 0.0:
                    first_nonzero = i
                    break
            
            # Find last non-zero bin from the right
            last_nonzero = n_bins - 1
            for i in range(n_bins - 1, -1, -1):
                if lp[i] != 0.0:
                    last_nonzero = i
                    break
            
            # Set limits to include one zero bin before first non-zero and after last non-zero
            # (if they exist and we're not excluding everything)
            if first_nonzero == last_nonzero and lp[first_nonzero] == 0.0:
                # All values are zero, use full range
                x_min, x_max = x[0], x[-1]
            else:
                # Include one zero bin before first non-zero (if exists)
                left_idx = max(0, first_nonzero - 1) if first_nonzero > 0 else 0
                # Include one zero bin after last non-zero (if exists)
                right_idx = min(n_bins - 1, last_nonzero + 1) if last_nonzero < n_bins - 1 else n_bins - 1
                
                x_min = x[left_idx]
                x_max = x[right_idx + 1]  # +1 because x has n_bins+1 elements (energy boundaries)
            
            ax.set_xlim(x_min, x_max)
        
    def plot(self, ax=None, xlim=None, xlog=False):
        """Create a new plot of sensitivity coefficients.

        :param ax: Optional existing axis to plot on
        :type ax: matplotlib.axes.Axes, optional
        :param xlim: Optional x-axis limits as (min, max). If None, smart limits are applied.
        :type xlim: tuple, optional
        :param xlog: Whether to use logarithmic scale for x-axis
        :type xlog: bool, optional
        :returns: The axis containing the plot
        :rtype: matplotlib.axes.Axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 4))
        self._plot_on_ax(ax, xlim=xlim, xlog=xlog)
        return ax