from dataclasses import dataclass
from mcnpy.input.parse_input import read_mcnp  # Fix import path
from mcnpy.mctal.parse_mctal import read_mctal
from typing import Dict, Union, List
import numpy as np
import matplotlib.pyplot as plt
import math


@dataclass
class SensitivityData:
    """Container class for sensitivity analysis data.

    :ivar tally_id: ID of the tally used for sensitivity calculation
    :type tally_id: int
    :ivar pert_energies: List of perturbation energy boundaries
    :type pert_energies: List[float]
    :ivar label: Label for the sensitivity data set
    :type label: str
    :ivar tally_name: Name of the tally
    :type tally_name: str
    :ivar data: Nested dictionary containing sensitivity coefficients organized by energy and reaction number
    :type data: Dict[Union[float,str], Dict[int, Coefficients]]
    """
    tally_id: int
    pert_energies: list[float]
    label: str
    tally_name: str = None
    data: Dict[Union[float,str], Dict[int, 'Coefficients']] = None

    @property
    def lethargy(self):
        """Calculate lethargy intervals between perturbation energies.

        Returns:
            List[float]: List of lethargy intervals.
        """
        return [np.log(self.pert_energies[i+1]/self.pert_energies[i]) for i in range(len(self.pert_energies)-1)]
        
    def plot(self, energy: Union[float, str, List[float], List[str]] = None, 
             reactions: Union[List[int], int] = None, xlim: tuple = None):
        """Plot sensitivity coefficients for specified energies and reactions.

        Args:
            energy (float, str, or list): Energy value(s) to plot. If None, plots all energies.
            reactions (int or list): Reaction number(s) to plot. If None, plots all reactions.
            xlim (tuple): Optional x-axis limits as (min, max).

        Raises:
            ValueError: If specified energies are not found in the data.
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
        if reactions is None:
            # Get unique reactions from all energy data
            reactions = list(set().union(*[d.keys() for d in self.data.values()]))
        elif not isinstance(reactions, list):
            reactions = [reactions]

        # Create a separate figure for each energy
        for e in energies:
            coeffs_dict = self.data[e]
            n = len(reactions)
            
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
            
            fig.suptitle(f"Energy = {e}")
            
            for i, rxn in enumerate(reactions):
                ax = axes[i]
                if rxn not in coeffs_dict:
                    ax.text(0.5, 0.5, f"Reaction {rxn} not found", ha='center', va='center')
                    ax.axis('off')
                else:
                    coef = coeffs_dict[rxn]
                    coef.plot(ax=ax, xlim=xlim)

            # Hide any extra subplots
            for j in range(n, len(axes)):
                axes[j].axis('off')
            
            plt.tight_layout()
            plt.show()

    @classmethod
    def plot_comparison(cls, sens_list: List['SensitivityData'], 
                      energy: Union[float, str, List[float], List[str]] = None, 
                      reactions: Union[List[int], int] = None, 
                      xlim: tuple = None):
        """Plot comparison of multiple sensitivity datasets.

        Args:
            sens_list (List[SensitivityData]): List of sensitivity datasets to compare.
            energy (float, str, or list): Energy value(s) to plot.
            reactions (int or list): Reaction number(s) to plot.
            xlim (tuple): Optional x-axis limits as (min, max).
            
        """
        # If no energy specified, use all energies
        if energy is None:
            energy = list(sens_list[0].data.keys())
        elif not isinstance(energy, list):
            energy = [energy]
        
        # Ensure reactions is always a list
        if reactions is None:
            sample_energy = energy[0]
            reactions = list(sens_list[0].data[sample_energy].keys())
        elif not isinstance(reactions, list):
            reactions = [reactions]

        colors_list = plt.rcParams['axes.prop_cycle'].by_key()['color']

        # Create a separate figure for each energy
        for e in energy:
            n = len(reactions)
            
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
            
            fig.suptitle(f"Energy = {e}")
            
            for i, rxn in enumerate(reactions):
                ax = axes[i]
                has_data = False
                
                for idx, sens in enumerate(sens_list):
                    if e in sens.data and rxn in sens.data[e]:
                        has_data = True
                        coef = sens.data[e][rxn]
                        color = colors_list[idx % len(colors_list)]
                        lp = np.array(coef.values_per_lethargy)
                        leth = np.array(coef.lethargy)
                        error_bars = np.array(coef.values) * np.array(coef.errors) / leth
                        x = np.array(coef.pert_energies)
                        y = np.append(lp, lp[-1])
                        ax.step(x, y, where='post', color=color, linewidth=2, label=sens.label)
                        x_mid = (x[:-1] + x[1:]) / 2.0
                        ax.errorbar(x_mid, lp, yerr=np.abs(error_bars), fmt=' ', 
                                  elinewidth=1.5, ecolor=color, capsize=2.5)
                
                if not has_data:
                    ax.text(0.5, 0.5, f"Reaction {rxn} not found", ha='center', va='center')
                    ax.axis('off')
                else:
                    ax.grid(True, alpha=0.3)
                    ax.set_title(f"MT = {rxn}")
                    ax.set_xlabel("Energy (MeV)")
                    ax.set_ylabel("Sensitivity per lethargy")
                    if xlim is not None:
                        ax.set_xlim(xlim)
                    ax.legend()

            # Hide any extra subplots
            for j in range(n, len(axes)):
                axes[j].axis('off')
            
            plt.tight_layout()
            plt.show()

    def export_plot_data(self) -> dict:
        """Export sensitivity data for plotting.

        Returns:
            dict: A dictionary with keys:
                label (str): The sensitivity data label.
                tally_name (str): The tally name.
                data (dict): Mapping energy values to reaction dictionaries, where each reaction dictionary maps a reaction number to a dict with:
                data[x] (list): Energy boundaries.
                data[y] (list): Sensitivity values.
                data[errors] (list): Error values.

        """
        export_data = {
            'label': self.label,
            'tally_name': self.tally_name,
            'data': {}
        }

        for energy_val, rxn_dict in self.data.items():
            export_data['data'][energy_val] = {}
            for rxn, coef in rxn_dict.items():
                x = coef.pert_energies
                # Use the computed values per lethargy as y values, and append the last value for step plotting
                lp = np.array(coef.values_per_lethargy)
                y = np.append(lp, lp[-1]).tolist()
                # Compute error bars: note that they are computed from values, errors and lethargy
                leth = np.array(coef.lethargy)
                error_bars = (np.array(coef.values) * np.array(coef.errors) / leth).tolist()

                export_data['data'][energy_val][rxn] = {
                    'x': x,
                    'y': y,
                    'errors': error_bars
                }

        return export_data


@dataclass
class Coefficients:
    """Container for sensitivity coefficients for a specific energy and reaction.

    :ivar energy: Energy value or label
    :type energy: Union[float, str]
    :ivar reaction: Reaction number
    :type reaction: int
    :ivar pert_energies: Perturbation energy boundaries
    :type pert_energies: List[float]
    :ivar values: Raw sensitivity coefficient values
    :type values: List[float]
    :ivar errors: Relative errors for the sensitivity coefficients
    :type errors: List[float]
    """
    energy: Union[float, str]
    reaction: int
    pert_energies: list[float]
    values: list[float]
    errors: list[float]

    @property
    def lethargy(self):
        """Calculate lethargy intervals between perturbation energies.

        Returns:
            List[float]: List of lethargy intervals.
        """
        return [np.log(self.pert_energies[i+1]/self.pert_energies[i]) for i in range(len(self.pert_energies)-1)]

    @property
    def values_per_lethargy(self):
        """Calculate sensitivity coefficients per unit lethargy.

        Returns:
            List[float]: Sensitivity coefficients normalized by lethargy intervals.
        """
        lethargy_vals = self.lethargy
        return [self.values[i]/lethargy_vals[i] for i in range(len(lethargy_vals))]
    
    # New helper method to plot onto a provided axis
    def _plot_on_ax(self, ax, xlim=None):
        """Plot sensitivity coefficients on a given matplotlib axis.

        Args:
            ax (matplotlib.axes.Axes): The axis to plot on.
            xlim (tuple): Optional x-axis limits as (min, max).
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
        if xlim is not None:
            ax.set_xlim(xlim)
        
    def plot(self, ax=None, xlim=None):
        """Create a new plot of sensitivity coefficients.

        Args:
            ax (matplotlib.axes.Axes): Optional existing axis to plot on.
            xlim (tuple): Optional x-axis limits as (min, max).

        Returns:
            matplotlib.axes.Axes: The axis containing the plot.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 4))
        self._plot_on_ax(ax, xlim=xlim)
        return ax
    

def compute_senstivity(input_path: str, mctal_path: str, tally: int, label: str) -> SensitivityData:
    """Compute sensitivity coefficients from MCNP input and output files.

    :param input_path: Path to MCNP input file
    :type input_path: str
    :param mctal_path: Path to MCNP MCTAL output file
    :type mctal_path: str
    :param tally: Tally number to analyze
    :type tally: int
    :param label: Label for the sensitivity data set
    :type label: str

    :returns: Object containing computed sensitivity coefficients
    :rtype: SensitivityData
    """
    input = read_mcnp(input_path)
    mctal = read_mctal(mctal_path)
    
    pert_energies = input.pert.pert_energies
    reactions = input.pert.reactions
    group_dict = input.pert.group_perts_by_reaction(2)
    
    energy = mctal.tally[tally].energies 
    r0 = np.array(mctal.tally[tally].results)
    e0 = np.array(mctal.tally[tally].errors)
       
    sens_result = SensitivityData(
        tally_id=tally,
        pert_energies=pert_energies,
        tally_name=mctal.tally[tally].name,
        label=label,
        data={}
    )

    for i in range(len(energy)):            # Loop over detector energies
        energy_data = {}
        for rxn in reactions:               # Loop over unique reaction
            sensCoef = np.zeros(len(group_dict[rxn]))
            sensErr = np.zeros(len(group_dict[rxn]))
            for j, pert in enumerate(group_dict[rxn]):    # Loop over list of perturbations - one per pert energy bin
                c1 = mctal.tally[tally].pert_data[pert].results[i]
                e1 = mctal.tally[tally].pert_data[pert].errors[i]
                sensCoef[j] = c1/r0[i]
                sensErr[j] = np.sqrt(e0[i]**2 + e1**2)
            
            energy_data[rxn] = Coefficients(
                energy=energy[i],
                reaction=rxn,
                pert_energies=pert_energies,
                values=sensCoef,
                errors=sensErr
            )
        
        sens_result.data[energy[i]] = energy_data

    if mctal.tally[tally].integral_result is not None:
        integral_data = {}
        for rxn in reactions:
            sensCoef_int = np.zeros(len(group_dict[rxn]))
            sensErr_int = np.zeros(len(group_dict[rxn]))
            for j, pert in enumerate(group_dict[rxn]):
                c1_int = mctal.tally[tally].pert_data[pert].integral_result
                e1_int = mctal.tally[tally].pert_data[pert].integral_error
                sensCoef_int[j] = c1_int / mctal.tally[tally].integral_result
                sensErr_int[j] = np.sqrt(mctal.tally[tally].integral_error**2 + e1_int**2)
            integral_data[rxn] = Coefficients(
                energy=mctal.tally[tally].integral_result,
                reaction=rxn,
                pert_energies=pert_energies,
                values=sensCoef_int,
                errors=sensErr_int
            )
        sens_result.data["integral"] = integral_data
    
    return sens_result
