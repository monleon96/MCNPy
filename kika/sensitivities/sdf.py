from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
import os
import numpy as np
from kika._constants import MT_TO_REACTION, ATOMIC_NUMBER_TO_SYMBOL
from kika.sensitivities.sensitivity import SensitivityData
from kika.energy_grids.utils import _identify_energy_grid
from kika.plotting import MultigroupXSPlotData, UncertaintyBand


@dataclass
class SDFReactionData:
    """Container for sensitivity data for a specific nuclide and reaction.
    
    :ivar zaid: ZAID of the nuclide
    :type zaid: int
    :ivar mt: MT reaction number
    :type mt: int
    :ivar sensitivity: List of sensitivity coefficients
    :type sensitivity: List[float]
    :ivar error: List of relative errors
    :type error: List[float]
    :ivar nuclide: Nuclide symbol (calculated from ZAID)
    :type nuclide: str
    :ivar reaction_name: Reaction name (calculated from MT)
    :type reaction_name: str
    """
    zaid: int
    mt: int
    sensitivity: List[float]
    error: List[float]
    nuclide: str = field(init=False)
    # reaction_name can be provided (e.g. for unknown MT numbers not in mapping);
    # if None it's inferred from MT_TO_REACTION in __post_init__.
    reaction_name: str | None = None
    
    def __post_init__(self):
        """Calculate and store nuclide symbol and reaction name after initialization.

        If the MT number is not present in ``MT_TO_REACTION`` and no
        ``reaction_name`` was supplied, we fall back to a generic label
        ``MT<mt>`` so that previously unknown / extended MT numbers do not
        cause parsing to fail. When a custom ``reaction_name`` is supplied
        (e.g. by the parser reading the legacy file), it is preserved as is.
        """
        # Calculate nuclide symbol
        z = self.zaid // 1000
        a = self.zaid % 1000
        
        if z not in ATOMIC_NUMBER_TO_SYMBOL:
            raise KeyError(f"Atomic number {z} not found in ATOMIC_NUMBER_TO_SYMBOL dictionary")
            
        self.nuclide = f"{ATOMIC_NUMBER_TO_SYMBOL[z]}-{a}"
        
        # Calculate reaction name
        if self.reaction_name is None:
            if self.mt in MT_TO_REACTION:
                self.reaction_name = MT_TO_REACTION[self.mt]
            else:
                # Fallback: preserve unknown MT with generic name
                self.reaction_name = f"MT{self.mt}"
    
    def __repr__(self) -> str:
        """Returns a formatted string representation of the reaction data.
        
        This method is called when the object is evaluated in interactive environments
        like Jupyter notebooks or the Python interpreter.
        
        :return: Formatted string representation of the reaction data
        :rtype: str
        """
        # Create a visually appealing header with a border
        header_width = 60
        header = "=" * header_width + "\n"
        header += f"{'SDF Reaction Data':^{header_width}}\n"
        header += "=" * header_width + "\n\n"
        
        # Create aligned key-value pairs with consistent width
        label_width = 25  # Width for labels
        
        info_lines = []
        info_lines.append(f"{'Nuclide:':{label_width}} {self.nuclide} (ZAID {self.zaid})")
        info_lines.append(f"{'Reaction:':{label_width}} {self.reaction_name} (MT {self.mt})")
        info_lines.append(f"{'Energy groups:':{label_width}} {len(self.sensitivity)}")
        
        stats = "\n".join(info_lines)
        
        # Data preview - show first few and last few sensitivity values
        data_preview = "\n\nSensitivity coefficients (preview):\n"
        data_preview += "  Group      Sensitivity       Rel. Error\n"
        data_preview += "  -----    --------------    ------------\n"
        
        # Show first 3 and last 3 groups, if available
        n_groups = len(self.sensitivity)
        preview_count = min(3, n_groups)
        
        for i in range(preview_count):
            data_preview += f"  {i+1:<5d}    {self.sensitivity[i]:14.6e}    {self.error[i]:12.6e}\n"
            
        # Add ellipsis if there are more than 6 groups
        if n_groups > 6:
            data_preview += "  ...\n"
            
        # Show last 3 groups if there are more than 3 groups
        if n_groups > 3:
            for i in range(max(preview_count, n_groups-3), n_groups):
                data_preview += f"  {i+1:<5d}    {self.sensitivity[i]:14.6e}    {self.error[i]:12.6e}\n"
        
        return header + stats + data_preview

    def to_plot_data(
        self,
        pert_energies,
        per_lethargy: bool = True,
        uncertainty: bool = True,
        sigma: float = 1.0,
        uncertainty_style: str = 'errorbar',
        label: str = None,
        **styling_kwargs
    ) -> Union['MultigroupXSPlotData', Tuple['MultigroupXSPlotData', 'UncertaintyBand']]:
        """Convert sensitivity data for this reaction into PlotData objects.

        Returns a :class:`~kika.plotting.MultigroupXSPlotData` (step plot) and,
        optionally, an :class:`~kika.plotting.UncertaintyBand` suitable for
        :class:`~kika.plotting.PlotBuilder`.

        :param pert_energies: Energy bin boundaries (n+1 values, ascending order in MeV).
        :type pert_energies: array-like
        :param per_lethargy: If True, normalise sensitivity values by the lethargy
            width of each bin (matches ``Coefficients.plot()`` behaviour).
        :type per_lethargy: bool
        :param uncertainty: If True, return an ``UncertaintyBand`` alongside the
            nominal data.
        :type uncertainty: bool
        :param sigma: Sigma multiplier for the uncertainty band.
        :type sigma: float
        :param uncertainty_style: Rendering style for the uncertainty band:
            ``'errorbar'`` (default for sensitivity data) or ``'band'``.
        :type uncertainty_style: str
        :param label: Legend label. Auto-generated as ``"<nuclide> <reaction_name>"``
            (e.g. ``"Fe-56 (n,el)"``) when *None*.
        :type label: str, optional
        :param styling_kwargs: Forwarded to ``MultigroupXSPlotData``
            (``color``, ``linestyle``, ``linewidth``, etc.).
        :returns: ``MultigroupXSPlotData`` when *uncertainty=False*,
            or ``(MultigroupXSPlotData, UncertaintyBand)`` when *uncertainty=True*.
        :rtype: MultigroupXSPlotData or Tuple[MultigroupXSPlotData, UncertaintyBand]
        """
        energies = np.asarray(pert_energies, dtype=float)
        sens = np.asarray(self.sensitivity, dtype=float)
        n = len(sens)

        if per_lethargy:
            lethargy = np.log(energies[1:] / energies[:-1])
            y_vals = sens / lethargy
        else:
            y_vals = sens.copy()

        # Step plot: n+1 x-points, repeat last y value
        x = energies
        y = np.append(y_vals, y_vals[-1])

        if label is None:
            label = f"{self.nuclide} {self.reaction_name}"

        plot_data = MultigroupXSPlotData(
            x=x,
            y=y,
            label=label,
            plot_type='step',
            step_where='post',
            zaid=self.zaid,
            mt=self.mt,
            energy_bins=energies,
            **styling_kwargs,
        )

        if not uncertainty:
            return plot_data

        # Build UncertaintyBand from relative errors.
        # self.error stores relative errors (fractional) on the raw sensitivity.
        # We need relative errors on the plotted y values. Since per-lethargy
        # divides by a constant per bin, the *relative* error is unchanged.
        rel_err = np.asarray(self.error, dtype=float)
        rel_err_extended = np.append(rel_err, rel_err[-1])

        band = UncertaintyBand(
            x=x,
            relative_uncertainty=rel_err_extended,
            sigma=sigma,
            label=f"{label} ({sigma}\u03c3)" if sigma != 1.0 else None,
            style=uncertainty_style,
        )

        return plot_data, band


@dataclass
class SDFData:
    """Container for SDF data.
    
    :ivar title: Title of the SDF dataset
    :type title: str
    :ivar energy: Energy value or label
    :type energy: str
    :ivar pert_energies: List of perturbation energy boundaries
    :type pert_energies: List[float]
    :ivar r0: Unperturbed tally result (reference response value)
    :type r0: float
    :ivar e0: Relative error of the unperturbed tally result (σ/μ)
    :type e0: float
    :ivar data: List of reaction-specific sensitivity data
    :type data: List[SDFReactionData]
    """
    title: str
    energy: str
    pert_energies: List[float]
    r0: float = None
    e0: float = None
    data: List[SDFReactionData] = field(default_factory=list)

    def __repr__(self) -> str:
        """Returns a detailed formatted string representation of the SDF data.
        
        This method is called when the object is evaluated in interactive environments
        like Jupyter notebooks or the Python interpreter.
        
        :return: Formatted string representation of the SDF data
        :rtype: str
        """
        # Create a visually appealing header with a border
        header_width = 70
        header = "=" * header_width + "\n"
        header += f"{'SDF Data: ' + self.title:^{header_width}}\n"
        header += f"{'Energy range: ' + self.energy:^{header_width}}\n"
        header += "=" * header_width + "\n\n"
        
        # Create aligned key-value pairs with consistent width
        label_width = 25  # Width for labels
        
        # Basic information section
        info_lines = []
        info_lines.append(f"{'Response value:':{label_width}} {self.r0:.6e} ± {self.e0*100:.2f}% (rel)")
        info_lines.append(f"{'Energy groups:':{label_width}} {len(self.pert_energies) - 1}")
        
        # Add energy grid structure identification
        grid_name = _identify_energy_grid(self.pert_energies)
        if grid_name:
            info_lines.append(f"{'Energy structure:':{label_width}} {grid_name}")
            
        info_lines.append(f"{'Sensitivity profiles:':{label_width}} {len(self.data)}")
        
        # Count unique nuclides
        nuclides = {react.nuclide for react in self.data}
        info_lines.append(f"{'Unique nuclides:':{label_width}} {len(nuclides)}")
        
        stats = "\n".join(info_lines)
        
        # Energy grid preview
        energy_preview = "\n\nEnergy grid (preview):"
        energy_grid = "  " + ", ".join(f"{e:.6e}" for e in self.pert_energies[:3])
        
        if len(self.pert_energies) > 6:
            energy_grid += ", ... , " 
            energy_grid += ", ".join(f"{e:.6e}" for e in self.pert_energies[-3:])
        elif len(self.pert_energies) > 3:
            energy_grid += ", " + ", ".join(f"{e:.6e}" for e in self.pert_energies[3:])
            
        energy_preview += "\n  " + energy_grid
        
        # Data summary - most important nuclides and reactions with indices
        data_summary = "\n\nNuclides and reactions (with access indices):\n"
        
        # Group by nuclide
        nuclide_reactions = {}
        nuclide_indices = {}
        
        # Store all reaction data with their indices
        for idx, react in enumerate(self.data):
            if react.nuclide not in nuclide_reactions:
                nuclide_reactions[react.nuclide] = []
                nuclide_indices[react.nuclide] = []
            nuclide_reactions[react.nuclide].append((react.reaction_name, react.mt))
            nuclide_indices[react.nuclide].append(idx)
        
        # Determine width for consistent alignment
        reaction_width = 30  # Base width for reaction name + MT
        
        # Show data for each nuclide (limit to first 5 nuclides)
        for i, nuclide in enumerate(sorted(nuclide_reactions.keys())):
            if i >= 5:
                data_summary += f"\n  ... ({len(nuclides) - 5} more nuclides) ...\n"
                break
                
            data_summary += f"\n  {nuclide}:\n"
            reactions = nuclide_reactions[nuclide]
            indices = nuclide_indices[nuclide]
            
            # Sort by MT number but keep track of original indices
            sorted_data = sorted(zip(reactions, indices), key=lambda x: x[0][1])
            
            for j, ((name, mt), idx) in enumerate(sorted_data):
                if j >= 10:  # Limit to 10 reactions per nuclide
                    data_summary += f"    ... ({len(reactions) - 10} more reactions) ...\n"
                    break
                # Format the reaction info with consistent alignment for the "access with" part
                reaction_info = f"{name} (MT={mt})"
                data_summary += f"    {reaction_info:{reaction_width}} access with .data[{idx}]\n"
        
        # Footer with available methods
        footer = "\n\nAvailable methods:\n"
        footer += "- .to_plot_data() - Get PlotData for use with PlotBuilder\n"
        footer += "- .write_file() - Write SDF data to a file\n"
        footer += "- .group_inelastic_reactions() - Group MT 51-91 into MT 4\n"
        footer += "- SDFData.merge() - Merge multiple SDF objects\n"
        footer += "- SDFData.can_merge() - Check if SDF objects can be merged\n"

        return header + stats + energy_preview + data_summary + footer

    def to_plot_data(
        self,
        index: int = None,
        zaid: int = None,
        mt: int = None,
        per_lethargy: bool = True,
        uncertainty: bool = True,
        sigma: float = 1.0,
        uncertainty_style: str = 'errorbar',
        label: str = None,
        **styling_kwargs
    ) -> Union['MultigroupXSPlotData', Tuple['MultigroupXSPlotData', 'UncertaintyBand']]:
        """Convert a reaction's sensitivity data into PlotData objects.

        Look up a reaction either by *index* into :attr:`data` or by
        *(zaid, mt)* pair, then delegate to
        :meth:`SDFReactionData.to_plot_data`.

        :param index: Direct index into ``self.data``.
        :type index: int, optional
        :param zaid: ZAID of the nuclide (requires *mt* as well).
        :type zaid: int, optional
        :param mt: MT reaction number (requires *zaid* as well).
        :type mt: int, optional
        :param per_lethargy: Normalise by lethargy width (default ``True``).
        :type per_lethargy: bool
        :param uncertainty: Include an ``UncertaintyBand`` (default ``True``).
        :type uncertainty: bool
        :param sigma: Sigma multiplier for the uncertainty band.
        :type sigma: float
        :param uncertainty_style: Rendering style for the uncertainty band:
            ``'errorbar'`` (default for sensitivity data) or ``'band'``.
        :type uncertainty_style: str
        :param label: Legend label (auto-generated if *None*).
        :type label: str, optional
        :param styling_kwargs: Forwarded to ``MultigroupXSPlotData``.
        :returns: See :meth:`SDFReactionData.to_plot_data`.
        :rtype: MultigroupXSPlotData or Tuple[MultigroupXSPlotData, UncertaintyBand]
        :raises ValueError: If neither *index* nor *(zaid, mt)* is given, or if
            the requested reaction is not found.
        """
        if index is not None:
            if index < 0 or index >= len(self.data):
                raise ValueError(
                    f"index {index} out of range (0..{len(self.data) - 1})"
                )
            reaction = self.data[index]
        elif zaid is not None and mt is not None:
            reaction = None
            for rd in self.data:
                if rd.zaid == zaid and rd.mt == mt:
                    reaction = rd
                    break
            if reaction is None:
                raise ValueError(
                    f"No reaction found with ZAID={zaid}, MT={mt}"
                )
        else:
            raise ValueError(
                "Provide either 'index' or both 'zaid' and 'mt'"
            )

        return reaction.to_plot_data(
            self.pert_energies,
            per_lethargy=per_lethargy,
            uncertainty=uncertainty,
            sigma=sigma,
            uncertainty_style=uncertainty_style,
            label=label,
            **styling_kwargs,
        )

    @classmethod
    def merge(cls, sdf_list: List['SDFData'],
              title: Optional[str] = None,
              energy: Optional[str] = None,
              r0: Optional[float] = None,
              e0: Optional[float] = None) -> 'SDFData':
        """Merge multiple SDFData objects into a single one.

        All SDFs must share the same perturbation energy grid.  The merged
        object contains every ``SDFReactionData`` entry from the inputs;
        duplicate ``(zaid, mt)`` pairs are not allowed.

        :param sdf_list: List of SDFData objects to combine.
        :type sdf_list: List[SDFData]
        :param title: Override title. If *None*, uses the common title when all
            inputs match, otherwise joins distinct titles with ``" + "``.
        :type title: Optional[str]
        :param energy: Override energy label. If *None*, uses the common label
            when all inputs match, otherwise raises ``ValueError``.
        :type energy: Optional[str]
        :param r0: Override response value. If *None*, uses the common value
            when all inputs agree, otherwise raises ``ValueError``.
        :type r0: Optional[float]
        :param e0: Override relative error. If *None*, uses the common value
            when all inputs agree, otherwise raises ``ValueError``.
        :type e0: Optional[float]
        :returns: A new merged SDFData instance.
        :rtype: SDFData
        :raises ValueError: If *sdf_list* is empty, energy grids differ,
            r0/e0 values conflict (and none was given), energy labels differ
            (and none was given), or duplicate (zaid, mt) pairs are found.
        """
        if not sdf_list:
            raise ValueError("Cannot merge an empty list of SDFData objects")

        # --- energy grid ---
        ref_energies = sdf_list[0].pert_energies
        for i, sdf in enumerate(sdf_list[1:], 1):
            if sdf.pert_energies != ref_energies:
                raise ValueError(
                    f"Energy grids do not match: SDF 0 ('{sdf_list[0].title}') has "
                    f"{len(ref_energies)} boundaries, SDF {i} ('{sdf.title}') has "
                    f"{len(sdf.pert_energies)} boundaries"
                )

        # --- r0 / e0 ---
        if r0 is None:
            r0_values = {s.r0 for s in sdf_list if s.r0 is not None}
            if len(r0_values) > 1:
                raise ValueError(
                    f"SDFs have different r0 values {r0_values}. "
                    f"Provide the 'r0' parameter explicitly."
                )
            r0 = r0_values.pop() if r0_values else None

        if e0 is None:
            e0_values = {s.e0 for s in sdf_list if s.e0 is not None}
            if len(e0_values) > 1:
                raise ValueError(
                    f"SDFs have different e0 values {e0_values}. "
                    f"Provide the 'e0' parameter explicitly."
                )
            e0 = e0_values.pop() if e0_values else None

        # --- energy label ---
        if energy is None:
            labels = {s.energy for s in sdf_list}
            if len(labels) == 1:
                energy = labels.pop()
            else:
                raise ValueError(
                    f"SDFs have different energy labels {labels}. "
                    f"Provide the 'energy' parameter explicitly."
                )

        # --- title ---
        if title is None:
            titles = list(dict.fromkeys(s.title for s in sdf_list))  # unique, order-preserving
            title = " + ".join(titles)

        # --- duplicate check & collect data ---
        seen = {}
        combined_data = []
        for i, sdf in enumerate(sdf_list):
            for rd in sdf.data:
                key = (rd.zaid, rd.mt)
                if key in seen:
                    raise ValueError(
                        f"Duplicate reaction: {rd.nuclide} {rd.reaction_name} "
                        f"(ZAID={rd.zaid}, MT={rd.mt}) appears in SDF {seen[key]} "
                        f"and SDF {i} ('{sdf.title}')"
                    )
                seen[key] = i
                combined_data.append(rd)

        return cls(
            title=title,
            energy=energy,
            pert_energies=ref_energies,
            r0=r0,
            e0=e0,
            data=combined_data,
        )

    @classmethod
    def can_merge(cls, sdf_list: List['SDFData'],
                  energy: Optional[str] = None,
                  r0: Optional[float] = None,
                  e0: Optional[float] = None) -> Tuple[bool, str]:
        """Check whether a list of SDFData objects can be merged.

        Runs the same validation as :meth:`merge` but never raises.
        Pass the same override parameters you would pass to :meth:`merge`
        to check whether the merge would succeed with those overrides.

        :param sdf_list: List of SDFData objects to check.
        :type sdf_list: List[SDFData]
        :param energy: Override energy label (same as in :meth:`merge`).
        :type energy: Optional[str]
        :param r0: Override response value (same as in :meth:`merge`).
        :type r0: Optional[float]
        :param e0: Override relative error (same as in :meth:`merge`).
        :type e0: Optional[float]
        :returns: ``(True, "")`` if the merge would succeed, or
            ``(False, reason)`` with a human-readable explanation otherwise.
        :rtype: Tuple[bool, str]
        """
        try:
            cls.merge(sdf_list, energy=energy, r0=r0, e0=e0)
            return True, ""
        except ValueError as exc:
            return False, str(exc)

    def __add__(self, other: 'SDFData') -> 'SDFData':
        """Merge two SDFData objects using the ``+`` operator.

        :param other: Another SDFData object.
        :type other: SDFData
        :returns: A new merged SDFData instance.
        :rtype: SDFData
        """
        if not isinstance(other, SDFData):
            return NotImplemented
        return SDFData.merge([self, other])

    def write_file(self, output_dir: Optional[str] = None):
        """
        Write the SDF data to a file using the legacy format.
        
        :param output_dir: Directory where the SDF file will be written. If None, uses current directory.
        :type output_dir: Optional[str]
        """
        # Use current directory if output_dir is not provided
        if output_dir is None:
            output_dir = os.getcwd()
        
        # Ensure directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a clean filename from title and energy
        filename = f"{self.title}_{self.energy}.sdf"
        # Ensure filename is valid by removing problematic characters
        filename = filename.replace(' ', '_').replace('/', '_').replace('\\', '_')
        
        # Create full path to file
        filepath = os.path.join(output_dir, filename)
        
        ngroups = len(self.pert_energies) - 1
        nprofiles = len(self.data)
        
        # Sort the data by ZAID and then by MT number
        sorted_data = sorted(self.data, key=lambda x: (x.zaid, x.mt))
        
        with open(filepath, 'w', encoding='utf-8') as file:
            # Write header
            file.write(f"{self.title} MCNP to SCALE sdf {ngroups}gr\n")
            file.write(f"       {ngroups} number of neutron groups\n")
            file.write(f"       {nprofiles}  number of sensitivity profiles         {nprofiles} are region integrated\n")
            
            # Ensure r0 and e0 are properly formatted
            r0_value = 0.0 if self.r0 is None else self.r0
            e0_value = 0.0 if self.e0 is None else self.e0
            file.write(f"  {r0_value:.6E} +/-   {e0_value:.6E}\n")
            
            # Write energy grid data - reversed to be in descending order
            file.write("energy boundaries:\n")
            energy_lines = ""
            # Create reversed list of energies for writing in descending order
            descending_energies = list(reversed(self.pert_energies))
            for idx, energy in enumerate(descending_energies):
                if idx > 0 and idx % 5 == 0:
                    energy_lines += "\n"
                energy_lines += f"{energy: >14.6E}"
            energy_lines += "\n"
            file.write(energy_lines)
            
            # Write sensitivity coefficient and standard deviations data for each reaction
            # using the sorted data
            for reaction in sorted_data:
                file.write(self._format_reaction_data(reaction))
        
        # Add print message indicating where the file was saved
        print(f"SDF file saved successfully: {filepath}")

    def _format_reaction_data(self, reaction: SDFReactionData) -> str:
        """
        Format a single SDFReactionData block to match the legacy file structure.
        
        :param reaction: The reaction data to format
        :type reaction: SDFReactionData
        :returns: Formatted string for the reaction data block
        :rtype: str
        """
        # Use the properties to get the nuclide symbol and reaction name
        form = reaction.nuclide
        reac = reaction.reaction_name
        
        # Format the header line for this reaction
        block = f"{form:<13}{reac:<17}{reaction.zaid:>5}{reaction.mt:>7}\n"
        block += "      0      0\n"
        block += "  0.000000E+00  0.000000E+00      0      0\n"
        
        # Calculate the 5 scalar values according to SDF specification
        # 1. Energy-integrated sensitivity coefficient (sum of groupwise sensitivities)
        s_int = sum(reaction.sensitivity)
        
        # 2. Standard deviation of S_int (error propagation for sum)
        # For absolute errors: σ_total = sqrt(Σ σ_i²)
        s_int_std = (sum(err**2 for err in reaction.error))**0.5
        
        # 3. Sum of absolute values of groupwise sensitivities
        sum_abs = sum(abs(s) for s in reaction.sensitivity)
        
        # 4. "osc" = sum of sensitivities with sign opposite to S_int
        if s_int == 0:
            # If S_int is zero, all terms contribute to oscillation
            osc = sum_abs
        else:
            # Collect terms with opposite sign to S_int
            s_int_sign = 1 if s_int >= 0 else -1
            osc = sum(s for s in reaction.sensitivity if (s >= 0) != (s_int_sign > 0))
        
        # 5. Standard deviation of "osc" (error propagation for the oscillating terms)
        if s_int == 0:
            # If S_int is zero, all errors contribute to osc uncertainty
            osc_std = s_int_std
        else:
            # Only include errors for terms that contribute to osc
            s_int_sign = 1 if s_int >= 0 else -1
            osc_std = (sum(err**2 for s, err in zip(reaction.sensitivity, reaction.error) 
                          if (s >= 0) != (s_int_sign > 0)))**0.5
        
        block += f"{s_int:>14.6E}{s_int_std:>14.6E}{sum_abs:>14.6E}{osc:>14.6E}{osc_std:>14.6E}\n"
        
        # Reverse sensitivity and error arrays to match the descending energy order
        reversed_sensitivity = list(reversed(reaction.sensitivity))
        reversed_error = list(reversed(reaction.error))
        
        # Write sensitivity coefficients with 5 per line (in reversed order)
        for idx, sens in enumerate(reversed_sensitivity):
            if idx > 0 and idx % 5 == 0:
                block += "\n"
            block += f"{sens:>14.6E}"
        block += "\n"
        
        # Write standard deviations with 5 per line (in reversed order)
        for idx, err in enumerate(reversed_error):
            if idx > 0 and idx % 5 == 0:
                block += "\n"
            block += f"{err:>14.6E}"
        block += "\n"
        return block

    def group_inelastic_reactions(self, replace: bool = False, remove_originals: bool = True) -> None:
        """Group inelastic reactions (MT 51-91) into MT 4 for each nuclide.
        
        This method combines all inelastic scattering reactions (MT 51-91) into 
        the total inelastic scattering reaction (MT 4) for each nuclide.
        
        :param replace: If True, replace existing MT 4 data if present.
                        If False, raise an error when MT 4 is already present.
        :type replace: bool, optional
        :param remove_originals: If True, remove the original MT 51-91 reactions
                                after combining them.
        :type remove_originals: bool, optional
        :raises ValueError: If MT 4 already exists for a nuclide and replace=False
        """
        # Group data by ZAID
        nuclide_reactions = {}
        for react in self.data:
            if react.zaid not in nuclide_reactions:
                nuclide_reactions[react.zaid] = []
            nuclide_reactions[react.zaid].append(react)
        
        # Process each nuclide
        for zaid, reactions in nuclide_reactions.items():
            # Find MT 4 if it exists
            mt4_exists = False
            mt4_reaction = None
            for react in reactions:
                if react.mt == 4:
                    mt4_exists = True
                    mt4_reaction = react
                    break
            
            # Find inelastic reactions (MT 51-91)
            inelastic_reactions = [r for r in reactions if 51 <= r.mt <= 91]
            
            # Skip if no inelastic reactions found for this nuclide
            if not inelastic_reactions:
                continue
            
            # Handle existing MT 4 reaction
            if mt4_exists and not replace:
                # Calculate the nuclide symbol for more informative error message
                z = zaid // 1000
                a = zaid % 1000
                symbol = ATOMIC_NUMBER_TO_SYMBOL.get(z, f"unknown_{z}")
                nuclide = f"{symbol}-{a}"
                
                raise ValueError(
                    f"MT 4 already exists for nuclide {nuclide} (ZAID {zaid}). "
                    f"Set replace=True to overwrite."
                )
            
            # Sum sensitivity and error values from all inelastic reactions
            n_groups = len(inelastic_reactions[0].sensitivity)
            summed_sensitivity = [0.0] * n_groups
            summed_error_squared = [0.0] * n_groups  
            
            for react in inelastic_reactions:
                for i in range(n_groups):
                    summed_sensitivity[i] += react.sensitivity[i]
                    # Convert relative error to absolute error (multiply by sensitivity), then square
                    absolute_error = react.sensitivity[i] * react.error[i]
                    summed_error_squared[i] += absolute_error ** 2 
            
            # Take square root of summed squared errors and convert back to relative errors
            summed_error = []
            for i in range(n_groups):
                absolute_error = summed_error_squared[i] ** 0.5
                # Convert back to relative error (divide by sensitivity)
                # Handle potential division by zero
                if summed_sensitivity[i] != 0:
                    relative_error = absolute_error / abs(summed_sensitivity[i])
                else:
                    relative_error = 0.0
                summed_error.append(relative_error)
            
            # Create or update MT 4 reaction
            if mt4_exists:
                mt4_reaction.sensitivity = summed_sensitivity
                mt4_reaction.error = summed_error
                print(f"Updated MT 4 for {mt4_reaction.nuclide} (ZAID {zaid})")
            else:
                new_mt4 = SDFReactionData(
                    zaid=zaid,
                    mt=4,
                    sensitivity=summed_sensitivity,
                    error=summed_error
                )
                self.data.append(new_mt4)
                print(f"Created MT 4 for {new_mt4.nuclide} (ZAID {zaid})")
            
            # Remove original MT 51-91 reactions if requested
            if remove_originals:
                mt_values = [r.mt for r in inelastic_reactions]
                print(f"Removed MT {', '.join(map(str, mt_values))} for {inelastic_reactions[0].nuclide} (ZAID {zaid})")
                
                # Remove the reactions from self.data
                self.data = [r for r in self.data if not (r.zaid == zaid and 51 <= r.mt <= 91)]
