"""Utility functions for sensitivity analysis and processing.

This module contains functions for computing sensitivity coefficients from MCNP files,
creating SDF data objects from sensitivity data, and other related utility functions.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from kika.mcnp.parse_input import read_mcnp
from kika.mcnp.parse_mctal import read_mctal
from kika._constants import ATOMIC_NUMBER_TO_SYMBOL, MT_TO_REACTION
from kika.sensitivities.sensitivity import SensitivityData, Coefficients, TaylorCoefficients
from kika.sensitivities.sdf import SDFData, SDFReactionData
from kika.serpent.sens import SensitivityFile
import math
import matplotlib.pyplot as plt


def compute_sensitivity(inputfile: str, mctalfile: str, tally: int,
                        zaid: int = None, label: str = '',
                        material: Optional[int] = None,
                        cell: int = 0,
                        pert_metadata: Optional[List[Tuple[int, int, int, int]]] = None
                        ) -> SensitivityData:
    """Compute sensitivity coefficients from MCNP input and output files.

    :param inputfile: Path to MCNP input file containing the PERT cards
    :type inputfile: str
    :param mctalfile: Path to MCNP MCTAL output file
    :type mctalfile: str
    :param tally: Tally number to analyze
    :type tally: int
    :param zaid: ZAID of the nuclide being perturbed. If None and pert_metadata is
        provided, extracted from the metadata (must be a single unique ZAID).
    :type zaid: int, optional
    :param label: Label for the sensitivity data set
    :type label: str
    :param material: Optional material number to filter perturbations by.
        If None, all perturbations are used (backward-compatible behavior).
    :type material: int, optional
    :param cell: Cell ID to use for multi-cell tallies. Default is 0, which is
        the total-over-cells bin in MCNP (added by the T parameter in the tally
        definition). For single-cell tallies this parameter is ignored.
    :type cell: int
    :param pert_metadata: Optional list of (start_pert, end_pert, zaid, material) tuples.
        Assigns zaid and original_material to PERT cards in the given ranges.
        When provided, zaid and material can be inferred from the metadata.
    :type pert_metadata: list of tuple, optional
    :returns: Object containing computed sensitivity coefficients
    :rtype: SensitivityData
    :raises ValueError: If no zaid is given and none can be resolved from
        pert_metadata or PERT-card metadata, or if pert_metadata contains
        multiple ZAIDs (use compute_total_sensitivity instead).
    """
    input_data = read_mcnp(inputfile, pert_metadata=pert_metadata)
    mctal = read_mctal(mctalfile)

    # Resolve zaid and material from pert_metadata if not explicitly provided
    if pert_metadata is not None:
        meta_zaids = sorted({z for _, _, z, _ in pert_metadata})
        meta_mats = sorted({m for _, _, _, m in pert_metadata})
        if zaid is None:
            if len(meta_zaids) != 1:
                raise ValueError(
                    f"pert_metadata contains multiple ZAIDs {meta_zaids}. "
                    "Pass zaid= explicitly or use compute_total_sensitivity()."
                )
            zaid = meta_zaids[0]
        if material is None and len(meta_mats) == 1:
            material = meta_mats[0]
        elif material is None and len(meta_mats) > 1:
            raise ValueError(
                f"pert_metadata contains multiple materials {meta_mats}. "
                "Pass material= explicitly or use compute_total_sensitivity()."
            )

    # Check that we have zaid info
    if zaid is None:
        raise ValueError(
            "No zaid provided and no ZAID metadata found in PERT cards. "
            "Provide pert_metadata=[(start, end, zaid, material), ...] "
            "or add 'c kika:pert_zaid=ZAID pert_mat=MAT' comments to the input file."
        )

    if not input_data.perturbation.has_zaid_info:
        # No per-card ZAID metadata (no pert_metadata argument and no
        # 'c kika:pert_zaid=' comments in the input file). A single ZAID was
        # provided explicitly, so treat this as a single-nuclide input and
        # assign that ZAID to every PERT card. Multi-nuclide inputs must supply
        # per-card metadata (or use compute_total_sensitivity()).
        for p in input_data.perturbation.pert.values():
            p.zaid = zaid

    # Guard: if no material specified and multiple materials exist, raise error
    if material is None:
        detected_mats = input_data.perturbation.materials_for_zaid(zaid)
        if len(detected_mats) > 1:
            raise ValueError(
                f"Multiple materials {detected_mats} found for ZAID {zaid}. "
                "Use compute_total_sensitivity() to sum across materials, "
                "or pass material= to select a single material."
            )
        elif len(detected_mats) == 1:
            material = detected_mats[0]

    return _compute_sensitivity_impl(input_data, mctal, tally, zaid, label, material, cell)


def _compute_sensitivity_impl(input_data, mctal, tally: int,
                              zaid: int, label: str,
                              material: Optional[int] = None,
                              cell: int = 0) -> SensitivityData:
    """Core sensitivity computation on pre-parsed MCNP input and mctal data.

    This is the internal implementation shared by ``compute_sensitivity`` and
    ``compute_total_sensitivity``.  Callers are responsible for parsing files
    and validating *zaid* / *material* before calling this function.

    Parameters
    ----------
    cell : int
        Cell ID to select for multi-cell tallies. Default 0 (total-over-cells
        in MCNP, added by the T parameter in the tally definition).
    """
    pert_energies = input_data.perturbation.pert_energies
    reactions = input_data.perturbation.reactions_for_zaid(zaid)
    group_dict_first = input_data.perturbation._group_perts_by_reaction(2, material=material, zaid=zaid)

    # Check if second-order perturbations are available
    has_second_order = False
    try:
        group_dict_second = input_data.perturbation._group_perts_by_reaction(3, material=material, zaid=zaid)
        has_second_order = bool(group_dict_second)
    except:
        group_dict_second = {}

    tally_obj = mctal.tally[tally]
    energy = tally_obj.energies
    n_energies = len(energy)
    # For tallies with no energy bins (e.g. DPA with FM multiplier), results
    # are stored as one value per cell.  Use n_energies_stride = max(1, ...)
    # so cell_offset computation is correct.
    n_energies_stride = max(n_energies, 1)

    # Handle multi-cell tallies: select requested cell
    n_cells = tally_obj.n_cells_surfaces if tally_obj.n_cells_surfaces > 0 else 1
    cell_offset = 0
    cell_idx = 0
    if n_cells > 1:
        if tally_obj.cell_surface_ids and cell in tally_obj.cell_surface_ids:
            cell_idx = tally_obj.cell_surface_ids.index(cell)
        else:
            raise ValueError(
                f"Cell {cell} not found in tally {tally}. "
                f"Available cells: {tally_obj.cell_surface_ids}"
            )
        cell_offset = cell_idx * n_energies_stride

    all_r0 = np.array(tally_obj.results)
    all_e0 = np.array(tally_obj.errors)
    r0 = all_r0[cell_offset:cell_offset + n_energies_stride]
    e0 = all_e0[cell_offset:cell_offset + n_energies_stride]

    # Prepare all the data first before creating the SensitivityData object
    full_data = {}
    coefficients = {}  # Store Taylor coefficients by energy and reaction

    for i in range(len(energy)):            # Loop over detector energies
        energy_data = {}
        coeff_data = {}

        # Calculate energy boundaries for the energy string
        if i == 0:
            lower_bound = 0.0
        else:
            lower_bound = energy[i-1]
        upper_bound = energy[i]
        # Format energy as string in the required format
        energy_str = f"{lower_bound:.2e}_{upper_bound:.2e}"

        for rxn in reactions:               # Loop over unique reaction
            # First-order processing
            sensCoef = np.zeros(len(group_dict_first[rxn]))
            sensErr = np.zeros(len(group_dict_first[rxn]))

            for j, pert in enumerate(group_dict_first[rxn]):
                c1 = tally_obj.perturbation[pert].results[cell_offset + i]
                e1 = tally_obj.perturbation[pert].errors[cell_offset + i]
                sensCoef[j] = c1/r0[i]
                sensErr[j] = np.sqrt(e0[i]**2 + e1**2)

            # Store first-order coefficients
            energy_data[rxn] = Coefficients(
                energy=energy_str,
                reaction=rxn,
                pert_energies=pert_energies,
                values=sensCoef,
                errors=sensErr,
                r0=float(r0[i]),
                e0=float(e0[i])
            )

            # Second-order processing (if available)
            if has_second_order and rxn in group_dict_second:
                c2_values = []
                c1_values = []  # Store the actual Taylor coefficients c1
                c1_errors = []  # Store errors of Taylor coefficients c1
                c2_errors = []  # Store errors of Taylor coefficients c2

                for j, pert in enumerate(group_dict_second[rxn]):
                    # Get first-order Taylor coefficient directly (not the sensitivity)
                    c1 = tally_obj.perturbation[group_dict_first[rxn][j]].results[cell_offset + i]
                    c1_err = tally_obj.perturbation[group_dict_first[rxn][j]].errors[cell_offset + i]
                    c1_values.append(c1)
                    c1_errors.append(c1_err)

                    # Get second-order Taylor coefficient directly
                    c2 = tally_obj.perturbation[pert].results[cell_offset + i]
                    c2_err = tally_obj.perturbation[pert].errors[cell_offset + i]
                    c2_values.append(c2)
                    c2_errors.append(c2_err)

                # Calculate the ratio c2/c1 directly for each energy bin
                ratio_values = []
                for j in range(len(c1_values)):
                    if c1_values[j] != 0:  # Avoid division by zero
                        ratio_values.append(c2_values[j] / c1_values[j])
                    else:
                        ratio_values.append(float('nan'))

                coeff_data[rxn] = TaylorCoefficients(
                    energy=energy_str,
                    reaction=rxn,
                    pert_energies=pert_energies,
                    c1=c1_values,
                    c2=c2_values,
                    ratio=ratio_values,
                    c1_errors=c1_errors,
                    c2_errors=c2_errors
                )

        full_data[energy_str] = energy_data
        if coeff_data:  # Only add if there are any coefficients
            coefficients[energy_str] = coeff_data

    # Process integral results if available
    has_integral = False
    integral_r0 = None
    integral_e0 = None

    if n_energies == 0:
        # No energy bins — the single per-cell value IS the integral
        has_integral = True
        integral_r0 = float(r0[0])
        integral_e0 = float(e0[0])
    elif n_cells <= 1 and tally_obj.integral_result is not None:
        has_integral = True
        integral_r0 = tally_obj.integral_result
        integral_e0 = tally_obj.integral_error
    elif n_cells > 1 and hasattr(tally_obj, 'total_energy_values') and tally_obj.total_energy_values:
        has_integral = True
        integral_r0 = tally_obj.total_energy_values['results'][cell_idx]
        integral_e0 = tally_obj.total_energy_values['errors'][cell_idx]

    if has_integral:
        integral_data = {}
        integral_coeff_data = {}

        for rxn in reactions:
            sensCoef_int = np.zeros(len(group_dict_first[rxn]))
            sensErr_int = np.zeros(len(group_dict_first[rxn]))

            # Process first-order coefficients for integral results
            for j, pert in enumerate(group_dict_first[rxn]):
                pert_obj = tally_obj.perturbation[pert]
                if n_energies == 0:
                    # No energy bins — result is directly per-cell
                    c1_int = pert_obj.results[cell_idx]
                    e1_int = pert_obj.errors[cell_idx]
                elif n_cells > 1 and hasattr(pert_obj, 'total_energy_values') and pert_obj.total_energy_values:
                    c1_int = pert_obj.total_energy_values['results'][cell_idx]
                    e1_int = pert_obj.total_energy_values['errors'][cell_idx]
                else:
                    c1_int = pert_obj.integral_result
                    e1_int = pert_obj.integral_error
                sensCoef_int[j] = c1_int / integral_r0
                sensErr_int[j] = np.sqrt(integral_e0**2 + e1_int**2)

            integral_data[rxn] = Coefficients(
                energy="integral",
                reaction=rxn,
                pert_energies=pert_energies,
                values=sensCoef_int,
                errors=sensErr_int,
                r0=integral_r0,
                e0=integral_e0
            )

            # Process second-order coefficients for integral results (if available)
            if has_second_order and rxn in group_dict_second:
                c2_values_int = []
                c1_values_int = []  # Store the actual Taylor coefficients
                c1_errors_int = []  # Store errors of Taylor coefficients c1
                c2_errors_int = []  # Store errors of Taylor coefficients c2

                for j, pert in enumerate(group_dict_second[rxn]):
                    # Get first-order Taylor coefficient directly
                    pert1_obj = tally_obj.perturbation[group_dict_first[rxn][j]]
                    if n_energies == 0:
                        c1_int_val = pert1_obj.results[cell_idx]
                        c1_int_err = pert1_obj.errors[cell_idx]
                    elif n_cells > 1 and hasattr(pert1_obj, 'total_energy_values') and pert1_obj.total_energy_values:
                        c1_int_val = pert1_obj.total_energy_values['results'][cell_idx]
                        c1_int_err = pert1_obj.total_energy_values['errors'][cell_idx]
                    else:
                        c1_int_val = pert1_obj.integral_result
                        c1_int_err = pert1_obj.integral_error
                    c1_values_int.append(c1_int_val)
                    c1_errors_int.append(c1_int_err)

                    # Get second-order Taylor coefficient directly
                    pert2_obj = tally_obj.perturbation[pert]
                    if n_energies == 0:
                        c2_int_val = pert2_obj.results[cell_idx]
                        c2_int_err = pert2_obj.errors[cell_idx]
                    elif n_cells > 1 and hasattr(pert2_obj, 'total_energy_values') and pert2_obj.total_energy_values:
                        c2_int_val = pert2_obj.total_energy_values['results'][cell_idx]
                        c2_int_err = pert2_obj.total_energy_values['errors'][cell_idx]
                    else:
                        c2_int_val = pert2_obj.integral_result
                        c2_int_err = pert2_obj.integral_error
                    c2_values_int.append(c2_int_val)
                    c2_errors_int.append(c2_int_err)

                # Calculate ratios for integral results
                ratio_values_int = []
                for j in range(len(c1_values_int)):
                    if c1_values_int[j] != 0:
                        ratio_values_int.append(c2_values_int[j] / c1_values_int[j])
                    else:
                        ratio_values_int.append(float('nan'))

                integral_coeff_data[rxn] = TaylorCoefficients(
                    energy="integral",
                    reaction=rxn,
                    pert_energies=pert_energies,
                    c1=c1_values_int,
                    c2=c2_values_int,
                    ratio=ratio_values_int,
                    c1_errors=c1_errors_int,
                    c2_errors=c2_errors_int
                )

        full_data["integral"] = integral_data
        if integral_coeff_data:
            coefficients["integral"] = integral_coeff_data

    # Create SensitivityData object after all data is prepared
    return SensitivityData(
        tally_id=tally,
        pert_energies=pert_energies,
        tally_name=mctal.tally[tally].name,
        zaid=zaid,
        label=label,
        data=full_data,
        coefficients=coefficients
    )


def compute_total_sensitivity(
    inputfile: str, mctalfile: str, tally: int, zaid: int = None, label: str = '',
    materials: Optional[List[int]] = None,
    cell: int = 0,
    pert_metadata: Optional[List[Tuple[int, int, int, int]]] = None,
) -> SensitivityData:
    """Compute total sensitivity by summing contributions from multiple materials.

    For an isotope present in multiple materials, the total sensitivity is the
    sum of the per-material sensitivities:

    .. math::

        S_{\\text{total}}(E) = \\sum_m S_m(E) = \\sum_m \\frac{c_{1,m}(E)}{r_0}

    Errors are propagated in quadrature (assuming statistical independence
    between material perturbations).

    Parameters
    ----------
    inputfile : str
        Path to MCNP input file containing the PERT cards.
    mctalfile : str
        Path to MCNP MCTAL output file.
    tally : int
        Tally number to analyze.
    zaid : int, optional
        ZAID of the nuclide being perturbed. If None and pert_metadata is
        provided, extracted from the metadata (all entries must share the same ZAID).
    label : str
        Label for the resulting sensitivity data set.
    materials : list of int, optional
        Material IDs to sum over. If *None*, auto-detected from the PERT cards
        in *inputfile* (or from *pert_metadata* if provided).
    cell : int
        Cell ID to use for multi-cell tallies. Default is 0, which is the
        total-over-cells bin in MCNP (added by the T parameter in the tally
        definition). For single-cell tallies this parameter is ignored.
    pert_metadata : list of tuple, optional
        List of (start_pert, end_pert, zaid, material) tuples. When provided,
        zaid and materials are extracted from the metadata.

    Returns
    -------
    SensitivityData
        Total sensitivity (summed across materials).

    Raises
    ------
    ValueError
        If no materials are found, or perturbation energy grids differ
        between materials.
    """
    # Parse files once — shared across all per-material computations
    input_data = read_mcnp(inputfile, pert_metadata=pert_metadata)
    mctal = read_mctal(mctalfile)

    # Extract zaid and materials from pert_metadata if provided
    if pert_metadata is not None:
        meta_zaids = sorted({z for _, _, z, _ in pert_metadata})
        meta_mats = sorted({m for _, _, _, m in pert_metadata})
        if zaid is None:
            if len(meta_zaids) != 1:
                raise ValueError(
                    f"pert_metadata contains multiple ZAIDs {meta_zaids}. "
                    "All entries must share the same ZAID for compute_total_sensitivity()."
                )
            zaid = meta_zaids[0]
        if materials is None:
            materials = meta_mats

    # Auto-detect materials from PERT cards if not provided
    if materials is None:
        materials = input_data.perturbation.materials_for_zaid(zaid)

    if not materials:
        raise ValueError("No materials found in perturbation cards")

    # Single material — delegate directly (reuse already-parsed data)
    if len(materials) == 1:
        return _compute_sensitivity_impl(input_data, mctal, tally, zaid, label,
                                         material=materials[0], cell=cell)

    # Compute per-material sensitivities using pre-parsed data
    per_material = {}
    for mat in materials:
        per_material[mat] = _compute_sensitivity_impl(
            input_data, mctal, tally, zaid,
            label=f"{label}_mat{mat}", material=mat, cell=cell,
        )

    # Use the first material as reference for structure validation
    ref = per_material[materials[0]]
    for mat in materials[1:]:
        if per_material[mat].pert_energies != ref.pert_energies:
            raise ValueError(
                f"Perturbation energy grids differ between materials "
                f"{materials[0]} and {mat}"
            )

    # --- Sum sensitivities and propagate errors ---
    full_data: Dict[str, Dict[int, Coefficients]] = {}
    coefficients: Dict[str, Dict[int, TaylorCoefficients]] = {}

    for energy_str in ref.data:
        energy_data: Dict[int, Coefficients] = {}

        # Collect all reactions present in any material for this energy
        all_rxns: set = set()
        for mat in materials:
            mat_sens = per_material[mat]
            if energy_str in mat_sens.data:
                all_rxns.update(mat_sens.data[energy_str].keys())

        for rxn in sorted(all_rxns):
            # Determine number of bins from any material that has this reaction
            n_bins = None
            ref_coef = None
            for mat in materials:
                mat_sens = per_material[mat]
                if energy_str in mat_sens.data and rxn in mat_sens.data[energy_str]:
                    ref_coef = mat_sens.data[energy_str][rxn]
                    n_bins = len(ref_coef.values)
                    break
            if n_bins is None:
                continue

            # Sum sensitivity values: S_total = sum_m S_m
            total_values = np.zeros(n_bins)
            total_abs_err_sq = np.zeros(n_bins)

            for mat in materials:
                mat_sens = per_material[mat]
                if energy_str in mat_sens.data and rxn in mat_sens.data[energy_str]:
                    coef = mat_sens.data[energy_str][rxn]
                    vals = np.array(coef.values)
                    total_values += vals
                    # Absolute error on S_m = S_m * e_m (e_m is relative)
                    abs_err = vals * np.array(coef.errors)
                    total_abs_err_sq += abs_err ** 2

            # Total relative error: |abs_err_total| / |S_total|
            total_abs_err = np.sqrt(total_abs_err_sq)
            with np.errstate(divide="ignore", invalid="ignore"):
                total_rel_err = np.where(
                    total_values != 0,
                    total_abs_err / np.abs(total_values),
                    0.0,
                )

            energy_data[rxn] = Coefficients(
                energy=energy_str,
                reaction=rxn,
                pert_energies=ref_coef.pert_energies,
                values=total_values.tolist(),
                errors=total_rel_err.tolist(),
                r0=ref_coef.r0,
                e0=ref_coef.e0,
            )

        full_data[energy_str] = energy_data

        # --- Taylor coefficients summation (second-order) ---
        all_coeff_rxns: set = set()
        for mat in materials:
            mat_sens = per_material[mat]
            if energy_str in mat_sens.coefficients:
                all_coeff_rxns.update(mat_sens.coefficients[energy_str].keys())

        for rxn in sorted(all_coeff_rxns):
            # Determine number of bins from any material
            n_bins = None
            ref_tc = None
            for mat in materials:
                mat_sens = per_material[mat]
                if (energy_str in mat_sens.coefficients
                        and rxn in mat_sens.coefficients[energy_str]):
                    ref_tc = mat_sens.coefficients[energy_str][rxn]
                    n_bins = len(ref_tc.c1)
                    break
            if n_bins is None:
                continue

            total_c1 = np.zeros(n_bins)
            total_c2 = np.zeros(n_bins)
            total_c1_abs_err_sq = np.zeros(n_bins)
            total_c2_abs_err_sq = np.zeros(n_bins)

            for mat in materials:
                mat_sens = per_material[mat]
                if (energy_str in mat_sens.coefficients
                        and rxn in mat_sens.coefficients[energy_str]):
                    tc = mat_sens.coefficients[energy_str][rxn]
                    c1_arr = np.array(tc.c1)
                    c2_arr = np.array(tc.c2)
                    total_c1 += c1_arr
                    total_c2 += c2_arr
                    total_c1_abs_err_sq += (c1_arr * np.array(tc.c1_errors)) ** 2
                    total_c2_abs_err_sq += (c2_arr * np.array(tc.c2_errors)) ** 2

            with np.errstate(divide="ignore", invalid="ignore"):
                c1_rel_err = np.where(
                    total_c1 != 0,
                    np.sqrt(total_c1_abs_err_sq) / np.abs(total_c1),
                    0.0,
                )
                c2_rel_err = np.where(
                    total_c2 != 0,
                    np.sqrt(total_c2_abs_err_sq) / np.abs(total_c2),
                    0.0,
                )

            ratio = []
            for j in range(n_bins):
                if total_c1[j] != 0:
                    ratio.append(total_c2[j] / total_c1[j])
                else:
                    ratio.append(float("nan"))

            if energy_str not in coefficients:
                coefficients[energy_str] = {}
            coefficients[energy_str][rxn] = TaylorCoefficients(
                energy=energy_str,
                reaction=rxn,
                pert_energies=ref_tc.pert_energies,
                c1=total_c1.tolist(),
                c2=total_c2.tolist(),
                ratio=ratio,
                c1_errors=c1_rel_err.tolist(),
                c2_errors=c2_rel_err.tolist(),
            )

    return SensitivityData(
        tally_id=tally,
        pert_energies=ref.pert_energies,
        tally_name=ref.tally_name,
        zaid=zaid,
        label=label,
        data=full_data,
        coefficients=coefficients,
    )


def plot_sens_comparison(sens_list: List[SensitivityData],
                  energy: Union[str, List[str]] = None, 
                  reactions: Union[List[int], int] = None, 
                  energy_range: tuple = None, xlog: bool = False, ylog: bool = False):
    """Plot comparison of multiple sensitivity datasets.

    Parameters
    ----------
    sens_list : List[SensitivityData]
        List of sensitivity datasets to compare.
    energy : Union[str, List[str]], optional
        Energy string(s) to plot. If None, uses first dataset's energies.
    reactions : Union[List[int], int], optional
        Reaction number(s) to plot. If None, uses reactions from first dataset.
    energy_range : tuple, optional
        Optional x-axis limits as (min, max).
    xlog : bool, optional
        Whether to use logarithmic scale for x-axis. Default is False.
    ylog : bool, optional
        Whether to use logarithmic scale for y-axis. Default is False.
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
                if energy_range is not None:
                    ax.set_xlim(energy_range)
                ax.legend()

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







def create_sdf_data(
    sens_list: Union[List[SensitivityData], List[Tuple[SensitivityData, List[int]]]], 
    energy: str,
    title: str,
    response_values: Tuple[float, float] = None
    ) -> SDFData:
    """Create a SDFData object from a list of SensitivityData objects.
    
    :param sens_list: List of SensitivityData objects or tuples of (SensitivityData, reactions_list)
    :type sens_list: Union[List[SensitivityData], List[Tuple[SensitivityData, List[int]]]]
    :param energy: Energy value to use for sensitivity data
    :type energy: str
    :param title: Title for the SDF dataset
    :type title: str
    :param response_values: Optional tuple of (r0, e0) to override the reference values from sensitivity data.
                           This allows combining data from different sources that might have different base values.
                           r0 is the unperturbed tally result (reference response value),
                           e0 is the absolute error of the unperturbed tally result (not relative).
                           Use this to ensure consistency when merging sensitivity data from different calculations.
    :type response_values: Tuple[float, float], optional
    :returns: SDFData object containing the combined sensitivity data
    :rtype: SDFData
    :raises ValueError: If pert_energies don't match across sensitivity data objects
    :raises ValueError: If r0 and e0 values don't match across sensitivity data objects and no response_values are provided
    """
    # Check if we have a list of SensitivityData objects or tuples
    has_tuples = any(isinstance(item, tuple) for item in sens_list)
    
    # Extract SensitivityData objects and reaction filters
    sens_data = []
    reaction_filters = []
    
    if has_tuples:
        for item in sens_list:
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError("Expected tuple of (SensitivityData, List[int])")
            sens_obj, reactions = item
            sens_data.append(sens_obj)
            reaction_filters.append(reactions)
    else:
        sens_data = sens_list
        # No reaction filters means use all reactions for each SensitivityData
        reaction_filters = [None] * len(sens_data)
    
    # Verify that all sensitivity data objects have matching pert_energies
    reference_energies = sens_data[0].pert_energies
    for sd in sens_data[1:]:
        if sd.pert_energies != reference_energies:
            raise ValueError("All SensitivityData objects must have the same perturbation energies")
    
    # Determine r0 and e0 values (unperturbed tally result and its error)
    r0 = None
    e0 = None
    
    if response_values is not None:
        # Use provided response values
        r0, e0 = response_values
    else:
        # Verify that all sensitivity data objects have matching r0 and e0
        for sd in sens_data:
            # Find the first available reaction to get r0 and e0
            if energy in sd.data:
                for mt in sd.data[energy]:
                    if r0 is None and e0 is None:
                        # First sensitivity data object with reaction - set reference values
                        r0 = sd.data[energy][mt].r0
                        e0 = sd.data[energy][mt].e0
                    else:
                        # Compare with reference values
                        if sd.data[energy][mt].r0 != r0 or sd.data[energy][mt].e0 != e0:
                            raise ValueError(
                                "All SensitivityData objects must have the same r0 (unperturbed tally result) "
                                "and e0 (error) values. Use the response_values parameter to specify common values."
                            )
                    break  # Only need to check one reaction per sensitivity data object
    
    # Create a new SDFData object
    sdf_data = SDFData(
        title=title,
        energy=energy,
        pert_energies=reference_energies,
        r0=r0,
        e0=e0,
        data=[]
    )
    
    # Process each SensitivityData object
    for sd, reaction_filter in zip(sens_data, reaction_filters):
        # Check if energy exists in this sensitivity data
        if energy not in sd.data:
            continue
        
        # Get the reactions to process
        if reaction_filter is None:
            reactions_to_process = list(sd.data[energy].keys())
        else:
            reactions_to_process = [r for r in reaction_filter if r in sd.data[energy]]
        
        # Process each reaction
        for mt in reactions_to_process:
            coef_data = sd.data[energy][mt]
            
            # Check if all sensitivity coefficients are zero
            if all(value == 0.0 for value in coef_data.values):
                # Calculate the nuclide symbol for more informative message
                z = sd.zaid // 1000
                a = sd.zaid % 1000
                symbol = ATOMIC_NUMBER_TO_SYMBOL.get(z, f"unknown_{z}")
                nuclide = f"{symbol}-{a}"
                
                # Print message that reaction was skipped
                reaction_name = MT_TO_REACTION.get(mt, f"Unknown(MT={mt})")
                print(f"Skipping {nuclide} {reaction_name} (MT={mt}): All sensitivity coefficients are zero")
                continue
            
            # Create SDFReactionData object
            reaction_data = SDFReactionData(
                zaid=sd.zaid,
                mt=mt,
                sensitivity=coef_data.values,
                error=coef_data.errors
            )
            
            # Add to SDF data
            sdf_data.data.append(reaction_data)
    
    return sdf_data


def create_sdf_from_serpent(
    serpent_file: Union['SensitivityFile', List['SensitivityFile']],
    response_name: Union[str, List[str]],
    title: str,
    material_filter: Union[str, List[str]] = None,
    nuclide_filter: Union[int, List[int]] = None,
    mt_filter: Union[int, List[int]] = None,
    response_values: Tuple[float, float] = None
) -> SDFData:
    """Create a SDFData object from SERPENT sensitivity results.
    
    Note: SERPENT provides relative errors (σ/μ) which are preserved as relative errors
    to maintain consistency with nuclear data uncertainty propagation methods.
    
    :param serpent_file: SERPENT sensitivity file object(s). Can be a single file or list of files.
    :type serpent_file: Union[SensitivityFile, List[SensitivityFile]]
    :param response_name: Name(s) of the response to extract. Can be a single response name 
                         (used for all files) or a list matching the number of files.
                         Examples: 'sens_ratio_BIN_2' or ['sens_ratio_BIN_1', 'sens_ratio_BIN_2']
    :type response_name: Union[str, List[str]]
    :param title: Title for the SDF dataset
    :type title: str
    :param material_filter: Material name(s) to include. If None, uses all materials.
    :type material_filter: Union[str, List[str]], optional
    :param nuclide_filter: Nuclide ZAI(s) to include. If None, uses all nuclides.
    :type nuclide_filter: Union[int, List[int]], optional
    :param mt_filter: MT reaction number(s) to include. If None, uses all MT reactions (including Legendre coefficients MT=4001+).
    :type mt_filter: Union[int, List[int]], optional
    :param response_values: Tuple of (r0, e0) reference response values. If None, uses (1.0, 0.01).
                           r0 is the unperturbed tally result (reference response value),
                           e0 is the relative error of the unperturbed tally result (e.g., 0.01 for 1%).
                           Note: e0 is stored as relative error for consistency with nuclear data uncertainties.
    :type response_values: Tuple[float, float], optional
    :returns: SDFData object containing the SERPENT sensitivity data
    :rtype: SDFData
    :raises ValueError: If the specified response is not found or file lists don't match
    """
    from kika.serpent.sens import SensitivityFile
    import numpy as np
    
    # Normalize inputs to lists for uniform processing
    if not isinstance(serpent_file, list):
        serpent_files = [serpent_file]
    else:
        serpent_files = serpent_file
    
    if not isinstance(response_name, list):
        response_names = [response_name] * len(serpent_files)
    else:
        response_names = response_name
        if len(response_names) != len(serpent_files):
            raise ValueError(f"Number of response names ({len(response_names)}) must match number of files ({len(serpent_files)})")
    
    if not serpent_files:
        raise ValueError("At least one SERPENT file must be provided")
    
    # Validate energy grids match across all files
    first_energies = serpent_files[0].energy_grid
    for i, sfile in enumerate(serpent_files[1:], 1):
        if not np.allclose(sfile.energy_grid, first_energies):
            raise ValueError(f"Energy grids don't match between files. File {i+1} has different energy grid than file 1.")
    
    # Set default response values
    if response_values is None:
        response_values = (1.0, 0.01)  # Default: r0=1.0, e0=1% relative error
    
    r0, e0_relative = response_values
    
    # Store relative error directly in SDF data for consistency with nuclear data uncertainties
    # Both statistical and nuclear data uncertainties should be handled as relative values
    
    # Convert energy edges to perturbation energies (SDF format expects MeV)
    pert_energies = first_energies.tolist()
    
    # Create energy string for SDF including response name(s)
    # For single response, use response name
    # For multiple responses, use "MultiResponse"
    if len(set(response_names)) == 1:
        # Single unique response name
        response_part = response_names[0]
    else:
        # Multiple different response names
        response_part = "MultiResponse"
    
    energy_str = f"{response_part}"
    
    # Create SDFData object
    sdf_data = SDFData(
        title=title,
        energy=energy_str,
        pert_energies=pert_energies,
        r0=r0,
        e0=e0_relative,  # Store relative error for consistency
        data=[]
    )
    
    # Process each SERPENT file
    for file_idx, (sfile, resp_name) in enumerate(zip(serpent_files, response_names)):
        print(f"Processing file {file_idx+1}/{len(serpent_files)} with response '{resp_name}'...")
        
        # Validate response exists in this file
        available_base_names = list(sfile.data.keys())
        available_full_names = sfile.responses
        
        # If resp_name is a full name (like "sens_ratio_BIN_2"), extract the base name
        if resp_name in available_full_names:
            # It's a full response name, extract the base name
            base_name = resp_name.split('_BIN_')[0] if '_BIN_' in resp_name else resp_name
            current_response_name = resp_name
        elif resp_name in available_base_names:
            # It's already a base name
            base_name = resp_name
            current_response_name = available_full_names[0]  # Use first available full name
        else:
            raise ValueError(f"Response '{resp_name}' not found in file {file_idx+1}. Available responses: {available_full_names}")
        
        # Process this file using the existing single-file logic
        file_data = _process_single_serpent_file(
            sfile, current_response_name, material_filter, nuclide_filter, mt_filter
        )
        
        # Add reaction data to combined SDF
        sdf_data.data.extend(file_data)
    
    print(f"Combined SDF contains {len(sdf_data.data)} sensitivity profiles from {len(serpent_files)} files")
    
    return sdf_data


def _process_single_serpent_file(
    serpent_file: 'SensitivityFile',
    response_name: str,
    material_filter: Union[str, List[str]] = None,
    nuclide_filter: Union[int, List[int]] = None,
    mt_filter: Union[int, List[int]] = None
) -> List['SDFReactionData']:
    """Process a single SERPENT file and return list of SDFReactionData objects."""
    from kika.serpent.sens import SensitivityFile
    import numpy as np
    
    # Prepare filters (existing logic)
    if material_filter is not None:
        if isinstance(material_filter, str):
            material_filter = [material_filter]
        try:
            material_indices = [serpent_file._material_index(mat) for mat in material_filter]
        except KeyError as e:
            print(f"Warning: {e}. Skipping material filter for this file.")
            material_indices = list(range(serpent_file.n_materials))
    else:
        material_indices = list(range(serpent_file.n_materials))
    
    if nuclide_filter is not None:
        if isinstance(nuclide_filter, int):
            nuclide_filter = [nuclide_filter]
        try:
            nuclide_indices = [serpent_file._nuclide_index(nuc) for nuc in nuclide_filter]
        except KeyError as e:
            print(f"Warning: {e}. Skipping nuclide filter for this file.")
            nuclide_indices = list(range(serpent_file.n_nuclides))
    else:
        nuclide_indices = list(range(serpent_file.n_nuclides))
    
    # Filter perturbations - include all MT reactions by default (including Legendre coefficients)
    if mt_filter is not None:
        if isinstance(mt_filter, int):
            mt_filter = [mt_filter]
        perturbation_indices = serpent_file._collect_perturbations(mt=mt_filter)
    else:
        # Include all MT reactions by default (both standard reactions and Legendre coefficients)
        perturbation_indices = [
            p.index for p in serpent_file.perturbations 
            if p.mt is not None
        ]
    
    reaction_data_list = []
    
    # Process each combination of material, nuclide, and perturbation (existing logic)
    for mat_idx in material_indices:
        for nuc_idx in nuclide_indices:
            # Get nuclide ZAI
            nuclide = serpent_file.nuclides[nuc_idx]
            zaid = nuclide.zai
            
            # Group perturbations by MT number for this nuclide
            # Note: Legendre moments are stored as MT 4001, 4002, 4003, ... (L=1, L=2, L=3, ...)
            mt_groups = {}
            for pert_idx in perturbation_indices:
                pert = serpent_file.perturbations[pert_idx]
                if pert.mt is not None:
                    if pert.mt not in mt_groups:
                        mt_groups[pert.mt] = []
                    mt_groups[pert.mt].append(pert_idx)
            
            # Create SDFReactionData for each MT number (including Legendre coefficients if enabled)
            for mt, pert_indices in mt_groups.items():
                # For multiple perturbations with same MT, we need to decide how to combine them
                # For now, let's take the first one or average if there are multiple
                if len(pert_indices) == 1:
                    pert_idx = pert_indices[0]
                    
                    try:
                        # Get energy-dependent sensitivity data
                        values, rel_errors = serpent_file.get_energy_dependent(
                            response_name, 
                            mat=mat_idx, 
                            zai=nuc_idx, 
                            mt=mt
                        )
                        
                        # Extract 1D arrays (remove any singleton dimensions)
                        sens_values = np.squeeze(values).tolist()
                        sens_errors = np.squeeze(rel_errors).tolist()

                        # Skip if all sensitivity coefficients are zero
                        if all(abs(v) < 1e-15 for v in sens_values):
                            continue

                        # Create SDFReactionData. SERPENT provides relative errors
                        # (σ/μ) on the sensitivity coefficients; SDFReactionData.error
                        # is the relative-error convention used by the rest of kika
                        # (to_plot_data, group_inelastic_reactions, the SDF writer's
                        # scalar block, and the kika-app frontend), so we store them
                        # verbatim.
                        reaction_data = SDFReactionData(
                            zaid=zaid,
                            mt=mt,
                            sensitivity=sens_values,
                            error=sens_errors
                        )
                        
                        reaction_data_list.append(reaction_data)
                        
                    except Exception as e:
                        # Skip reactions that cause errors (e.g., not available in this file)
                        print(f"Warning: Skipping MT={mt} for ZAID={zaid}: {e}")
                        continue
                        
                else:
                    # Multiple perturbations for same MT - average them
                    print(f"Warning: Multiple perturbations found for MT={mt}, ZAI={zaid}. Taking average.")
                    
                    try:
                        all_values = []
                        all_errors = []
                        
                        for pert_idx in pert_indices:
                            values, rel_errors = serpent_file.get_energy_dependent(
                                response_name, 
                                mat=mat_idx, 
                                zai=nuc_idx, 
                                mt=mt
                            )
                            all_values.append(np.squeeze(values))
                            all_errors.append(np.squeeze(rel_errors))
                        
                        # Average the values and relative errors (properly handling relative error averaging)
                        avg_values = np.mean(all_values, axis=0).tolist()
                        avg_rel_errors = np.sqrt(np.mean(np.array(all_errors)**2, axis=0)).tolist()

                        # Skip if all sensitivity coefficients are zero
                        if all(abs(v) < 1e-15 for v in avg_values):
                            continue

                        reaction_data = SDFReactionData(
                            zaid=zaid,
                            mt=mt,
                            sensitivity=avg_values,
                            error=avg_rel_errors
                        )
                        
                        reaction_data_list.append(reaction_data)
                        
                    except Exception as e:
                        print(f"Warning: Skipping averaged MT={mt} for ZAID={zaid}: {e}")
                        continue
    
    return reaction_data_list