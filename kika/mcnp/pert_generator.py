import os
import re
from typing import Optional, Union
from .parse_input import read_mcnp
from kika.materials.parse_mcnp import read_material
from kika.materials import Material, MaterialCollection
from kika._constants import MCNPY_HEADER, MCNPY_FOOTER, ATOMIC_MASS, N_AVOGADRO


def perturb_material(materials: MaterialCollection, material_id: int, nuclide: Union[int, str], 
                     density: Optional[float] = None, pert_mat_id: Optional[int] = None, 
                     in_place: bool = True, fraction_type: Optional[str] = None) -> Union[MaterialCollection, Material]:
    """Creates a perturbed material with 100% increase in the specified nuclide's fraction.
    
    Creates a new material with a 100% increase in the fraction of the specified nuclide.
    The perturbation is applied after normalizing the original material composition.
    
    Parameters
    ----------
    materials : MaterialCollection
        The material collection containing the material to perturb.
    material_id : int
        Material ID number to be perturbed.
    nuclide : int or str
        ZAID (int, e.g., 26056) or symbol (str, e.g., 'Fe56') of the nuclide to perturb.
    density : float, optional
        Density of the original material. If positive, interpreted as atoms/barn-cm,
        if negative, interpreted as g/cm³. Used for density recalculation info.
        If None, density calculations are skipped.
    pert_mat_id : int, optional
        ID for the perturbed material. If None, uses material_id * 100 + 1.
    in_place : bool, optional
        If True, adds the perturbed material to the existing collection.
        If False, returns a new MaterialCollection with both original and perturbed materials.
        Default is True.
    fraction_type : str, optional
        Output fraction type: 'atomic'/'ao' or 'weight'/'wo'.
        If None, uses the same format as the original material.
    
    Returns
    -------
    MaterialCollection or Material
        If in_place is True, returns the perturbed Material (collection is modified).
        If in_place is False, returns a new MaterialCollection containing original and perturbed.
    
    Raises
    ------
    ValueError
        If the material or nuclide is not found.
        
    Examples
    --------
    >>> # Modify collection in place
    >>> perturbed = kika.perturb_material(input_data.materials, 1007, 'Fe56')
    >>> 
    >>> # Create new collection without modifying original
    >>> new_collection = kika.perturb_material(input_data.materials, 1007, 26056, in_place=False)
    """
    # Validate material exists
    if material_id not in materials.by_id:
        raise ValueError(f"Material {material_id} not found in collection")
    
    original_material = materials.by_id[material_id]
    
    # Validate nuclide exists in material
    if nuclide not in original_material.nuclide:
        raise ValueError(f"Nuclide {nuclide} not found in material {material_id}")
    
    # Create perturbed material ID
    new_material_id = pert_mat_id if pert_mat_id is not None else material_id * 100 + 1
    
    # Check if new ID already exists
    if new_material_id in materials.by_id:
        raise ValueError(f"Material ID {new_material_id} already exists in collection")
    
    # Create a copy of the original material
    perturbed_mat = original_material.copy(new_material_id)
    
    # Store original fraction type
    original_is_weight = perturbed_mat.is_weight
    
    # Convert to atomic fractions for perturbation (standard practice)
    if original_is_weight:
        perturbed_mat.to_atomic_fraction()
    
    # Normalize the material composition
    total_fraction = sum(nuc.fraction for nuc in perturbed_mat.nuclide.values())
    if abs(total_fraction - 1.0) > 1e-6:
        normalization_factor = 1.0 / total_fraction
        for symbol in perturbed_mat.nuclide:
            perturbed_mat.nuclide[symbol].fraction *= normalization_factor
    
    # Apply 100% perturbation to the specified nuclide
    perturbed_mat.nuclide[nuclide].fraction *= 2.0
    
    # Calculate new total (will be > 1.0 due to perturbation)
    new_total = sum(nuc.fraction for nuc in perturbed_mat.nuclide.values())
    
    # Calculate density changes if density provided
    if density is not None:
        _calculate_perturbed_density(original_material, perturbed_mat, density, 
                                     new_total, original_is_weight)
    
    # Re-normalize to sum = 1.0
    renorm_factor = 1.0 / new_total
    for symbol in perturbed_mat.nuclide:
        perturbed_mat.nuclide[symbol].fraction *= renorm_factor
    
    # Apply output fraction type
    if fraction_type is None:
        # Use same format as original
        if original_is_weight and perturbed_mat.is_atomic:
            perturbed_mat.to_weight_fraction()
    elif fraction_type.lower() in ('weight', 'wo'):
        if perturbed_mat.is_atomic:
            perturbed_mat.to_weight_fraction()
    elif fraction_type.lower() in ('atomic', 'ao'):
        if perturbed_mat.is_weight:
            perturbed_mat.to_atomic_fraction()
    else:
        print(f"WARNING: Unrecognized fraction_type '{fraction_type}'. Using atomic fractions.")
    
    # Add metadata about perturbation
    perturbed_mat.metadata['perturbation'] = {
        'original_material_id': material_id,
        'perturbed_nuclide': nuclide,
        'perturbation_factor': 2.0,
    }
    
    if in_place:
        # Add to existing collection
        materials.by_id[new_material_id] = perturbed_mat
        print(f"Perturbed material {new_material_id} added to collection (100% increase in {nuclide})")
        return perturbed_mat
    else:
        # Create new collection with both materials
        new_collection = MaterialCollection()
        new_collection.by_id[material_id] = original_material.copy(material_id)
        new_collection.by_id[new_material_id] = perturbed_mat
        print(f"Created new collection with original ({material_id}) and perturbed ({new_material_id}) materials")
        return new_collection


def _calculate_perturbed_density(original_material: Material, perturbed_material: Material,
                                  density: float, new_total: float, 
                                  original_is_weight: bool) -> None:
    """Calculate and print density information for perturbed material.
    
    This is a helper function that calculates how the density changes
    after perturbation, sets the density attributes on the material objects,
    and prints the information.
    
    The density parameter follows MCNP convention:
    - Negative value: mass density in g/cm³
    - Positive value: atomic density in atoms/barn-cm
    """
    # Compute average atomic mass for original material
    avg_atomic_mass = 0.0
    
    if original_is_weight:
        sum_w_over_A = 0.0
        for symbol, nuclide_obj in original_material.nuclide.items():
            fraction = abs(nuclide_obj.fraction)
            zaid = nuclide_obj.zaid
            atomic_mass = ATOMIC_MASS.get(zaid, float(zaid % 1000) if zaid % 1000 > 0 else 1.0)
            sum_w_over_A += fraction / atomic_mass
        avg_atomic_mass = 1.0 / sum_w_over_A
    else:
        total_original = sum(nuc.fraction for nuc in original_material.nuclide.values())
        for symbol, nuclide_obj in original_material.nuclide.items():
            fraction = nuclide_obj.fraction / total_original
            zaid = nuclide_obj.zaid
            atomic_mass = ATOMIC_MASS.get(zaid, float(zaid % 1000) if zaid % 1000 > 0 else 1.0)
            avg_atomic_mass += fraction * atomic_mass
    
    # Convert between density types
    abs_density = abs(density)
    if density < 0:
        mass_density = abs_density
        atomic_density = mass_density * N_AVOGADRO / avg_atomic_mass * 1e-24
    else:
        atomic_density = abs_density
        mass_density = atomic_density * avg_atomic_mass / N_AVOGADRO * 1e24
    
    # Calculate new densities
    new_atomic_density = atomic_density * new_total
    
    # Recalculate average atomic mass for perturbed material
    new_avg_atomic_mass = 0.0
    for symbol, nuclide_obj in perturbed_material.nuclide.items():
        fraction = nuclide_obj.fraction / new_total
        zaid = nuclide_obj.zaid
        atomic_mass = ATOMIC_MASS.get(zaid, float(zaid % 1000) if zaid % 1000 > 0 else 1.0)
        new_avg_atomic_mass += fraction * atomic_mass
    
    new_mass_density = new_atomic_density * new_avg_atomic_mass / N_AVOGADRO * 1e24
    
    # Store in metadata
    perturbed_material.metadata['density_info'] = {
        'original_mass_density': mass_density,
        'original_atomic_density': atomic_density,
        'perturbed_mass_density': new_mass_density,
        'perturbed_atomic_density': new_atomic_density,
    }
    
    # Set density on the original material if not already set
    # This ensures the original material has density info for MCNP output
    if original_material.density is None:
        original_material.density = mass_density
        original_material.density_unit = 'g/cc'
    
    # Always set the perturbed density on the perturbed material
    perturbed_material.density = new_mass_density
    perturbed_material.density_unit = 'g/cc'
    
    print(f"Density change: {mass_density:.4e} → {new_mass_density:.4e} g/cm³")


def generate_PERTcards(inputfile, cell, reactions, material, energies=None, density=None,
                       order=2, errors=False, in_place=True, nuclide=None):
    """Generate PERT cards for MCNP sensitivity analysis.

    Creates PERT cards based on the provided parameters for first and/or second order
    perturbation calculations. The density (RHO) value is obtained from the material's
    density attribute in the input file.

    Can handle multiple materials by providing lists for cell, density, and material 
    parameters. When lists are provided, all list parameters must have the same length 
    as the materials list. The reactions and energies parameters are applied uniformly
    to all materials.

    Parameters
    ----------
    inputfile : str or Path
        Path to the MCNP input file. The file must contain the material definitions
        with density information for the specified material(s).
    cell : int, str, list[int], or list[list[int]]
        Cell number(s) for PERT card application. Can be:
        - Single cell: ``3`` or ``'3'``
        - Multiple cells for one material: ``[3, 5, 7]``
        - Multiple materials: ``[[3, 5], [7, 9]]`` (one list per material)
    reactions : list[int]
        Reaction MT numbers to perturb. Common values:
        - 1: total cross-section
        - 2: elastic scattering
        - 4: inelastic scattering
        - 18: fission
        - 51-91: discrete inelastic levels
        - 102: radiative capture (n,γ)
    material : int, str, or list
        Material identifier(s) for perturbation. Must exist in the input file.
        Can be single value or list for multiple materials.
    energies : array-like, optional
        Energy bin boundaries in eV, must be in ascending order. Used in consecutive
        pairs to define energy bins. If None, ERG keyword is omitted (energy-integrated).
        Built-in grids available via ``kika.energy_grids`` (e.g., SCALE44, VITAMINJ175).
    density : float or list[float], optional
        Override density value(s) for RHO in PERT cards. If None (default), uses the
        density from the material definition in the input file. Following MCNP convention:
        - Negative value: mass density in g/cm³
        - Positive value: atomic density in atoms/barn-cm
    order : int or list[int], optional
        Perturbation order (default: 2):
        - 1: First-order only (METHOD=2)
        - 2: First and second order (METHOD=2 and METHOD=3)
    errors : bool, optional
        If True, include exact error method cards (METHOD=-2, -3, 1). These are
        typically negligible and computationally expensive. Default is False.
    in_place : bool, optional
        If True (default), append PERT cards to the original input file.
        If False, create a new file with suffix ``_pert_cards``.

    Returns
    -------
    None
        PERT cards are written to the file.

    Raises
    ------
    ValueError
        If energies are not in ascending order, if list parameters have inconsistent
        lengths, if list-of-lists is provided for reactions/energies, or if material
        density is not available and not provided.

    Notes
    -----
    The RHO parameter in MCNP PERT cards represents the material density, not a
    perturbation magnitude. MCNP uses this density along with the perturbed material
    composition to calculate sensitivity coefficients.

    Examples
    --------
    >>> # Basic usage with single material
    >>> kika.generate_PERTcards(
    ...     inputfile='input.i',
    ...     cell=[3, 5, 7],
    ...     reactions=[1, 2, 102],
    ...     material=100701,
    ...     energies=kika.energy_grids.SCALE44
    ... )

    >>> # Multiple materials with different cells
    >>> kika.generate_PERTcards(
    ...     inputfile='input.i',
    ...     cell=[[3, 5], [7, 9]],
    ...     reactions=[1, 2, 102],
    ...     material=[100701, 100801],
    ...     energies=kika.energy_grids.SCALE44
    ... )

    See Also
    --------
    perturb_material : Create perturbed material compositions for sensitivity analysis.
    compute_sensitivity : Compute sensitivity coefficients from MCNP output.
    """
    
    # Validate that reactions and energies are not list-of-lists
    if isinstance(reactions, list) and len(reactions) > 0 and isinstance(reactions[0], list):
        raise ValueError("Reactions parameter cannot be a list of lists. Use a single list that applies to all materials.")
    
    if isinstance(energies, list) and len(energies) > 0 and isinstance(energies[0], list):
        raise ValueError("Energies parameter cannot be a list of lists. Use a single list that applies to all materials.")
    
    # Validate energies if provided
    if energies is not None:
        for i in range(len(energies) - 1):
            if energies[i] >= energies[i + 1]:
                raise ValueError(f"Energy values must be in ascending order. Found {energies[i]} >= {energies[i + 1]} at positions {i} and {i + 1}")
    
    # Convert material to list to determine number of materials
    if isinstance(material, list):
        material_list = material
        num_materials = len(material_list)
    else:
        material_list = [material]
        num_materials = 1
    
    # Validate and prepare cell parameter
    if isinstance(cell, list):
        if num_materials == 1:
            # Single material case: cell should be a simple list of cell numbers
            if any(isinstance(c, list) for c in cell):
                raise ValueError("List of lists for cell parameter is only allowed when multiple materials are provided.")
            cell_list = [cell]
        else:
            # Multiple materials case: check length consistency
            if len(cell) != num_materials:
                raise ValueError(f"Cell list length ({len(cell)}) must match number of materials ({num_materials})")
            cell_list = cell
    else:
        # Single value: replicate for all materials
        cell_list = [cell] * num_materials
    
    # Validate and prepare density parameter
    # Density will be resolved from materials if not provided
    if isinstance(density, list):
        if num_materials == 1:
            raise ValueError("List of densities is only allowed when multiple materials are provided.")
        if len(density) != num_materials:
            raise ValueError(f"Density list length ({len(density)}) must match number of materials ({num_materials})")
        density_list = density
    elif density is not None:
        # Single value: replicate for all materials
        density_list = [density] * num_materials
    else:
        # None: will be resolved from materials later
        density_list = [None] * num_materials
    
    # Validate and prepare order parameter
    if isinstance(order, list):
        if num_materials == 1:
            raise ValueError("List of orders is only allowed when multiple materials are provided.")
        if len(order) != num_materials:
            raise ValueError(f"Order list length ({len(order)}) must match number of materials ({num_materials})")
        order_list = order
    else:
        # Single value: replicate for all materials
        order_list = [order] * num_materials

    # --- Use the parser to get the highest existing PERT card number ---
    input_data = read_mcnp(inputfile)
    if hasattr(input_data, "perturbation") and hasattr(input_data.perturbation, "pert"):
        existing_pert_ids = list(input_data.perturbation.pert.keys())
        max_pert_num = max(existing_pert_ids) if existing_pert_ids else 0
    else:
        max_pert_num = 0

    # --- Resolve densities from materials if not provided ---
    resolved_density_list = []
    for mat_idx, mat_id in enumerate(material_list):
        user_density = density_list[mat_idx]
        
        # Get material from input file
        if mat_id not in input_data.materials.by_id:
            raise ValueError(f"Material {mat_id} not found in input file '{inputfile}'")
        
        mat_obj = input_data.materials.by_id[mat_id]
        mat_density = mat_obj.density
        mat_density_unit = mat_obj.density_unit
        
        if user_density is not None:
            # User provided density - use it, but warn if different from material's density
            if mat_density is not None:
                # Compare densities properly (convert to same unit for comparison)
                # User density: negative = g/cm³, positive = atoms/barn-cm
                # Material density is stored as positive with unit info
                
                # Convert material density to MCNP convention for comparison
                if mat_density_unit == 'g/cc':
                    mat_density_mcnp = -abs(mat_density)  # Mass density is negative in MCNP
                else:
                    mat_density_mcnp = abs(mat_density)   # Atomic density is positive
                
                # Check if they're effectively the same (within 0.1% tolerance)
                if abs(user_density) > 1e-10 and abs(mat_density_mcnp) > 1e-10:
                    rel_diff = abs((user_density - mat_density_mcnp) / mat_density_mcnp)
                    if rel_diff > 0.001:  # More than 0.1% difference
                        print(f"WARNING: Material {mat_id} - User-provided density ({user_density:.6e}) "
                              f"differs from material's density ({mat_density_mcnp:.6e}). "
                              f"Using user-provided value.")
            
            resolved_density_list.append(user_density)
        else:
            # No user density - must get from material
            if mat_density is None:
                raise ValueError(
                    f"Material {mat_id} has no density defined. "
                    f"Please either:\n"
                    f"  1. Set the material density: material.set_density(value, 'g/cc')\n"
                    f"  2. Provide density via the 'density' parameter in generate_PERTcards()"
                )
            
            # Convert to MCNP convention (negative for mass density)
            if mat_density_unit == 'g/cc':
                resolved_density_list.append(-abs(mat_density))
            else:
                # Assume atomic density if not g/cc
                resolved_density_list.append(abs(mat_density))
    
    # Replace density_list with resolved values
    density_list = resolved_density_list

    # Determine output file path
    if in_place:
        # Read the original file content to ensure we start on a new line
        with open(inputfile, 'r') as f:
            content = f.read()
        
        # Append to the original input file
        output_file = inputfile
        mode = "a"  # Append mode
        
        # Check if file ends with a newline, if not add one
        needs_newline = not content.endswith('\n')
    else:
        # Create a new file in the same directory
        input_dir = os.path.dirname(inputfile) or "."
        base_name = os.path.basename(inputfile)
        filename, ext = os.path.splitext(base_name)
        new_filename = f"{filename}_pert_cards{ext}"
        output_file = os.path.join(input_dir, new_filename)
        mode = "w"  # Write mode
        needs_newline = False
    
    # Generate all the PERT card content in memory
    content_to_write = []
    
    # First add a newline if needed
    if needs_newline:
        content_to_write.append("\n")

    # Initialize the perturbation counter
    pert_counter = max_pert_num + 1
    
    # Write header - always include it
    content_to_write.append(MCNPY_HEADER)
    content_to_write.append("c \n")

    # Emit ZAID comment if nuclide is provided
    if nuclide is not None:
        if isinstance(nuclide, int):
            zaid_value = nuclide
        elif isinstance(nuclide, str):
            # Try to resolve from material's nuclide accessor
            first_mat = material_list[0]
            if first_mat in input_data.materials.by_id:
                mat_obj = input_data.materials.by_id[first_mat]
                if nuclide in mat_obj.nuclide:
                    zaid_value = mat_obj.nuclide[nuclide].zaid
                else:
                    raise ValueError(f"Nuclide '{nuclide}' not found in material {first_mat}")
            else:
                raise ValueError(f"Material {first_mat} not found for nuclide lookup")
        else:
            zaid_value = int(nuclide)
        content_to_write.append(f"c kika:pert_zaid={zaid_value}\n")

    # Loop over each material
    for mat_idx in range(num_materials):
        cell_iter = cell_list[mat_idx]
        density_iter = density_list[mat_idx]
        material_iter = material_list[mat_idx]
        order_iter = order_list[mat_idx]
        
        # Format cell parameter to string
        if isinstance(cell_iter, list):
            cell_str = ','.join(map(str, cell_iter))
        else:
            cell_str = str(cell_iter)
        
        # Loop over each reaction for this material
        for reaction in reactions:
            if energies is not None:
                # Go through the energy list and use consecutive pairs
                for i in range(len(energies) - 1):
                    E1 = energies[i]
                    E2 = energies[i + 1]

                    # Create properly formatted METHOD=2 PERT card
                    pert_card = f"PERT{pert_counter}:n CELL={cell_str} MAT={material_iter} RHO={density_iter:.6e} METHOD=2 RXN={reaction} ERG={E1:.6e} {E2:.6e}"
                    content_to_write.append(_format_mcnp_line(pert_card) + "\n")
                    pert_counter += 1

                    if order_iter == 2:
                        # Create properly formatted METHOD=3 PERT card
                        pert_card = f"PERT{pert_counter}:n CELL={cell_str} MAT={material_iter} RHO={density_iter:.6e} METHOD=3 RXN={reaction} ERG={E1:.6e} {E2:.6e}"
                        content_to_write.append(_format_mcnp_line(pert_card) + "\n")
                        pert_counter += 1
            
                    if errors:
                        # Create properly formatted METHOD=-2 PERT card
                        pert_card = f"PERT{pert_counter}:n CELL={cell_str} MAT={material_iter} RHO={density_iter:.6e} METHOD=-2 RXN={reaction} ERG={E1:.6e} {E2:.6e}"
                        content_to_write.append(_format_mcnp_line(pert_card) + "\n")
                        pert_counter += 1

                        if order_iter == 2:
                            # Create properly formatted METHOD=-3 PERT card
                            pert_card = f"PERT{pert_counter}:n CELL={cell_str} MAT={material_iter} RHO={density_iter:.6e} METHOD=-3 RXN={reaction} ERG={E1:.6e} {E2:.6e}"
                            content_to_write.append(_format_mcnp_line(pert_card) + "\n")
                            pert_counter += 1

                            # Create properly formatted METHOD=1 PERT card
                            pert_card = f"PERT{pert_counter}:n CELL={cell_str} MAT={material_iter} RHO={density_iter:.6e} METHOD=1 RXN={reaction} ERG={E1:.6e} {E2:.6e}"
                            content_to_write.append(_format_mcnp_line(pert_card) + "\n")
                            pert_counter += 1
            else:
                # No energies provided, omit ERG part
                # Create properly formatted METHOD=2 PERT card
                pert_card = f"PERT{pert_counter}:n CELL={cell_str} MAT={material_iter} RHO={density_iter:.6e} METHOD=2 RXN={reaction}"
                content_to_write.append(_format_mcnp_line(pert_card) + "\n")
                pert_counter += 1

                if order_iter == 2:
                    # Create properly formatted METHOD=3 PERT card
                    pert_card = f"PERT{pert_counter}:n CELL={cell_str} MAT={material_iter} RHO={density_iter:.6e} METHOD=3 RXN={reaction}"
                    content_to_write.append(_format_mcnp_line(pert_card) + "\n")
                    pert_counter += 1

                if errors:
                    # Create properly formatted METHOD=-2 PERT card
                    pert_card = f"PERT{pert_counter}:n CELL={cell_str} MAT={material_iter} RHO={density_iter:.6e} METHOD=-2 RXN={reaction}"
                    content_to_write.append(_format_mcnp_line(pert_card) + "\n")
                    pert_counter += 1

                    if order_iter == 2:
                        # Create properly formatted METHOD=-3 PERT card
                        pert_card = f"PERT{pert_counter}:n CELL={cell_str} MAT={material_iter} RHO={density_iter:.6e} METHOD=-3 RXN={reaction}"
                        content_to_write.append(_format_mcnp_line(pert_card) + "\n")
                        pert_counter += 1

                        # Create properly formatted METHOD=1 PERT card
                        pert_card = f"PERT{pert_counter}:n CELL={cell_str} MAT={material_iter} RHO={density_iter:.6e} METHOD=1 RXN={reaction}"
                        content_to_write.append(_format_mcnp_line(pert_card) + "\n")
                        pert_counter += 1

    # Write footer - always include it
    content_to_write.append("c \n")
    content_to_write.append(MCNPY_FOOTER)
    
    # Write all content at once
    with open(output_file, mode) as stream:
        stream.writelines(content_to_write)
    
    print(f"\nSuccess! PERT cards written to: {output_file}")
    
    return


def _format_mcnp_line(line, max_length=120):
    """Helper function to format MCNP input lines to stay under the character limit.

    :param line: The full line to format
    :type line: str
    :param max_length: Maximum line length (default: 120 for MCNP6)
    :type max_length: int

    :return: Formatted string with proper line breaks
    :rtype: str
    """
    if len(line) <= max_length:
        return line
    
    result = []
    remaining = line.strip()
    
    while remaining:
        # If this is not the first line, add 5 spaces for indentation
        indent = 5 if result else 0
        available = max_length - indent
        
        if len(remaining) <= available:
            # The remaining content fits in one line
            if indent:
                result.append(" " * indent + remaining)
            else:
                result.append(remaining)
            break
        
        # Find a good splitting point
        split_pos = available
        
        # Try to find a space to split on
        while split_pos > 0 and remaining[split_pos] != " ":
            split_pos -= 1
        
        if split_pos == 0:
            # No good space found, force split at available length
            split_pos = available
        
        # Add the current line with a continuation character
        if indent:
            result.append(" " * indent + remaining[:split_pos].rstrip() + " &")
        else:
            result.append(remaining[:split_pos].rstrip() + " &")
        
        # Process the remaining part
        remaining = remaining[split_pos:].strip()
    
    return "\n".join(result)
