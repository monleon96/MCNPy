from .mctal import Mctal, Tally, TallyPert

def _parse_fixed_width(line, specs):
    """Internal helper function to parse a fixed-width formatted line into fields.

    :param line: Input line to parse
    :type line: str
    :param specs: List of (field_name, width) tuples
    :type specs: List[Tuple[str, int]]
    :returns: Dictionary mapping field names to their values
    :rtype: dict
    """
    fields = {}
    pos = 0
    for name, width in specs:
        fields[name] = line[pos:pos+width].strip()
        pos += width
    return fields

def read_mctal(filename):
    """Read and parse an MCNP MCTAL file.

    :param filename: Path to the MCTAL file
    :type filename: str
    :returns: An Mctal object containing the parsed data
    :rtype: Mctal
    :raises ValueError: If the file format is invalid or parsing fails
    """
    mctal = Mctal()

    with open(filename, "r") as f:
        # --- Parse Header Information ---
        header_line = f.readline().rstrip("\n")
        if len(header_line) >= 83:
            header_specs = [("code_name", 8), ("skip1", 3), ("ver", 5), ("probid", 19),
                            ("knod", 11), ("skip2", 1), ("nps", 20), ("skip3", 1), ("rnr", 15)]
        else:
            header_specs = [("code_name", 8), ("skip1", 3), ("ver", 5), ("probid", 19),
                            ("knod", 5), ("skip2", 1), ("nps", 15), ("skip3", 1), ("rnr", 15)]
        header_fields = _parse_fixed_width(header_line, header_specs)
        mctal.code_name = header_fields["code_name"]
        mctal.ver = header_fields["ver"]
        mctal.probid = header_fields["probid"]
        mctal.knod = int(header_fields["knod"])
        mctal.nps = int(header_fields["nps"])
        mctal.rnr = int(header_fields["rnr"])

        # --- Parse Problem Identification Line ---
        mctal.problem_id = f.readline().strip()

        # --- Parse Tally Header ---
        tally_header = f.readline().strip()
        parts = tally_header.split()
        
        # Check format: should start with "ntal" followed by a number
        if len(parts) < 2 or parts[0].lower() != "ntal":
            raise ValueError("Invalid tally header format: should start with 'ntal'")
        
        try:
            mctal.ntal = int(parts[1])
        except ValueError:
            raise ValueError("Unable to parse number of tallies (ntal)")
        
        # Check if npert is present (it's optional)
        if len(parts) >= 4 and parts[2].lower() == "npert":
            try:
                mctal.npert = int(parts[3])
            except ValueError:
                raise ValueError("Unable to parse number of perturbations (npert)")

        # --- Parse Tally Numbers ---
        tally_numbers = []
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            stripped = line.strip()
            # If the first token isn’t numeric, assume we’ve passed the tally numbers section.
            if not stripped or not stripped.split()[0].replace('-', '').isdigit():
                f.seek(pos)
                break
            for num in stripped.split():
                if num.replace('-', '').isdigit():
                    tally_numbers.append(int(num))
        mctal.tally_numbers = tally_numbers

        # Read all tallies
        tally_ids = set(tally_numbers)

        # --- Parse Tally Sections ---
        found_tally_ids = set()
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            if line.lstrip().lower().startswith("tally "):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        tid = int(parts[1])
                    except ValueError:
                        continue
                    if tid in tally_ids:
                        # Use pos (saved before reading this line) as the start position
                        # Always read TFC and perturbation data
                        tally = parse_tally(tid, f, pos, True, True)
                        mctal.tally[tid] = tally  # Store in dict using tid as key
                        found_tally_ids.add(tid)
                        if found_tally_ids == tally_ids:
                            break

        missing_ids = tally_ids - found_tally_ids
        if missing_ids:
            raise ValueError(f"Could not find or parse the following tally IDs: {missing_ids}")

    return mctal

def _parse_scientific_notation(numbers_str):
    """Internal helper function to parse space-separated scientific notation numbers.

    :param numbers_str: String containing space-separated numbers
    :type numbers_str: str
    :returns: List of parsed floating point numbers
    :rtype: List[float]
    :raises ValueError: If parsing fails for any number
    """
    return [float(num) for num in numbers_str.split()]

def parse_tally(tally_id, file_obj, start_pos, tfc=True, pert=True):
    """Parse a single tally section from an MCTAL file.

    :param tally_id: ID number of the tally to parse
    :type tally_id: int
    :param file_obj: Open file object positioned at tally start
    :type file_obj: file
    :param start_pos: File position where tally section starts
    :type start_pos: int
    :param tfc: Whether to parse TFC data
    :type tfc: bool
    :param pert: Whether to parse perturbation data
    :type pert: bool
    :returns: A Tally object containing the parsed data
    :rtype: Tally
    :raises ValueError: If tally format is invalid or parsing fails
    """
    # Rewind to the start of the tally section
    file_obj.seek(start_pos)
    tally = Tally(tally_id=tally_id)
    tally._start_pos = start_pos

    # Now read the tally header line containing "tally", the id, and 3 numbers.
    line = file_obj.readline()
    parts = line.split()
    if len(parts) < 5:  # Expecting "tally", id, and 3 numbers.
        raise ValueError("Invalid tally header line format")
    try:
        header_numbers = [int(x) for x in parts[2:]]
    except ValueError:
        raise ValueError("Unable to parse numbers in tally header line")
    # Verify that the first number is -1.
    if header_numbers[0] != -1:
        raise ValueError(f"Tally {tally_id} has unsupported number of particle types (expected -1 but got {header_numbers[0]})")

    # Read the next line that should contain 37 numbers.
    particle_line = file_obj.readline()
    try:
        particle_numbers = [int(x) for x in particle_line.split()]
    except ValueError:
        raise ValueError("Unable to parse particle numbers in tally")
    if len(particle_numbers) != 37 or particle_numbers[0] != 1 or any(x != 0 for x in particle_numbers[1:]):
        raise ValueError(f"Tally {tally_id} particle configuration is invalid (expected 37 numbers with first 1 and rest 0)")

    # Parse tally name
    line = file_obj.readline()
    if line[0].isspace():  # If line starts with space, it's a name
        tally.name = line.strip()
        line = file_obj.readline()
    else:
        tally.name = ""
        # line already contains the next section to process

    # Parse 'f' line - cells or surfaces where tally is scored
    while not line.lower().startswith('f '):
        line = file_obj.readline()
        if not line:
            raise ValueError(f"Unexpected end of file while looking for 'f' line for tally {tally_id}")

    parts = line.split()
    if len(parts) < 2 or not parts[1].isdigit():
        raise ValueError(f"Invalid format for 'f' line in tally {tally_id}")
    
    n_cells_surfaces = int(parts[1])
    tally.n_cells_surfaces = n_cells_surfaces
    
    # Parse cell/surface numbers - can span multiple lines
    cell_surface_ids = []
    while len(cell_surface_ids) < n_cells_surfaces:
        line = file_obj.readline()
        if not line:
            raise ValueError(f"Unexpected end of file while parsing cell/surface IDs for tally {tally_id}")
        
        # If we've reached the 'd' line, we've gone too far
        if line.lstrip().lower().startswith('d '):
            raise ValueError(f"Reached 'd' line before finding all {n_cells_surfaces} cell/surface IDs for tally {tally_id}")
        
        try:
            cell_surface_ids.extend([int(x) for x in line.strip().split()])
        except ValueError:
            raise ValueError(f"Error parsing cell/surface IDs for tally {tally_id}")
    
    tally.cell_surface_ids = cell_surface_ids
    
    # Parse 'd' line - direct vs. total or flagged vs. unflagged bins
    while not line.lower().startswith('d '):
        line = file_obj.readline()
        if not line:
            raise ValueError(f"Unexpected end of file while looking for 'd' line for tally {tally_id}")
    
    parts = line.split()
    if len(parts) < 2 or not parts[1].isdigit():
        raise ValueError(f"Invalid format for 'd' line in tally {tally_id}")
    
    n_direct_bins = int(parts[1])
    tally.n_direct_bins = n_direct_bins
    
    # Currently, only allow n_direct_bins = 1 as specified in the requirements
    if n_direct_bins != 1:
        raise ValueError(f"Only n_direct_bins = 1 is currently supported, got {n_direct_bins} for tally {tally_id}")
    
    # Parse 'u', 'ut', or 'uc' line - number of user bins
    line = file_obj.readline()
    if not line.lower().startswith(('u ', 'ut ', 'uc ')):
        raise ValueError(f"Expected 'u', 'ut', or 'uc' line but got '{line}' for tally {tally_id}")
    
    parts = line.split()
    if len(parts) < 2 or not parts[1].isdigit():
        raise ValueError(f"Invalid format for '{parts[0]}' line in tally {tally_id}")
    
    is_total_user_bin = line.lower().startswith('ut ')
    is_cumulative_user_bin = line.lower().startswith('uc ')
    n_user_bins = int(parts[1])
    
    tally.n_user_bins = n_user_bins
    tally._has_total_user_bin = is_total_user_bin  # Fix: use private attribute
    tally._has_cumulative_user_bin = is_cumulative_user_bin  # Fix: use private attribute
    
    # Parse 's', 'st', or 'sc' line - number of segment bins
    line = file_obj.readline()
    if not line.lower().startswith(('s ', 'st ', 'sc ')):
        raise ValueError(f"Expected 's', 'st', or 'sc' line but got '{line}' for tally {tally_id}")
    
    parts = line.split()
    if len(parts) < 2 or not parts[1].isdigit():
        raise ValueError(f"Invalid format for '{parts[0]}' line in tally {tally_id}")
    
    is_total_segment_bin = line.lower().startswith('st ')
    is_cumulative_segment_bin = line.lower().startswith('sc ')
    n_segment_bins = int(parts[1])
    
    tally.n_segment_bins = n_segment_bins
    tally._has_total_segment_bin = is_total_segment_bin  # Fix: use private attribute
    tally._has_cumulative_segment_bin = is_cumulative_segment_bin  # Fix: use private attribute
    
    # Parse 'm', 'mt', or 'mc' line - number of multiplier bins
    line = file_obj.readline()
    if not line.lower().startswith(('m ', 'mt ', 'mc ')):
        raise ValueError(f"Expected 'm', 'mt', or 'mc' line but got '{line}' for tally {tally_id}")
    
    parts = line.split()
    if len(parts) < 2 or not parts[1].isdigit():
        raise ValueError(f"Invalid format for '{parts[0]}' line in tally {tally_id}")
    
    is_total_multiplier_bin = line.lower().startswith('mt ')
    is_cumulative_multiplier_bin = line.lower().startswith('mc ')
    n_multiplier_bins = int(parts[1])
    
    tally.n_multiplier_bins = n_multiplier_bins
    tally._has_total_multiplier_bin = is_total_multiplier_bin  # Fix: use private attribute
    tally._has_cumulative_multiplier_bin = is_cumulative_multiplier_bin  # Fix: use private attribute
    
    # Parse 'c', 'ct', or 'cc' line - number of cosine bins
    line = file_obj.readline()
    if not line.lower().startswith(('c ', 'ct ', 'cc ')):
        raise ValueError(f"Expected 'c', 'ct', or 'cc' line but got '{line}' for tally {tally_id}")
    
    parts = line.split()
    if len(parts) < 2 or not parts[1].isdigit():
        raise ValueError(f"Invalid format for '{parts[0]}' line in tally {tally_id}")
    
    is_total_cosine_bin = line.lower().startswith('ct ')
    is_cumulative_cosine_bin = line.lower().startswith('cc ')
    n_cosine_bins = int(parts[1])
    
    tally.n_cosine_bins = n_cosine_bins
    tally._has_total_cosine_bin = is_total_cosine_bin  # Fix: use private attribute
    tally._has_cumulative_cosine_bin = is_cumulative_cosine_bin  # Fix: use private attribute
    
    # Parse energy bins
    while not line.lower().startswith(('e ', 'ec ', 'et ')):
        line = file_obj.readline()
        if not line:
            raise ValueError(f"Unexpected end of file while parsing energy bins for tally {tally_id}")

    # Parse number of energy bins
    parts = line.split()
    if len(parts) < 2 or not parts[1].isdigit():
        raise ValueError(f"Invalid energy bin count line format for tally {tally_id}")
    n_energy_bins = int(parts[1])
    
    # Check if this is a total bin case (et) or zero bins case
    is_total = line.lower().startswith('et ')
    is_cumulative = line.lower().startswith('ec ')
    is_zero_bins = (n_energy_bins == 0)
    expected_energies = 0 if is_zero_bins else (n_energy_bins - 1 if is_total else n_energy_bins)
    
    tally.n_energy_bins = n_energy_bins
    tally._has_total_energy_bin = is_total  # Fix: use private attribute
    tally._has_cumulative_energy_bin = is_cumulative  # Fix: use private attribute

    # Parse energy values if we have any bins
    energies = []
    if not is_zero_bins:
        while True:
            line = file_obj.readline()
            if not line:
                raise ValueError(f"Unexpected end of file while parsing energies for tally {tally_id}")
            
            # Check if we've reached the next section
            if line.lstrip().lower().startswith(('t ', 'tt ', 'tc ')):
                break

            # Parse scientific notation numbers from the line
            try:
                energies.extend(_parse_scientific_notation(line.strip()))
            except ValueError as e:
                raise ValueError(f"Error parsing energy values for tally {tally_id}: {str(e)}")

        if len(energies) != expected_energies:
            raise ValueError(
                f"Expected {expected_energies} energy bins "
                f"({'total bin case' if is_total else 'normal case'}) "
                f"but found {len(energies)} for tally {tally_id}"
            )
    else:
        # For zero energy bins, we need to explicitly read the next line 
        # which should be the time bins line
        line = file_obj.readline()
        if not line:
            raise ValueError(f"Unexpected end of file after zero energy bins for tally {tally_id}")

    tally.energies = energies
    
    # Parse 't', 'tt', or 'tc' line - number of time bins (line already read in energy parsing)
    if not line.lower().startswith(('t ', 'tt ', 'tc ')):
        raise ValueError(f"Expected 't', 'tt', or 'tc' line but got '{line}' for tally {tally_id}")
    
    parts = line.split()
    if len(parts) < 2 or not parts[1].isdigit():
        raise ValueError(f"Invalid format for '{parts[0]}' line in tally {tally_id}")
    
    is_total_time_bin = line.lower().startswith('tt ')
    is_cumulative_time_bin = line.lower().startswith('tc ')
    n_time_bins = int(parts[1])
    
    tally.n_time_bins = n_time_bins
    tally._has_total_time_bin = is_total_time_bin  # Fix: use private attribute
    tally._has_cumulative_time_bin = is_cumulative_time_bin  # Fix: use private attribute
    
    # Parse time values if we have any bins
    times = []
    is_zero_time_bins = (n_time_bins == 0)
    expected_times = 0 if is_zero_time_bins else (n_time_bins - 1 if is_total_time_bin else n_time_bins)
    
    if not is_zero_time_bins:
        while True:
            line = file_obj.readline()
            if not line:
                raise ValueError(f"Unexpected end of file while parsing time bins for tally {tally_id}")
            
            # Check if we've reached the vals section
            if line.lstrip().lower().startswith('vals'):
                break
            
            # Parse scientific notation numbers from the line
            try:
                times.extend(_parse_scientific_notation(line.strip()))
            except ValueError as e:
                raise ValueError(f"Error parsing time values for tally {tally_id}: {str(e)}")
    
        if len(times) != expected_times:
            raise ValueError(
                f"Expected {expected_times} time bins "
                f"({'total bin case' if is_total_time_bin else 'normal case'}) "
                f"but found {len(times)} for tally {tally_id}"
            )
    else:
        # If we have zero time bins, we should directly reach the vals section
        line = file_obj.readline()
        if not line.lstrip().lower().startswith('vals'):
            raise ValueError(f"Expected 'vals' line after zero time bins but got '{line}' for tally {tally_id}")
    
    tally.times = times

    # After parsing all bin dimensions, we're at the 'vals' line
    if not line.lstrip().lower().startswith('vals'):
        # If we're not at vals yet, read until we find it
        while True:
            line = file_obj.readline()
            if not line:
                raise ValueError(f"Unexpected end of file while looking for vals in tally {tally_id}")
            if line.lstrip().lower().startswith('vals'):
                break

    # Calculate the total number of expected result/error pairs
    # For dimensions with 0 bins, treat as 1 bin for multiplication
    # For dimensions with total bins, use the actual count including total
    def effective_bin_count(count, has_total):
        if count == 0:
            return 1
        return count

    expected_result_count = (
        effective_bin_count(tally.n_cells_surfaces, False) *
        effective_bin_count(tally.n_direct_bins, False) *
        effective_bin_count(tally.n_user_bins, tally._has_total_user_bin) *
        effective_bin_count(tally.n_segment_bins, tally._has_total_segment_bin) *
        effective_bin_count(tally.n_multiplier_bins, tally._has_total_multiplier_bin) *
        effective_bin_count(tally.n_cosine_bins, tally._has_total_cosine_bin) *
        effective_bin_count(tally.n_energy_bins, tally._has_total_energy_bin) *
        effective_bin_count(tally.n_time_bins, tally._has_total_time_bin)
    )

    # Parse results and errors
    results = []
    errors = []
    while True:
        line = file_obj.readline()
        if not line:
            raise ValueError(f"Unexpected end of file while parsing vals in tally {tally_id}")
        
        if line.lstrip().lower().startswith('tfc'):
            break

        # Parse pairs of values (result and error)
        values = line.strip().split()
        for i in range(0, len(values), 2):
            try:
                if i+1 < len(values):  # Make sure we have both result and error
                    results.append(float(values[i]))
                    errors.append(float(values[i + 1]))
            except (ValueError, IndexError) as e:
                raise ValueError(f"Error parsing result/error pairs in tally {tally_id}: {str(e)}")

    # Verify we have the expected number of results
    if len(results) != expected_result_count or len(errors) != expected_result_count:
        raise ValueError(
            f"Expected {expected_result_count} result/error pairs based on bin structure, "
            f"but found {len(results)}/{len(errors)} for tally {tally_id}"
        )

    # Store the raw results and errors
    tally.results = results.copy()
    tally.errors = errors.copy()

    # For backward compatibility and simpler access, extract the total energy bin where applicable
    if tally._has_total_energy_bin:
        if (tally.n_cells_surfaces <= 1 and
            tally.n_direct_bins <= 1 and
            tally.n_user_bins <= 1 and
            tally.n_segment_bins <= 1 and
            tally.n_multiplier_bins <= 1 and
            tally.n_cosine_bins <= 1 and
            effective_bin_count(tally.n_time_bins, tally._has_total_time_bin) <= 1):
            
            # For simple tallies with energy bins and a total:
            # 1. Store the total energy bin values
            tally.total_energy_result = results[-1]
            tally.total_energy_error = errors[-1]
            
            # 2. Remove total from regular results (keep only energy-specific values)
            tally.results = results[:-1]
            tally.errors = errors[:-1]
            
            # 3. Keep the total value also in integral_result for backward compatibility
            tally.integral_result = tally.total_energy_result
            tally.integral_error = tally.total_energy_error
        else:
            # For multi-dimensional tallies with energy total bins, we need to extract the total bin values
            # In MCNP MCTAL files, the rightmost dimension (energy) varies fastest
            # Extract total energy results and remove them from the main results array
            regular_results, total_results = separate_total_energy_bins(
                results, 
                tally.n_energy_bins, 
                tally._has_total_energy_bin
            )
            regular_errors, total_errors = separate_total_energy_bins(
                errors, 
                tally.n_energy_bins, 
                tally._has_total_energy_bin
            )
            
            # Store regular results (without totals)
            tally.results = regular_results
            tally.errors = regular_errors
            
            # Store all total energy values in a dictionary
            tally.total_energy_values = {
                'results': total_results,
                'errors': total_errors
            }
            
            # Store the first total value for backward compatibility
            if total_results and total_errors:
                tally.total_energy_result = total_results[0]
                tally.total_energy_error = total_errors[0]
            else:
                tally.total_energy_result = None
                tally.total_energy_error = None
            
            # For multi-dimensional tallies, don't set an integral_result
            tally.integral_result = None
            tally.integral_error = None

    elif expected_result_count == 1:
        # If we have exactly one result (all dimensions are zero or one with no totals)
        # Single result case
        tally.integral_result = results[0]
        tally.integral_error = errors[0]
    else:
        # Multi-dimensional case without total energy bins
        tally.integral_result = None
        tally.integral_error = None

    # Parse TFC data if requested
    if tfc:
        # Parse TFC header line
        parts = line.strip().split()
        if len(parts) < 10:  # "tfc" + 9 numbers
            raise ValueError(f"Invalid TFC header format in tally {tally_id}")
        
        try:
            tfc_n_steps = int(parts[1])
        except ValueError:
            raise ValueError(f"Unable to parse number of TFC steps in tally {tally_id}")

        # Parse TFC data lines
        for _ in range(tfc_n_steps):
            line = file_obj.readline()
            if not line:
                raise ValueError(f"Unexpected end of file while parsing TFC data in tally {tally_id}")
            
            try:
                nps, result, error, fom = line.strip().split()
                tally.tfc_nps.append(int(nps))
                tally.tfc_results.append(float(result))
                tally.tfc_errors.append(float(error))
                tally.tfc_fom.append(float(fom))
            except (ValueError, IndexError) as e:
                raise ValueError(f"Error parsing TFC data line in tally {tally_id}: {str(e)}")
    else:
        # Skip TFC data if not requested
        while True:
            line = file_obj.readline()
            if not line:
                break
            if line.lstrip().lower().startswith(('tally', 'surf', 'cell', 'vals')):
                file_obj.seek(file_obj.tell() - len(line))
                break

    # --- Parse perturbed results if available (npert > 0) and requested ---
    if not pert:
        # Skip all perturbation data
        while True:
            pos_skip = file_obj.tell()
            line = file_obj.readline()
            if not line:
                break
            if line.lstrip().lower().startswith(('tally', 'surf', 'cell')):
                file_obj.seek(pos_skip)
                break
    else:
        # Look for blocks starting with "vals pert"
        while True:
            pos_pert = file_obj.tell()
            line = file_obj.readline()
            if not line:
                break
            if not line.lstrip().lower().startswith("vals pert"):
                file_obj.seek(pos_pert)
                break  # no more pert blocks
            
            # Parse vals pert header
            parts = line.strip().split()
            if len(parts) < 3:
                raise ValueError(f"Invalid pert vals header in tally {tally_id}")
            try:
                pert_index = int(parts[2])
            except ValueError:
                raise ValueError(f"Unable to parse perturbation index in tally {tally_id}")
            
            # Parse pert result/error pairs
            pert_results = []
            pert_errors = []
            while True:
                line = file_obj.readline().strip()
                if not line:
                    raise ValueError(f"Unexpected end of file while parsing pert vals in tally {tally_id}")
                
                # Check if we've reached the TFC section
                if line.lower().startswith('tfc'):
                    break
                
                # Parse pairs of values
                values = line.split()
                try:
                    for i in range(0, len(values), 2):
                        pert_results.append(float(values[i]))
                        pert_errors.append(float(values[i+1]))
                except (ValueError, IndexError) as e:
                    raise ValueError(f"Error parsing pert result/error pairs in tally {tally_id}: {str(e)}")

            # Validate pert results — expected count matches unperturbed
            # (all dimensions contribute: cells × energy × etc.)
            _ebc = lambda c: 1 if c == 0 else c
            expected_pert_count = (
                _ebc(tally.n_cells_surfaces) *
                _ebc(tally.n_direct_bins) *
                _ebc(tally.n_user_bins) *
                _ebc(tally.n_segment_bins) *
                _ebc(tally.n_multiplier_bins) *
                _ebc(tally.n_cosine_bins) *
                _ebc(tally.n_energy_bins) *
                _ebc(tally.n_time_bins)
            )

            if len(pert_results) != expected_pert_count or len(pert_errors) != expected_pert_count:
                raise ValueError(
                    f"Expected {expected_pert_count} results/errors for pert "
                    f"but found {len(pert_results)}/{len(pert_errors)} for tally {tally_id}"
                )

            pert_total_energy_values = None
            n_pert_cells = _ebc(tally.n_cells_surfaces)

            if is_zero_bins:
                pert_integral_result = pert_results[0]
                pert_integral_error = pert_errors[0]
            elif is_total:
                if n_pert_cells <= 1:
                    # Single cell: last value is the total energy bin
                    pert_integral_result = pert_results.pop()
                    pert_integral_error = pert_errors.pop()
                else:
                    # Multi-cell: separate total energy bins from regular results
                    pert_results, pert_total_results = separate_total_energy_bins(
                        pert_results, tally.n_energy_bins, True)
                    pert_errors, pert_total_errors = separate_total_energy_bins(
                        pert_errors, tally.n_energy_bins, True)
                    pert_integral_result = None
                    pert_integral_error = None
                    pert_total_energy_values = {
                        'results': pert_total_results,
                        'errors': pert_total_errors
                    }
            else:
                pert_integral_result = None
                pert_integral_error = None

            # Parse pert TFC data - line already contains the TFC header
            if tfc:  # Use the same TFC flag for both main and pert data
                parts_tfc = line.split()
                if len(parts_tfc) < 10:
                    raise ValueError(f"Invalid TFC header for pert in tally {tally_id}")
                try:
                    pert_tfc_n_steps = int(parts_tfc[1])
                except ValueError:
                    raise ValueError(f"Unable to parse number of pert TFC steps in tally {tally_id}")

                pert_tfc_nps = []
                pert_tfc_results = []
                pert_tfc_errors = []
                pert_tfc_fom = []
                
                # Parse TFC data lines
                for _ in range(pert_tfc_n_steps):
                    line = file_obj.readline()
                    if not line:
                        raise ValueError(f"Unexpected end of file while parsing pert TFC data in tally {tally_id}")
                    try:
                        nps_val, res_val, err_val, fom_val = line.strip().split()
                        pert_tfc_nps.append(int(nps_val))
                        pert_tfc_results.append(float(res_val))
                        pert_tfc_errors.append(float(err_val))
                        pert_tfc_fom.append(float(fom_val))
                    except (ValueError, IndexError) as e:
                        raise ValueError(f"Error parsing pert TFC data line in tally {tally_id}: {str(e)}")
            else:
                # Skip TFC data
                pert_tfc_n_steps = None
                pert_tfc_nps = []
                pert_tfc_results = []
                pert_tfc_errors = []
                pert_tfc_fom = []
                # Skip the TFC lines if present
                parts_tfc = line.split()
                try:
                    n_skip = int(parts_tfc[1])
                    for _ in range(n_skip):
                        file_obj.readline()
                except (ValueError, IndexError):
                    pass

            # Create a new TallyPert object from the current tally.
            new_tally_pert = TallyPert(
                tally_id=tally.tally_id,
                name=tally.name,
                n_cells_surfaces=tally.n_cells_surfaces,
                cell_surface_ids=tally.cell_surface_ids.copy() if tally.cell_surface_ids else None,
                energies=energies.copy(),
                results=pert_results,
                errors=pert_errors,
                integral_result=pert_integral_result,
                integral_error=pert_integral_error,
                tfc_nps=pert_tfc_nps,
                tfc_results=pert_tfc_results,
                tfc_errors=pert_tfc_errors,
                tfc_fom=pert_tfc_fom,
                perturbation={},
                perturbation_number=pert_index
            )
            if pert_total_energy_values is not None:
                new_tally_pert.total_energy_values = pert_total_energy_values
            tally.perturbation[pert_index] = new_tally_pert

    tally._end_pos = file_obj.tell()
    return tally

def separate_total_energy_bins(values, n_energy_bins, has_total_energy_bin):
    """Separate the total energy bin values from the regular results array.
    
    In MCNP MCTAL files with multiple dimensions, energy is the rightmost dimension
    and varies fastest. For tallies with total energy bins, every nth value
    (where n = n_energy_bins) is a total bin value.
    
    Args:
        values (list): The flat array of values (results or errors)
        n_energy_bins (int): Number of energy bins including total bins
        has_total_energy_bin (bool): Whether the tally has total energy bins
        
    Returns:
        tuple: (regular_values, total_values) - values with totals removed, and the extracted totals
    """
    if not has_total_energy_bin or n_energy_bins <= 1:
        return values, []
    
    # For energy bins with a total, the number of regular bins is n_energy_bins - 1
    n_regular_bins = n_energy_bins - 1
    
    if n_regular_bins == 0:
        # Special case: If there are no regular energy bins, all values are total bins
        return [], values
    
    # Initialize regular and total value lists
    regular_values = []
    total_values = []
    
    # Extract values: in each group of n_energy_bins values, the last one is the total
    for i in range(0, len(values), n_energy_bins):
        # Add all but the last value to regular_values
        group_end = min(i + n_energy_bins, len(values))
        
        if i + n_regular_bins <= len(values):
            regular_values.extend(values[i:i + n_regular_bins])
        
        # Add the total value if it exists in this group
        if i + n_regular_bins < group_end:
            total_values.append(values[i + n_regular_bins])
    
    return regular_values, total_values