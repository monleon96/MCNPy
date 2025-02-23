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

def read_mctal(filename, tally_ids=None, tfc=True, pert=True):
    """Read and parse an MCNP MCTAL file.

    :param filename: Path to the MCTAL file
    :type filename: str
    :param tally_ids: List of tally IDs to parse. If None, parse all
    :type tally_ids: List[int] or None
    :param tfc: Whether to parse TFC (tally fluctuation chart) data
    :type tfc: bool
    :param pert: Whether to parse perturbation data
    :type pert: bool
    :returns: An Mctal object containing the parsed data
    :rtype: Mctal
    :raises ValueError: If the file format is invalid or parsing fails
    """
    if tally_ids is None:
        tally_ids = []
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

        # If no tally_ids are provided, use all found tally numbers.
        if not tally_ids:
            tally_ids = set(tally_numbers)
        else:
            tally_ids = set(tally_ids)

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
                        tally = parse_tally(tid, f, pos, tfc, pert)
                        mctal.tally[tid] = tally  # Store in dict using tid as key
                        found_tally_ids.add(tid)
                        if found_tally_ids == tally_ids:
                            break

        missing_ids = tally_ids - found_tally_ids
        if missing_ids:
            raise ValueError(f"Could not find or parse the following requested tally IDs: {missing_ids}")

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
    is_zero_bins = (n_energy_bins == 0)
    expected_energies = 0 if is_zero_bins else (n_energy_bins - 1 if is_total else n_energy_bins)

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

    tally.energies = energies

    # After energy parsing, look for 'vals' line
    while True:
        line = file_obj.readline()
        if not line:
            raise ValueError(f"Unexpected end of file while looking for vals in tally {tally_id}")
        if line.lstrip().lower().startswith('vals'):
            break

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
                results.append(float(values[i]))
                errors.append(float(values[i + 1]))
            except (ValueError, IndexError) as e:
                raise ValueError(f"Error parsing result/error pairs in tally {tally_id}: {str(e)}")

    # Handle results based on case
    if is_zero_bins:
        if len(results) != 1 or len(errors) != 1:
            raise ValueError(
                f"Expected 1 result/error pair for zero bins case "
                f"but found {len(results)}/{len(errors)} for tally {tally_id}"
            )
        # For zero bins, save the single value in both places
        tally.integral_result = results[0]
        tally.integral_error = errors[0]
        tally.results = results
        tally.errors = errors
    else:
        expected_length = len(energies)
        if is_total:
            if len(results) != expected_length + 1 or len(errors) != expected_length + 1:
                raise ValueError(
                    f"Expected {expected_length + 1} results/errors (total bin case) "
                    f"but found {len(results)}/{len(errors)} for tally {tally_id}"
                )
            # Split off the total values
            tally.integral_result = results.pop()
            tally.integral_error = errors.pop()
        else:
            if len(results) != expected_length or len(errors) != expected_length:
                raise ValueError(
                    f"Expected {expected_length} results/errors but found "
                    f"{len(results)}/{len(errors)} for tally {tally_id}"
                )
        tally.results = results
        tally.errors = errors

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

            # Validate pert results using same expected lengths as unperturbed
            if is_zero_bins:
                if len(pert_results) != 1 or len(pert_errors) != 1:
                    raise ValueError(
                        f"Expected 1 result/error pair for zero bins pert case "
                        f"but found {len(pert_results)}/{len(pert_errors)} for tally {tally_id}"
                    )
                pert_integral_result = pert_results[0]
                pert_integral_error = pert_errors[0]
            else:
                expected_length = len(energies)
                if is_total:
                    if len(pert_results) != expected_length + 1 or len(pert_errors) != expected_length + 1:
                        raise ValueError(
                            f"Expected {expected_length + 1} results/errors for pert (total bin case) "
                            f"but found {len(pert_results)}/{len(pert_errors)} for tally {tally_id}"
                        )
                    pert_integral_result = pert_results.pop()
                    pert_integral_error = pert_errors.pop()
                else:
                    if len(pert_results) != expected_length or len(pert_errors) != expected_length:
                        raise ValueError(
                            f"Expected {expected_length} results/errors for pert but found "
                            f"{len(pert_results)}/{len(pert_errors)} for tally {tally_id}"
                        )
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
                energies=energies.copy(),
                results=pert_results,
                errors=pert_errors,
                integral_result=pert_integral_result,
                integral_error=pert_integral_error,
                tfc_nps=pert_tfc_nps,
                tfc_results=pert_tfc_results,
                tfc_errors=pert_tfc_errors,
                tfc_fom=pert_tfc_fom,
                pert_data={}  # New pert instance doesn't have its own pert_data
            )
            tally.pert_data[pert_index] = new_tally_pert

    tally._end_pos = file_obj.tell()
    return tally