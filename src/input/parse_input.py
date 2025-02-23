from .input import Input, Perturbation, Pert
import re

def read_PERT(lines, start_index):
    """
    Read and parse a PERT card from the input lines.
    
    Args:
        lines: List of input lines
        start_index: Starting index of the PERT card
        
    Returns:
        tuple: (perturbation_object, new_index) or (None, new_index) if parsing fails
    """
    i = start_index
    line = lines[i].strip()
    card_lines = []
    
    # Accumulate lines that end with '&'
    while line.endswith("&"):
        card_lines.append(line.rstrip("&").strip())
        i += 1
        if i < len(lines):
            line = lines[i].strip()
        else:
            break
    card_lines.append(line)
    full_line = " ".join(card_lines)
    
    # Tokenize the full card line
    tokens = full_line.split()
    # Parse header: e.g. "PERT1:n"
    header_match = re.match(r'PERT(\d+):(\w)', tokens[0])
    if not header_match:
        return None, i + 1
        
    pert_num = int(header_match.group(1))
    particle = header_match.group(2)
    
    allowed_keywords = {"CELL", "MAT", "RHO", "METHOD", "RXN", "ERG"}
    pert_attrs = {}
    j = 1
    while j < len(tokens):
        token = tokens[j]
        if "=" in token:
            key, val = token.split("=", 1)
            key = key.upper()
            if key not in allowed_keywords:
                break
            if key == "CELL":
                cell_vals = val.replace(',', ' ').split()
                pert_attrs['cell'] = [int(x) for x in cell_vals]
            elif key == "MAT":
                pert_attrs['material'] = int(val)
            elif key == "RHO":
                pert_attrs['rho'] = float(val)
            elif key == "METHOD":
                pert_attrs['method'] = int(val)
            elif key == "RXN":
                pert_attrs['reaction'] = int(val)
            elif key == "ERG":
                erg_numbers = [val]
                j += 1
                while j < len(tokens) and "=" not in tokens[j]:
                    erg_numbers.append(tokens[j])
                    j += 1
                if len(erg_numbers) >= 2:
                    pert_attrs['energy'] = (float(erg_numbers[0]), float(erg_numbers[1]))
                else:
                    pert_attrs['energy'] = None
                continue
            j += 1
        else:
            break
    
    return Perturbation(id=pert_num, particle=particle, **pert_attrs), i + 1

def read_mcnp(file_path):
    inst = Input()  # instance of the input class
    inst.pert = Pert()
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("PERT"):
            pert_obj, i = read_PERT(lines, i)
            if pert_obj:
                inst.pert.perturbation[pert_obj.id] = pert_obj
        else:
            i += 1
            
    return inst

