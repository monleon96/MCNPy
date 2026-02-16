from .input import Input
from .perturbations import Perturbation, Pert
from kika.materials import Material, MaterialCollection
from kika.materials.parse_mcnp import read_material
import re

def _read_PERT(lines, start_index):
    """Internal helper function to read and parse a PERT card from MCNP input lines.

    :param lines: List of input lines from the MCNP input file
    :type lines: List[str]
    :param start_index: Starting index of the PERT card in the lines list
    :type start_index: int

    :returns: A tuple containing:
        - Pert object with parsed data, or None if parsing fails
        - The new index after processing the PERT card
    :rtype: tuple[Pert, int]
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
    
    return Pert(id=pert_num, particle=particle, **pert_attrs), i + 1

def read_mcnp(file_path):
    """Reads and parses an MCNP input file.

    This function reads an MCNP input file and parses its contents, focusing on
    PERT cards and material definitions. It creates an Input object containing 
    all parsed data.

    :param file_path: Path to the MCNP input file
    :type file_path: str

    :returns: An Input object containing all parsed data
    :rtype: Input
    """
    inst = Input()  # instance of the input class
    inst.perturbation = Perturbation()
    inst.materials = MaterialCollection()
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    i = 0
    current_zaid = None
    while i < len(lines):
        line = lines[i].strip()
        # Detect kika:pert_zaid comment
        zaid_match = re.search(r'kika:pert_zaid=(\d+)', line)
        if zaid_match:
            current_zaid = int(zaid_match.group(1))
            i += 1
        elif line.startswith("PERT"):
            pert_obj, i = _read_PERT(lines, i)
            if pert_obj:
                if current_zaid is not None:
                    pert_obj.zaid = current_zaid
                inst.perturbation.pert[pert_obj.id] = pert_obj
        elif line.startswith("m"):
            material_obj, i = read_material(lines, i)
            if material_obj:
                inst.materials.by_id[material_obj.id] = material_obj
        else:
            i += 1
            
    return inst
