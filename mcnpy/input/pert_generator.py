import sys
from mcnpy._grids import ENERGY_GRIDS

def generate_PERTcards(cell, rho, reactions, energies, mat=None, order=2, errors=False, output_path=None):
    """Generates PERT cards for MCNP input files.

    Generates PERT cards based on the provided parameters. Can generate both first and
    second order perturbations, as well as cards for exact uncertainty calculations.
    Note that exact uncertainties are usually negligible, so verify their necessity
    before running long calculations.

    :param cell: Cell number(s) for PERT card application
    :type cell: int or str or list[int]
    :param rho: Density value for the perturbation
    :type rho: float
    :param reactions: List of reaction identifiers
    :type reactions: list[str]
    :param energies: Energy values. Used in consecutive pairs for energy bins
    :type energies: list[float]
    :param mat: Material identifier, defaults to None
    :type mat: str, optional
    :param order: Order of PERT card method (1 or 2), defaults to 2
    :type order: int, optional
    :param errors: Whether to include error methods (-2, -3, 1), defaults to False
    :type errors: bool, optional
    :param output_path: Path to output file. If None, prints to stdout
    :type output_path: str, optional
    
    :returns: None
    
    :note: Prints PERT cards to either stdout or specified file with sequential numbering
    """
    # Determine output stream
    if output_path:
        stream = open(output_path, "w")
    else:
        stream = sys.stdout

    # Initialize the perturbation counter
    pert_counter = 1
    if type(cell) == list: 
        cell_str = ','.join(map(str, cell)) if isinstance(cell, list) else str(cell)
    else: 
        cell_str = str(cell)
    # Loop over each combination of cell, rho, and reaction
    for reaction in reactions:
        # Go through the energy list and use consecutive pairs
        for i in range(len(energies) - 1):
            E1 = energies[i]
            E2 = energies[i + 1]

            if mat is None:
                # Print the output for METHOD=2
                stream.write(f"PERT{pert_counter}:n CELL={cell_str} &\nRHO={rho:.6e} METHOD=2 RXN={reaction} ERG={E1:.6e} {E2:.6e}\n")
                pert_counter += 1

                # Print the output for METHOD=3
                if order == 2:
                    stream.write(f"PERT{pert_counter}:n CELL={cell_str} &\nRHO={rho:.6e} METHOD=3 RXN={reaction} ERG={E1:.6e} {E2:.6e}\n")
                    pert_counter += 1

                if errors:
                    # Print the output for METHOD=-2
                    stream.write(f"PERT{pert_counter}:n CELL={cell_str} &\nRHO={rho:.6e} METHOD=-2 RXN={reaction} ERG={E1:.6e} {E2:.6e}\n")
                    pert_counter += 1

                    if order == 2:
                        # Print the output for METHOD=-3
                        stream.write(f"PERT{pert_counter}:n CELL={cell_str} &\nRHO={rho:.6e} METHOD=-3 RXN={reaction} ERG={E1:.6e} {E2:.6e}\n")
                        pert_counter += 1

                        # Print the output for METHOD=1
                        stream.write(f"PERT{pert_counter}:n CELL={cell_str} &\nRHO={rho:.6e} METHOD=1 RXN={reaction} ERG={E1:.6e} {E2:.6e}\n")
                        pert_counter += 1

            else:
                # Print the output for METHOD=2 with MAT
                stream.write(f"PERT{pert_counter}:n CELL={cell_str} MAT={mat} &\nRHO={rho:.6e} METHOD=2 RXN={reaction} ERG={E1:.6e} {E2:.6e}\n")
                pert_counter += 1

                if order == 2:
                    # Print the output for METHOD=3 with MAT
                    stream.write(f"PERT{pert_counter}:n CELL={cell_str} MAT={mat} &\nRHO={rho:.6e} METHOD=3 RXN={reaction} ERG={E1:.6e} {E2:.6e}\n")
                    pert_counter += 1
            
                if errors:
                    # Print the output for METHOD=-2 with MAT
                    stream.write(f"PERT{pert_counter}:n CELL={cell_str} MAT={mat} &\nRHO={rho:.6e} METHOD=-2 RXN={reaction} ERG={E1:.6e} {E2:.6e}\n")
                    pert_counter += 1

                    if order == 2:
                        # Print the output for METHOD=-3 with MAT
                        stream.write(f"PERT{pert_counter}:n CELL={cell_str} MAT={mat} &\nRHO={rho:.6e} METHOD=-3 RXN={reaction} ERG={E1:.6e} {E2:.6e}\n")
                        pert_counter += 1

                        # Print the output for METHOD=1 with MAT
                        stream.write(f"PERT{pert_counter}:n CELL={cell_str} MAT={mat} &\nRHO={rho:.6e} METHOD=1 RXN={reaction} ERG={E1:.6e} {E2:.6e}\n")
                        pert_counter += 1

    # Close the file if it was opened
    if output_path:
        stream.close()