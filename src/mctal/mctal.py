from dataclasses import dataclass
from typing import Optional, List, Dict
import matplotlib.pyplot as plt

@dataclass
class Mctal:
    # Header information
    code_name: Optional[str] = None
    ver: Optional[str] = None
    probid: Optional[str] = None
    knod: Optional[int] = None
    nps: Optional[int] = None
    rnr: Optional[int] = None

    # Problem identification line
    problem_id: Optional[str] = None

    # Tally information
    ntal: Optional[int] = None  # Number of tallies
    npert: Optional[int] = 0    # Number of perturbations (optional)
    
    # Remove n and m as they were incorrectly used
    # Tally numbers list
    tally_numbers: List[int] = None

    # Change tally to be a dictionary
    tally: Dict[int, 'Tally'] = None

    def __post_init__(self):
        if self.tally_numbers is None:
            self.tally_numbers = []
        if self.tally is None:
            self.tally = {}  

@dataclass
class Tally:
    tally_id: int
    name: str = ""
    energies: List[float] = None
    results: List[float] = None
    errors: List[float] = None
    integral_result: Optional[float] = None
    integral_error: Optional[float] = None
    # TFC data (unperturbed)
    tfc_nps: List[int] = None
    tfc_results: List[float] = None
    tfc_errors: List[float] = None
    tfc_fom: List[float] = None
    # Perturbation data
    pert_data: Dict[int, dict] = None
    
    def __post_init__(self):
        if self.energies is None:
            self.energies = []
        if self.results is None:
            self.results = []
        if self.errors is None:
            self.errors = []
        if self.tfc_nps is None:
            self.tfc_nps = []
        if self.tfc_results is None:
            self.tfc_results = []
        if self.tfc_errors is None:
            self.tfc_errors = []
        if self.tfc_fom is None:
            self.tfc_fom = []
        if self.pert_data is None:
            self.pert_data = {}

    def plot_tfc_data(self, figsize=(15, 5)):
        """
        Creates a figure with 3 subplots showing TFC results, errors, and FOM.
        
        Args:
            figsize (tuple): Figure size in inches (width, height)
        """
        if not self.tfc_nps:
            raise ValueError("No TFC data available for plotting")

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
        
        # Plot results
        ax1.plot(self.tfc_nps, self.tfc_results, 'b.-')
        ax1.set_xlabel('NPS')
        ax1.set_ylabel('Results')
        ax1.set_title('Result vs NPS')
        ax1.grid(True)
        
        # Plot errors
        ax2.plot(self.tfc_nps, self.tfc_errors, 'r.-')
        ax2.set_xlabel('NPS')
        ax2.set_ylabel('Relative Error')
        ax2.set_title('Error vs NPS')
        ax2.grid(True)
        
        # Plot FOM
        ax3.plot(self.tfc_nps, self.tfc_fom, 'g.-')
        ax3.set_xlabel('NPS')
        ax3.set_ylabel('Figure of Merit')
        ax3.set_title('FOM vs NPS')
        ax3.grid(True)
        
        plt.tight_layout()
        return fig


@dataclass
class TallyPert(Tally):
    def __post_init__(self):
        # Call parent post init to initialize lists/dict as needed.
        super().__post_init__()
