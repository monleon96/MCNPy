"""
ENDF file writer and section replacement utilities.

This module provides functionality to modify specific sections (MF or MT) 
in ENDF files while preserving the rest of the file content.
"""
import os
from typing import Dict, List, Optional, Union, Tuple
from ..classes.mt import MT
from ..classes.mf1.mf1mt import MT451
from ..classes.mf import MF
from ..classes.mf4.base import MF4MT
from ..utils import parse_endf_id
from ...utils import get_endf_logger
from .update_directory import update_mf1_directory

# Initialize logger for this module
logger = get_endf_logger(__name__)


class ENDFWriter:
    """
    Class for writing modified ENDF files.
    
    This class provides methods to replace specific MF or MT sections
    in ENDF files while preserving the rest of the content.
    """
    
    def __init__(self, original_filepath: str):
        """
        Initialize the ENDF writer with an original file.
        
        Args:
            original_filepath: Path to the original ENDF file
        """
        self.original_filepath = original_filepath
        self.original_lines = None
        self._load_original_file()
    
    def _load_original_file(self):
        """Load the original ENDF file into memory."""
        if not os.path.exists(self.original_filepath):
            raise FileNotFoundError(f"Original ENDF file not found: {self.original_filepath}")
        
        with open(self.original_filepath, 'r') as f:
            self.original_lines = f.readlines()
        
        logger.debug(f"Loaded {len(self.original_lines)} lines from {self.original_filepath}")
    
    def find_mf_boundaries(self, mf_number: int) -> List[Tuple[int, int]]:
        """
        Find the line boundaries for all occurrences of a specific MF section.
        
        Args:
            mf_number: The MF number to find
            
        Returns:
            List of (start_line, end_line) tuples for each MF section
        """
        boundaries = []
        current_start = None
        
        for i, line in enumerate(self.original_lines):
            mat, mf, mt = parse_endf_id(line)
            
            if mf == mf_number and current_start is not None:
                # Continue within MF section
                pass
            elif mf == mf_number and current_start is None:
                # Start of MF section
                current_start = i
            elif current_start is not None:
                # End of MF section
                if mf == 0 and mt == 0:
                    # FEND line — include it in the boundary
                    boundaries.append((current_start, i))
                else:
                    boundaries.append((current_start, i - 1))
                current_start = None
        
        # Handle case where MF section goes to end of file
        if current_start is not None:
            boundaries.append((current_start, len(self.original_lines) - 1))
        
        logger.debug(f"Found {len(boundaries)} MF{mf_number} sections at lines: {boundaries}")
        return boundaries
    
    def find_mt_boundaries_in_mf(self, mf_number: int, mt_number: int) -> List[Tuple[int, int]]:
        """
        Find the line boundaries for a specific MT section within an MF section.
        
        Args:
            mf_number: The MF number containing the MT section
            mt_number: The MT number to find
            
        Returns:
            List of (start_line, end_line) tuples for each matching MT section
        """
        boundaries = []
        current_start = None
        
        for i, line in enumerate(self.original_lines):
            mat, mf, mt = parse_endf_id(line)
            
            if mf == mf_number and mt == mt_number:
                if current_start is None:
                    # Start of target MT section
                    current_start = i
                # else: continue within MT section
            elif current_start is not None:
                if mf == mf_number and mt == 0:
                    # SEND line — include it in the boundary
                    boundaries.append((current_start, i))
                elif mf == 0 and mt == 0:
                    # FEND line — include it in the boundary
                    boundaries.append((current_start, i))
                else:
                    # Different MT in same MF (no SEND between them)
                    boundaries.append((current_start, i - 1))
                current_start = None
        
        # Handle case where MT section goes to end of file
        if current_start is not None:
            boundaries.append((current_start, len(self.original_lines) - 1))
        
        logger.debug(f"Found {len(boundaries)} MF{mf_number}/MT{mt_number} sections at lines: {boundaries}")
        return boundaries
    
    def replace_mf_section(self, modified_mf: MF, output_filepath: Optional[str] = None,
                           update_directory: bool = True) -> bool:
        """
        Replace an entire MF section with a modified version.

        Args:
            modified_mf: The modified MF object with new content
            output_filepath: Output file path (if None, overwrites original)
            update_directory: If True, update MF1/MT451 directory after writing

        Returns:
            True if replacement succeeded, False otherwise
        """
        try:
            # Find boundaries of the target MF section
            boundaries = self.find_mf_boundaries(modified_mf.number)
            
            if not boundaries:
                logger.error(f"MF{modified_mf.number} section not found in original file")
                return False
            
            if len(boundaries) > 1:
                logger.warning(f"Found {len(boundaries)} MF{modified_mf.number} sections, replacing the first one")
            
            start_line, end_line = boundaries[0]
            
            # Get the modified content as lines
            modified_content = str(modified_mf)
            if not modified_content.endswith('\n'):
                modified_content += '\n'
            modified_lines = modified_content.split('\n')[:-1]  # Remove empty last element

            # Create new file content
            new_lines = (
                self.original_lines[:start_line] +
                [line + '\n' for line in modified_lines] +
                self.original_lines[end_line + 1:]
            )
            
            # Write the result
            output_path = output_filepath if output_filepath else self.original_filepath
            with open(output_path, 'w') as f:
                f.writelines(new_lines)
            
            logger.debug(f"Successfully replaced MF{modified_mf.number} section in {output_path}")

            if update_directory:
                update_mf1_directory(output_path)

            return True

        except Exception as e:
            logger.error(f"Error replacing MF{modified_mf.number} section: {e}")
            return False

    def replace_mt_section(self, modified_mt: Union[MT, MT451, MF4MT], mf_number: int,
                          output_filepath: Optional[str] = None,
                          update_directory: bool = True) -> bool:
        """
        Replace a specific MT section within an MF section.

        Args:
            modified_mt: The modified MT object with new content
            mf_number: The MF number containing this MT section
            output_filepath: Output file path (if None, overwrites original)
            update_directory: If True, update MF1/MT451 directory after writing

        Returns:
            True if replacement succeeded, False otherwise
        """
        try:
            # Find boundaries of the target MT section
            boundaries = self.find_mt_boundaries_in_mf(mf_number, modified_mt.number)
            
            if not boundaries:
                logger.error(f"MF{mf_number}/MT{modified_mt.number} section not found in original file")
                return False
            
            if len(boundaries) > 1:
                logger.warning(f"Found {len(boundaries)} MF{mf_number}/MT{modified_mt.number} sections, replacing the first one")
            
            start_line, end_line = boundaries[0]
            
            # Get the modified content as lines
            modified_content = str(modified_mt)
            if not modified_content.endswith('\n'):
                modified_content += '\n'
            modified_lines = modified_content.split('\n')[:-1]  # Remove empty last element

            # Create new file content
            new_lines = (
                self.original_lines[:start_line] +
                [line + '\n' for line in modified_lines] +
                self.original_lines[end_line + 1:]
            )
            
            # Write the result
            output_path = output_filepath if output_filepath else self.original_filepath
            with open(output_path, 'w') as f:
                f.writelines(new_lines)
            
            logger.debug(f"Successfully replaced MF{mf_number}/MT{modified_mt.number} section in {output_path}")

            if update_directory:
                update_mf1_directory(output_path)

            return True

        except Exception as e:
            logger.error(f"Error replacing MF{mf_number}/MT{modified_mt.number} section: {e}")
            return False


# Convenience functions for direct use without instantiating the class
def replace_mf_section(original_filepath: str, modified_mf: MF,
                      output_filepath: Optional[str] = None,
                      update_directory: bool = True) -> bool:
    """
    Replace an MF section in an ENDF file.

    Args:
        original_filepath: Path to the original ENDF file
        modified_mf: The modified MF object
        output_filepath: Output file path (if None, overwrites original)
        update_directory: If True, update MF1/MT451 directory after writing

    Returns:
        True if replacement succeeded, False otherwise
    """
    writer = ENDFWriter(original_filepath)
    return writer.replace_mf_section(modified_mf, output_filepath, update_directory)


def replace_mt_section(original_filepath: str, modified_mt: Union[MT, MT451, MF4MT],
                      mf_number: int, output_filepath: Optional[str] = None,
                      update_directory: bool = True) -> bool:
    """
    Replace an MT section in an ENDF file.

    Args:
        original_filepath: Path to the original ENDF file
        modified_mt: The modified MT object
        mf_number: The MF number containing this MT section
        output_filepath: Output file path (if None, overwrites original)
        update_directory: If True, update MF1/MT451 directory after writing

    Returns:
        True if replacement succeeded, False otherwise
    """
    writer = ENDFWriter(original_filepath)
    return writer.replace_mt_section(modified_mt, mf_number, output_filepath, update_directory)
