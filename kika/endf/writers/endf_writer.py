"""
ENDF file writer and section replacement utilities.

This module provides functionality to modify specific sections (MF or MT) 
in ENDF files while preserving the rest of the file content.
"""
import os
from typing import Dict, List, Optional, Union, Tuple
from ..classes.mt import MT
from ..classes.mf1.mf1mt451 import MF1MT451
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
        #: What the last ``resum_redundant=True`` replacement did to the
        #: summation MTs, one
        #: :class:`~kika.endf.writers.redundant.RedundantUpdate` each. Empty
        #: when the option was not used. It lives here rather than in the
        #: return value because these methods answer ``bool``, and changing
        #: that would break every caller for the sake of a diagnostic.
        self.redundant_updates = []
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

    def replace_mt_section(self, modified_mt: Union[MT, MF1MT451, MF4MT], mf_number: int,
                          output_filepath: Optional[str] = None,
                          update_directory: bool = True,
                          resum_redundant: bool = False) -> bool:
        """
        Replace a specific MT section within an MF section.

        Args:
            modified_mt: The modified MT object with new content
            mf_number: The MF number containing this MT section
            output_filepath: Output file path (if None, overwrites original)
            update_directory: If True, update MF1/MT451 directory after writing
            resum_redundant: MF3 only. Rebuild the summation cross sections
                this replacement invalidated -- transfer MT52 in and MT4
                changes, then MT3 and MT1 because MT4 did. What it did lands in
                ``self.redundant_updates``; the rules are in
                :func:`~kika.endf.writers.redundant.recompute_redundant_mf3`.

        Returns:
            True if replacement succeeded, False otherwise

        ``resum_redundant`` is **off by default and has to stay off.** A
        replacement is a byte operation on one section; a resummation restates
        values the caller never named, and doing it silently would mean someone
        who moved MT52 got MT1 moved too without being told.

        When it is on, two guards come with it and neither is optional. The
        replaced MT is **protected** from the rebuild -- transferring a total
        explicitly and then overwriting it with the local sum would discard the
        very section that was moved. And the file as it was **before** this
        replacement is passed as the baseline, so a redundant MT is only rebuilt
        where the invariant held beforehand: a tape cut down to a few sections
        (``micro_fe56_structural.endf`` keeps MT1, MT2 and MT102 out of a full
        Fe-56, and its MT1 sits 63% above MT2+MT102) is reported and left alone
        instead of having its total replaced by a sum over the survivors.

        **A transfer of several sections is safe one call at a time**, and it
        is the baseline that makes it so rather than the protected set. Since
        the protected set only ever holds this call's MT, moving MT4 in and then
        editing MT52 with the option on looks like it should rebuild MT4 from
        the partials and discard the MT4 just placed. It does not: the second
        call's baseline is the file as that call found it, which already carries
        the transferred MT4 stating a value its partials do not make, so the
        "restore the invariant only where it held" rule declines to touch it. The two guards
        look interchangeable and are not -- one is about what the caller named,
        the other about what the file already claimed -- and only the second
        sees across calls.

        Not offered on :meth:`replace_mf_section`: replacing a whole MF3 makes
        every MT dirty at once, so "which redundants did this invalidate" has no
        answer narrower than "all of them". A caller who wants that can say it
        outright, with ``changed_mts=None``.
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

            # Before the directory, not after: a resummation changes line
            # counts, and the directory has to describe the file that is
            # finally on disk.
            self.redundant_updates = []
            if resum_redundant and mf_number == 3:
                self._resum_redundant(output_path, int(modified_mt.number))

            if update_directory:
                update_mf1_directory(output_path)

            return True

        except Exception as e:
            logger.error(f"Error replacing MF{mf_number}/MT{modified_mt.number} section: {e}")
            return False


    def _resum_redundant(self, output_path: str, replaced_mt: int) -> None:
        """Rebuild the summation MTs that replacing *replaced_mt* invalidated."""
        from .redundant import recompute_redundant_mf3

        with open(output_path, "r") as fh:
            edited = fh.read()

        rewritten, updates = recompute_redundant_mf3(
            edited,
            changed_mts=[replaced_mt],
            protected_mts=[replaced_mt],
            baseline_content="".join(self.original_lines),
        )
        self.redundant_updates = updates
        for update in updates:
            logger.info(update.describe())
        if rewritten != edited:
            with open(output_path, "w") as fh:
                fh.write(rewritten)

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


def replace_mt_section(original_filepath: str, modified_mt: Union[MT, MF1MT451, MF4MT],
                      mf_number: int, output_filepath: Optional[str] = None,
                      update_directory: bool = True,
                      resum_redundant: bool = False) -> bool:
    """
    Replace an MT section in an ENDF file.

    Args:
        original_filepath: Path to the original ENDF file
        modified_mt: The modified MT object
        mf_number: The MF number containing this MT section
        output_filepath: Output file path (if None, overwrites original)
        update_directory: If True, update MF1/MT451 directory after writing
        resum_redundant: MF3 only. Rebuild the summation cross sections this
            replacement invalidated. :meth:`ENDFWriter.replace_mt_section`
            explains why it is off by default; instantiate the class when you
            want to read back what it did.

    Returns:
        True if replacement succeeded, False otherwise
    """
    writer = ENDFWriter(original_filepath)
    return writer.replace_mt_section(modified_mt, mf_number, output_filepath,
                                     update_directory, resum_redundant)
