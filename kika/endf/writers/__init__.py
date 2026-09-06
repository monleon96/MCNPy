"""
ENDF writers module for modifying and writing ENDF files.

This module provides utilities for:
- Modifying existing ENDF files (replace sections)
- Creating MF33 (cross-section covariance) and MF34 (angular distribution
  covariance) sections from scratch
- Writing covariance data to ENDF format
"""

from .endf_writer import ENDFWriter, replace_mf_section, replace_mt_section
from .mf34_writer import (
    create_mf34_from_covariance,
    write_mf34_to_file,
    remove_mf34_from_file,
    merge_mf34,
)
from .mf33_writer import (
    create_mf33_from_covariance,
    merge_mf33_covariance_into_host,
    write_mf33_to_file,
)
from .update_directory import update_mf1_directory
from .section_ops import remove_sections
from .redundant import (
    RedundantUpdate,
    recompute_redundant_mf3,
    resolve_sum_components,
)

__all__ = [
    # ENDF file modification
    'ENDFWriter',
    'replace_mf_section',
    'replace_mt_section',
    # MF1 directory maintenance
    'update_mf1_directory',
    # Section operations
    'remove_sections',
    # MF3 summation cross sections
    'RedundantUpdate',
    'recompute_redundant_mf3',
    'resolve_sum_components',
    # MF34 covariance creation and manipulation
    'create_mf34_from_covariance',
    'write_mf34_to_file',
    'remove_mf34_from_file',
    'merge_mf34',
    # MF33 covariance creation
    'create_mf33_from_covariance',
    'merge_mf33_covariance_into_host',
    'write_mf33_to_file',
]
