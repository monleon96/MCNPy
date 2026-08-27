"""
NJOY reprocessing module for generating ACE files from existing perturbed ENDF files.

This module provides functionality to process already-generated perturbed ENDF files
through NJOY to create ACE files at new temperatures without regenerating the ENDF files.
It maintains the same directory structure as the original perturbation workflow.

⚠ THIS IS THE MF4/MF34 ROUTE ONLY. It re-runs RECONR on the perturbed ENDF, so
it only carries a perturbation that lives *in the ENDF tape*. A set whose
cross sections were perturbed through the PENDF (``perturb_PENDF_files``, and
the MF33 half of ``perturb_combined_files``) keeps that perturbation in
``output_dir/pendf/``, and reconstructing from the ENDF instead would quietly
produce unperturbed cross sections. Those sets are reprocessed by re-running
stage B — ``pendf_perturbation._stage_b_run_njoy_for_pair`` — against the
stored PENDF. ``scripts/reprocess_ace.py`` picks the route from what is on
disk and drives either one.
"""
import os
import re
from typing import List, Union, Optional, Dict, Tuple, Any
from multiprocessing import Pool
from datetime import datetime
from pathlib import Path

from kika.endf.read_endf import read_endf
from kika.sampling.utils import (
    DualLogger,
    _get_logger,
    _set_logger
)

# Import the NJOY processing function from endf_perturbation
from kika.sampling.endf_perturbation import _process_njoy_for_sample

# Global logger instance
_logger = None


def _construct_endf_file_paths(
    output_dir: str,
    zaid: int,
    num_samples: int,
    base_filename: str,
) -> List[Tuple[str, int, int]]:
    """
    Construct file paths for perturbed ENDF files based on naming convention.
    
    Expected structure: output_dir/endf/{zaid}/{sample_num:04d}/{base_filename}_{sample_num:04d}.{ext}
    Examples:
        - 260560_0001.endf
        - jeff40_260560_0001.txt
    
    Parameters
    ----------
    output_dir : str
        Base output directory containing the endf/ subdirectory
    zaid : int
        ENDF ZAID of the isotope, Z*1000+A (26056 for Fe-56)
    num_samples : int
        Number of samples to process (generates paths for samples 1 to num_samples)
    base_filename : str
        Base filename without extension and without the _NNNN suffix
        Examples: "260560", "jeff40_260560"
    # File extensions are ignored; any extension is accepted for matching
        
    Returns
    -------
    List[Tuple[str, int, int]]
        List of (endf_file_path, zaid, sample_index) tuples
        
    Notes
    -----
    Sample indices are 0-based internally but file naming uses 1-based numbering (0001, 0002, etc.)
    """
    logger = _get_logger()
    endf_dir = os.path.join(output_dir, "endf")

    if not os.path.exists(endf_dir):
        if logger:
            logger.error(f"ENDF directory not found: {endf_dir}")
        return []

    constructed_files = []
    missing_files = []

    for sample_num in range(1, num_samples + 1):
        # 0-based index for internal use
        sample_index = sample_num - 1

        # 1-based, 4-digit zero-padded for file naming
        sample_str = f"{sample_num:04d}"

        # Directory containing the sample files
        sample_dir = os.path.join(endf_dir, str(zaid), sample_str)

        # Expected prefix for files in this directory (ignore extension)
        prefix = f"{base_filename}_{sample_str}"

        if not os.path.isdir(sample_dir):
            missing_files.append(os.path.join(sample_dir, f"{prefix}.*"))
            continue

        try:
            entries = os.listdir(sample_dir)
        except OSError:
            missing_files.append(os.path.join(sample_dir, f"{prefix}.*"))
            continue

        # Find any file starting with the prefix (ignores extension)
        candidates = [os.path.join(sample_dir, f) for f in entries if f.startswith(prefix)]
        if candidates:
            constructed_files.append((candidates[0], zaid, sample_index))
        else:
            missing_files.append(os.path.join(sample_dir, f"{prefix}.*"))

    # Report missing files
    if missing_files and logger:
        logger.warning(f"[REPROCESS] [DISCOVERY] {len(missing_files)} expected ENDF files not found")
        # Show first few missing files as examples
        for missing_file in missing_files[:5]:
            logger.warning(f"[REPROCESS] [DISCOVERY]   Missing: {missing_file}")
        if len(missing_files) > 5:
            logger.warning(f"[REPROCESS] [DISCOVERY]   ... and {len(missing_files) - 5} more")

    return constructed_files


#: Boltzmann constant in MeV/K — the units NJOY writes kT with on the ACE
#: header's first line.
_K_BOLTZMANN_MEV = 8.617333262e-11
#: Two ACE temperatures are the same one when they agree to this many kelvin.
_TEMP_TOL_K = 1.0
#: An ACE file is `<zaid>.<ext>` (current layout, e.g. `26056.06c`) or
#: `<zaid*10>_<lib>_<NNNN>.<ext>` (legacy, e.g. `260560_40_0001.06c`). Neither
#: ends in `.ace`, which is what the first version of this check looked for —
#: so it never matched anything and `skip_existing` silently never skipped.
_ACE_NAME_RE = re.compile(r"^\d+(?:_\d+_\d{4})?\.\d{2}[a-z]$")


def _ace_temperature_K(ace_path: str) -> Optional[float]:
    """Kelvin from an ACE file's header, or None when it cannot be read.

    ACE 1.0 carries `zaid.ext awr kT date` on line 1; ACE 2.0 moves that to
    line 2 behind a version token. Only those two lines are read — this runs
    once per sample per temperature and must not pay for the XSS array.
    """
    try:
        with open(ace_path, "r", errors="ignore") as fh:
            first = fh.readline()
            second = fh.readline()
    except OSError:
        return None

    parts = first.split()
    if parts and parts[0][:1].isdigit() and "." in parts[0] and len(parts) >= 3:
        token = parts[2]
    elif parts and parts[0].startswith("2."):
        second_parts = second.split()
        if len(second_parts) < 2:
            return None
        token = second_parts[1]
    else:
        return None

    try:
        return float(token) / _K_BOLTZMANN_MEV
    except ValueError:
        return None


def _sample_ace_files(output_dir: str, zaid: int, sample_index: int) -> List[str]:
    """Every ACE file belonging to one sample, across both on-disk layouts.

    Current layout keeps all temperatures of a sample together in
    ``ace/<NNNN>/``; the pre-2026-05 one split them as
    ``ace/<temp>/<zaid>/<NNNN>/``. Sets built by either are still around, so
    both are looked at.
    """
    sample_str = f"{sample_index+1:04d}"
    dirs = [os.path.join(output_dir, "ace", sample_str)]

    ace_root = os.path.join(output_dir, "ace")
    try:
        for temp_entry in os.listdir(ace_root):
            legacy = os.path.join(ace_root, temp_entry, str(zaid), sample_str)
            if os.path.isdir(legacy):
                dirs.append(legacy)
    except OSError:
        pass

    found = []
    for d in dirs:
        try:
            names = os.listdir(d)
        except OSError:
            continue
        for n in names:
            path = os.path.join(d, n)
            if (_ACE_NAME_RE.match(n) or n.endswith(".ace")) and os.path.isfile(path):
                found.append(path)
    return found


def _ace_file_exists(
    output_dir: str,
    zaid: int,
    sample_index: int,
    temperature: float,
    endf_file: str
) -> bool:
    """Does this sample already have an ACE **at this temperature**?

    The temperature is read out of the ACE header rather than inferred from
    the file name or the directory, because an extension is a label the writer
    chooses: a set processed at the wrong temperature and stamped with the
    right extension looks correct from a listing. Reading kT is what makes
    ``skip_existing`` mean "already done at this temperature" instead of
    "a file with roughly that name is there".

    Parameters
    ----------
    output_dir : str
        Base output directory
    zaid : int
        ENDF ZAID of the isotope (26056 for Fe-56 — Z*1000+A, not Z*10000+A*10)
    sample_index : int
        Sample index (0-based)
    temperature : float
        Temperature in Kelvin
    endf_file : str
        Unused; kept so the signature stays stable for existing callers.

    Returns
    -------
    bool
        True when an ACE at that temperature is already on disk.
    """
    for path in _sample_ace_files(output_dir, zaid, sample_index):
        temp_K = _ace_temperature_K(path)
        if temp_K is not None and abs(temp_K - float(temperature)) <= _TEMP_TOL_K:
            return True
    return False


def _process_sample_worker(args):
    """
    Worker function for parallel processing of a single ENDF file at all temperatures.
    
    Parameters
    ----------
    args : tuple
        Tuple of (endf_file, zaid, sample_index, temperatures, output_dir, njoy_exe,
                  library_name, njoy_version, xsdir_file, skip_existing, extensions)
        
    Returns
    -------
    Tuple[int, int, Dict[str, Any]]
        (zaid, sample_index, result_dict) where result_dict contains success status,
        processed temperatures, errors, and warnings
    """
    (endf_file, zaid, sample_index, temperatures, output_dir, njoy_exe,
     library_name, njoy_version, xsdir_file, skip_existing, extensions) = args
    
    logger = _get_logger()
    sample_str = f"{sample_index+1:04d}"
    
    # Filter temperatures if skip_existing is enabled
    temps_to_process = []
    skipped_temps = []
    
    for temp in temperatures:
        if skip_existing and _ace_file_exists(output_dir, zaid, sample_index, temp, endf_file):
            skipped_temps.append(temp)
            if logger:
                logger.debug(f"[NJOY] ZAID {zaid} Sample {sample_str}: Skipping existing ACE at {temp}K")
        else:
            temps_to_process.append(temp)
    
    # If all temperatures are skipped, return early
    if not temps_to_process:
        return (zaid, sample_index, {
            "success": True,
            "temperatures_processed": [],
            "temperatures_skipped": skipped_temps,
            "errors": [],
            "warnings": []
        })
    
    # Call the existing NJOY processing function
    try:
        result = _process_njoy_for_sample(
            out_endf=endf_file,
            sample_index=sample_index,
            njoy_exe=njoy_exe,
            temperatures=temps_to_process,
            library_name=library_name,
            njoy_version=njoy_version,
            output_dir=output_dir,
            xsdir_file=xsdir_file,
            extensions=extensions,
        )
        
        # Add skipped temperatures to result
        result["temperatures_skipped"] = skipped_temps
        
        return (zaid, sample_index, result)
        
    except Exception as e:
        if logger:
            logger.error(f"[NJOY] ZAID {zaid} Sample {sample_str}: Processing failed - {e}")
        
        return (zaid, sample_index, {
            "success": False,
            "temperatures_processed": [],
            "temperatures_skipped": skipped_temps,
            "errors": [f"Processing exception: {e}"],
            "warnings": []
        })


def reprocess_endf_to_ace(
    output_dir: str,
    zaid: int,
    num_samples: int,
    base_filename: str,
    temperatures: Union[float, List[float]],
    njoy_exe: str,
    library_name: str,
    njoy_version: str = "NJOY 2016.78",
    nprocs: int = 1,
    skip_existing: bool = False,
    verbose: bool = True,
    extensions: Optional[List[str]] = None,
):
    """
    Reprocess existing perturbed ENDF files to generate ACE files at new temperatures.
    
    This function constructs paths to perturbed ENDF files based on the standard naming
    convention and processes them through NJOY to create ACE files at the specified
    temperatures. It maintains the same directory structure as the original perturbation
    workflow.
    
    Parameters
    ----------
    output_dir : str
        Base output directory containing the endf/ subdirectory with perturbed ENDF files.
        Expected structure: output_dir/endf/{zaid}/{sample_num:04d}/{base_filename}_{sample_num:04d}.{ext}
    zaid : int
        ENDF ZAID of the isotope, Z*1000+A — 26056 for Fe-56. This is the key
        the endf/ and ace/ trees are built on; 260560 (Z*10000+A*10) is the
        ACE *file name* stem and will not find anything here.
    num_samples : int
        Number of samples to process (processes samples 1 to num_samples)
    base_filename : str
        Base filename without extension and without the _NNNN suffix.
        Examples: "260560", "jeff40_260560"
    temperatures : Union[float, List[float]]
        Temperature(s) in Kelvin for ACE generation. Can be a single float or list of floats.
    njoy_exe : str
        Path to NJOY executable
    library_name : str
        Nuclear data library name (e.g., 'endfb81', 'jeff40')
    njoy_version : str, default "NJOY 2016.78"
        NJOY version string for metadata and titles
    nprocs : int, default 1
        Number of parallel processes for sample processing
    skip_existing : bool, default False
        If True, skip samples whose ACE at the requested temperature is already
        on disk — judged by the kT in the ACE header, not by the file name.
        If False, always regenerate ACE files.
    verbose : bool, default True
        Enable verbose logging output
    extensions : list of str, optional
        ACE extension per temperature (e.g. ``['06c']``). When given, the ACE
        is written as ``<ZAID>.<ext>`` in the per-sample directory, which is
        the naming a set built with ``extensions=`` already uses — so a
        reprocessing run overwrites in place and every xsdir line and MCNP
        input stays valid. When None, the legacy
        ``<ZAID*10>_<lib>_<NNNN>.<ext>`` naming is produced instead. Pass
        whatever the set already carries.
        
    Notes
    -----
    File naming convention:
    - ENDF files: {base_filename}_{sample_num:04d}.* (any extension)
    - Examples: "260560_0001.endf", "jeff40_260560_0042.txt"
    
    XSDIR file handling:
    The function automatically detects and updates existing xsdir files:
    1. Checks for existing xsdir files in output_dir/xsdir/
       - If found: Updates them with new temperature entries for this isotope
       - If not found: Creates only per-sample xsdir files (no master files)
    2. Always creates per-sample xsdir files: output_dir/ace/temp/zaid/sample_num/{base}_{sample_num}.xsdir
       - One file per ACE file, containing only that specific isotope/temperature
       - Uses relative paths for portability
    
    This allows you to reprocess at new temperatures and automatically update
    the xsdir files from previous perturbation runs.
    
    Output directory structure will include:
    - output_dir/endf/zaid/sample_num/ : existing perturbed ENDF files (not modified)
    - output_dir/ace/temp/zaid/sample_num/ : newly generated ACE files
    - output_dir/njoy_files/temp/zaid/sample_num/ : NJOY auxiliary files
    - output_dir/xsdir/ : xsdir files (updated if they exist from previous runs)
    - output_dir/*.log : log files
    
    Examples
    --------
    # Process Fe-56 samples at a single temperature
    reprocess_endf_to_ace(
        output_dir="/path/to/output",
        zaid=260560,
        num_samples=100,
        base_filename="260560",
        temperatures=600.0,
        njoy_exe="/usr/local/bin/njoy",
        library_name="endfb81"
    )
    
    # Process JEFF-4.0 library files (extensions are ignored)
    reprocess_endf_to_ace(
        output_dir="/path/to/output",
        zaid=260560,
        num_samples=50,
        base_filename="jeff40_260560",
        temperatures=[300.0, 600.0, 900.0],
        njoy_exe="/usr/local/bin/njoy",
        library_name="jeff40",
        nprocs=4,
        skip_existing=True
    )
    """
    global _logger
    
    # Validate output directory
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"Output directory not found: {output_dir}")
    
    endf_dir = os.path.join(output_dir, "endf")
    if not os.path.exists(endf_dir):
        raise FileNotFoundError(f"ENDF directory not found: {endf_dir}")
    
    # Validate NJOY executable
    if not os.path.exists(njoy_exe):
        raise FileNotFoundError(f"NJOY executable not found: {njoy_exe}")
    
    # Detect existing xsdir files in output_dir/xsdir/
    # These files from the original perturbation have names like: xsdir40-irdff2_0001
    # We'll update them in-place by stripping the suffix and letting create_xsdir_files_for_ace add it back
    xsdir_dir = os.path.join(output_dir, "xsdir")
    detected_xsdir_files = []
    if os.path.exists(xsdir_dir):
        try:
            entries = os.listdir(xsdir_dir)
            # Look for files matching pattern: *_NNNN (e.g., xsdir40-irdff2_0001)
            # where NNNN is a 4-digit sample number
            for entry in entries:
                if os.path.isfile(os.path.join(xsdir_dir, entry)):
                    # Check if filename ends with _NNNN pattern
                    parts = entry.split('_')
                    if len(parts) >= 2 and parts[-1].isdigit() and len(parts[-1]) == 4:
                        detected_xsdir_files.append(entry)
        except OSError:
            pass
    
    # Convert temperature to list if it's a single float
    if isinstance(temperatures, (int, float)):
        temperatures = [float(temperatures)]
    elif isinstance(temperatures, list):
        if len(temperatures) == 0:
            raise ValueError("temperatures list cannot be empty")
        temperatures = [float(t) for t in temperatures]
    else:
        raise ValueError("temperatures must be a float or list of floats")
    
    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(output_dir, f'reprocess_endf_to_ace_{timestamp}.log')
    _logger = DualLogger(log_file)
    _set_logger(_logger)
    
    # Console: Basic start message
    print(f"[INFO] Starting ENDF to ACE reprocessing job")
    print(f"[INFO] Log file: {log_file}")
    print(f"[INFO] Output directory: {os.path.abspath(output_dir)}")
    
    # Print run parameters to log file
    separator = "=" * 80
    _logger.info(f"\n{separator}")
    _logger.info(f"[REPROCESS] [PARAMETERS] Run Configuration")
    _logger.info(f"{separator}")
    _logger.info(f"Output directory: {os.path.abspath(output_dir)}")
    _logger.info(f"ZAID: {zaid}")
    _logger.info(f"Number of samples: {num_samples}")
    _logger.info(f"Base filename: {base_filename}")
    _logger.info(f"Temperatures: {temperatures}")
    _logger.info(f"NJOY executable: {njoy_exe}")
    _logger.info(f"Library name: {library_name}")
    _logger.info(f"NJOY version: {njoy_version}")
    if detected_xsdir_files:
        _logger.info(f"XSDIR files detected: {len(detected_xsdir_files)} file(s) in {xsdir_dir}")
        _logger.info(f"  → Existing xsdir files will be updated in-place")
    else:
        _logger.info(f"XSDIR files detected: None (will create only per-sample xsdir files)")
    _logger.info(f"Parallel processes: {nprocs}")
    _logger.info(f"Skip existing: {skip_existing}")
    _logger.info(f"{separator}")
    
    # Construct ENDF file paths
    _logger.info(f"\n[REPROCESS] [DISCOVERY] Constructing ENDF file paths...")
    discovered_files = _construct_endf_file_paths(
        output_dir, zaid, num_samples, base_filename
    )
    
    if not discovered_files:
        _logger.error(f"[REPROCESS] [DISCOVERY] No ENDF files found for ZAID {zaid}")
        _logger.error(f"[REPROCESS] [DISCOVERY] Expected pattern: {output_dir}/endf/{zaid}/NNNN/{base_filename}_NNNN.*")
        print(f"[ERROR] No ENDF files found for ZAID {zaid}")
        return
    
    _logger.info(f"[REPROCESS] [DISCOVERY] Found {len(discovered_files)} ENDF files for ZAID {zaid}")
    
    print(f"[INFO] Found {len(discovered_files)} ENDF files to process")
    print(f"[INFO] ZAID: {zaid}, Samples: {len(discovered_files)}/{num_samples}, Temperatures: {len(temperatures)}")
    
    # Validate a sample of ENDF files
    _logger.info(f"\n[REPROCESS] [VALIDATION] Validating ENDF files...")
    validation_errors = []
    
    for endf_file, file_zaid, sample_index in discovered_files[:min(5, len(discovered_files))]:
        try:
            endf_data = read_endf(endf_file)
            if endf_data.zaid != zaid:
                validation_errors.append(
                    f"ZAID mismatch: expected={zaid}, file={endf_data.zaid} in {endf_file}"
                )
        except Exception as e:
            validation_errors.append(f"Failed to parse {endf_file}: {e}")
    
    if validation_errors:
        _logger.warning(f"[REPROCESS] [VALIDATION] Found {len(validation_errors)} validation errors:")
        for error in validation_errors:
            _logger.warning(f"[REPROCESS] [VALIDATION]   {error}")
    else:
        _logger.info(f"[REPROCESS] [VALIDATION] Validation passed")
    
    # Process all ENDF files
    _logger.info(f"\n[REPROCESS] [PROCESSING] Starting NJOY processing...")
    
    # Prepare arguments for workers
    # For xsdir files: strip the _XXXX suffix before passing to create_xsdir_files_for_ace
    # so when it adds the suffix back, we get the same filename (updating in-place)
    worker_args = []
    for endf_file, file_zaid, sample_index in discovered_files:
        # Look for xsdir file matching this sample
        sample_str = f"{sample_index+1:04d}"
        sample_xsdir = None
        if detected_xsdir_files:
            # Try to find a matching xsdir file for this sample
            # Pattern: any file ending with _NNNN where NNNN matches sample
            for xsdir_name in detected_xsdir_files:
                if xsdir_name.endswith(f"_{sample_str}"):
                    # Strip the _XXXX suffix to get the base name
                    # create_xsdir_files_for_ace will add it back, resulting in the same filename
                    base_xsdir_name = xsdir_name.rsplit('_', 1)[0]
                    sample_xsdir = os.path.join(xsdir_dir, base_xsdir_name)
                    break
        
        worker_args.append((
            endf_file, file_zaid, sample_index, temperatures, output_dir, njoy_exe,
            library_name, njoy_version, sample_xsdir, skip_existing, extensions
        ))
    
    
    # Process samples with optional parallelization
    results = []
    
    if nprocs > 1 and len(discovered_files) > 1:
        _logger.info(f"[REPROCESS] [PROCESSING] Using {nprocs} processes for parallel processing")
        print(f"[INFO] Processing with {nprocs} parallel processes...")
        
        try:
            with Pool(processes=nprocs) as pool:
                futures = []
                for args in worker_args:
                    future = pool.apply_async(_process_sample_worker, args=(args,))
                    futures.append(future)
                
                # Wait for all processes to complete
                pool.close()
                pool.join()
                
                # Collect results
                for future in futures:
                    try:
                        result = future.get()
                        results.append(result)
                    except Exception as e:
                        _logger.error(f"[REPROCESS] [PROCESSING] Worker failed: {e}")
                        
        except Exception as e:
            _logger.error(f"[REPROCESS] [PROCESSING] Parallel processing failed: {e}")
            _logger.info(f"[REPROCESS] [PROCESSING] Falling back to serial processing...")
            
            # Fall back to serial processing
            results = []
            for args in worker_args:
                try:
                    result = _process_sample_worker(args)
                    results.append(result)
                except Exception as e:
                    _logger.error(f"[REPROCESS] [PROCESSING] Sample processing failed: {e}")
    else:
        # Serial processing
        _logger.info(f"[REPROCESS] [PROCESSING] Using serial processing")
        print(f"[INFO] Processing samples sequentially...")
        
        for i, args in enumerate(worker_args):
            try:
                result = _process_sample_worker(args)
                results.append(result)
                
                # Periodic progress update
                if (i + 1) % 10 == 0 or (i + 1) == len(worker_args):
                    print(f"[INFO] Progress: {i+1}/{len(worker_args)} samples processed")
                    
            except Exception as e:
                _logger.error(f"[REPROCESS] [PROCESSING] Sample processing failed: {e}")
    
    # Generate summary report
    _logger.info(f"\n{separator}")
    _logger.info(f"[REPROCESS] [SUMMARY] Processing Results")
    _logger.info(f"{separator}")
    
    # Calculate summary statistics
    summary = {
        'total_samples': 0,
        'successful_samples': 0,
        'failed_samples': 0,
        'temps_processed': {temp: 0 for temp in temperatures},
        'temps_skipped': {temp: 0 for temp in temperatures},
        'xsdir_files_updated': 0 if detected_xsdir_files else None,
        'per_sample_xsdir_created': 0,
        'errors': [],
        'warnings': []
    }
    
    for file_zaid, sample_index, result in results:
        summary['total_samples'] += 1
        
        if result.get('success', False):
            summary['successful_samples'] += 1
        else:
            summary['failed_samples'] += 1
        
        # Count processed temperatures
        for temp in result.get('temperatures_processed', []):
            summary['temps_processed'][temp] += 1
        
        # Count skipped temperatures
        for temp in result.get('temperatures_skipped', []):
            summary['temps_skipped'][temp] += 1
        
        # Count xsdir file operations (one per processed temperature)
        num_temps_processed = len(result.get('temperatures_processed', []))
        if num_temps_processed > 0:
            summary['per_sample_xsdir_created'] += num_temps_processed
            if summary['xsdir_files_updated'] is not None:
                summary['xsdir_files_updated'] += num_temps_processed
        
        # Accumulate errors and warnings
        summary['errors'].extend(result.get('errors', []))
        summary['warnings'].extend(result.get('warnings', []))
    
    # Print summary
    _logger.info(f"\n  ZAID {zaid}:")
    _logger.info(f"  {'-' * 60}")
    _logger.info(f"  Total samples: {summary['total_samples']}")
    _logger.info(f"  Successful: {summary['successful_samples']}")
    _logger.info(f"  Failed: {summary['failed_samples']}")
    
    # Temperature-specific results
    for temp in temperatures:
        processed = summary['temps_processed'][temp]
        skipped = summary['temps_skipped'][temp]
        temp_str = str(temp).rstrip('0').rstrip('.') if '.' in str(temp) else str(temp)
        
        _logger.info(f"  Temperature {temp_str}K: {processed} processed, {skipped} skipped")
    
    # XSDIR file results
    _logger.info(f"  Per-sample xsdir files created: {summary['per_sample_xsdir_created']}")
    if summary['xsdir_files_updated'] is not None:
        _logger.info(f"  Existing xsdir files updated: {summary['xsdir_files_updated']} entries")
        if os.path.exists(xsdir_dir):
            _logger.info(f"  XSDIR directory: {xsdir_dir}")
    
    # Errors
    if summary['errors']:
        _logger.info(f"  Errors ({len(summary['errors'])}):")
        # Show unique errors
        unique_errors = list(set(summary['errors']))
        for error in unique_errors[:10]:  # Limit to first 10 unique errors
            _logger.info(f"    • {error}")
        if len(unique_errors) > 10:
            _logger.info(f"    ... and {len(unique_errors) - 10} more")
    
    # Warnings
    if summary['warnings']:
        _logger.info(f"  Warnings ({len(summary['warnings'])}):")
        # Show unique warnings
        unique_warnings = list(set(summary['warnings']))
        for warning in unique_warnings[:10]:  # Limit to first 10 unique warnings
            _logger.info(f"    • {warning}")
        if len(unique_warnings) > 10:
            _logger.info(f"    ... and {len(unique_warnings) - 10} more")
    
    _logger.info(f"\n{separator}")
    
    # Console: Final summary
    total_samples = summary['total_samples']
    successful_samples = summary['successful_samples']
    failed_samples = summary['failed_samples']
    
    print(f"\n[INFO] ENDF to ACE reprocessing completed!")
    print(f"[INFO] Total samples: {total_samples}")
    print(f"[INFO] Successful: {successful_samples}")
    if failed_samples > 0:
        print(f"[WARNING] Failed: {failed_samples}")
    print(f"[INFO] Detailed log saved to: {log_file}")
    
    if summary.get('xsdir_files_updated') is not None and summary['xsdir_files_updated'] > 0:
        print(f"[INFO] Existing xsdir files updated: {summary['xsdir_files_updated']} entries")
    if summary.get('per_sample_xsdir_created', 0) > 0:
        print(f"[INFO] Per-sample xsdir files created: {summary['per_sample_xsdir_created']}")
