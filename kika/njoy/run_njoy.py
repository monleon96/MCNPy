import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from textwrap import dedent
from .launcher import build_njoy_command
from .templates import NJOY_INPUT_TEMPLATE, NJOY_INPUT_TEMPLATE_WITH_PENDF
from kika.endf.read_endf import read_endf
from kika._constants import K_TO_SUFFIX, NDLIBRARY_TO_SUFFIX
from kika._utils import MeV_to_kelvin

def _resolve_nd_suffix(library_name: str) -> str:
    """Return the ACE suffix for a library name, falling back to '00' if unknown."""
    import warnings
    library_key = library_name.lower().replace('-', '').replace('/', '').replace('.', '')
    if library_key in NDLIBRARY_TO_SUFFIX:
        return NDLIBRARY_TO_SUFFIX[library_key]
    warnings.warn(
        f"library_name={library_name!r} not in NDLIBRARY_TO_SUFFIX "
        f"({sorted(NDLIBRARY_TO_SUFFIX.keys())}); using fallback suffix '00'.",
        stacklevel=2,
    )
    return "00"


def _get_temperature_info(temperature: float) -> tuple[float, str]:
    """
    Helper function to get temperature in Kelvin and corresponding suffix.

    Parameters
    ----------
    temperature : float
        Temperature in Kelvin or MeV (if < 1.0, assumed to be MeV)

    Returns
    -------
    tuple[float, str]
        (temperature_in_kelvin, temperature_suffix)
    """
    # Convert temperature to Kelvin if needed
    if temperature < 1.0:
        temp_K = MeV_to_kelvin(temperature)
    else:
        temp_K = temperature

    # Find the corresponding temperature suffix
    if K_TO_SUFFIX:
        closest_temp = min(K_TO_SUFFIX.keys(), key=lambda k: abs(k - temp_K))
        if abs(closest_temp - temp_K) <= 1.0:
            temp_suffix = K_TO_SUFFIX[closest_temp]
        else:
            temp_suffix = ".00"
    else:
        temp_suffix = ".00"

    return temp_K, temp_suffix


def _suff_from_extension(extension: str) -> str:
    """Convert a user-supplied ACE extension (e.g. '06c') to the NJOY ACER
    `{suff}` slot value (e.g. '.06'). Trailing 'c' is stripped; anything
    else is taken verbatim after the leading dot."""
    ext = str(extension).strip()
    if ext.endswith("c"):
        return "." + ext[:-1]
    return "." + ext

def _generate_unique_filename(base_filename: str, ace_dir: Path, njoy_files_dir: Path, 
                             temp_suffix: str, additional_suffix: str = None) -> tuple[str, str]:
    """
    Generate a unique filename by checking for existing files and adding a suffix if needed.
    
    Parameters
    ----------
    base_filename : str
        Base filename without extension (e.g., "260560_81.02")
    ace_dir : Path
        Directory where ACE files are stored
    njoy_files_dir : Path
        Directory where NJOY files are stored
    temp_suffix : str
        Temperature suffix (e.g., ".02")
    additional_suffix : str | None
        Optional additional suffix provided by user
        
    Returns
    -------
    tuple[str, str]
        (final_base_filename, final_ace_filename)
    """
    # Start with the original base filename
    zaid_lib_part = base_filename.replace(temp_suffix, "")  # e.g., "260560_81"
    
    if additional_suffix:
        # User provided a specific suffix
        final_base = f"{zaid_lib_part}_{additional_suffix}{temp_suffix}"
        final_ace = f"{final_base}c"
    else:
        # Check if files already exist and generate unique suffix
        final_base = base_filename
        final_ace = f"{final_base}c"
        
        # Check if files exist
        ace_file_path = ace_dir / final_ace
        input_file_path = njoy_files_dir / f"{final_base}.input"
        
        if ace_file_path.exists() or input_file_path.exists():
            # Find the next available suffix
            counter = 1
            while True:
                auto_suffix = f"v{counter:02d}"
                final_base = f"{zaid_lib_part}_{auto_suffix}{temp_suffix}"
                final_ace = f"{final_base}c"
                
                ace_file_path = ace_dir / final_ace
                input_file_path = njoy_files_dir / f"{final_base}.input"
                
                if not ace_file_path.exists() and not input_file_path.exists():
                    break
                counter += 1
    
    return final_base, final_ace

def _render_title(isotope_label: str, T: float, library_name: str, njoy_version: str = "NJOY 2016.78") -> str:
    # Example: "FE56 - 293.6 K - JEFF-40 (NJOY 2016.78)"
    return f"{isotope_label} - {T:.1f} K - {library_name.upper()} ({njoy_version})"

def _render_njoy_input(mat: int, T: float, title: str, suff: str = None,
                       template: str = NJOY_INPUT_TEMPLATE) -> str:
    if suff is None:
        # Use helper function to get temperature suffix (without 'c')
        _, temp_suffix = _get_temperature_info(T)
        suff = temp_suffix
    # Fill the template with the run-specific parameters
    return dedent(template).format(mat=mat, T=T, title=title, suff=suff)

# ---- 2) Runner ----
def run_njoy(
    njoy_exe: str,
    endf_path: str | Path,
    temperature: float,
    library_name: str,
    output_dir: str | Path,
    *,
    suff: str = None,
    njoy_version: str = "NJOY 2016.78",
    additional_suffix: str = None,
    extension: str | None = None,
    ace_dir: str | Path | None = None,
    xsdir_dir: str | Path | None = None,
    njoy_files_dir: str | Path | None = None,
) -> dict:
    """
    Run NJOY once using the provided ENDF file and parameters.

    Parameters
    ----------
    njoy_exe : str
        Path to the NJOY executable.
    endf_path : str | Path
        Path to the ENDF file (this will be copied to tape20).
    temperature : float
        Temperature in Kelvin used in broadr/thermr/acer blocks.
    library_name : str
        For the title line, e.g., 'jeff40' or 'endfb81'.
    suff : str | None
        Suffix for the ACE file, e.g., '.01c' or '.02c'
    output_dir : str | Path
        Base output directory where ace/ and njoy_files/ will be created.
    njoy_version : str
        Version string shown in the title.
    additional_suffix : str | None
        Optional additional suffix to add to filenames to prevent overwrites.
        If None and files exist, a unique suffix will be generated automatically.

    Returns
    -------
    dict with paths to outputs and the NJOY return code.
    """

    endf_path = Path(endf_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse ENDF file to get MAT and isotope information
    endf = read_endf(str(endf_path))
    
    if endf.mat is not None:
        mat = endf.mat
    else:
        raise ValueError(f"Could not determine MAT number from ENDF file {endf_path}")
    
    # Get ZAID from ENDF data
    if endf.zaid is not None:
        zaid = endf.zaid
    else:
        raise ValueError(f"Could not determine ZAID from ENDF file {endf_path}. MAT: {endf.mat}")
    
    nd_suffix = _resolve_nd_suffix(library_name)

    # Get isotope symbol for the title
    isotope_symbol = endf.isotope

    # Get temperature information using helper function
    temp_K, temp_suffix_legacy = _get_temperature_info(temperature)

    # New layout: caller supplies extension + ace_dir/xsdir_dir.
    # Legacy layout: derive from library_name + temp_K (K_TO_SUFFIX).
    use_new_layout = extension is not None
    if use_new_layout:
        temp_suffix = _suff_from_extension(extension)
        ace_dir = Path(ace_dir) if ace_dir is not None else (
            output_dir / "ace" / library_name / f"{temp_K:.1f}K"
        )
        xsdir_dir = Path(xsdir_dir) if xsdir_dir is not None else ace_dir / "xsdir"
        njoy_files_dir = Path(njoy_files_dir) if njoy_files_dir is not None else (
            output_dir / "njoy_files" / library_name / f"{temp_K:.1f}K"
        )
        ace_filename = f"{zaid}.{extension}"
        xsdir_filename = f"{zaid}.{extension}.xsdir"
        base_filename = ace_filename
    else:
        temp_suffix = temp_suffix_legacy
        ace_dir = Path(ace_dir) if ace_dir is not None else (
            output_dir / "ace" / library_name / f"{temp_K:.1f}K"
        )
        njoy_files_dir = Path(njoy_files_dir) if njoy_files_dir is not None else (
            output_dir / "njoy_files" / library_name / f"{temp_K:.1f}K"
        )
        xsdir_dir = Path(xsdir_dir) if xsdir_dir is not None else njoy_files_dir
        zaid_extended = zaid * 10
        base_filename_initial = f"{zaid_extended}_{nd_suffix}{temp_suffix}"
        base_filename, ace_filename = _generate_unique_filename(
            base_filename_initial, ace_dir, njoy_files_dir, temp_suffix, additional_suffix
        )
        xsdir_filename = f"{base_filename}.xsdir"

    ace_dir.mkdir(parents=True, exist_ok=True)
    xsdir_dir.mkdir(parents=True, exist_ok=True)
    njoy_files_dir.mkdir(parents=True, exist_ok=True)

    if suff is None:
        suff = temp_suffix

    title = _render_title(isotope_label=isotope_symbol, T=temp_K, library_name=library_name, njoy_version=njoy_version)
    njoy_input = _render_njoy_input(mat=mat, T=temperature, title=title, suff=suff)

    workdir = Path(tempfile.mkdtemp(prefix="njoy_", dir=output_dir))

    try:
        # NJOY expects tape files with specific unit numbers
        tape20 = workdir / "tape20"
        shutil.copy2(endf_path, tape20)

        # Write the input deck
        deck_path = workdir / "njoy.inp"
        deck_path.write_text(njoy_input)

        # Run NJOY: feed stdin from the input deck
        result = subprocess.run(
            build_njoy_command(njoy_exe),
            cwd=workdir,
            input=njoy_input.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )

        # Save organized output files
        results = {
            "returncode": result.returncode,
            "zaid": zaid,
            "ace_filename": ace_filename,
            "base_filename": base_filename,
        }
        
        # Save NJOY input file
        input_file = njoy_files_dir / f"{base_filename}.input"
        shutil.copy2(deck_path, input_file)
        results["njoy_input"] = str(input_file)
        
        # Save NJOY output (log)
        output_file = njoy_files_dir / f"{base_filename}.output"
        output_file.write_bytes(result.stdout)
        results["njoy_output"] = str(output_file)
        
        # Process tape files (second ACER pass writes final ACE/xsdir)
        tape43 = workdir / "tape43"  # ACE file (final, from second ACER pass)
        tape44 = workdir / "tape44"  # xsdir file (final, from second ACER pass)

        if tape43.exists():
            ace_file = ace_dir / ace_filename
            shutil.move(str(tape43), ace_file)
            results["ace_file"] = str(ace_file)
        else:
            results["ace_file"] = None

        if tape44.exists():
            xsdir_file = xsdir_dir / xsdir_filename
            shutil.move(str(tape44), xsdir_file)
            results["xsdir_file"] = str(xsdir_file)
        else:
            results["xsdir_file"] = None

        return results

    finally:
        # Always clean up temporary directory
        shutil.rmtree(workdir, ignore_errors=True)


def run_njoy_with_pendf(
    njoy_exe: str,
    endf_path: str | Path,
    pendf_path: str | Path,
    temperature: float,
    library_name: str,
    output_dir: str | Path,
    *,
    suff: str = None,
    njoy_version: str = "NJOY 2016.78",
    additional_suffix: str = None,
    extension: str | None = None,
    ace_dir: str | Path | None = None,
    xsdir_dir: str | Path | None = None,
    njoy_files_dir: str | Path | None = None,
) -> dict:
    """Run NJOY against a pre-existing (typically perturbed) PENDF, skipping RECONR.

    Same chain and outputs as :func:`run_njoy`, but the resonance-reconstruction
    step is replaced by a ``moder`` that reads the user-provided ASCII PENDF and
    routes it into the binary tape (-21) that ``broadr`` consumes downstream.

    Use this when the PENDF has already been generated externally — e.g. the
    MF3 perturbation pipeline produces a perturbed PENDF, then calls this to
    chain through broadr/heatr/purr/gaspr/acer to ACE.

    Parameters
    ----------
    endf_path : str | Path
        Original (or, in the future combined pipeline, perturbed) ENDF tape.
        Provides MF1/MF2/MF4/MF5/MF6 — everything except MF3 σ(E).
    pendf_path : str | Path
        Pre-existing ASCII PENDF tape to inject. Replaces RECONR's output.
    Other parameters: see :func:`run_njoy`.
    """
    endf_path = Path(endf_path).resolve()
    pendf_path = Path(pendf_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    endf = read_endf(str(endf_path))
    if endf.mat is None:
        raise ValueError(f"Could not determine MAT number from ENDF file {endf_path}")
    if endf.zaid is None:
        raise ValueError(f"Could not determine ZAID from ENDF file {endf_path}")
    mat = endf.mat
    zaid = endf.zaid

    nd_suffix = _resolve_nd_suffix(library_name)
    isotope_symbol = endf.isotope

    temp_K, temp_suffix_legacy = _get_temperature_info(temperature)

    use_new_layout = extension is not None
    if use_new_layout:
        temp_suffix = _suff_from_extension(extension)
        ace_dir = Path(ace_dir) if ace_dir is not None else (
            output_dir / "ace" / library_name / f"{temp_K:.1f}K"
        )
        xsdir_dir = Path(xsdir_dir) if xsdir_dir is not None else ace_dir / "xsdir"
        njoy_files_dir = Path(njoy_files_dir) if njoy_files_dir is not None else (
            output_dir / "njoy_files" / library_name / f"{temp_K:.1f}K"
        )
        ace_filename = f"{zaid}.{extension}"
        xsdir_filename = f"{zaid}.{extension}.xsdir"
        base_filename = ace_filename
    else:
        temp_suffix = temp_suffix_legacy
        ace_dir = Path(ace_dir) if ace_dir is not None else (
            output_dir / "ace" / library_name / f"{temp_K:.1f}K"
        )
        njoy_files_dir = Path(njoy_files_dir) if njoy_files_dir is not None else (
            output_dir / "njoy_files" / library_name / f"{temp_K:.1f}K"
        )
        xsdir_dir = Path(xsdir_dir) if xsdir_dir is not None else njoy_files_dir
        zaid_extended = zaid * 10
        base_filename_initial = f"{zaid_extended}_{nd_suffix}{temp_suffix}"
        base_filename, ace_filename = _generate_unique_filename(
            base_filename_initial, ace_dir, njoy_files_dir, temp_suffix, additional_suffix,
        )
        xsdir_filename = f"{base_filename}.xsdir"

    ace_dir.mkdir(parents=True, exist_ok=True)
    xsdir_dir.mkdir(parents=True, exist_ok=True)
    njoy_files_dir.mkdir(parents=True, exist_ok=True)

    if suff is None:
        suff = temp_suffix

    title = _render_title(
        isotope_label=isotope_symbol, T=temp_K, library_name=library_name,
        njoy_version=njoy_version,
    )
    njoy_input = _render_njoy_input(
        mat=mat, T=temperature, title=title, suff=suff,
        template=NJOY_INPUT_TEMPLATE_WITH_PENDF,
    )

    workdir = Path(tempfile.mkdtemp(prefix="njoy_pendf_", dir=output_dir))

    try:
        tape20 = workdir / "tape20"
        tape26 = workdir / "tape26"
        shutil.copy2(endf_path, tape20)
        shutil.copy2(pendf_path, tape26)

        deck_path = workdir / "njoy.inp"
        deck_path.write_text(njoy_input)

        result = subprocess.run(
            build_njoy_command(njoy_exe),
            cwd=workdir,
            input=njoy_input.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )

        results = {
            "returncode": result.returncode,
            "zaid": zaid,
            "ace_filename": ace_filename,
            "base_filename": base_filename,
        }

        input_file = njoy_files_dir / f"{base_filename}.input"
        shutil.copy2(deck_path, input_file)
        results["njoy_input"] = str(input_file)

        output_file = njoy_files_dir / f"{base_filename}.output"
        output_file.write_bytes(result.stdout)
        results["njoy_output"] = str(output_file)

        tape43 = workdir / "tape43"
        tape44 = workdir / "tape44"

        if tape43.exists():
            ace_file = ace_dir / ace_filename
            shutil.move(str(tape43), ace_file)
            results["ace_file"] = str(ace_file)
        else:
            results["ace_file"] = None

        if tape44.exists():
            xsdir_file = xsdir_dir / xsdir_filename
            shutil.move(str(tape44), xsdir_file)
            results["xsdir_file"] = str(xsdir_file)
        else:
            results["xsdir_file"] = None

        return results

    finally:
        shutil.rmtree(workdir, ignore_errors=True)