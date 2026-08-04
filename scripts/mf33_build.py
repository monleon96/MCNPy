"""Build and write the MF33 elastic-magnitude covariance products.

Shared by the pipeline (``exfor_to_endf_sampling_v2``) and the offline rebuild
(``rebuild_mf33``) so both produce byte-identical MF33 sections from the same
inputs.  Three stages, each usable on its own:

1. :func:`build_mf33_denominator` — the host central per bin, from the
   RECONR-reconstructed PENDF folded through each bin's TOF kernel.
2. :func:`build_mf33_matrices` — the fine relative covariance plus its own
   adaptive multigroup collapse.
3. :func:`write_mf33_products` — range-merge both into their host ENDF files.

Why the denominator is not simply ``File 3`` averaged over the bin
---------------------------------------------------------------------
MF33 is relative, so it needs a central value to divide by, and two independent
things make the obvious choice wrong:

* **Resolution.**  The fitted ``c0`` is what a detector with a finite TOF
  resolution measured, not a box average of the true cross section over a 1 keV
  analysis bin.  On the Fe-56 setup ``sigma_E`` runs 4–41 keV, i.e. 4–40x the
  bin width.  Folding the denominator through the same kernel cuts the residual
  scatter against ``4*pi*c0`` from ~33% to ~18%.
* **Resonance range.**  File 3 holds only the smooth background inside a
  resolved resonance range, so ``MF3 MT2`` is identically zero below the RRR
  upper bound (850 keV for JEFF-4.0 Fe-56).  Dividing by it produced NaN rows
  that reached ``np.linalg.eigvalsh`` in the ENDF writer and surfaced as
  "Eigenvalues did not converge".

Neither fix is sufficient alone: folding File 3 near the RRR edge integrates the
artificial zeros and biases the denominator *low* (worst-case ratio 11.0 vs 5.1
for a plain box average).  The PENDF supplies real cross sections there, and the
combination is well behaved (worst-case 2.1).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from kika.endf.writers import (
    merge_mf33_covariance_into_host,
    write_mf33_to_file,
)
from kika.processing.njoy_pendf_cache import (
    get_or_create_pendf,
    read_pendf_mf3_sections,
)
from kika.utils.numerics import average_over_intervals, fold_tabulated
from scripts.mf33_diagnostics import FOUR_PI, bin_average_xs
from scripts.multigroup_collapse import (
    MF33MultigroupResult,
    perform_mf33_multigroup_collapse,
)


def fold_xs_over_bins(
    e_ev: np.ndarray,
    xs_b: np.ndarray,
    energy_bins: Sequence[Any],
    *,
    n_nodes: int = 12,
    logger=None,
) -> np.ndarray:
    """Project a pointwise cross section onto analysis bins through the TOF kernel.

    The per-bin counterpart of
    :func:`scripts.mf33_diagnostics.fold_host_mf3_at_points`, and the correct
    denominator for recentring a DCS-derived MF33 on the host File 3 — see the
    module docstring for why a box average is not.

    The bin width itself adds nothing worth modelling: convolving the Gaussian
    with a box of width ``w`` gives ``sqrt(sigma_E^2 + w^2/12)``, which at
    ``w = 1 keV`` and ``sigma_E = 4 keV`` is a 0.3% change in the kernel width.

    Parameters
    ----------
    e_ev, xs_b : np.ndarray
        Pointwise cross section (eV, barns), ascending.  For a target whose
        resolved resonance range reaches into the analysis window this must be
        the **reconstructed** (PENDF) cross section: File 3 alone is the smooth
        background there and is identically zero inside the RRR.
    energy_bins : Sequence[EnergyBinInfo]
        Bins carrying ``energy_mev``, ``sigma_E_mev`` and the bin edges.
    n_nodes : int, default 12
        Gauss-Hermite nodes.
    logger : optional
        Sink for the count of bins that fell back to a box average.

    Returns
    -------
    np.ndarray
        Folded cross section per bin (barns), shape ``(len(energy_bins),)``.

    Notes
    -----
    Bins with ``sigma_E_mev <= 0`` fall back to an average over the bin.  The
    Gaussian fold degenerates to a *point sample* at zero width, which on a
    resonant cross section is worse than a box average.
    """
    e_ev = np.asarray(e_ev, dtype=float)
    xs_b = np.asarray(xs_b, dtype=float)

    out = np.empty(len(energy_bins), dtype=float)
    fallback: List[int] = []
    for i, eb in enumerate(energy_bins):
        sigma_E_mev = float(getattr(eb, "sigma_E_mev", 0.0) or 0.0)
        if sigma_E_mev > 0.0:
            out[i] = fold_tabulated(
                e_ev, xs_b, float(eb.energy_mev) * 1e6, sigma_E_mev * 1e6,
                n_nodes=n_nodes,
            )
        else:
            fallback.append(i)
            edges = np.array(
                [eb.bin_lower_mev * 1e6, eb.bin_upper_mev * 1e6], dtype=float,
            )
            out[i] = average_over_intervals(e_ev, xs_b, edges)[0]

    if fallback and logger is not None:
        logger.warning(
            f"  [MF33] {len(fallback)}/{len(out)} bins have sigma_E <= 0 and "
            f"fell back to a box average over the bin."
        )
    return out


@dataclass
class MF33Denominator:
    """The host central per bin, plus the pointwise data it came from.

    The pointwise arrays are carried so callers can reuse them for the
    folded-comparison diagnostics without a second PENDF read (and so those
    diagnostics see the same reconstructed cross section the denominator did,
    rather than the File-3 background).
    """
    sigma_host_bin: np.ndarray   # (n_bins,) barns
    pendf_path: Path
    e_ev: np.ndarray             # pointwise energies (eV)
    xs_b: np.ndarray             # pointwise cross section (barns)


@dataclass
class MF33Products:
    """Everything needed to write MF33 into the fine and multigroup files."""
    rel_fine: np.ndarray            # (n_fine, n_fine) relative, host-recentred
    grid_fine_ev: np.ndarray        # (n_fine + 1,) fine (MF4) grid
    c0_host: np.ndarray             # (n_fine,) host central per bin, c0 units
    sigma_host_bin: np.ndarray      # (n_fine,) host central per bin, barns
    multigroup: MF33MultigroupResult


def build_mf33_denominator(
    endf_file: str,
    energy_bins: Sequence[Any],
    *,
    mt: int = 2,
    njoy_exe: str,
    pendf_tolerance: float = 0.001,
    pendf_cache_dir: Optional[str] = None,
    grid_ev: Optional[np.ndarray] = None,
    logger=None,
) -> MF33Denominator:
    """Host central per bin (barns), from the PENDF folded through the TOF kernel.

    Parameters
    ----------
    endf_file : str
        Host ENDF tape; reconstructed with RECONR and cached on its sha256, so
        repeat calls with the same tolerance cost nothing.
    energy_bins : Sequence[EnergyBinInfo]
        Analysis bins, carrying ``energy_mev``, ``sigma_E_mev`` and the bin edges.
    grid_ev : np.ndarray, optional
        Fine grid; when given, the box-average denominator is computed too and
        the two are compared in the log.  Diagnostic only.

    Returns
    -------
    MF33Denominator

    Raises
    ------
    ValueError
        If any bin ends up with a non-positive central.  With the PENDF this
        should be impossible; failing loudly here is deliberate, because
        silently passing zeros downstream is what broke the MF33 write.
    """
    pendf_path = get_or_create_pendf(
        endf_file,
        tolerance=pendf_tolerance,
        njoy_exe=njoy_exe,
        cache_dir=pendf_cache_dir,
    )
    section = read_pendf_mf3_sections(pendf_path)[mt]
    e_ev = np.asarray(section.energies, dtype=float)
    xs_b = np.asarray(section.cross_sections, dtype=float)
    if logger:
        logger.info(
            f"  [MF33] Denominator from PENDF {Path(pendf_path).name}: "
            f"{len(e_ev)} points, [{e_ev[0]:.4g}, {e_ev[-1]:.4g}] eV"
        )

    sigma_host_bin = fold_xs_over_bins(e_ev, xs_b, energy_bins, logger=logger)

    bad = np.flatnonzero(~(np.isfinite(sigma_host_bin) & (sigma_host_bin > 0.0)))
    if bad.size:
        if grid_ev is not None:
            spans = ", ".join(
                f"{grid_ev[i] / 1e6:.4f}-{grid_ev[i + 1] / 1e6:.4f} MeV"
                for i in bad[:5]
            )
        else:
            spans = ", ".join(f"{energy_bins[i].energy_mev:.4f} MeV" for i in bad[:5])
        raise ValueError(
            f"[MF33] {bad.size} bin(s) have a non-positive host central after "
            f"PENDF folding; cannot build a relative covariance. First: {spans}"
        )

    if grid_ev is not None and logger:
        sigma_box = bin_average_xs(e_ev, xs_b, np.asarray(grid_ev, dtype=float))
        logger.info(
            "  [MF33] Denominator spread (barns): folded "
            f"[{sigma_host_bin.min():.3f}, {sigma_host_bin.max():.3f}] vs "
            f"box-average [{sigma_box.min():.3f}, {sigma_box.max():.3f}]"
        )

    return MF33Denominator(
        sigma_host_bin=sigma_host_bin,
        pendf_path=Path(pendf_path),
        e_ev=e_ev,
        xs_b=xs_b,
    )


def build_mf33_matrices(
    cov_abs_fine: np.ndarray,
    sigma_host_bin: np.ndarray,
    energy_bins: Sequence[Any],
    grid_fine_ev: np.ndarray,
    *,
    c0_dcs: Optional[np.ndarray] = None,
    regularize_near_zero: bool = True,
    snr_threshold: float = 1.0,
    n_neighbors: int = 3,
    rho_min: float = 0.85,
    sigma_ratio_max: Optional[float] = None,
    diagnostics_file: Optional[Path] = None,
    logger=None,
) -> MF33Products:
    """Fine relative covariance plus the adaptive multigroup collapse.

    ``cov_abs_fine`` is the DCS-derived **absolute** covariance of ``c0``.  It is
    recentred on the host central so that ``C_rel * sigma_host^2`` reproduces the
    absolute uncertainty that was actually measured — the correct choice while
    MF3 itself stays at the host value.

    The near-zero SNR guard (``regularize_near_zero_relative_covariance``) runs
    on both the fine and the collapsed matrix.  It rescales rows whose relative
    std is an artefact of a small central rather than a real uncertainty, via a
    congruence transform that preserves correlations and PSD.
    """
    from scripts.exfor_utils import (
        recentre_relative_covariance,
        regularize_near_zero_relative_covariance,
    )

    cov_abs_fine = np.asarray(cov_abs_fine, dtype=float)
    sigma_host_bin = np.asarray(sigma_host_bin, dtype=float)
    grid_fine_ev = np.asarray(grid_fine_ev, dtype=float)
    c0_host = sigma_host_bin / FOUR_PI

    rel_fine = recentre_relative_covariance(cov_abs_fine, c0_host)

    if logger and c0_dcs is not None:
        ratio = np.where(c0_host > 0, np.asarray(c0_dcs, dtype=float) / c0_host, np.nan)
        logger.info(
            "  [MF33] Relative covariance recentred on the folded PENDF central: "
            f"median c0_DCS/c0_host = {np.nanmedian(ratio):.3f}, "
            f"mean |ratio-1| = {100 * np.nanmean(np.abs(ratio - 1)):.1f}%, "
            f"max = {np.nanmax(ratio):.2f}"
        )

    if regularize_near_zero:
        rel_fine, _diag = regularize_near_zero_relative_covariance(
            cov_rel=rel_fine,
            mean_params=c0_host,
            cov_abs=cov_abs_fine,
            max_order=1,
            snr_threshold=snr_threshold,
            n_neighbors=n_neighbors,
            logger=logger,
        )

    if logger:
        rel_std = np.sqrt(np.clip(np.diag(rel_fine), 0.0, None))
        logger.info(
            f"  [MF33] Fine relative sigma: median {100 * np.median(rel_std):.2f}%, "
            f"range {100 * rel_std.min():.2f}-{100 * rel_std.max():.2f}%"
        )

    mg = perform_mf33_multigroup_collapse(
        cov_abs_fine=cov_abs_fine,
        means_fine=c0_host,
        energy_bins=list(energy_bins),
        rho_min=rho_min,
        sigma_ratio_max=sigma_ratio_max,
        logger=logger,
        diagnostics_file=diagnostics_file,
        label="MF33 MT2 (elastic magnitude)",
    )

    if regularize_near_zero:
        mg.cov_rel_grouped, _ = regularize_near_zero_relative_covariance(
            cov_rel=mg.cov_rel_grouped,
            mean_params=mg.means_grouped,
            cov_abs=mg.cov_abs_grouped,
            max_order=1,
            snr_threshold=snr_threshold,
            n_neighbors=n_neighbors,
            logger=logger,
        )

    if logger:
        rel_std_mg = np.sqrt(np.clip(np.diag(mg.cov_rel_grouped), 0.0, None))
        logger.info(
            f"  [MF33] Multigroup relative sigma: "
            f"median {100 * np.median(rel_std_mg):.2f}%, "
            f"range {100 * rel_std_mg.min():.2f}-{100 * rel_std_mg.max():.2f}%"
        )

    return MF33Products(
        rel_fine=rel_fine,
        grid_fine_ev=grid_fine_ev,
        c0_host=c0_host,
        sigma_host_bin=sigma_host_bin,
        multigroup=mg,
    )


def _project_lb5(matrix: np.ndarray, src_grid: np.ndarray,
                 dst_grid: np.ndarray) -> np.ndarray:
    """Lift an LB=5 relative matrix from ``src_grid`` onto ``dst_grid``.

    LB=5 is piecewise-constant in both indices, so a destination group takes the
    value of whichever source group contains its midpoint.  Exact whenever the
    destination is a refinement of the source, which is the case here (the
    partials are on coarser host grids than the analysis window).
    """
    mid = 0.5 * (dst_grid[:-1] + dst_grid[1:])
    idx = np.clip(np.searchsorted(src_grid, mid, side="right") - 1,
                  0, src_grid.size - 2)
    return matrix[np.ix_(idx, idx)]


def build_mt1_from_partials(
    host_endf: str,
    *,
    elastic_rel: np.ndarray,
    grid_ev: np.ndarray,
    pendf_path: Any,
    elastic_mt: int = 2,
    total_mt: int = 1,
    budget_rtol: float = 1e-3,
    logger=None,
) -> Tuple[np.ndarray, dict]:
    """Rebuild the total (MT1) relative covariance from the partials.

    The splice replaces MF33 MT2 inside the analysis window but leaves MT1 as
    the host wrote it, so the file ends up claiming a total uncertainty that its
    own elastic no longer supports.  This rebuilds MT1 over the same window as
    the exact sandwich over the partial reactions with **cross terms zeroed**::

        A_tot(E,E') = sum_i sigma_i(E) sigma_i(E') C_i(E,E')
        C_tot       = A_tot / (sigma_tot (x) sigma_tot)

    with ``C_elastic`` taken from this pipeline and every other partial kept at
    the host's.  PSD by construction: a sum of PSD blocks conjugated by positive
    diagonal scalings.

    Zeroing the cross terms is not an approximation forced on us — it is the
    host's own convention.  On JEFF-4.0T Fe-56 the shipped MT1 sits within
    **1.4%** of the uncorrelated sum of its own partials, so the evaluator built
    it the same way.

    Cross sections come from the **PENDF**, not File 3: inside a resolved
    resonance range File 3 holds only the smooth background (see the module
    docstring), and the window's lower edge is inside the Fe-56 RRR.

    Parameters
    ----------
    host_endf : str
        File carrying the host MF33 partials.  Safe to point at the file being
        written — MT1 and the partials are read before MT1 is replaced.
    elastic_rel : np.ndarray
        This pipeline's relative covariance for ``elastic_mt``, on ``grid_ev``.
    grid_ev : np.ndarray
        Boundaries of the window (eV), ``len == elastic_rel.shape[0] + 1``.
    pendf_path : str or Path
        RECONR'd PENDF for the host tape.
    budget_rtol : float, default 1e-3
        Tolerance on ``sum(partials) == total`` before warning.

    Returns
    -------
    (cov_rel, diagnostics)
        Relative covariance for ``total_mt`` on ``grid_ev``, and a dict of
        per-MT variance shares plus the reaction-budget residual.
    """
    from kika.endf import read_endf

    log = logger
    grid_ev = np.asarray(grid_ev, dtype=float)
    elastic_rel = np.asarray(elastic_rel, dtype=float)
    n = grid_ev.size - 1
    if elastic_rel.shape != (n, n):
        raise ValueError(
            f"elastic_rel {elastic_rel.shape} does not match grid_ev "
            f"({n} groups)"
        )

    sections = read_pendf_mf3_sections(pendf_path)

    def group_xs(mt: int) -> Optional[np.ndarray]:
        sec = sections.get(mt)
        if sec is None:
            return None
        return average_over_intervals(
            np.asarray(sec.energies, dtype=float),
            np.asarray(sec.cross_sections, dtype=float),
            grid_ev,
        )

    sigma_tot = group_xs(total_mt)
    if sigma_tot is None or np.any(sigma_tot <= 0):
        raise ValueError(
            f"[MF33] MT{total_mt} central is missing or non-positive on the "
            f"window grid; cannot build a relative total covariance."
        )

    host = read_endf(host_endf, mf_numbers=[33])
    mf33_file = host.get_file(33)
    if mf33_file is None:
        raise ValueError(f"[MF33] {host_endf} carries no MF33 to rebuild MT1 from")

    # Every self-covariance partial the host ships, minus the total itself.
    partial_mts = sorted(
        mt for mt, sec in mf33_file.sections.items()
        if mt not in (0, total_mt) and not (sec._mtl or 0)
    )

    absolute = np.zeros((n, n))
    shares: dict = {}
    used: List[int] = []
    sigma_sum = np.zeros(n)

    for mt in [elastic_mt] + [m for m in partial_mts if m != elastic_mt]:
        sigma = group_xs(mt)
        if sigma is None or not np.any(sigma > 0):
            continue
        sigma_sum = sigma_sum + sigma
        if mt == elastic_mt:
            cov_rel = elastic_rel
        else:
            section = mf33_file.sections.get(mt)
            if section is None:
                continue
            decoded = section.to_xs_covmat()
            if not decoded.matrices:
                continue
            cov_rel = _project_lb5(
                np.asarray(decoded.matrices[0], dtype=float),
                np.asarray(decoded.energy_grids[0], dtype=float),
                grid_ev,
            )
        block = np.outer(sigma, sigma) * cov_rel
        absolute += block
        shares[mt] = np.diag(block).copy()
        used.append(mt)

    residual = sigma_tot - sigma_sum
    rel_residual = float(np.max(np.abs(residual) / sigma_tot))
    if rel_residual > budget_rtol and log:
        log.warning(
            f"  [MF33] Reaction budget for MT{total_mt}: partials {used} miss "
            f"{100 * rel_residual:.2f}% of the total at worst — the rebuilt "
            f"MT{total_mt} attributes no uncertainty to the missing part."
        )

    cov_rel_total = absolute / np.outer(sigma_tot, sigma_tot)
    cov_rel_total = 0.5 * (cov_rel_total + cov_rel_total.T)

    total_var = np.diag(absolute)
    diagnostics = {
        "partials": used,
        "budget_max_rel_residual": rel_residual,
        "variance_share": {
            mt: float(np.median(v / np.where(total_var > 0, total_var, np.nan)))
            for mt, v in shares.items()
        },
    }

    if log:
        rel_std = np.sqrt(np.clip(np.diag(cov_rel_total), 0.0, None))
        share_txt = ", ".join(
            f"MT{mt} {100 * s:.1f}%"
            for mt, s in diagnostics["variance_share"].items() if s == s
        )
        log.info(
            f"  [MF33] MT{total_mt} rebuilt from partials {used} "
            f"(cross terms zeroed): relative sigma median "
            f"{100 * np.median(rel_std):.2f}%, "
            f"range {100 * rel_std.min():.2f}-{100 * rel_std.max():.2f}%; "
            f"median variance share {share_txt}"
        )

    return cov_rel_total, diagnostics


def write_mf33_products(
    products: MF33Products,
    *,
    fine_endf: Optional[str],
    mg_endf: Optional[str],
    mt: int,
    za: Optional[float] = None,
    awr: Optional[float] = None,
    mat: Optional[int] = None,
    mg_representation: str = "multigroup",
    rebuild_mt1: bool = False,
    pendf_path: Any = None,
    total_mt: int = 1,
    logger=None,
) -> List[str]:
    """Range-merge and write MF33 into the fine and/or multigroup ENDF files.

    Each file gets the representation that matches its MF34: the fine file the
    MF4-grid matrix, the multigroup file its own adaptive grid.  ENDF lets every
    MF/MT section carry its own boundaries, so the two need not agree — and they
    should not, since the shape and magnitude channels have different
    correlation lengths.

    ``mg_representation="fine"`` goes further and gives the ``_mg`` file the
    fine MF33 while its MF34 stays collapsed.  MF34 is what makes that file
    large and genuinely needs grouping; MF33 does not, and the collapse is
    irreversible — on run 82 it pulled the peak elastic sigma from 19.3% down to
    12.2%.  Shipping the fine matrix lets the consumer collapse to whatever
    group structure it actually wants.

    The merge keeps the host MF33 outside the analysis range and zeroes the
    in/out cross terms (documented factorization); the per-MT section writer
    preserves sibling MF33 sections (MT 1, 4, 5, 16, 102, 103).

    With ``rebuild_mt1``, MT1 is rebuilt over the same window from the partials
    (:func:`build_mt1_from_partials`) and spliced the same way, so the file's
    total no longer contradicts its own elastic.  Needs ``pendf_path``.
    """
    if mg_representation not in ("multigroup", "fine"):
        raise ValueError(
            f"mg_representation must be 'multigroup' or 'fine', "
            f"got {mg_representation!r}"
        )
    if rebuild_mt1 and pendf_path is None:
        raise ValueError("rebuild_mt1=True requires pendf_path")

    written: List[str] = []

    targets = []
    if fine_endf:
        targets.append(("fine", fine_endf, products.rel_fine, products.grid_fine_ev))
    if mg_endf:
        if mg_representation == "fine":
            targets.append((
                "multigroup-file (ungrouped)",
                mg_endf, products.rel_fine, products.grid_fine_ev,
            ))
        else:
            targets.append((
                "multigroup", mg_endf,
                products.multigroup.cov_rel_grouped,
                products.multigroup.group_boundaries_ev,
            ))

    for kind, path, matrix, grid in targets:
        section = merge_mf33_covariance_into_host(
            host_endf=path,
            cov_matrix=matrix,
            energy_grid_ev=grid,
            mt=mt,
            za=za, awr=awr, mat=mat,
            logger=logger,
        )
        write_mf33_to_file(path, section, path)
        written.append(path)
        if logger:
            logger.info(
                f"  [MF33] {kind} MF33 MT{mt} ({matrix.shape[0]} groups) "
                f"written to: {path}"
            )

        if rebuild_mt1:
            # Read back from `path`: MT1 and the other partials are still the
            # host's there, only MT2 has been replaced — which is exactly the
            # elastic we want the new total built on.
            cov_total, _diag = build_mt1_from_partials(
                path,
                elastic_rel=matrix,
                grid_ev=grid,
                pendf_path=pendf_path,
                elastic_mt=mt,
                total_mt=total_mt,
                logger=logger,
            )
            section_total = merge_mf33_covariance_into_host(
                host_endf=path,
                cov_matrix=cov_total,
                energy_grid_ev=grid,
                mt=total_mt,
                za=za, awr=awr, mat=mat,
                logger=logger,
            )
            write_mf33_to_file(path, section_total, path)
            if logger:
                logger.info(
                    f"  [MF33] {kind} MF33 MT{total_mt} "
                    f"({cov_total.shape[0]} groups) written to: {path}"
                )

    return written
