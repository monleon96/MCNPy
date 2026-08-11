"""Perturbing a prompt fission neutron spectrum from its MF35 covariance.

The pipeline, per incident-energy node of an MF5 LF=1 table:

1. integrate the original table over the band's outgoing groups → ``P⁰``;
2. measure ``∫χ`` and ``Σ P⁰`` and record both, before anything is changed;
3. freeze groups with ``P⁰ = 0`` — no ratio is defined there;
4. form ratios ``r_j = 1 + δ_j / P⁰_j`` from the sampled absolute deltas;
5. project onto ``{r ≥ 0, Σ P⁰ r = Σ P⁰}`` (:mod:`kika.sampling.pfns_positivity`);
6. insert the group boundaries the factor steps at, plus a shoulder below each,
   and scale;
7. rescale the whole table so ``∫χ`` is exactly what it was.

**Step 6 is load-bearing and is the single most likely silent bug.** Scaling the
existing nodes without first inserting the group boundary applies the factor
over the wrong interval: the panel straddling a boundary gets one group's factor
across both groups. The realised group integral then stops being ``r_j × P⁰_j``,
which is the property that makes the whole construction preserve normalisation —
and step 7 will absorb the difference into a scalar, so the run completes,
NJOY is happy, and the perturbation is not the one that was computed.

**Everything inserted is an exact refinement.** New outgoing nodes are valued by
interpolation on the original table, new incident nodes by
``evaluate_at_incident``, which blends the two bracketing tables on their union
grid. With lin-lin on both axes those are identities, not approximations, so
with ``δ ≡ 0`` the whole path reproduces χ pointwise — which is the strongest
test in the suite.

**Linear space, not log.** ``nubar_perturbation`` samples in log space and it is
right to, because a nu-bar factor is multiplicative and must stay positive. Here
the covariance is of quantities that *sum to one*, so a linear draw satisfies
``1ᵀδ ≈ 0`` automatically and preserves normalisation to within the measured
drift budget. A log draw destroys that property outright.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..endf.classes.mf5.partials import MF5PartialTabulated
from .core import draw_samples
from .model_blocks import covariance_suite_blocks
from .pfns_positivity import check_ratios, project_ratios_to_simplex

__all__ = [
    "OUTGOING_STEP_SHOULDER", "INCIDENT_STEP_SHOULDER", "ENDF_ENERGY_RTOL",
    "SAMPLING_SPACE", "SIGMA_CLAMP",
    "load_pfns_covariance", "build_pfns_covariance", "covariance_suite_blocks",
    "band_grids",
    "generate_pfns_samples", "normalisation_residual", "perturb_pfns_partial",
    "realised_group_probabilities", "realised_covariance_report",
    "row_sum_residual", "normalisation_drift",
]

#: Where the shoulder node goes below an outgoing group boundary — as a
#: fraction of the **lower group's width**, not of the boundary energy.
#:
#: A duplicate-free lin-lin table cannot express a step, so the perturbed curve
#: ramps between two adjacent groups' factors and the shoulder confines that
#: ramp to a sliver. ``NUBAR_STEP_SHOULDER`` measures the sliver as a fraction
#: of the *energy* (``E_b·(1-s)``) and is right to, because MF31 bins span
#: decades so ``s·E`` is always a small part of a bin.
#:
#: **That does not carry over, and copying it was measurably wrong.** MF35's
#: outgoing grids are linear at the top: Cf-252's highest groups are 2e5 wide at
#: 2e7 eV, where ``s·E`` = 2e4 is a **tenth of the group**. The ramp then spans
#: a tenth of the group and the realised group integral misses ``r_j·P⁰_j`` by
#: percent, which is the same size as the perturbation. Measured on Cf-252: max
#: group error 1.5 with the energy-relative shoulder, 8e-4 with this one.
#:
#: Scaling by the group width instead makes the ramp error ~½·s·|Δr| whatever
#: the grid geometry does.
OUTGOING_STEP_SHOULDER = 1.0e-3

#: The same, for the step in incident energy at a band edge.
INCIDENT_STEP_SHOULDER = 1.0e-3

#: Two energies closer than this are the same energy once written in ENDF's
#: 11-character field. Inserting a node inside this of an existing one produces
#: a duplicate energy, which is the thing the shoulder construction exists to
#: avoid.
ENDF_ENERGY_RTOL = 1.0e-6

#: Stated as a constant so it cannot be "fixed" back to "log" by analogy with
#: nu-bar. See the module docstring: linear is what preserves normalisation
#: here, because ``C·1 ≈ 0``.
SAMPLING_SPACE = "linear"

#: Deliberately absent: a relative floor on P⁰ like ``SIGMA_FLOOR_REL = 1e-3``
#: in ``mf33_sampling``. Cf-252's group probabilities run down to 1e-17 and a
#: floor would freeze the whole low-energy tail. It is not needed either — σ/P
#: is finite there, and the positivity projection handles a wild draw. Only
#: groups with ``P⁰`` exactly zero are frozen.


# ---------------------------------------------------------------------------
# Reading the covariance
# ---------------------------------------------------------------------------

def load_pfns_covariance(
    endf_path,
    mt: int = 18,
    *,
    energy_unit: str = "eV",
    covariance_override=None,
    logger=None,
):
    """Read MF5 and MF35 from *endf_path*.

    Returns ``(covariance_suite, mf5_section, bands)`` where ``bands`` is the
    list of ``(E1, E2)`` incident ranges, in section order.
    """
    from ..endf import read_endf

    endf = read_endf(str(endf_path), mf_numbers=[5, 35])
    return build_pfns_covariance(
        endf, mt=mt, energy_unit=energy_unit,
        covariance_override=covariance_override, logger=logger,
    )


def build_pfns_covariance(
    endf_obj,
    mt: int = 18,
    *,
    energy_unit: str = "eV",
    covariance_override=None,
    logger=None,
):
    """The object form, so a 36 MB tape is not parsed twice.

    The GNDS ``CovarianceSuite`` is the container — one ``CovarianceSection``
    per band, the band carried as an incident-energy slice on the row link.
    There is no PFNS-specific covariance class: the point of the model layer is
    that one format-agnostic object serves every source format, and a bespoke
    container here would have been the fourth.
    """
    if energy_unit != "eV":
        raise NotImplementedError(
            f"energy_unit={energy_unit!r}: ENDF is native eV and this pipeline "
            f"writes ENDF back, so no conversion happens here. Unit handling "
            f"belongs to the model layer, not to the sampler."
        )
    if covariance_override is not None:
        raise NotImplementedError(
            "covariance_override is reserved for the cross-pipeline external "
            "covariance resolver (roadmap P5) and is not implemented. It is in "
            "the signature now so that adding it is not an API break. Note "
            "that MF35 will need its own grid reconciliation: its entries are "
            "absolute covariances of integrated probabilities, so collapsing "
            "groups must SUM the block, not average it."
        )

    # Function-scope, and that is the rule rather than a style choice: the GNDS
    # model must stay dormant on `import kika` and on `import kika.sampling`.
    #
    # Written absolute rather than relative on purpose. The layering ratchet in
    # `test_nothing_imports_the_adapter.py` matches import statements by module
    # name, so `from ..endf.model_adapter import ...` slips past it — this
    # module would have evaded the allowlist on a technicality. It is on the
    # allowlist instead, which is where a decision like this belongs.
    from kika.endf.model_adapter import decodeCovarianceSuite

    mf5 = endf_obj.mf.get(5) if hasattr(endf_obj, "mf") else None
    mf35 = endf_obj.mf.get(35) if hasattr(endf_obj, "mf") else None
    if mf5 is None or mt not in getattr(mf5, "mt", {}):
        raise ValueError(f"no MF5/MT{mt} in this evaluation: nothing to perturb")
    if mf35 is None or mt not in getattr(mf35, "mt", {}):
        raise ValueError(
            f"no MF35/MT{mt} in this evaluation: nothing to sample from. MF35 "
            f"exists for MT18 only on every library checked, and there is no "
            f"MF35/MT455 anywhere — delayed spectra have no covariance to draw "
            f"on, which is why this pipeline is prompt-only."
        )

    suite, report = decodeCovarianceSuite(endf_obj)
    sections = [s for s in suite if s.rowData is not None
                and s.rowData.ENDF_MFMT == f"35/{mt}"]
    bands = [s.rowData.incidentEnergyBand for s in sections]

    if logger is not None:
        logger.info(
            f"  [PFNS] MF35/MT{mt}: {len(sections)} band(s), "
            f"orders {[s.form.matrix.shape[0] for s in sections]}"
        )
        for line in report.unsupported:
            logger.info(f"  [PFNS] {line}")

    suite.covarianceSections = sections
    return suite, mf5.mt[mt], bands


def band_grids(suite) -> List[np.ndarray]:
    """The outgoing-energy boundaries of every band, in section order."""
    return [np.asarray(section.form.rowGrid, dtype=float) for section in suite]


# ---------------------------------------------------------------------------
# Diagnostics on the covariance itself
# ---------------------------------------------------------------------------

def row_sum_residual(matrix: np.ndarray) -> float:
    """``max_i |Σ_j C_ij| / max|C|`` — the test of the group-probability reading."""
    matrix = np.asarray(matrix, dtype=float)
    scale = float(np.max(np.abs(matrix))) if matrix.size else 0.0
    if scale == 0.0:
        return 0.0
    return float(np.max(np.abs(matrix.sum(axis=1))) / scale)


def normalisation_drift(matrix: np.ndarray) -> float:
    """``sqrt(|1ᵀC1|)`` — the standard deviation of a draw's sum-rule drift."""
    return float(np.sqrt(abs(float(np.asarray(matrix, dtype=float).sum()))))


def normalisation_residual(mf5_section, partial_index: int = 0) -> Dict[str, Any]:
    """The *input* evaluation's own departure from ``∫χ = 1``.

    The analogue of ``sum_rule_residual`` for nu-bar, and it exists for the same
    reason: it is how you find out the file was already off before blaming your
    own code. Measured 4.3e-7 on Cf-252 and 4.8e-8 on ENDF/B-VIII.1 U-235.
    """
    partial = mf5_section.partials[partial_index]
    residuals = np.array([
        partial.normalisation(k) - 1.0
        for k in range(len(partial.incident_energies))
    ])
    worst = int(np.argmax(np.abs(residuals))) if residuals.size else -1
    return {
        "per_node": residuals,
        "max_abs": float(np.max(np.abs(residuals))) if residuals.size else 0.0,
        "argmax_node": worst,
        "argmax_energy": (
            float(partial.incident_energies[worst]) if worst >= 0 else float("nan")
        ),
        "n_nodes": len(partial.incident_energies),
    }


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

#: How many stated standard deviations a drawn delta may reach before it is
#: treated as repair debris rather than evaluated uncertainty. See
#: :func:`generate_pfns_samples` for why this is needed and why 5.
SIGMA_CLAMP = 5.0


def generate_pfns_samples(
    suite,
    n_samples: int,
    *,
    isotope: Any = None,
    mt: int = 18,
    decomposition_method: str = "svd",
    sampling_method: str = "sobol",
    seed: Optional[int] = None,
    psd_method: str = "auto",
    null_tol: float = 1e-10,
    sigma_clamp: Optional[float] = SIGMA_CLAMP,
    verbose: bool = True,
    logger=None,
) -> Tuple[Dict[Tuple, np.ndarray], Dict[Tuple, Dict[str, Any]]]:
    """Absolute group-probability deltas per ``(isotope, mt, band)``.

    Mostly a thin adapter: unpack the suite into blocks, call the core sampler,
    hand back what it returned. Returns ``float64`` **absolute deltas** — not
    factors, and emphatically not ``float32``, because an MF35 delta stored as
    ``1 + δ`` in single precision is annihilated for all but the largest groups.

    **The one thing it adds: a clamp at** ``sigma_clamp`` **stated standard
    deviations**, and it is here rather than in the core because it is a
    PFNS-specific judgement, not a property of sampling.

    Why it is needed. These bands carry tiny negative eigenvalues
    (``|λmin|/λmax`` ≈ 1e-4), which ``psd_method="auto"`` sends to ``clip``.
    Clipping rebuilds ``V·clip(Λ,0)·Vᵀ``, which preserves the eigenvectors —
    the property that matters — but does **not** preserve the diagonal: it adds
    a little variance in directions where the file states almost none. In
    absolute terms that addition is ~1e-13 of a spectrum normalised to 1, which
    is nothing.

    It is not nothing here, because this pipeline divides each delta by its
    group probability, and a fission spectrum's lowest groups hold ~1e-17 of
    the total. Measured on Cf-252 before the clamp: groups whose stated σ/P⁰ is
    0.12 were drawing perturbation ratios of ~1e+4, putting a spike in the
    low-energy tail that MF35 never asked for.

    Why 5σ, and why this is not the outlier heuristic ``generate_samples``
    applies. That one flags a *bin* whose variance is atypical, which fires on
    a PFNS variance vector spanning 30 decades of genuine structure. This
    compares each drawn component against **its own** stated σ, so it makes no
    assumption about the shape of the variance vector at all.

    What it actually touches, measured on Cf-252 with 512 samples: **4 or 5
    groups out of 122 per band**, which between them hold ~1e-14 of the
    spectrum. Every group holding more than 1e-4 of the spectrum comes back
    with a realised standard deviation within 1 % of the stated one, clamp or
    no clamp. So this is not shaping the distribution — it is removing repair
    debris from groups that carry no mass. It is counted and reported per band
    rather than applied quietly, and ``sigma_clamp=None`` turns it off.

    ``sigma_clamp=None`` disables it. ``psd_method="higham"`` would remove the
    need for it, since Higham preserves the diagonal, but it is far too slow to
    be a default: four 122×122 Cf-252 bands did not finish in two minutes.
    """
    blocks = covariance_suite_blocks(suite, isotope=isotope, mt=mt)
    samples, diagnostics = draw_samples(
        blocks, n_samples,
        space=SAMPLING_SPACE, returns="deltas",
        decomposition_method=decomposition_method,
        sampling_method=sampling_method, seed=seed,
        psd_method=psd_method, null_tol=null_tol,
        dtype=np.float64, verbose=verbose, logger=logger,
    )

    for (key, matrix) in blocks:
        info = diagnostics[key]
        info["sigma_clamp"] = sigma_clamp
        if sigma_clamp is None:
            info["n_clamped"] = 0
            info["clamped_fraction"] = 0.0
            continue

        sigma = np.sqrt(np.clip(np.diag(matrix), 0.0, None))
        limit = float(sigma_clamp) * sigma
        drawn = samples[key]
        exceeded = np.abs(drawn) > limit
        n_exceeded = int(exceeded.sum())
        if n_exceeded:
            samples[key] = np.clip(drawn, -limit, limit)
        info["n_clamped"] = n_exceeded
        info["clamped_fraction"] = float(n_exceeded / drawn.size) if drawn.size else 0.0
        if n_exceeded and logger is not None:
            logger.info(
                f"  [PFNS] [{sigma_clamp:g}σ CLAMP] band {key[-1]}: "
                f"{n_exceeded} of {drawn.size} drawn components "
                f"({info['clamped_fraction']:.2e}) exceeded the stated marginal"
            )

    return samples, diagnostics


# ---------------------------------------------------------------------------
# Applying one sample to one MF5 partial
# ---------------------------------------------------------------------------

def realised_group_probabilities(partial: MF5PartialTabulated, k: int,
                                 boundaries: Sequence[float]) -> np.ndarray:
    """``P_j`` of the table as it now stands — used to check what was written."""
    return partial.group_integrals(k, boundaries)


def _band_of(energy: float, bands: Sequence[Tuple[float, float]]) -> Optional[int]:
    """Half-open ``[E1, E2)``, with the last band closed at the top."""
    last = len(bands) - 1
    for index, (lo, hi) in enumerate(bands):
        if lo <= energy < hi:
            return index
        if index == last and energy == hi:
            return index
    return None


def _insert_incident_shoulders(partial: MF5PartialTabulated,
                               bands: Sequence[Tuple[float, float]]) -> int:
    """Give every interior band edge a node just below it.

    The factor is piecewise constant in incident energy, so at a band edge it
    steps. Band edges are already MF5 nodes on every tape measured, but that
    only puts a node *at* the step — without one just below it, the lower band's
    factor ramps across the whole preceding incident interval.

    The new node's table comes from ``evaluate_at_incident``, which is an exact
    refinement. Copying the edge's own table instead is ~0.1 % wrong and would
    surface as a spurious central-value shift in a ``δ ≡ 0`` run — a bug that
    looks like physics.
    """
    edges = sorted({float(lo) for lo, _ in bands[1:]})
    inserted = 0
    for edge in edges:
        shoulder = edge * (1.0 - INCIDENT_STEP_SHOULDER)
        existing = np.asarray(partial.incident_energies, dtype=float)
        if existing.size == 0 or not (existing[0] < shoulder < existing[-1]):
            continue
        nearest = existing[np.argmin(np.abs(existing - shoulder))]
        if np.isclose(shoulder, nearest, rtol=ENDF_ENERGY_RTOL, atol=0.0):
            continue
        partial.insert_incident_node(shoulder)
        inserted += 1
    return inserted


def _apply_factors_to_table(
    x: np.ndarray, y: np.ndarray,
    boundaries: np.ndarray, ratios: np.ndarray,
    *, max_points: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Refine the table at stepping boundaries, then scale each node.

    Returns ``(x_new, y_new, n_inserted, n_steps_dropped)``.
    """
    lo, hi = float(x[0]), float(x[-1])

    # Boundary ``interior[i]`` separates groups i and i+1, so the factor steps
    # there iff ``ratios[i+1] != ratios[i]``. Only steps strictly inside the
    # table's own support can be represented at all.
    interior = boundaries[1:-1]
    widths = np.diff(boundaries)[:-1]            # width of the group *below* each
    jumps = np.abs(ratios[1:] - ratios[:-1])
    stepping = (jumps != 0.0) & (interior > lo) & (interior < hi)
    candidates = interior[stepping]
    magnitudes = jumps[stepping]
    shoulder_widths = widths[stepping]

    n_dropped = 0
    if max_points is not None and candidates.size:
        # Two nodes per surviving step. Drop the smallest steps first, and log
        # how many — a silent cap reads as "the factor was applied everywhere",
        # which is exactly the claim that would then be false.
        budget = max(int((max_points - x.size) // 2), 0)
        if budget < candidates.size:
            keep = np.argsort(magnitudes)[::-1][:budget]
            n_dropped = int(candidates.size - keep.size)
            order = np.sort(keep)
            candidates = candidates[order]
            shoulder_widths = shoulder_widths[order]

    wanted: List[float] = []
    for edge, width in zip(candidates, shoulder_widths):
        wanted.append(float(edge))
        shoulder = float(edge) - OUTGOING_STEP_SHOULDER * float(width)
        if shoulder > lo:
            wanted.append(shoulder)

    n_inserted = 0
    if wanted:
        new = np.unique(np.asarray(wanted, dtype=float))
        new = new[(new > lo) & (new < hi)]
        if new.size:
            nearest = x[np.clip(np.searchsorted(x, new), 1, x.size - 1)]
            below = x[np.clip(np.searchsorted(x, new) - 1, 0, x.size - 1)]
            duplicate = (np.isclose(new, nearest, rtol=ENDF_ENERGY_RTOL, atol=0.0)
                         | np.isclose(new, below, rtol=ENDF_ENERGY_RTOL, atol=0.0))
            new = new[~duplicate]
        if new.size:
            values = np.interp(new, x, y)
            x = np.concatenate([x, new])
            y = np.concatenate([y, values])
            order = np.argsort(x, kind="stable")
            x, y = x[order], y[order]
            n_inserted = int(new.size)

    # Group of each node: side='right' puts a node sitting exactly on a
    # boundary into the group *above* it, and the shoulder just below into the
    # group beneath — which is the assignment the step construction needs.
    group = np.clip(np.searchsorted(boundaries, x, side="right") - 1,
                    0, ratios.size - 1)
    inside = (x >= boundaries[0]) & (x <= boundaries[-1])
    scale = np.where(inside, ratios[group], 1.0)
    return x, y * scale, n_inserted, n_dropped


def perturb_pfns_partial(
    partial: MF5PartialTabulated,
    deltas_by_band: Dict[int, np.ndarray],
    bands: Sequence[Tuple[float, float]],
    boundaries_by_band: Sequence[np.ndarray],
    *,
    metric: str = "probability",
    max_outgoing_points: Optional[int] = None,
    logger=None,
) -> Tuple[MF5PartialTabulated, Dict[str, Any]]:
    """Apply one sample's deltas to every incident node of *partial*, in place.

    *deltas_by_band* maps band index → absolute group-probability deltas for
    that band, one entry per group of ``boundaries_by_band[band]``.
    """
    inserted_incident = _insert_incident_shoulders(partial, bands)

    per_node: List[Dict[str, Any]] = []
    for k, energy in enumerate(list(partial.incident_energies)):
        band = _band_of(float(energy), bands)
        if band is None or band not in deltas_by_band:
            continue

        boundaries = np.asarray(boundaries_by_band[band], dtype=float)
        deltas = np.asarray(deltas_by_band[band], dtype=float)

        p0 = partial.group_integrals(k, boundaries)
        if p0.size != deltas.size:
            raise ValueError(
                f"band {band}: {deltas.size} deltas for {p0.size} groups"
            )

        total = partial.normalisation(k)
        covered = float(p0.sum())

        # Frozen groups first, so the projection re-closes the sum that
        # dropping their deltas re-opens.
        empty = p0 == 0.0
        n_frozen = int(empty.sum())
        deltas = np.where(empty, 0.0, deltas)

        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = np.where(empty, 1.0, 1.0 + deltas / np.where(empty, 1.0, p0))

        raw_ratios = ratios
        before = check_ratios(raw_ratios, p0, covered)
        ratios, shift, n_clipped = project_ratios_to_simplex(
            raw_ratios, p0, covered, metric=metric,
        )
        after = check_ratios(ratios, p0, covered)

        # The honest measure of what clipping cost is the *mass* it moved, not
        # the number of groups: zeroing thirty groups that hold 1e-15 of the
        # spectrum between them is not the same event as zeroing one that holds
        # a per cent, and a group count cannot tell them apart.
        clipped = (raw_ratios + shift < 0.0) & (p0 > 0.0)
        clipped_mass = (
            float(p0[clipped].sum() / covered) if covered and clipped.any() else 0.0
        )

        x, y = partial.table(k)
        x_new, y_new, n_inserted, n_dropped = _apply_factors_to_table(
            x, y, boundaries, ratios, max_points=max_outgoing_points,
        )
        partial.replace_table(k, x_new, y_new, [(len(x_new), 2)])

        realised = partial.normalisation(k)
        rescale = (total / realised) if realised != 0.0 else 1.0
        if rescale != 1.0:
            partial.replace_table(
                k, x_new, y_new * rescale, [(len(x_new), 2)],
            )

        # Did step 6 actually deliver ``P_j = rescale · r*_j · P⁰_j``?
        #
        # This is the self-check for the failure the module docstring names:
        # scaling nodes without inserting the stepping boundary applies a
        # factor over the wrong interval, the realised group integrals stop
        # being what was computed, and step 7 hides the difference in a scalar.
        # Nothing outside this function can see that, so it is measured here.
        # The residual that legitimately remains is the shoulder ramp — the
        # sliver between ``g_j(1-s)`` and ``g_j`` carries a blend of two
        # factors — and it is bounded by ~½·s·max|Δr|.
        wanted = rescale * ratios * p0
        realised_groups = partial.group_integrals(k, boundaries)
        scored = wanted > 0.0
        group_error = (
            float(np.max(np.abs(realised_groups[scored] / wanted[scored] - 1.0)))
            if np.any(scored) else 0.0
        )
        # The same discrepancy as a fraction of the whole spectrum. **This is
        # the one to gate on.** The relative form above is unbounded wherever
        # the projection drove a group's ratio towards zero — its denominator
        # goes to nothing while the neighbouring shoulder ramp does not — so it
        # reports percent-level numbers on groups holding 1e-5 of the spectrum
        # and says nothing about whether the perturbation is right. Measured:
        # relative ≤1.6e-2, mass-weighted ≤8.4e-7 on Cf-252.
        group_mass_error = (
            float(np.max(np.abs(realised_groups - wanted)) / covered)
            if covered else 0.0
        )

        per_node.append({
            "max_group_ratio_error": group_error,
            "max_group_mass_error": group_mass_error,
            "node": k,
            "incident_energy": float(energy),
            "band": band,
            "integral_before": total,
            "covered_before": covered,
            "mass_outside_mf35": float(total - covered),
            "frac_mass_outside_mf35": float((total - covered) / total) if total else 0.0,
            "n_groups_frozen": n_frozen,
            "sum_error_before_projection": before["sum_error"],
            "sum_error_after_projection": after["sum_error"],
            "projection_shift": shift,
            "n_clipped": n_clipped,
            "clipped_mass_fraction": clipped_mass,
            "renormalisation_scalar": rescale,
            "renormalisation_error": abs(rescale - 1.0),
            "n_outgoing_inserted": n_inserted,
            "n_steps_dropped": n_dropped,
            "n_outgoing_points": int(len(x_new)),
        })

    diagnostics = _summarise_nodes(per_node)
    diagnostics["n_incident_inserted"] = inserted_incident
    diagnostics["per_node"] = per_node

    if logger is not None and per_node:
        logger.info(
            f"  [PFNS] {len(per_node)} incident node(s) perturbed; "
            f"max |1 - N/N'| = {diagnostics['max_renormalisation_error']:.3e}, "
            f"max |t| = {diagnostics['max_projection_shift']:.3e}, "
            f"clipped mass = {diagnostics['max_clipped_mass_fraction']:.3e}"
        )
        if diagnostics["total_steps_dropped"]:
            logger.info(
                f"  [PFNS] [CAP] {diagnostics['total_steps_dropped']} factor "
                f"step(s) dropped to stay inside max_outgoing_points"
            )

    return partial, diagnostics


def _summarise_nodes(per_node: List[Dict[str, Any]]) -> Dict[str, Any]:
    def collect(field: str) -> np.ndarray:
        return np.array([abs(entry[field]) for entry in per_node], dtype=float)

    if not per_node:
        return {
            "n_nodes_perturbed": 0,
            "max_renormalisation_error": 0.0,
            "mean_renormalisation_error": 0.0,
            "max_projection_shift": 0.0,
            "mean_projection_shift": 0.0,
            "max_sum_error_before_projection": 0.0,
            "max_sum_error_after_projection": 0.0,
            "total_clipped": 0,
            "max_clipped_mass_fraction": 0.0,
            "total_groups_frozen": 0,
            "total_outgoing_inserted": 0,
            "total_steps_dropped": 0,
            "max_frac_mass_outside_mf35": 0.0,
            "max_group_ratio_error": 0.0,
            "max_group_mass_error": 0.0,
        }

    return {
        "n_nodes_perturbed": len(per_node),
        "max_renormalisation_error": float(collect("renormalisation_error").max()),
        "mean_renormalisation_error": float(collect("renormalisation_error").mean()),
        "max_projection_shift": float(collect("projection_shift").max()),
        "mean_projection_shift": float(collect("projection_shift").mean()),
        "max_sum_error_before_projection": float(
            collect("sum_error_before_projection").max()
        ),
        "max_sum_error_after_projection": float(
            collect("sum_error_after_projection").max()
        ),
        "total_clipped": int(sum(e["n_clipped"] for e in per_node)),
        "max_clipped_mass_fraction": float(collect("clipped_mass_fraction").max()),
        "total_groups_frozen": int(sum(e["n_groups_frozen"] for e in per_node)),
        "total_outgoing_inserted": int(sum(e["n_outgoing_inserted"] for e in per_node)),
        "total_steps_dropped": int(sum(e["n_steps_dropped"] for e in per_node)),
        "max_frac_mass_outside_mf35": float(collect("frac_mass_outside_mf35").max()),
        "max_group_ratio_error": float(collect("max_group_ratio_error").max()),
        "max_group_mass_error": float(collect("max_group_mass_error").max()),
    }


# ---------------------------------------------------------------------------
# The acceptance gate
# ---------------------------------------------------------------------------

def realised_covariance_report(
    deltas: np.ndarray,
    matrix: np.ndarray,
    *,
    null_tol: float = 1e-10,
) -> Dict[str, Any]:
    """Does the realised ensemble reproduce MF35 where MF35 says anything?

    With two thirds of a band null, an elementwise comparison of ``cov(Δ)`` to
    ``C`` is meaningless: the null directions contribute to a norm over the
    whole matrix but carry no information, so a bad ensemble can score well.
    Restricting to the retained eigenvectors measures what was sampled.

    Two numbers matter.

    ``spectral_fidelity_median`` — the median over retained modes of
    ``|Var_s(a_k)/λ_k − 1|``, where ``a_k = v_kᵀΔ``. Gate it against
    ``3·sqrt(2/n_samples)``, the 3σ band of a χ²ₙ₋₁ variance estimate. The
    **median**, not the max: over hundreds of modes the max is dominated by
    sampling noise on the smallest retained λ and says nothing.

    ``null_leakage`` — ``‖Δ − V_K V_Kᵀ Δ‖ / ‖Δ‖``. This is what catches the
    projection or the shoulder ramp having pushed the ensemble somewhere MF35
    never authorised.
    """
    deltas = np.asarray(deltas, dtype=float)
    matrix = np.asarray(matrix, dtype=float)

    values, vectors = np.linalg.eigh(matrix)
    largest = float(values.max()) if values.size else 0.0
    keep = values > null_tol * largest if largest > 0 else np.zeros_like(values, bool)
    n_keep = int(keep.sum())

    if n_keep == 0 or deltas.shape[0] < 2:
        return {
            "n_modes_retained": n_keep,
            "n_modes_null": int(values.size - n_keep),
            "spectral_fidelity_median": float("nan"),
            "spectral_fidelity_p05": float("nan"),
            "spectral_fidelity_p95": float("nan"),
            "gate_tolerance": float("nan"),
            "passes_spectral_gate": False,
            "null_leakage": float("nan"),
        }

    basis = vectors[:, keep]
    amplitudes = deltas @ basis
    realised = amplitudes.var(axis=0, ddof=1)
    ratio = realised / values[keep]

    residual = deltas - (amplitudes @ basis.T)
    norm = float(np.linalg.norm(deltas))
    leakage = float(np.linalg.norm(residual) / norm) if norm > 0 else 0.0

    tolerance = 3.0 * np.sqrt(2.0 / deltas.shape[0])
    median = float(np.median(np.abs(ratio - 1.0)))

    return {
        "n_modes_retained": n_keep,
        "n_modes_null": int(values.size - n_keep),
        "spectral_fidelity_median": median,
        "spectral_fidelity_p05": float(np.quantile(ratio, 0.05)),
        "spectral_fidelity_p50": float(np.quantile(ratio, 0.50)),
        "spectral_fidelity_p95": float(np.quantile(ratio, 0.95)),
        "gate_tolerance": float(tolerance),
        "passes_spectral_gate": bool(median <= tolerance),
        "null_leakage": leakage,
        "max_abs_delta_sum": float(np.max(np.abs(deltas.sum(axis=1)))),
    }
