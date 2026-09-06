"""
SINBAD shielding benchmarks as objects.

The design follows the rest of kika: typed objects with a ``to_dataframe()``
method, so the explicit path and the quick path both exist. The quick path
matters most here -- a user who wants the measured data of a benchmark should
not have to learn a document model to get at it.

Example
-------
    >>> import kika.sinbad as sinbad
    >>> b = sinbad.open("SINBAD-ASPIS-IRON88")
    >>> b.to_dataframe()
    >>> b.to_dataframe("Al27")          # one foil
    >>> b.ce(wide=True)

Notes
-----
Scalar and descriptive content -- measurements, uncertainties, calculations,
C/E -- is always readable. Only arrays (spectra, sensitivity coefficients,
covariance matrices) live in a binary sidecar and need ``h5py`` installed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from kika.sinbad._constants import GENERATED_DESCRIPTION
from kika.sinbad.exceptions import ArrayBackendMissingError
from kika.sinbad.package import SinbadPackage

__all__ = [
    "SinbadBenchmark",
    "Measurement",
    "MeasurementSystem",
    "Calculation",
    "CalculationInput",
    "SensitivitySet",
]

#: Correlation scopes that generate off-diagonal covariance. Anything else --
#: ``none``, ``unknown``, ``partiallyCorrelated`` -- contributes to the diagonal
#: only. The reader never invents a correlation and never drops a declared one.
_CORRELATING = (
    "fullWithinEntry",
    "fullWithinMeasurementSystem",
    "fullWithinRepository",
)


# ---------------------------------------------------------------------------
# leaf objects
# ---------------------------------------------------------------------------


@dataclass
class MeasurementSystem:
    """
    One detector system of a benchmark -- here, one activation foil.

    A benchmark entry usually holds several. They share the assembly and the
    source but not the reaction, the energy response, or the calibration
    uncertainty, so they are separate objects rather than a column.

    Attributes
    ----------
    id : str
        Identifier within the entry, e.g. ``"FOIL-AL27"``.
    target_nuclide : str
        The nuclide the reaction acts on, e.g. ``"Al27"``.
    reaction : str
        Full reaction string, e.g. ``"Al27(n,alpha)Na24"``.
    threshold_mev : float or None
        Approximate energy above which the reaction responds. ``None`` for a
        capture reaction, which has no threshold.
    dosimetry_evaluation : str
        The activation cross-section evaluation the calculations used. A
        property of the analysis, not of the experiment.
    """

    id: str
    target_nuclide: str
    reaction: str
    product_nuclide: Optional[str] = None
    threshold_mev: Optional[float] = None
    dosimetry_evaluation: Optional[str] = None
    foil: dict = field(default_factory=dict, repr=False)

    def __repr__(self) -> str:
        thr = f", >{self.threshold_mev} MeV" if self.threshold_mev else ""
        return f"<MeasurementSystem {self.id}: {self.reaction}{thr}>"


@dataclass
class Measurement:
    """
    One measured point of a benchmark.

    Attributes
    ----------
    id : str
        Identifier within the entry, e.g. ``"EXP-AL-A3"``.
    system : str
        Identifier of the :class:`MeasurementSystem` this point belongs to.
    position : str
        Detector position label, e.g. ``"A3"``.
    depth_cm : float
        Axial depth of the position.
    value : float
        The reported value, in :attr:`unit`.
    unit : str
        Unit of :attr:`value`.
    rel_unc : float
        Total relative uncertainty, 1-sigma.
    components : list of dict
        The uncertainty broken down by cause, each with its correlation scope
        and whether it was reported or reconstructed. This is the part that
        turns a column of numbers into a covariance matrix.
    missing : list of dict
        What the entry explicitly records as not known about this point.
    """

    id: str
    system: str
    position: str
    depth_cm: float
    value: float
    unit: str
    rel_unc: float
    components: list = field(default_factory=list, repr=False)
    reported_value: Optional[float] = None
    missing: list = field(default_factory=list, repr=False)

    @property
    def abs_unc(self) -> float:
        """Absolute uncertainty, ``value * rel_unc``."""
        return self.value * self.rel_unc

    @property
    def correlated_rel_unc(self) -> float:
        """
        The part of the uncertainty that survives averaging.

        Components declared correlated across a foil or across the entry do not
        shrink with the number of points. This is the number that decides what
        an aggregate is worth.
        """
        parts = [
            c["relative"] / c["coverageFactor"]
            for c in self.components
            if c["correlationScope"]["kind"] in _CORRELATING
        ]
        return float(np.sqrt(sum(p * p for p in parts)))

    @property
    def independent_rel_unc(self) -> float:
        """The part of the uncertainty that does average down."""
        parts = [
            c["relative"] / c["coverageFactor"]
            for c in self.components
            if c["correlationScope"]["kind"] not in _CORRELATING
        ]
        return float(np.sqrt(sum(p * p for p in parts)))

    def __repr__(self) -> str:
        return (
            f"<Measurement {self.id}: {self.value:.4e} {self.unit} "
            f"+/- {self.rel_unc * 100:.1f}% at {self.depth_cm} cm>"
        )


@dataclass
class CalculationInput:
    """
    One file a calculation needed in order to run.

    Attributes
    ----------
    role : str
        ``transportInput``, ``weightWindow``, ``weightWindowGenerator``,
        ``crossSectionDirectory``, ``tallySpecification`` or
        ``calculationOutput``.
    applies_to : str or None
        Measurement this input is specific to, when variance reduction was
        generated per detector position.
    available : bool
        Whether the file is obtainable from the package or its recorded path.
        ``False`` is a legitimate and common answer.
    sha256 : str or None
        Digest of the referenced file. A path is a statement about one machine;
        a digest is a statement about the file.
    derived_from : str or None
        Identifier of the artifact this one is a modified copy of.
    modification : str or None
        What was changed relative to ``derived_from``.
    """

    role: str
    artifact: str
    format: str
    available: bool
    applies_to: Optional[str] = None
    sha256: Optional[str] = None
    derived_from: Optional[str] = None
    modification: Optional[str] = None
    note: str = ""

    def __repr__(self) -> str:
        mark = "available" if self.available else "MISSING"
        scope = f" for {self.applies_to}" if self.applies_to else ""
        return f"<CalculationInput {self.role}{scope}: {self.artifact} [{mark}]>"


@dataclass
class Calculation:
    """
    One transport calculation of the benchmark.

    Attributes
    ----------
    library : str
        Nuclear data library used, e.g. ``"ENDF/B-VIII.1"``.
    library_selector : str or None
        The library as the code actually resolved it -- the cross-section
        directory name. ``"JEFF-4.0"`` is a claim; this is evidence.
    system : str or None
        Which measurement system this run computed.
    variance_reduction : dict
        Method, generator and scope. In a deep-penetration benchmark this is
        part of the result, not metadata about it.
    inputs : list of CalculationInput
        Every file this run needed.
    results : list of dict
        Calculated values, each referencing the measurement it corresponds to.
    """

    id: str
    code: str
    code_version: str
    library: str
    variance_reduction: dict
    inputs: list
    results: list
    system: Optional[str] = None
    library_selector: Optional[str] = None
    histories: Optional[str] = None

    @property
    def reproducible(self) -> bool:
        """True when every declared input is actually obtainable."""
        return bool(self.inputs) and all(i.available for i in self.inputs)

    def __repr__(self) -> str:
        got = sum(i.available for i in self.inputs)
        return (
            f"<Calculation {self.id}: {self.code} {self.code_version}, "
            f"{self.library}, {len(self.results)} results, "
            f"{got}/{len(self.inputs)} inputs available>"
        )


@dataclass
class SensitivitySet:
    """
    A set of sensitivity coefficients, with its array loaded on demand.

    Attributes
    ----------
    parameter : dict
        Structured parameter identity -- nuclide, reactions, and the optional
        Legendre-order and secondary-energy slots.
    convention : str
        Whether the coefficients are relative-relative, per group or per unit
        lethargy, and so on. Reading this before using the numbers is not
        optional.
    data_origin : str
        ``syntheticDemo`` marks a placeholder that must never be used as physics.
    system : str or None
        Measurement system the profile belongs to, so a set can be found by
        foil rather than by remembering its identifier.
    position : str or None
        Detector position label, e.g. ``"A7"``.
    measurement : str or None
        The measurement this profile is the sensitivity *of*. ``None`` when the
        profile was computed at a position the measurement set does not carry --
        which happens, and is worth being able to see.
    nuclear_data_library : str or None
        The library the perturbation run used. A profile belongs to one library
        and comparing it against another one's C/E is a category error.
    interchange : dict
        The same coefficients in a community format (SCALE SDF here), by
        reference and checksum. Never the normative copy.
    """

    id: str
    parameter: dict
    convention: str
    method: str
    includes_implicit_effects: bool
    data_origin: str
    system: Optional[str] = None
    position: Optional[str] = None
    measurement: Optional[str] = None
    nuclear_data_library: Optional[str] = None
    response_value: Optional[float] = None
    response_rel_unc: Optional[float] = None
    interchange: dict = field(default_factory=dict, repr=False)
    _benchmark: "SinbadBenchmark" = field(repr=False, default=None)
    _coefficients_ref: str = field(repr=False, default=None)
    _uncertainties_ref: str = field(repr=False, default=None)
    _group_ref: str = field(repr=False, default=None)

    @property
    def target_nuclide(self) -> str:
        return self.parameter["targetNuclide"]

    @property
    def reactions(self) -> list:
        """Reaction labels, one per row of :attr:`coefficients`."""
        names = self.parameter.get("reactionNames")
        if names:
            return names
        return [f"MT{mt}" for mt in self.parameter["reactions"]]

    @property
    def mts(self) -> list:
        """MT numbers, one per row of :attr:`coefficients`."""
        return list(self.parameter["reactions"])

    @property
    def coefficients(self) -> np.ndarray:
        """Coefficient array, shape ``(len(reactions), n_groups)``."""
        return self._benchmark.array(self._coefficients_ref)

    @property
    def relative_uncertainties(self) -> Optional[np.ndarray]:
        """Per-group relative uncertainties, same shape as the coefficients."""
        if not self._uncertainties_ref:
            return None
        return self._benchmark.array(self._uncertainties_ref)

    @property
    def group_boundaries(self) -> np.ndarray:
        """Energy group boundaries, ascending."""
        return self._benchmark.array(self._group_ref)

    @property
    def integral(self) -> pd.Series:
        """Energy-integrated sensitivity per reaction."""
        return pd.Series(
            np.atleast_2d(self.coefficients).sum(axis=1),
            index=self.reactions, name=self.id,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """
        Return the coefficients as a tidy DataFrame.

        Returns
        -------
        pandas.DataFrame
            Columns ``reaction``, ``mt``, ``group``, ``e_low``, ``e_high``,
            ``e_mid``, ``sensitivity`` and, when present,
            ``relativeUncertainty``.
        """
        edges = self.group_boundaries
        coeffs = np.atleast_2d(self.coefficients)
        unc = self.relative_uncertainties
        unc = np.atleast_2d(unc) if unc is not None else None
        rows = []
        for r, (reaction, mt) in enumerate(zip(self.reactions, self.mts)):
            for g in range(coeffs.shape[1]):
                row = {
                    "reaction": reaction,
                    "mt": mt,
                    "group": g,
                    "e_low": edges[g],
                    "e_high": edges[g + 1],
                    "e_mid": float(np.sqrt(edges[g] * edges[g + 1])),
                    "sensitivity": coeffs[r, g],
                }
                if unc is not None:
                    row["relativeUncertainty"] = unc[r, g]
                rows.append(row)
        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        warn = " SYNTHETIC" if self.data_origin == "syntheticDemo" else ""
        where = f" {self.position}" if self.position else ""
        return (
            f"<SensitivitySet {self.id}:{where} {self.target_nuclide} "
            f"{self.reactions}{warn}>"
        )


# ---------------------------------------------------------------------------
# the benchmark
# ---------------------------------------------------------------------------


class SinbadBenchmark:
    """
    One SINBAD shielding benchmark entry.

    Open a package by path with :meth:`from_path`, or by identifier out of the
    configured library with :meth:`open`. Both package forms -- a directory or
    a ``.sinbad`` archive -- are accepted and behave identically.

    An entry usually holds several measurement systems (for ASPIS Iron-88, five
    activation foils). Every table method takes an optional ``system`` argument
    that accepts an identifier, a target nuclide, or any unambiguous fragment,
    so ``b.ce(system="Al27")`` works without looking anything up.

    Parameters
    ----------
    path : str or pathlib.Path
        A package directory or ``.sinbad`` archive.

    Examples
    --------
    >>> import kika.sinbad as sinbad
    >>> b = sinbad.open("SINBAD-ASPIS-IRON88")
    >>> print(b.summary())
    >>> b.to_dataframe()
    >>> b.ce(wide=True)
    >>> b.covariance("Al27")
    """

    def __init__(self, path: Union[str, Path]):
        self.package = SinbadPackage(path)
        self._m = json.loads(self.package.read_text(GENERATED_DESCRIPTION))
        self._arrays = {}

    # -- constructors -----------------------------------------------------

    @classmethod
    def from_path(cls, path: Union[str, Path]) -> "SinbadBenchmark":
        """Open a package from an explicit path."""
        return cls(path)

    @classmethod
    def open(
        cls, identifier: Union[str, Path], path: Optional[str] = None
    ) -> "SinbadBenchmark":
        """
        Open a benchmark by identifier from the configured library.

        A path that exists is opened directly, so this also works as a
        general-purpose entry point.

        Parameters
        ----------
        identifier : str or pathlib.Path
            Benchmark identifier (e.g. ``"SINBAD-ASPIS-IRON88"``), or a path to
            a package.
        path : str, optional
            Library directory to search. Defaults to the configured one.

        Returns
        -------
        SinbadBenchmark
        """
        from kika.sinbad.library import find_package  # noqa: PLC0415

        candidate = Path(str(identifier)).expanduser()
        if candidate.exists():
            return cls(candidate)
        return cls(find_package(str(identifier), path=path))

    # -- identity ---------------------------------------------------------

    @property
    def id(self) -> str:
        """Benchmark identifier."""
        return self._m["identification"]["id"]

    @property
    def title(self) -> str:
        """Human-readable title."""
        return self._m["identification"]["title"]

    @property
    def facility(self) -> str:
        return self._m["identification"]["facility"]

    @property
    def year(self):
        return self._m["identification"]["year"]

    @property
    def model(self) -> dict:
        """The whole underlying model. Nothing is hidden behind the accessors."""
        return self._m

    # -- measurement systems ----------------------------------------------

    @property
    def systems(self) -> list:
        """List of :class:`MeasurementSystem` -- the foils of this entry."""
        return [
            MeasurementSystem(
                id=s["id"],
                target_nuclide=s["targetNuclide"],
                reaction=s["reaction"],
                product_nuclide=s.get("productNuclide"),
                threshold_mev=s.get("effectiveThreshold_MeV"),
                dosimetry_evaluation=s.get("dosimetryEvaluation"),
                foil=s.get("foil", {}),
            )
            for s in self._m["experiment"]["measurementSystems"]
        ]

    def system(self, key: str) -> MeasurementSystem:
        """Return one :class:`MeasurementSystem` by identifier or fragment."""
        resolved = self._resolve_system(key)
        return next(s for s in self.systems if s.id == resolved)

    def _resolve_system(self, key: Optional[str]) -> Optional[str]:
        """Map a user-supplied fragment onto a measurement system identifier."""
        if key is None:
            return None
        raw = self._m["experiment"]["measurementSystems"]
        ids = [s["id"] for s in raw]
        if key in ids:
            return key
        nuclides = {s["targetNuclide"].upper(): s["id"] for s in raw}
        if key.upper() in nuclides:
            return nuclides[key.upper()]
        hits = [i for i in ids if key.upper() in i.upper()]
        if len(hits) == 1:
            return hits[0]
        raise KeyError(
            f"'{key}' matches {hits or 'no measurement system'}. "
            f"Available: {ids} (or a target nuclide: {sorted(nuclides)})"
        )

    @property
    def observables(self) -> list:
        """What was measured, one entry per measurement system."""
        return self._m["experiment"]["observables"]

    def observable(self, system: Optional[str] = None) -> dict:
        """
        The observable of one measurement system.

        ``unit_confidence == "derived"`` means the unit was reconstructed rather
        than read from a labelled field. Worth checking before using the values.
        """
        obs = self.observables
        if system is None:
            return obs[0]
        key = self._resolve_system(system)
        return next(o for o in obs if o.get("systemRef") == key)

    # -- content ----------------------------------------------------------

    def measurements(self, system: Optional[str] = None) -> list:
        """
        List of :class:`Measurement`, optionally for one measurement system.

        Parameters
        ----------
        system : str, optional
            System identifier (``"FOIL-AL27"``), target nuclide (``"Al27"``) or
            any unambiguous fragment.
        """
        key = self._resolve_system(system)
        out = []
        for m in self._m["experiment"]["measurements"]:
            if key is not None and m["systemRef"] != key:
                continue
            comps = m["uncertainty"]["components"]
            total = m["uncertainty"].get("reportedTotal")
            if total is None:
                total = float(
                    np.sqrt(
                        sum(
                            (c["relative"] / c["coverageFactor"]) ** 2 for c in comps
                        )
                    )
                )
            out.append(
                Measurement(
                    id=m["id"],
                    system=m["systemRef"],
                    position=m["position"]["label"],
                    depth_cm=m["position"]["axialDepth_cm"],
                    value=m["value"],
                    unit=m["unit"],
                    rel_unc=total,
                    components=comps,
                    reported_value=m.get("reportedValue"),
                    missing=m["uncertainty"]["missing"],
                )
            )
        return out

    @property
    def calculations(self) -> list:
        """List of :class:`Calculation`."""
        arts = {a["id"]: a for a in self.artifacts}
        out = []
        for c in self._m["calculations"]:
            inputs = []
            for i in c.get("inputs", []):
                a = arts[i["artifactRef"]]
                inputs.append(
                    CalculationInput(
                        role=i["role"],
                        artifact=a["id"],
                        format=a["format"],
                        available=bool(a["includedInPackage"] or a["path"]),
                        applies_to=i.get("appliesTo"),
                        sha256=a.get("sha256"),
                        derived_from=a.get("derivedFrom"),
                        modification=a.get("modification"),
                        note=a["reason"],
                    )
                )
            out.append(
                Calculation(
                    id=c["id"],
                    code=c["code"],
                    code_version=c["codeVersion"],
                    library=c["nuclearDataLibrary"],
                    variance_reduction=c["varianceReduction"],
                    inputs=inputs,
                    results=c["results"],
                    system=c.get("systemRef"),
                    library_selector=c.get("librarySelector"),
                    histories=c.get("historiesRequested"),
                )
            )
        return out

    def sensitivities(self, system: Optional[str] = None) -> list:
        """
        Sensitivity sets, optionally restricted to one measurement system.

        Parameters
        ----------
        system : str, optional
            Measurement system id, target nuclide, or an unambiguous fragment
            of either -- resolved the same way as everywhere else.

        Returns
        -------
        list of SensitivitySet
        """
        sid = self._resolve_system(system) if system else None
        return [
            SensitivitySet(
                id=s["id"],
                parameter=s["parameter"],
                convention=s["convention"],
                method=s["method"],
                includes_implicit_effects=s["includesImplicitEffects"],
                data_origin=s["dataOrigin"],
                system=s.get("systemRef"),
                position=s.get("position"),
                measurement=s.get("measurementRef"),
                nuclear_data_library=s.get("nuclearDataLibrary"),
                response_value=s.get("responseValue"),
                response_rel_unc=s.get("responseRelUnc"),
                interchange=s.get("interchange") or {},
                _benchmark=self,
                _coefficients_ref=s["coefficientsRef"],
                _uncertainties_ref=s.get("uncertaintiesRef"),
                _group_ref=s["groupStructureRef"],
            )
            for s in self._m["sensitivities"]
            if sid is None or s.get("systemRef") == sid
        ]

    def sensitivity_profile(self, system: str, position: str) -> SensitivitySet:
        """
        The one sensitivity set at a given system and position.

        Raises
        ------
        KeyError
            If no set matches, listing the positions that do exist.
        """
        sets = self.sensitivities(system)
        for s in sets:
            if s.position == position:
                return s
        raise KeyError(
            f"no sensitivity profile at position {position!r}; "
            f"available: {[s.position for s in sets]}"
        )

    def sensitivity_summary(self, system: Optional[str] = None) -> pd.DataFrame:
        """
        Energy-integrated sensitivity per position and reaction.

        The whole set of profiles reduced to one number each -- how the
        response at a depth responds to a reaction overall. Wide, indexed by
        position, one column per reaction.

        Returns
        -------
        pandas.DataFrame
        """
        rows = []
        for s in self.sensitivities(system):
            row = {"system": s.system, "position": s.position,
                   "measurement": s.measurement}
            row.update(s.integral.to_dict())
            rows.append(row)
        df = pd.DataFrame(rows)
        return df.set_index(["system", "position"]) if not df.empty else df

    @property
    def artifacts(self) -> list:
        """External files the entry references, included or not."""
        return self._m["artifacts"]

    @property
    def libraries(self) -> list:
        """Nuclear data libraries this entry has calculations for."""
        return sorted({c["nuclearDataLibrary"] for c in self._m["calculations"]})

    def nuclides(self) -> set:
        """Every nuclide the benchmark mentions, whatever role it plays."""
        found = set()
        for m in self._m["experiment"]["materials"]:
            found |= {n["id"] for n in m["nuclides"]}
        for s in self._m["sensitivities"]:
            found.add(s["parameter"]["targetNuclide"])
        for ms in self._m["experiment"]["measurementSystems"]:
            found.add(ms["targetNuclide"])
        return found

    def has_sensitivity_to(self, nuclide: str) -> bool:
        """
        Whether the entry carries a sensitivity profile for a nuclide.

        Distinct from ``nuclide in self.nuclides()``: containing a nuclide is a
        material fact, being sensitive to it is a result.
        """
        return any(s.target_nuclide == nuclide for s in self.sensitivities())

    # -- arrays -----------------------------------------------------------

    def array(self, ref: str) -> np.ndarray:
        """
        Resolve a ``file#/path/inside`` reference to the array itself.

        Raises
        ------
        ArrayBackendMissingError
            If the payload is HDF5 and ``h5py`` is not installed.
        """
        fname, inner = ref.split("#/", 1)
        if fname not in self._arrays:
            buf = self.package.open(fname)
            if fname.endswith((".h5", ".hdf5")):
                try:
                    import h5py  # noqa: PLC0415
                except ImportError as exc:
                    raise ArrayBackendMissingError(
                        f"'{fname}' is HDF5 and h5py is not installed. "
                        "Install it with 'pip install h5py'. Everything except "
                        "arrays -- measurements, uncertainties, calculations, "
                        "C/E -- is readable without it."
                    ) from exc
                self._arrays[fname] = h5py.File(buf, "r")
            else:
                self._arrays[fname] = np.load(buf, allow_pickle=False)
        return np.asarray(self._arrays[fname][inner])

    def sensitivity(self, sensitivity_id: str) -> SensitivitySet:
        """Return one :class:`SensitivitySet` by identifier."""
        for s in self.sensitivities():
            if s.id == sensitivity_id:
                return s
        raise KeyError(
            f"No sensitivity set '{sensitivity_id}'. "
            f"Available: {len(self._m['sensitivities'])} sets, e.g. "
            f"{[s['id'] for s in self._m['sensitivities'][:3]]}"
        )

    # -- tables -----------------------------------------------------------

    def to_dataframe(self, system: Optional[str] = None) -> pd.DataFrame:
        """
        Return the experimental data as a DataFrame.

        The quick path: this is what most users want from a benchmark.

        Parameters
        ----------
        system : str, optional
            Restrict to one measurement system.

        Returns
        -------
        pandas.DataFrame
            Columns ``id``, ``system``, ``nuclide``, ``position``, ``depth_cm``,
            ``value``, ``unit``, ``rel_unc``, ``abs_unc``, ``correlated_unc``,
            ``independent_unc``.
        """
        nuc = {s.id: s.target_nuclide for s in self.systems}
        return pd.DataFrame(
            [
                {
                    "id": m.id,
                    "system": m.system,
                    "nuclide": nuc.get(m.system),
                    "position": m.position,
                    "depth_cm": m.depth_cm,
                    "value": m.value,
                    "unit": m.unit,
                    "rel_unc": m.rel_unc,
                    "abs_unc": m.abs_unc,
                    "correlated_unc": m.correlated_rel_unc,
                    "independent_unc": m.independent_rel_unc,
                }
                for m in self.measurements(system)
            ]
        )

    #: Explicit alias of :meth:`to_dataframe`, for readable scripts.
    experimental_data = to_dataframe

    def ce(
        self,
        library: Optional[str] = None,
        system: Optional[str] = None,
        wide: bool = False,
    ) -> pd.DataFrame:
        """
        Return C, E and C/E.

        Parameters
        ----------
        library : str, optional
            Restrict to one nuclear data library. Defaults to all of them.
        system : str, optional
            Restrict to one measurement system.
        wide : bool, default False
            If True, pivot to positions x libraries holding C/E only. With
            several systems in play the index becomes ``(system, position)``,
            since position labels repeat across foils.

        Returns
        -------
        pandas.DataFrame
        """
        if library is not None and library not in self.libraries:
            raise KeyError(
                f"No calculation for library '{library}'. Available: {self.libraries}"
            )
        key = self._resolve_system(system)
        nuc = {s.id: s.target_nuclide for s in self.systems}

        exp = {m.id: m for m in self.measurements()}
        calc_of = {c["id"]: c for c in self._m["calculations"]}
        results = {r["id"]: r for c in self._m["calculations"] for r in c["results"]}

        rows = []
        for c in self._m["comparisons"]:
            calc = calc_of[c["calculationResultRef"].split(":")[0]]
            if library is not None and calc["nuclearDataLibrary"] != library:
                continue
            m = exp[c["measurementRef"]]
            if key is not None and m.system != key:
                continue
            res = results[c["calculationResultRef"]]
            rows.append(
                {
                    "system": m.system,
                    "nuclide": nuc.get(m.system),
                    "position": m.position,
                    "depth_cm": m.depth_cm,
                    "library": calc["nuclearDataLibrary"],
                    "C": res["value"],
                    "E": m.value,
                    "ce": c["value"],
                    "calc_rel_unc": res.get("statisticalRelativeUncertainty"),
                    "exp_rel_unc": m.rel_unc,
                }
            )
        df = pd.DataFrame(rows).sort_values(
            ["system", "library", "depth_cm"], ignore_index=True
        )
        if wide:
            index = "position" if df["system"].nunique() == 1 else ["system", "position"]
            wide_df = df.pivot(index=index, columns="library", values="ce")
            # pivot sorts the index lexically, which puts A10 before A2. Restore
            # depth order: these are positions along a penetration axis, and the
            # order is the physics.
            order = df.drop_duplicates(
                subset=index if isinstance(index, list) else [index]
            )
            keys = (
                list(order[index])
                if isinstance(index, str)
                else list(order[index].itertuples(index=False, name=None))
            )
            return wide_df.reindex(keys)
        return df

    def covariance(
        self, system: Optional[str] = None, relative: bool = True
    ) -> pd.DataFrame:
        """
        Build the experimental covariance from the declared components.

        The matrix is generated from the component structure rather than stored:
        each component says what it is and how far it reaches. Components with
        scope ``unknown`` or ``none`` contribute to the diagonal only -- the
        reader will not invent a correlation that was never reported, and will
        not drop one that was. See :meth:`unresolved`.

        Parameters
        ----------
        system : str, optional
            Restrict to one measurement system.
        relative : bool, default True
            Return a relative covariance. If False, scale by the values.

        Returns
        -------
        pandas.DataFrame
            Square, labelled by measurement identifier.
        """
        meas = self.measurements(system)
        ids = [m.id for m in meas]
        n = len(ids)
        cov = np.zeros((n, n))

        def rel(c):
            return c["relative"] / c["coverageFactor"]

        for i, mi in enumerate(meas):
            for ci in mi.components:
                ri = rel(ci)
                cov[i, i] += ri * ri
                kind = ci["correlationScope"]["kind"]
                if kind not in _CORRELATING:
                    continue
                for j, mj in enumerate(meas):
                    if j == i:
                        continue
                    if kind == "fullWithinMeasurementSystem" and mi.system != mj.system:
                        continue
                    for cj in mj.components:
                        if (
                            cj["type"] == ci["type"]
                            and cj["correlationScope"]["kind"] == kind
                        ):
                            cov[i, j] += ri * rel(cj)

        if not relative:
            v = np.array([m.value for m in meas])
            cov = cov * np.outer(v, v)
        return pd.DataFrame(cov, index=ids, columns=ids)

    def uncertainty_of_mean(self, system: Optional[str] = None) -> pd.Series:
        """
        What the correlation structure is worth, as a number.

        Compares the uncertainty on an unweighted mean built from the full
        covariance against the same quantity built from its diagonal. The ratio
        is the factor by which an analysis that treats the points as independent
        understates itself.

        Returns
        -------
        pandas.Series
            ``n``, ``diagonal_only``, ``full``, ``factor``.
        """
        cov = self.covariance(system).to_numpy()
        n = len(cov)
        w = np.ones(n) / n
        full = float(np.sqrt(w @ cov @ w))
        diag = float(np.sqrt(w @ np.diag(np.diag(cov)) @ w))
        return pd.Series(
            {
                "n": n,
                "diagonal_only": diag,
                "full": full,
                "factor": full / diag if diag else np.nan,
            }
        )

    def uncertainty_budget(self) -> pd.DataFrame:
        """
        The cost of the correlation structure, per system and for the entry.

        Returns
        -------
        pandas.DataFrame
            Columns ``scope``, ``n``, ``diagonal_only``, ``full``, ``factor``.
        """
        rows = []
        for scope in [None] + [s.id for s in self.systems]:
            u = self.uncertainty_of_mean(scope)
            rows.append({"scope": scope or "whole entry", **u.to_dict()})
        return pd.DataFrame(rows).astype({"n": int})

    def reproducibility(self) -> pd.DataFrame:
        """
        Per calculation, how much of what it needed is actually obtainable.

        Returns
        -------
        pandas.DataFrame
            Columns ``calculation``, ``system``, ``library``, ``inputs``,
            ``available``, ``missing_roles``, ``reproducible``.
        """
        rows = []
        for c in self.calculations:
            rows.append(
                {
                    "calculation": c.id,
                    "system": c.system,
                    "library": c.library,
                    "inputs": len(c.inputs),
                    "available": sum(i.available for i in c.inputs),
                    "missing_roles": ", ".join(
                        sorted({i.role for i in c.inputs if not i.available})
                    )
                    or "-",
                    "reproducible": c.reproducible,
                }
            )
        return pd.DataFrame(rows)

    def findings(self) -> pd.DataFrame:
        """
        Data-quality findings the entry carries about itself.

        A caveat that lives only in the prose of a 1993 report is a caveat that
        will be lost. These have identifiers and an ``affects`` list, so a tool
        can surface them.

        Returns
        -------
        pandas.DataFrame
            Columns ``id``, ``severity``, ``affects``, ``summary``, ``detail``.
        """
        return pd.DataFrame(
            [
                {
                    "id": f["id"],
                    "severity": f["severity"],
                    "affects": f.get("affects"),
                    "summary": f["summary"],
                    "detail": f["detail"],
                }
                for f in self._m.get("dataQuality", [])
            ],
            columns=["id", "severity", "affects", "summary", "detail"],
        )

    def unresolved(self) -> pd.DataFrame:
        """
        Everything the entry records as not known, or known only by inference.

        Reading this is the honest first step of any analysis built on the
        entry. An empty frame means the entry is fully characterised, which for
        a legacy benchmark would be remarkable.

        Returns
        -------
        pandas.DataFrame
            Columns ``where``, ``what``, ``reason``.
        """
        rows = []
        for m in self.measurements():
            for mi in m.missing:
                rows.append({"where": m.id, "what": mi["what"], "reason": mi["reason"]})
            for c in m.components:
                if c["correlationScope"]["kind"] == "unknown":
                    rows.append(
                        {
                            "where": f"{m.id}/{c['type']}",
                            "what": "correlationScope",
                            "reason": "unknown",
                        }
                    )
        # A reconstructed component is not a missing one, but it is not a
        # reported one either, and an analysis is entitled to know the
        # difference. Reported once per type rather than once per point.
        recon = sorted(
            {
                c["type"]
                for m in self.measurements()
                for c in m.components
                if c.get("derivation") == "reconstructed"
            }
        )
        for t in recon:
            rows.append(
                {
                    "where": "all measurements",
                    "what": f"{t} component",
                    "reason": "reconstructed, not reported",
                }
            )
        for s in self._m.get("systematics", []):
            for f in s["factors"]:
                if f["relativeUncertainty"] is None:
                    rows.append(
                        {
                            "where": f"{s['id']}/{f['symbol']}",
                            "what": "relativeUncertainty",
                            "reason": f["uncertaintyMissingReason"],
                        }
                    )
        for c in self.calculations:
            for i in c.inputs:
                if not i.available:
                    rows.append(
                        {
                            "where": f"{c.id}/{i.artifact}",
                            "what": f"input file ({i.role})",
                            "reason": "notInThisExtract",
                        }
                    )
        for f in self._m.get("dataQuality", []):
            rows.append(
                {
                    "where": f"dataQuality/{f['id']}",
                    "what": f["summary"],
                    "reason": f["severity"],
                }
            )
        return pd.DataFrame(rows, columns=["where", "what", "reason"])

    # -- presentation -----------------------------------------------------

    def summary(self) -> str:
        """Return a formatted summary of the benchmark."""
        rep = self.reproducibility()
        unres = self.unresolved()
        budget = self.uncertainty_of_mean()
        systems = self.systems
        lines = [
            f"SINBAD benchmark: {self.id}",
            f"  Title:        {self.title}",
            f"  Facility:     {self.facility} ({self.year})",
            f"  Systems:      {len(systems)} "
            f"({', '.join(s.target_nuclide for s in systems)})",
            f"  Measurements: {len(self.measurements())} points",
            f"  Libraries:    {', '.join(self.libraries)}",
            f"  Calculations: {len(rep)}",
            f"  Package:      {self.package.kind} at {self.package.path}",
            "",
            f"  Reproducible calculations: {int(rep['reproducible'].sum())}/{len(rep)}",
            f"  Unresolved items:          {len(unres)}   (see .unresolved())",
            f"  Uncertainty on mean C/E:   {100 * budget['full']:.2f}% with the "
            f"declared correlations, {100 * budget['diagonal_only']:.2f}% without",
        ]
        if budget["factor"] > 1.5:
            lines.append(
                f"        -> ignoring the correlation structure understates it "
                f"by {budget['factor']:.1f}x   (see .uncertainty_budget())"
            )
        derived = [
            o for o in self.observables if o.get("unit_confidence") != "reported"
        ]
        if derived:
            lines += [
                "",
                f"  NOTE: {len(derived)} observable(s) have a derived unit, not one "
                "read from a labelled field.",
            ]
        warnings = [
            f for f in self._m.get("dataQuality", []) if f["severity"] != "info"
        ]
        if warnings:
            lines += [
                "",
                f"  Data-quality warnings: {len(warnings)}   (see .findings())",
            ]
            for f in warnings:
                lines.append(f"    - {f['id']}: {f['summary']}")
        return "\n".join(lines)

    def close(self) -> None:
        """Release array and archive handles."""
        for h in self._arrays.values():
            if hasattr(h, "close"):
                h.close()
        self._arrays.clear()
        self.package.close()

    def __enter__(self) -> "SinbadBenchmark":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"<SinbadBenchmark {self.id}: {len(self.systems)} systems, "
            f"{len(self.measurements())} measurements, "
            f"{len(self._m['calculations'])} calculations "
            f"({', '.join(self.libraries)})>"
        )
