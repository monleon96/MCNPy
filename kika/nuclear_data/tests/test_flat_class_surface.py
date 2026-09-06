"""The surface ratchet: the flat canonical classes may grow, never change shape.

**Why this exists.** Phase 3 of the GNDS roadmap rewrites ``kika.nuclear_data``
around a GNDS-shaped model, and Phase 3d reimplements these five classes on top
of it. The roadmap's acceptance for that step is "the façade passes the phase 0
tests without modifying them" — but the phase 0 tests do not actually pin the
surface. ``test_annotations.py`` checks that annotations *resolve*; nothing
anywhere checks that ``CrossSection`` still has seven fields in that order, that
``to_endf`` still takes keyword-only ``qm``/``qi``/``lr``, or that ``__repr__``
still produces the string the notebooks print. This file is that check, written
*before* the rewrite so the rewrite has something to be measured against.

**Why it matters more than the small consumer count suggests.** Nothing outside
this library imports ``kika.nuclear_data`` by name — 0 scripts, 0 kika-app
modules, 2 notebooks. But ``kika.processing.njoy_reconstruct`` *returns*
``Dict[int, CrossSection]``, and ``kika-api``'s ``endf_service`` and
``plot.py`` consume those objects and serialise them. A field rename here
breaks the desktop app with no grep hit anywhere. The coupling is real and it
is invisible; this table is where it is written down.

**Two directions.** Like ``kika/tests/test_layering.py``, this fails both ways:
a name that disappears fails, and a table entry that no longer matches reality
fails. Adding a new method or field is allowed and needs no edit here.

**Note on annotation strings.** These modules use ``from __future__ import
annotations``, so ``dataclasses.fields(cls)[i].type`` is already a string. It is
compared as a string on purpose: resolving it would import ``kika.endf``, which
is exactly what the calculation layer must not do. Entries that look
double-quoted (``"'MF3MT'"``) are faithful — the source writes the annotation as
a quoted forward reference and the future-import stringifies it again.
"""
from __future__ import annotations

import dataclasses
import importlib
import inspect

import numpy as np
import pytest

import kika.nuclear_data as nd

MODULES = (
    "kika.nuclear_data.cross_section",
    "kika.nuclear_data.angular_distribution",
    "kika.nuclear_data.resonance_parameters",
    "kika.nuclear_data.nuclide_info",
)

#: Dataclass fields, in declaration order: (name, annotation, default).
#: Order is load-bearing — these are constructed positionally outside the repo.
EXPECTED_FIELDS: dict[str, list[tuple[str, str, str]]] = {
    "CrossSection": [
        ("energies", "np.ndarray", "REQUIRED"),
        ("values", "np.ndarray", "REQUIRED"),
        ("reaction", "int", "REQUIRED"),
        ("nuclide_id", "int", "REQUIRED"),
        ("temperature", "float", "0.0"),
        ("interpolation", "str", "'linlin'"),
        ("metadata", "Dict", "factory:dict"),
    ],
    "AngularDistribution": [
        ("energies", "np.ndarray", "REQUIRED"),
        ("coefficients", "Dict[int, np.ndarray]", "factory:dict"),
        ("reaction", "int", "0"),
        ("nuclide_id", "int", "0"),
        ("frame", "str", "'CM'"),
        ("representation", "str", "'legendre'"),
        ("tabulated_data", "Optional[Dict]", "None"),
        ("metadata", "Dict", "factory:dict"),
        ("uncertainties", "Optional[Dict[int, np.ndarray]]", "None"),
        ("uncertainty_energies", "Optional[np.ndarray]", "None"),
        ("uncertainty_type", "str", "'relative'"),
    ],
    "ResonanceRecord": [
        ("energy", "float", "REQUIRED"),
        ("spin", "float", "REQUIRED"),
        # c3..c6 are ENDF record positions, not physics names: they are
        # GN,GG,GF,- under SLBW/MLBW and GN,GG,GFA,GFB under Reich-Moore. The
        # names stay until the GNDS per-formalism nodes replace them wholesale.
        ("c3", "float", "REQUIRED"),
        ("c4", "float", "REQUIRED"),
        ("c5", "float", "REQUIRED"),
        ("c6", "float", "REQUIRED"),
    ],
    "LGroup": [
        ("awri", "float", "REQUIRED"),
        ("l", "int", "REQUIRED"),
        ("resonances", "List[ResonanceRecord]", "factory:list"),
        ("ap", "Optional[float]", "None"),
    ],
    "ResonanceParameters": [
        ("nuclide_id", "int", "REQUIRED"),
        ("spin", "float", "REQUIRED"),
        ("scattering_radius", "float", "REQUIRED"),
        ("formalism", "str", "REQUIRED"),
        ("energy_range", "Tuple[float, float]", "REQUIRED"),
        ("l_groups", "List[LGroup]", "factory:list"),
        ("metadata", "Dict", "factory:dict"),
        ("scattering_radius_table", "Optional[Tuple[np.ndarray, np.ndarray, List]]", "None"),
    ],
    "URR_JGroup": [
        ("j", "float", "REQUIRED"),
        ("amun", "float", "1.0"),
        ("d", "Union[float, np.ndarray]", "0.0"),
        ("gn0", "Union[float, np.ndarray]", "0.0"),
        ("gg", "Union[float, np.ndarray]", "0.0"),
        ("gf", "Union[float, np.ndarray]", "0.0"),
        ("gx", "Union[float, np.ndarray]", "0.0"),
    ],
    "URR_LGroup": [
        ("awri", "float", "REQUIRED"),
        ("l", "int", "REQUIRED"),
        ("j_groups", "List[URR_JGroup]", "factory:list"),
    ],
    "UnresolvedResonanceParameters": [
        ("nuclide_id", "int", "REQUIRED"),
        ("spin", "float", "REQUIRED"),
        ("scattering_radius", "float", "REQUIRED"),
        ("energy_range", "Tuple[float, float]", "REQUIRED"),
        ("lssf", "int", "REQUIRED"),
        ("l_groups", "List[URR_LGroup]", "factory:list"),
        ("energy_grid", "Optional[np.ndarray]", "None"),
        ("metadata", "Dict", "factory:dict"),
    ],
    "NuclideInfo": [
        ("nuclide_id", "int", "0"),
        ("atomic_weight_ratio", "float", "0.0"),
        ("temperature", "float", "0.0"),
        ("evaluation_info", "Dict", "factory:dict"),
        ("metadata", "Dict", "factory:dict"),
    ],
}

#: Callables on each class. Single-underscore names are included deliberately:
#: they are private by convention but load-bearing outside their own module
#: (``_interp_single_scheme`` is the interpolation entry point;
#: ``_evaluate_pdf_*`` are what ``evaluate_pdf`` dispatches to).
EXPECTED_METHODS: dict[str, dict[str, str]] = {
    "CrossSection": {
        "from_endf": '(cls, mf3mt: "\'MF3MT\'") -> "\'CrossSection\'"',
        "to_endf": '(self, mat: \'Optional[int]\' = None, *, qm: \'Optional[float]\' = None, qi: \'Optional[float]\' = None, lr: \'Optional[int]\' = None) -> "\'MF3MT\'"',
        "from_endf_file": '(cls, endf: "\'ENDF\'", mt: \'int\', use_reconstructed: \'bool\' = True) -> "\'CrossSection\'"',
        "all_from_endf_file": '(cls, endf: "\'ENDF\'", use_reconstructed: \'bool\' = True) -> "Dict[int, \'CrossSection\']"',
        "from_ace": '(cls, ace: "\'Ace\'", mt: \'int\') -> "\'CrossSection\'"',
        "all_from_ace": '(cls, ace: "\'Ace\'", include_composites: \'bool\' = True) -> "Dict[int, \'CrossSection\']"',
        "to_plot_data": '(self, label: \'Optional[str]\' = None, **styling_kwargs) -> "\'CrossSectionPlotData\'"',
        "get_cross_section": "(self, energy: 'Union[float, ArrayLike]', out_of_range: 'str' = 'zero') -> 'Union[float, np.ndarray]'",
        "_interp_single_scheme": "(self, target: 'np.ndarray', scheme: 'str', out_of_range: 'str') -> 'np.ndarray'",
    },
    "AngularDistribution": {
        "from_endf": '(cls, mf4mt: "\'MF4MT\'") -> "\'AngularDistribution\'"',
        "from_ace": '(cls, ace: "\'Ace\'", mt: \'int\') -> "\'AngularDistribution\'"',
        "all_from_ace": '(cls, ace: "\'Ace\'") -> "Dict[int, \'AngularDistribution\']"',
        "project_to_legendre": "(self, max_order: 'int' = 6) -> 'None'",
        "evaluate_pdf": "(self, energy: 'float', cosines: 'Optional[np.ndarray]' = None, num_points: 'int' = 100) -> 'Tuple[np.ndarray, np.ndarray]'",
        "_evaluate_pdf_legendre": "(self, energy: 'float', cosines: 'np.ndarray') -> 'Tuple[np.ndarray, np.ndarray]'",
        "_evaluate_pdf_tabulated": "(self, energy: 'float', cosines: 'np.ndarray') -> 'Tuple[np.ndarray, np.ndarray]'",
        "to_endf": '(self, mat: \'Optional[int]\' = None) -> "\'MF4MT\'"',
        "to_plot_data": '(self, order: \'int\', label: \'Optional[str]\' = None, **styling_kwargs) -> "\'LegendreCoeffPlotData\'"',
        "attach_uncertainties": '(self, covariance: "\'LegendreCovariance\'") -> \'None\'',
        "to_uncertainty_plot_data": '(self, order: \'int\', sigma: \'float\' = 1.0, label: \'Optional[str]\' = None, **styling_kwargs) -> "\'LegendreUncertaintyPlotData\'"',
        "has_uncertainties": "property",
        "max_order": "property",
    },
    "ResonanceParameters": {
        "from_endf": '(cls, mf2: "\'MF2MT151\'") -> "List[Union[\'ResonanceParameters\', \'UnresolvedResonanceParameters\']]"',
        "num_resonances": "property",
    },
    "NuclideInfo": {
        "from_endf": '(cls, mt451: "\'MF1MT451\'") -> "\'NuclideInfo\'"',
        "from_ace": '(cls, ace: "\'Ace\'") -> "\'NuclideInfo\'"',
        # `docs/library/library-gaps.md` M2. A new method needs no entry here, but its
        # three siblings' `to_endf` are pinned, and a signature that may drift
        # while theirs may not is the odd one out rather than the free one.
        "to_endf": '(self, mat: \'Optional[int]\' = None) -> "\'MF1MT451\'"',
    },
}

#: Module-level helpers. Private by name, but ``_dominant_interpolation`` has
#: its own test file and ``_urr_from_endf`` is the whole URR decode path.
EXPECTED_MODULE_LEVEL: dict[str, dict[str, str]] = {
    "kika.nuclear_data.cross_section": {
        "_dominant_interpolation": "(regions: 'List[Tuple[int, int]]') -> 'str'",
    },
    "kika.nuclear_data.resonance_parameters": {
        "_urr_from_endf": "(er, za, awr, mat, abn)",
    },
}

#: ``kika.nuclear_data.__all__``, exactly.
EXPECTED_ALL = [
    "AngularDistribution",
    "CrossSection",
    "LGroup",
    "NuclideInfo",
    "ResonanceParameters",
    "ResonanceRecord",
    "URR_JGroup",
    "URR_LGroup",
    "UnresolvedResonanceParameters",
]

#: ``__repr__`` is printed in notebooks and scraped from logs, so its exact
#: text is API. Each entry is (constructor thunk, expected string).
_E = np.array([1.0, 10.0, 100.0])


def _repr_cases() -> list[tuple[str, object, str]]:
    from kika.nuclear_data import (
        AngularDistribution,
        CrossSection,
        LGroup,
        NuclideInfo,
        ResonanceParameters,
        ResonanceRecord,
        UnresolvedResonanceParameters,
        URR_JGroup,
        URR_LGroup,
    )

    resonances = [
        ResonanceRecord(1.0, 0.5, 1.0, 2.0, 3.0, 4.0),
        ResonanceRecord(2.0, 0.5, 1.0, 2.0, 3.0, 4.0),
    ]
    return [
        (
            "CrossSection",
            CrossSection(_E, np.array([1.5, 2.5, 3.5]), 2, 26056),
            "CrossSection(MT=2, ZA=26056, n_points=3, E=[1, 100] eV)",
        ),
        (
            "CrossSection-empty",
            CrossSection(np.array([]), np.array([]), 102, 26056),
            "CrossSection(MT=102, ZA=26056, empty)",
        ),
        (
            "AngularDistribution",
            AngularDistribution(
                energies=_E,
                coefficients={1: np.array([0.1, 0.2, 0.3]), 2: np.array([0.4, 0.5, 0.6])},
                reaction=2,
                nuclide_id=26056,
            ),
            "AngularDistribution(MT=2, ZA=26056, repr='legendre', frame='CM', "
            "max_L=2, n_energies=3)",
        ),
        (
            "ResonanceParameters",
            ResonanceParameters(
                26056, 0.0, 0.54, "MLBW", (1e-5, 1e6),
                l_groups=[LGroup(55.36, 0, resonances)],
            ),
            "ResonanceParameters(ZA=26056, formalism='MLBW', "
            "E=[1e-05, 1e+06] eV, n_resonances=2)",
        ),
        (
            "UnresolvedResonanceParameters",
            UnresolvedResonanceParameters(
                26056, 0.0, 0.54, (1e6, 2e6), 1,
                l_groups=[URR_LGroup(55.36, 0, [URR_JGroup(0.5)])],
            ),
            "UnresolvedResonanceParameters(ZA=26056, LSSF=1, E=[1e+06, 2e+06] eV)",
        ),
        (
            "NuclideInfo",
            NuclideInfo(26056, 55.36, 293.6),
            "NuclideInfo(ZA=26056, AWR=55.3600, T=293.6 K)",
        ),
    ]


# ---------------------------------------------------------------------------
# Introspection helpers — shared with the self-test at the bottom
# ---------------------------------------------------------------------------

def _default_repr(f: dataclasses.Field) -> str:
    if f.default is not dataclasses.MISSING:
        return repr(f.default)
    if f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
        return f"factory:{f.default_factory.__name__}"  # type: ignore[misc]
    return "REQUIRED"


def actual_fields(cls) -> list[tuple[str, str, str]]:
    return [(f.name, str(f.type), _default_repr(f)) for f in dataclasses.fields(cls)]


def actual_methods(cls) -> dict[str, str]:
    found: dict[str, str] = {}
    for name, raw in vars(cls).items():
        if name.startswith("__"):
            continue
        if isinstance(raw, property):
            found[name] = "property"
            continue
        fn = getattr(raw, "__func__", raw)
        if inspect.isfunction(fn):
            found[name] = str(inspect.signature(fn))
    return found


def _classes() -> dict[str, type]:
    found: dict[str, type] = {}
    for mod_name in MODULES:
        mod = importlib.import_module(mod_name)
        for name, obj in vars(mod).items():
            if inspect.isclass(obj) and obj.__module__ == mod.__name__:
                found[name] = obj
    return found


# ---------------------------------------------------------------------------
# The ratchet
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls_name", sorted(EXPECTED_FIELDS))
def test_dataclass_fields_are_unchanged(cls_name):
    """Name, order, annotation and default, all four, for every field."""
    cls = _classes().get(cls_name)
    assert cls is not None, f"{cls_name} has disappeared from kika.nuclear_data"
    assert actual_fields(cls) == EXPECTED_FIELDS[cls_name]


@pytest.mark.parametrize("cls_name", sorted(EXPECTED_FIELDS))
def test_the_flat_classes_are_still_dataclasses(cls_name):
    """Guards a silent failure mode in ``test_annotations.py``.

    ``test_dataclass_fields_resolve`` loops rather than parametrizes, and skips
    anything ``dataclasses.is_dataclass`` says no to. So a phase 3d façade built
    as properties over a ``_model`` attribute would not turn that test red — it
    would quietly drop out of it, and the coverage would vanish with nothing to
    show for it. This is the assertion that notices.
    """
    cls = _classes().get(cls_name)
    assert cls is not None, f"{cls_name} has disappeared from kika.nuclear_data"
    assert dataclasses.is_dataclass(cls), (
        f"{cls_name} is no longer a dataclass. test_annotations.py has stopped "
        f"checking its field annotations and said nothing."
    )


@pytest.mark.parametrize("cls_name", sorted(EXPECTED_METHODS))
def test_public_method_signatures_are_unchanged(cls_name):
    """Signatures may gain methods, never lose or reshape one."""
    cls = _classes().get(cls_name)
    assert cls is not None, f"{cls_name} has disappeared from kika.nuclear_data"
    actual = actual_methods(cls)

    missing = sorted(set(EXPECTED_METHODS[cls_name]) - set(actual))
    assert not missing, f"{cls_name} lost: {missing}"

    changed = {
        name: (expected, actual[name])
        for name, expected in EXPECTED_METHODS[cls_name].items()
        if actual[name] != expected
    }
    assert not changed, "\n".join(
        f"{cls_name}.{n}\n  was: {was}\n  now: {now}" for n, (was, now) in changed.items()
    )


@pytest.mark.parametrize("mod_name", sorted(EXPECTED_MODULE_LEVEL))
def test_module_level_helpers_are_unchanged(mod_name):
    mod = importlib.import_module(mod_name)
    for name, expected in EXPECTED_MODULE_LEVEL[mod_name].items():
        fn = getattr(mod, name, None)
        assert fn is not None, f"{mod_name}.{name} has disappeared"
        assert str(inspect.signature(fn)) == expected


def test_module_exports_are_unchanged():
    assert sorted(nd.__all__) == EXPECTED_ALL
    for name in EXPECTED_ALL:
        assert hasattr(nd, name), f"kika.nuclear_data.{name} is not importable"


@pytest.mark.parametrize(
    "label,obj,expected", _repr_cases(), ids=[c[0] for c in _repr_cases()]
)
def test_repr_format_is_unchanged(label, obj, expected):
    """Notebook output and log scraping depend on these exact strings."""
    assert repr(obj) == expected


# ---------------------------------------------------------------------------
# The ratchet itself
# ---------------------------------------------------------------------------

def test_the_surface_ratchet_catches_a_rename():
    """A ratchet nobody has seen fail is not a ratchet.

    Build a throwaway dataclass matching one table entry, rename a field, and
    require the comparison to notice. Mirrors
    ``test_layering.py::test_the_ratchet_catches_a_violation``.
    """
    @dataclasses.dataclass
    class Faithful:
        energy: float
        spin: float

    @dataclasses.dataclass
    class Renamed:
        energy_eV: float  # noqa: N815 — the planted rename
        spin: float

    table = [("energy", "float", "REQUIRED"), ("spin", "float", "REQUIRED")]

    assert actual_fields(Faithful) == table
    assert actual_fields(Renamed) != table

    # A changed default must be caught too, not just a rename.
    @dataclasses.dataclass
    class Redefaulted:
        energy: float = 0.0
        spin: float = 0.0

    assert actual_fields(Redefaulted) != table
