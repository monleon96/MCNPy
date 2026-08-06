"""The PEP 562 deprecation surface: what it fixes, and what it does not.

Phase 3d's last commit replaced ``kika/nuclear_data/__init__.py``'s four eager
imports with a module-level ``__getattr__``. Two things follow, and only one of
them is the advertised win, so both are asserted here rather than described.

**What it fixes.** ``kika/__init__.py`` does ``from . import nuclear_data`` at
module scope, so every consumer reaches this package on ``import kika``. The
package no longer drags four modules in behind it, and the deprecation warning
fires once per *name* on first access — never per instance (which would put the
``warnings`` registry on the reconstruction hot path and flood sbatch logs) and
never at import (which would warn every consumer at once for a message none of
them asked for).

**What it does not fix.** ``kika.processing.reconstruct`` imports
``kika.nuclear_data.cross_section`` and ``.resonance_parameters`` *by module
path* at module scope, so those two still load on ``import kika`` regardless of
this file. ``angular_distribution`` — 1000 lines, the largest of the four — and
``nuclide_info`` genuinely stay out. Making the other two lazy means moving
``reconstruct`` onto the model, which is phase 4. The test below pins the
current state so the claim is a measurement.
"""
from __future__ import annotations

import subprocess
import sys
import warnings

import pytest

import kika.nuclear_data as nd


def _inSubprocess(body: str) -> str:
    completed = subprocess.run(
        [sys.executable, "-c", "import sys\n" + body],
        capture_output=True, text=True, timeout=300,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed.stdout.strip()


# ---------------------------------------------------------------------------
# The surface is unchanged
# ---------------------------------------------------------------------------

def test_every_name_still_resolves():
    """Nine names, same nine, reachable the same way as before."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        for name in nd.__all__:
            assert getattr(nd, name) is not None


def test_an_unknown_name_still_raises_attribute_error():
    """``__getattr__`` must not turn a typo into something stranger."""
    with pytest.raises(AttributeError, match="NotAThing"):
        nd.NotAThing


def test_dir_lists_the_public_surface():
    """``dir()`` stops being useful the moment names are lazy, unless helped."""
    assert set(nd.__dir__()) == set(nd.__all__)


# ---------------------------------------------------------------------------
# The warning fires once per name, on access
# ---------------------------------------------------------------------------

def test_the_warning_fires_on_first_access_and_not_again():
    output = _inSubprocess(
        "import warnings\n"
        "with warnings.catch_warnings(record=True) as caught:\n"
        "    warnings.simplefilter('always')\n"
        "    import kika.nuclear_data as nd\n"
        "    nd.CrossSection; nd.CrossSection; nd.CrossSection\n"
        "print(len(caught), caught[0].category.__name__)"
    )
    assert output == "1 DeprecationWarning"


def test_the_warning_does_not_fire_at_import():
    """Importing the package must stay silent; only using a name warns."""
    output = _inSubprocess(
        "import warnings\n"
        "with warnings.catch_warnings(record=True) as caught:\n"
        "    warnings.simplefilter('always')\n"
        "    import kika.nuclear_data\n"
        "print(len([w for w in caught if w.category is DeprecationWarning]))"
    )
    assert output == "0"


def test_the_warning_names_a_replacement_and_admits_its_limits():
    """A deprecation that points nowhere is noise; one that overclaims is worse."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        nd._WARNED.discard("NuclideInfo")
        nd.__getattr__("NuclideInfo")

    message = str(caught[0].message)
    assert "kika.nuclear_data.model" in message
    assert "1.0" in message
    assert "reconstruction" in message, (
        "the message must say the model does not cover everything yet"
    )


def test_construction_does_not_warn():
    """Per-instance warnings would land on the reconstruction hot path.

    ``kika/processing/reconstruct.py`` builds one ``CrossSection`` per MT per
    call, per sample, per temperature. A ``__post_init__`` warning there is a
    performance problem and an unreadable log, which is why the deprecation is
    on the *name* and not on the object.
    """
    output = _inSubprocess(
        "import warnings, numpy as np\n"
        "from kika.nuclear_data.cross_section import CrossSection\n"
        "with warnings.catch_warnings(record=True) as caught:\n"
        "    warnings.simplefilter('always')\n"
        "    for _ in range(50):\n"
        "        CrossSection(np.array([1.0]), np.array([2.0]), 2, 26056)\n"
        "print(len([w for w in caught if w.category is DeprecationWarning]))"
    )
    assert output == "0"


# ---------------------------------------------------------------------------
# What actually loads on `import kika`
# ---------------------------------------------------------------------------

def test_import_kika_no_longer_loads_the_two_biggest_flat_modules():
    output = _inSubprocess(
        "import kika\n"
        "print(sorted(m for m in sys.modules if m.startswith('kika.nuclear_data.')))"
    )
    loaded = eval(output)  # noqa: S307 - our own output, a list literal

    assert "kika.nuclear_data.angular_distribution" not in loaded
    assert "kika.nuclear_data.nuclide_info" not in loaded


def test_the_other_two_still_load_and_here_is_who_pulls_them():
    """Pinned so the partial win is not mistaken for the whole one.

    ``kika.processing.reconstruct`` imports both by module path at module
    scope. They will go lazy when phase 4 moves reconstruction onto the model,
    and this test will then fail and should be tightened rather than deleted.
    """
    output = _inSubprocess(
        "import kika\n"
        "print(sorted(m for m in sys.modules if m.startswith('kika.nuclear_data.')))"
    )
    loaded = eval(output)  # noqa: S307

    assert "kika.nuclear_data.cross_section" in loaded
    assert "kika.nuclear_data.resonance_parameters" in loaded


def test_the_model_is_still_dormant_after_the_facade():
    """The invariant every increment since P4 has had to keep.

    The façades import the adapter *inside* their methods, so ``import kika``
    still does not build the model. If this fails, the cluster pipeline and the
    desktop app just got slower for no reason.
    """
    output = _inSubprocess(
        "import kika\n"
        "print('kika.nuclear_data.model' in sys.modules)"
    )
    assert output == "False"
