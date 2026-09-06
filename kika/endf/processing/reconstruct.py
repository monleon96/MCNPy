"""
Resonance reconstruction (ENDF adapter).

Thin wrapper around ``kika.processing.reconstruct`` that accepts and returns
ENDF-specific types (``MF2MT151``, ``MF3MT``).  The actual physics
lives in ``kika.processing.reconstruct``.

**Phase 4: the middle is the model.** This used to convert MF2 into the flat
``ResonanceParameters`` and hand those to the physics. It now decodes into the
GNDS-shaped resonance nodes and hands those over — which is the whole point of
the phase, and also what makes LRF=7 reachable at all, since ``c3..c6`` cannot
hold a five-channel spin group and the model can.

Supported formalisms
--------------------
- LRF=1  SLBW  (Single-Level Breit-Wigner)
- LRF=2  MLBW  (Multi-Level Breit-Wigner)
- LRF=3  RM    (Reich-Moore, eliminated capture channel)

Only resolved resonance ranges (LRU=1) and the unresolved range are processed.
"""

from typing import Dict, Optional

from ..classes.mf2.mf2mt151 import MF2MT151
from ..classes.mf3.mf3mt import MF3MT


def reconstruct(mf2_mt151: MF2MT151,
           mf3_data=None,
           tolerance: float = 1e-3) -> Dict[int, MF3MT]:
    """Reconstruct pointwise cross sections from MF2 resonance parameters.

    Parameters
    ----------
    mf2_mt151 : MF2MT151
        Parsed MF2/MT151 section.
    mf3_data : MF (MF3 file object), optional
        MF3 background cross-section data.  If provided, the smooth
        background is added to the reconstructed resonance cross sections
        inside the resonance region and preserved outside it.
    tolerance : float
        Linearization tolerance (default 0.1%).

    Returns
    -------
    Dict[int, MF3MT]
        Reconstructed pointwise cross sections keyed by MT number.
        Typically contains MT 1 (total), 2 (elastic), 18 (fission), 102 (capture).

    Raises
    ------
    ValueError
        If the section carries more than one isotope with a real abundance
        split. See :func:`_oneNuclide`.
    """
    # Lazy imports to avoid circular dependency at module load time, and to keep
    # the model off the ``read_endf`` critical path.
    from kika.endf.model_adapter import decodeMF2MT151, decodeMF3MT
    from kika.nuclear_data.model import EVAL_LABEL
    from kika.processing.reconstruct import reconstruct as _reconstruct

    # Convert ENDF types → the model
    resonances, provenance, _ = decodeMF2MT151(mf2_mt151)
    _oneNuclide(provenance)

    reactions = {}
    background = None
    if mf3_data is not None:
        background = {}
        for mt, section in mf3_data.sections.items():
            reaction, _ = decodeMF3MT(section)
            reactions[mt] = reaction
            background[mt] = reaction.crossSection[EVAL_LABEL]

    # Delegate to format-independent processing
    forms = _reconstruct(
        resonances, background, tolerance=tolerance,
        atomicWeightRatio=provenance.awr,
    )

    # Convert the model back → ENDF types
    return {mt: _toMF3MT(mt, form, provenance, reactions.get(mt))
            for mt, form in forms.items()}


def _oneNuclide(provenance) -> None:
    """Refuse an MF2/151 whose ranges belong to more than one nuclide.

    ENDF lets one MF2/151 carry several isotopes, each with an abundance, and
    the reconstruction used to weight every range by its ABN. GNDS gives each
    nuclide its own ``reactionSuite``, so the model merges the ranges into one
    list and the abundances have nowhere to go — the decoder says so in its own
    report.

    Rather than reconstruct an abundance-weighted sum that the model cannot
    describe, this refuses. Every evaluation on this machine is NIS=1 with
    ABN=1, so nothing that works today stops working; what would have happened
    silently is an elemental evaluation coming back as though each isotope were
    the whole material.
    """
    isotopes = (provenance.headerFields or {}).get("isotopes") or []
    weighted = [i for i in isotopes
                if i.get("abn") is not None and float(i["abn"]) != 1.0]
    if len(isotopes) > 1 or weighted:
        raise ValueError(
            f"MF2/151 carries {len(isotopes)} isotopes with abundances "
            f"{[i.get('abn') for i in isotopes]}. Reconstruction is per nuclide: "
            f"the model gives each its own resonances, so an abundance-weighted "
            f"sum over them is not something it can express. Split the material "
            f"and reconstruct each nuclide."
        )


#: MTs whose Q values are zero by definition rather than by omission. Elastic
#: scattering leaves the nucleus as it found it, and the total is a sum and not
#: a reaction, so ``QM = QI = 0`` for both is a statement and not a default.
_ZERO_Q = (1, 2)


def _reactionQ(mt: int, source):
    """``(QM, QI, LR)`` for a reconstructed section — inherited, never invented.

    A reconstructed MT102 *replaces* the file's own MT102 over the resonance
    region: same reaction, same evaluation, so it inherits that section's Q.
    That is the question ``model/output_channel.py`` left open ("whose Q does a
    reconstructed MT102 inherit"), and it only looks open while the answer has
    to come from somewhere other than the file being reconstructed.

    Where there is no such section and the MT is not one of the two whose Q is
    zero by definition, this raises. Writing zero was the old behaviour and is
    the defect: QM for (n,gamma) is the neutron separation energy — 7.6 MeV for
    Fe-56 — and a header claiming 0 is not a smaller error than no header.
    """
    if source is not None:
        return (getattr(source.provenance, "qm", None),
                source.outputChannel.Q.value,
                getattr(source.provenance, "lr", None))
    if mt in _ZERO_Q:
        return 0.0, 0.0, 0
    raise ValueError(
        f"MT{mt} was reconstructed but the input carried no MF3/MT{mt} to take "
        f"QM, QI and LR from, and they are not zero for this reaction. Pass the "
        f"file's MF3 as mf3_data; a header written with Q = 0 would be wrong "
        f"rather than incomplete."
    )


def _toMF3MT(mt: int, form, provenance, source) -> MF3MT:
    """One reconstructed ``XYs1d`` → an ``MF3MT``.

    Not ``encodeMF3MT``: that takes a whole ``Reaction``, and what comes back
    from the reconstruction is one form. The Q values come from the reaction the
    section replaces — see :func:`_reactionQ`.
    """
    qm, qi, lr = _reactionQ(mt, source)

    section = MF3MT(number=int(mt))
    section._za = float(provenance.za)
    section._awr = provenance.awr if provenance.awr is not None else 0.0
    section._mat = provenance.mat
    section._qm = qm
    section._qi = qi
    section._lr = lr
    section._energies = list(form.xs)
    section._cross_sections = list(form.ys)
    section._np = int(form.xs.size)
    section._nr = 1
    section._interpolation = [(int(form.xs.size), form.endfInterpolationCode)]
    return section
