"""``kika.endf.processing.reconstruct``: what the ENDF side of the sandwich owes.

The arithmetic is pinned elsewhere — ``kika/processing/tests/test_numeric_goldens.py``
freezes the numbers, including a sha256 over this adapter's full Fe-56 output.
What is checked here is everything *around* the numbers: the section header the
adapter writes, and the two inputs it refuses rather than guesses at.
"""
from __future__ import annotations

import pytest

from kika.endf.processing.reconstruct import reconstruct
from kika.endf.read_endf import read_endf


@pytest.fixture(scope="module")
def endf(micro_tape):
    return read_endf(str(micro_tape))


def test_a_reconstructed_section_inherits_the_q_of_the_one_it_replaces(endf):
    """The half of the Q = 0 defect phase 1 deferred.

    Every reconstructed section used to be written with ``QM = QI = 0`` and
    ``LR = 0``, whatever the reaction. For MT1 and MT2 that is right by
    accident; for MT102 it is wrong by 7.6 MeV, because QM for (n,gamma) is the
    neutron separation energy. A reconstructed MT102 replaces the file's own
    MT102 over the resonance region — same reaction, same evaluation — so it
    inherits that section's Q, and this asserts the values are *the file's* and
    not a plausible constant.
    """
    produced = reconstruct(endf.mf[2].mt[151], endf.files.get(3))
    source = endf.files[3].sections

    assert set(produced) == {1, 2, 102}
    for mt in sorted(produced):
        assert produced[mt]._qm == source[mt].q_mass_difference, f"MT{mt}: QM"
        assert produced[mt]._qi == source[mt].q_reaction, f"MT{mt}: QI"
        assert produced[mt]._lr == source[mt].breakup_flag, f"MT{mt}: LR"


def test_the_capture_q_is_not_zero_on_this_tape(endf):
    """Without this the assertion above would pass on a file where 0 was right.

    Fe-56 (n,gamma) has QM = 7.646 MeV. If this ever reads zero the test above
    has stopped distinguishing an inherited Q from the hardcode it replaced.
    """
    qm = endf.files[3].sections[102].q_mass_difference
    assert qm > 1.0e6, f"MF3/MT102 QM is {qm}, so this tape no longer pins anything"


def test_an_mt_with_no_section_to_inherit_from_is_refused(endf):
    """A header written with Q = 0 is wrong rather than incomplete.

    Reconstructing without MF3 leaves MT102 with no Q to take, and the adapter
    raises instead of writing zero. MT1 and MT2 are exempt and stay silent:
    elastic scattering leaves the nucleus as it found it and the total is a sum,
    so zero there is a statement rather than a default.
    """
    with pytest.raises(ValueError, match="MT102"):
        reconstruct(endf.mf[2].mt[151])


def test_a_multi_isotope_section_is_refused_rather_than_summed(endf):
    """GNDS gives each nuclide its own suite, so ABN has nowhere to go.

    The old path weighted each range by its abundance. The model merges the
    ranges of every isotope into one list — the decoder reports the loss — so an
    abundance-weighted sum is not something it can express, and reconstructing
    anyway would return an elemental evaluation as though each isotope were the
    whole material. Driven from a mutated header rather than from a tape,
    because every evaluation on this machine is NIS=1 with ABN=1.
    """
    section = endf.mf[2].mt[151]
    original = section._isotopes
    try:
        section._isotopes = list(original) * 2
        with pytest.raises(ValueError, match="isotopes"):
            reconstruct(section, endf.files.get(3))
    finally:
        section._isotopes = original
