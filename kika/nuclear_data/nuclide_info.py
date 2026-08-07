"""
Format-agnostic nuclide identity and evaluation metadata.

Wraps the information in ENDF MF1/MT451 (general information).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from kika.ace.classes.ace import Ace
    from kika.endf.classes.mf1.mf1mt451 import MF1MT451


@dataclass
class NuclideInfo:
    """Format-agnostic nuclide identity and evaluation metadata.

    Attributes
    ----------
    nuclide_id : int
        ZA = 1000*Z + A.
    atomic_weight_ratio : float
        Ratio of nucleus mass to neutron mass.
    temperature : float
        Target temperature in Kelvin (0 if unbroadened).
    evaluation_info : dict
        Human-readable evaluation metadata: ``"laboratory"``,
        ``"authors"``, ``"eval_date"``, ``"reference"``, etc.
    metadata : dict
        Format-specific extras for lossless round-trips.
    """

    nuclide_id: int = 0
    atomic_weight_ratio: float = 0.0
    temperature: float = 0.0
    evaluation_info: Dict = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # ENDF adapter
    # ------------------------------------------------------------------

    @classmethod
    def from_endf(cls, mt451: "MF1MT451") -> "NuclideInfo":
        """Create from an ENDF ``MT451`` object.

        **Phase 3d: this reads the file through the GNDS model.** The section is
        decoded into ``PoPs`` + an ``evaluated`` style + ``EndfProvenance``, and
        ``model.interop`` projects that back into the five fields this class has
        always had. The fields, their order and their defaults are unchanged —
        only the body is — which is what makes ``test_flat_class_surface.py``
        the proof that nothing downstream can tell.

        One thing does change, and it is a fix: ZA is **rounded** rather than
        truncated. ENDF's fixed-format floats do not round-trip exactly, so
        Th-232's ``9.023200+4`` reads back as ``90231.99999999999`` and the old
        ``int()`` named Ac-231. See ``docs/library-gaps.md`` D1.

        Parameters
        ----------
        mt451 : MT451
            Parsed MF1/MT451 section.

        Returns
        -------
        NuclideInfo
        """
        from kika.endf.model_adapter import decodeMF1MT451
        from kika.nuclear_data.model.interop import flatNuclideInfo

        _, _, provenance, _ = decodeMF1MT451(mt451)
        return cls(**flatNuclideInfo(provenance))

    def to_endf(self, mat: Optional[int] = None) -> "MF1MT451":
        """Convert back to an ENDF ``MF1MT451`` object.

        ``docs/library-gaps.md`` M2. MF1/451 was read-only: half the canonical
        layer could be parsed and not written, so the only way to change an
        evaluation's header was to mutate the ``MF1MT451`` dataclass in place —
        bypassing the format-agnostic layer for exactly the case it exists for.

        **Built from the flat fields, not from a stashed model**, for the same
        reason :meth:`CrossSection.to_endf` is: these fields are the source of
        truth for this API, so an edited ``temperature`` has to reach the file.

        **On the directory.** MF1/451 declares NC, a *line count*, for every
        section of the tape, so a directory is only true of the tape it was read
        from. This writes back the one it read. That is right for a round trip
        and wrong the moment a section changes length — at which point
        :func:`kika.endf.writers.update_directory.update_mf1_directory` rebuilds
        it by scanning the written file, which is the only place the true counts
        exist. Recomputing here would mean guessing at lengths this object
        cannot see.

        Parameters
        ----------
        mat : int, optional
            MAT number. If *None*, uses the value from ``metadata``.

        Returns
        -------
        MF1MT451

        Raises
        ------
        ValueError
            If ``metadata`` carries no ENDF header — an ACE-sourced object, or
            one built by hand. ACE records four header fields out of nineteen
            and no descriptive text at all, so the section this would write
            would be mostly invented. The same refusal, and the same reason, as
            :meth:`CrossSection.to_endf` on a missing Q.
        """
        from kika.endf.classes.mf1.mf1mt451 import MF1MT451

        required = ("lrp", "lfi", "nlib", "nmod", "elis", "sta", "lis", "liso",
                    "nfor", "awi", "emax", "lrel", "nsub", "nver", "ldrv")
        missing = [key for key in required if key not in self.metadata]
        if missing:
            source = self.metadata.get("source_format", "unknown")
            raise ValueError(
                f"NuclideInfo (source_format={source!r}) carries no "
                f"{'/'.join(missing)}, so an ENDF MF1/451 section cannot be "
                f"written for it. ACE records a handful of header fields and no "
                f"descriptive text, so most of the section would be invented. "
                f"Build the object from ENDF, where the header comes from the "
                f"file."
            )

        text = list(self.metadata.get("text") or [])
        directory = [tuple(entry) for entry in self.metadata.get("directory") or []]

        mt451 = MF1MT451(number=451)
        mt451._mat = mat if mat is not None else self.metadata.get("mat")
        mt451._za = float(self.nuclide_id)
        mt451._awr = self.atomic_weight_ratio
        mt451._temp = self.temperature
        for key in required:
            setattr(mt451, f"_{key}", self.metadata[key])

        # `__str__` re-emits records 5..4+NWD from `_text_lines`, which the
        # parser stores as the whole section including its four header records.
        # Only the slice from 4 is ever read, so the padding is never written.
        mt451._text_lines = [""] * 4 + text
        mt451._nwd = len(text)
        mt451._directory = directory
        mt451._nxc = len(directory)

        # Set from `evaluation_info` as well, and not instead: `__str__` falls
        # back to these seven when there is no text block, and a NuclideInfo
        # whose text survived should not carry a header that disagrees with it.
        for attr, key in (("_zsymam", "material_id"), ("_alab", "laboratory"),
                          ("_edate", "eval_date"), ("_auth", "authors"),
                          ("_ref", "reference"), ("_ddate", "dist_date"),
                          ("_rdate", "revision_date")):
            setattr(mt451, attr, self.evaluation_info.get(key) or "")

        return mt451

    # ------------------------------------------------------------------
    # ACE adapter
    # ------------------------------------------------------------------

    @classmethod
    def from_ace(cls, ace: "Ace") -> "NuclideInfo":
        """Create from an ACE object.

        Parameters
        ----------
        ace : Ace
            Parsed ACE file.

        Returns
        -------
        NuclideInfo
        """
        from kika._constants import BOLTZMANN_CONSTANT

        kT_MeV = ace.header.temperature or 0.0
        temperature = kT_MeV / BOLTZMANN_CONSTANT if kT_MeV else 0.0

        return cls(
            nuclide_id=ace.header.zaid or 0,
            atomic_weight_ratio=ace.header.atomic_weight_ratio or 0.0,
            temperature=temperature,
            evaluation_info={
                "date": ace.header.date,
                "source": getattr(ace.header, "source", None),
                "reference": ace.header.comment,
            },
            metadata={
                "source_format": "ace",
                "ace_extension": ace.header.extension,
                "ace_matid": ace.header.matid,
                "ace_format_version": ace.header.format_version,
            },
        )

    def __repr__(self) -> str:
        return (
            f"NuclideInfo(ZA={self.nuclide_id}, "
            f"AWR={self.atomic_weight_ratio:.4f}, "
            f"T={self.temperature:.1f} K)"
        )
