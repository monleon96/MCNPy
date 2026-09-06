"""
ENDF file representation.

Contains multiple MF files organized in a dictionary.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from .mf import MF
from ..plotting import endf_to_plot_data


@dataclass
class ENDF:
    """
    Data class representing an ENDF file.
    """
    files: Dict[int, MF] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    mat: Optional[int] = None  # MAT number from ENDF file
    _pendf: Optional[Dict] = field(default=None, repr=False)
    
    def add_file(self, mf: MF) -> None:
        """Add an MF file to this ENDF file"""
        self.files[mf.number] = mf
    
    def get_file(self, mf_number: int) -> Optional[MF]:
        """Get an MF file by number"""
        return self.files.get(mf_number)
    
    @property
    def zaid(self) -> Optional[int]:
        """
        Get the ZAID number derived from the MAT number.
        
        Returns
        -------
        int or None
            ZAID number if MAT is available and in the mapping, None otherwise
        """
        if self.mat is not None:
            from kika._constants import ENDF_MAT_TO_ZAID
            return ENDF_MAT_TO_ZAID.get(self.mat, None)
        return None
    
    @property
    def isotope(self) -> Optional[str]:
        """
        Get the isotope symbol (e.g., 'Fe56') from the ZAID.
        
        Returns
        -------
        str or None
            Isotope symbol like 'Fe56' if ZAID is available, None otherwise

        Notes
        -----
        ``None`` for a thermal scattering evaluation, and correctly so — see
        :attr:`is_thermal_scattering`.
        """
        if self.zaid is not None:
            from kika._utils import zaid_to_symbol
            return zaid_to_symbol(self.zaid)
        return None

    @property
    def is_thermal_scattering(self) -> bool:
        """Whether this tape is a thermal scattering (TSL) evaluation.

        Read from MF1/451's ``NSUB``, which is the format's own answer rather
        than a guess from the MAT range or the filename. It matters because
        :attr:`zaid` and :attr:`isotope` are ``None`` here and will stay that
        way: a TSL material is a bound scatterer in a compound — H in H₂O — not
        a nuclide. Its MAT (1-8399) is below the lowest key of
        ``ENDF_MAT_TO_ZAID``, and its ZA is a *pseudo*-ZA (126 for beryllium
        metal) that means nothing arithmetically. Use
        :func:`kika.endf.thermal_scatterer` for the identity instead.

        Requires MF1 to have been parsed; ``False`` if it was not.
        """
        mf1 = self.files.get(1)
        mt451 = mf1.sections.get(451) if mf1 is not None else None
        if mt451 is None:
            return False
        from .mf7.scatterer import THERMAL_SCATTERING_NSUB
        return mt451.sublibrary == THERMAL_SCATTERING_NSUB

    @property
    def mf(self) -> Dict[int, MF]:
        """
        Direct access to MF files dictionary.
        
        This allows accessing MF files like: endf.mf[1]
        """
        return self.files
    
    def to_plot_data(self, mf: int, mt: int, uncertainty: bool = None,
                     sigma: float = 1.0, reconstructed: bool = False,
                     tolerance: float = 1e-3, **kwargs):
        """
        Create a PlotData object from the specified MF and MT sections.

        This is a convenience method that delegates to the MF file's to_plot_data method.

        For MF3 (cross sections), supports:
        - ``reconstructed=True``: auto-reconstructs from MF2 resonance parameters + MF3
          background, producing the full pointwise cross section across all energies.
          Requires MF2 to be loaded; falls back to raw MF3 if unavailable.
        - ``uncertainty=True``: extracts uncertainty bands from MF33 covariance data.

        For MF4 (angular distributions), supports:
        - ``uncertainty=True``: extracts uncertainty bands from MF34 covariance data.
          Requires an 'order' parameter.

        Parameters
        ----------
        mf : int
            MF file number to extract data from
        mt : int
            MT section number to extract data from
        uncertainty : bool, optional
            If True, extract uncertainty bands from covariance data (MF33 for MF3,
            MF34 for MF4). If None (default), automatically set to True for MF3
            and MF4, False for other MF files.
        sigma : float, optional
            Number of sigma levels for uncertainty bands (default: 1.0 for 1σ).
            Only used when uncertainty=True.
        reconstructed : bool, optional
            If True and mf=3, reconstruct pointwise cross sections from MF2
            resonance parameters + MF3 background. Results are cached in
            ``self.pendf``. Default: False.
        tolerance : float, optional
            Linearization tolerance for resonance reconstruction (default 0.1%).
            Only used when reconstructed=True.
        **kwargs
            Additional parameters passed to the underlying to_plot_data method.
            For MF4, this should include 'order' (Legendre polynomial order).
            May also include styling parameters (label, color, linestyle, etc.)

        Returns
        -------
        PlotData or tuple of (PlotData, UncertaintyBand or None)
            - For MF3/MF4 with uncertainty=True (default): Returns tuple of
              (PlotData, UncertaintyBand or None). The UncertaintyBand will be None
              if covariance data (MF33/MF34) is not available.
            - For other MF files or uncertainty=False: Returns PlotData object only.

        Raises
        ------
        KeyError
            If the MF file or MT section doesn't exist
        ValueError
            If uncertainty=True for MF files other than MF3/MF4
            If reconstructed=True for MF files other than MF3

        Examples
        --------
        >>> endf = read_endf('fe56.endf', mf_numbers=[2, 3, 33])
        >>>
        >>> # MF3 with reconstruction (full cross section including resonance region)
        >>> data, unc = endf.to_plot_data(mf=3, mt=2, reconstructed=True)
        >>>
        >>> # MF3 raw background only, no uncertainties
        >>> data = endf.to_plot_data(mf=3, mt=2, uncertainty=False)
        >>>
        >>> # MF4 with uncertainties from MF34
        >>> data, unc = endf.to_plot_data(mf=4, mt=2, order=1)
        >>>
        >>> # Plot with PlotBuilder
        >>> from kika.plotting import PlotBuilder
        >>> builder = PlotBuilder()
        >>> builder.add_data(endf.to_plot_data(mf=3, mt=102, reconstructed=True))
        >>> fig = builder.build()

        Notes
        -----
        The orchestration lives in :mod:`kika.endf.plotting`, which is where
        the two associations — MF33 covaries MF3, MF34 covaries MF4 — and the
        two defects they carry are written down. This method stays because it
        is the documented entry point and the app's; it adds nothing.
        """
        return endf_to_plot_data(self, mf, mt, uncertainty=uncertainty,
                                 sigma=sigma, reconstructed=reconstructed,
                                 tolerance=tolerance, **kwargs)

    @property
    def pendf(self) -> Optional[Dict]:
        """Reconstructed pointwise cross sections, or *None*.

        Set this to give the resonance-reconstructed sigma(E) to whatever needs
        it — plotting, MF33 NC LTY=0 resolution, the sigma-weighted MF34
        multigroup collapse::

            endf.pendf = njoy_reconstruct(path, njoy_executable=njoy)

        Nothing populates it on its own any more. ``ENDF.reconstruct_xs()``
        used to, silently, using an in-Python reconstructor its own docstring
        called "not working correctly" — so every consumer of ``pendf`` was
        quietly weighted by numbers nobody trusted. Choosing the source is the
        caller's, and it is now visible in their code.

        Expects ``{MT: section}``. Both ``MF3MT`` and ``CrossSection`` qualify,
        but they do **not** spell sigma the same way: ``MF3MT`` exposes
        ``energies`` and ``cross_sections``, ``CrossSection`` exposes
        ``energies`` and ``values``. Only ``energies`` is common.

        The asymmetry is load-bearing, because the two producers disagree —
        ``kika.endf.processing.reconstruct`` yields ``MF3MT``,
        ``kika.processing.njoy_reconstruct`` yields ``CrossSection``. A
        consumer that reads one spelling silently rejects half its callers,
        which is what ``MF34_to_MG`` did until
        ``kika.cov.multigroup.collapse._pendf_grid`` was written. Read sigma
        through that helper, or handle both names explicitly.
        """
        return self._pendf

    @pendf.setter
    def pendf(self, sections: Optional[Dict]) -> None:
        self._pendf = sections

    def __repr__(self):
        return f"ENDF({len(self.files)} files)"
    
    def __getitem__(self, mf_number: int) -> MF:
        """
        Allow accessing MF files using dictionary-like syntax: endf[1]
        
        Args:
            mf_number: The MF file number to retrieve
            
        Returns:
            The requested MF file
            
        Raises:
            KeyError: If the MF file doesn't exist
        """
        if mf_number not in self.files:
            raise KeyError(f"MF file {mf_number} not found in ENDF")
        return self.files[mf_number]
