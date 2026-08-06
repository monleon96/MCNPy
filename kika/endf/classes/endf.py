"""
ENDF file representation.

Contains multiple MF files organized in a dictionary.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from .mf import MF


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
        """
        if self.zaid is not None:
            from kika._utils import zaid_to_symbol
            return zaid_to_symbol(self.zaid)
        return None
    
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
        """
        if mf not in self.files:
            raise KeyError(f"MF file {mf} not found in ENDF")

        # Validate reconstructed is only used with MF3
        if reconstructed and mf != 3:
            raise ValueError(
                f"reconstructed is only supported for MF3 (cross sections), not MF{mf}."
            )

        # Auto-set uncertainty based on MF file type
        if uncertainty is None:
            uncertainty = (mf in (3, 4))

        # Force uncertainty=False for covariance files themselves
        if mf in (33, 34) and uncertainty:
            uncertainty = False

        # Validate that uncertainty is only used with MF3 or MF4
        if uncertainty and mf not in (3, 4):
            raise ValueError(
                f"uncertainty is only supported for MF3 and MF4, not MF{mf}."
            )

        # --- MF3: cross sections with optional reconstruction and uncertainty ---
        if mf == 3 and (reconstructed or uncertainty):
            return self._mf3_plot_data(mt, uncertainty, sigma, reconstructed,
                                       tolerance, **kwargs)

        # Get the main plot data
        plot_data = self.files[mf].to_plot_data(mt=mt, **kwargs)

        # If uncertainties are not requested, return just the plot data
        if not uncertainty:
            return plot_data

        # --- MF4: angular distributions with MF34 uncertainty ---
        uncertainty_band = None

        # Check if MF34 exists and has the requested MT section
        if 34 in self.files and mt in self.files[34].mt:
            try:
                # Extract the order parameter (required for MF4)
                order = kwargs.get('order')
                if order is None:
                    raise ValueError("'order' parameter is required when uncertainty=True for MF4")

                # Get MF34 covariance data (pass MF4 to populate Legendre coefficients)
                mf34_mt = self.files[34].mt[mt]
                mf4 = self.files.get(4)
                mf34_covmat = mf34_mt.to_ang_covmat(mf4_data=mf4)

                # Get isotope ID (ZAID)
                isotope_id = self.zaid if self.zaid is not None else int(mf34_mt._za)

                # Prepare kwargs for LegendreCovariance.to_plot_data - remove parameters we're setting explicitly
                styling_kwargs = {k: v for k, v in kwargs.items() if k not in ['order', 'mt', 'nuclide', 'uncertainty_type']}

                # Get uncertainty data from MF34 (returns tuple: (None, unc_data))
                _, unc_native = mf34_covmat.to_plot_data(
                    nuclide=isotope_id,
                    mt=mt,
                    order=order,
                    uncertainty_type='relative',
                    color=plot_data.color,  # Match the main plot color
                    **styling_kwargs
                )

                if unc_native is not None:
                    # Return the native MF34 uncertainty data directly (on sparse grid)
                    # This allows it to be:
                    # 1. Plotted as a step plot (preserves native structure)
                    # 2. Used as uncertainty band (PlotBuilder will handle interpolation)

                    # Apply sigma multiplier if needed
                    if sigma != 1.0:
                        import numpy as np
                        unc_native.y = np.array(unc_native.y) * sigma
                        # Update label to reflect sigma level
                        if unc_native.label:
                            unc_native.label = unc_native.label.replace('(σ %)', f'({sigma}σ %)')

                    # Match color to nominal data
                    unc_native.color = plot_data.color

                    # Use the native uncertainty data directly
                    uncertainty_band = unc_native

                    # Append uncertainty info to the main plot label
                    if plot_data.label is not None:
                        sigma_suffix = f" (±{sigma}σ)" if sigma != 1.0 else " (±1σ)"
                        plot_data.label = plot_data.label + sigma_suffix
                else:
                    uncertainty_band = None

            except Exception as e:
                # If anything goes wrong, just skip the uncertainty band
                print(f"Warning: Could not create uncertainty band: {e}")
                uncertainty_band = None

        return plot_data, uncertainty_band

    def _mf3_plot_data(self, mt: int, uncertainty: bool, sigma: float,
                       reconstructed: bool, tolerance: float, **kwargs):
        """Handle MF3 plot data with optional reconstruction and uncertainty.

        Returns PlotData if uncertainty=False, or (PlotData, unc_band) if True.
        """
        import warnings

        # --- Nominal data: reconstructed or raw ---
        plot_data = None

        if reconstructed:
            # Consume what the caller supplied; never reconstruct behind their
            # back. This used to call reconstruct_xs(), which produced numbers
            # its own docstring said were wrong, and labelled the curve
            # "(reconstructed)" without qualification.
            if self._pendf is None:
                warnings.warn(
                    "reconstructed=True but endf.pendf is not set — returning "
                    "the raw MF3 background. Populate it first, e.g. "
                    "endf.pendf = kika.processing.njoy_reconstruct(path, "
                    "njoy_executable=...)."
                )

            if self._pendf is not None and mt in self._pendf:
                plot_data = self._pendf[mt].to_plot_data(**kwargs)
                # Mark as reconstructed if user didn't supply a custom label
                if 'label' not in kwargs and plot_data.label is not None:
                    plot_data.label += " (reconstructed)"
                plot_data.data_source = "pendf"
            elif self._pendf is not None:
                warnings.warn(
                    f"MT{mt} not available in reconstructed data "
                    f"(available: {sorted(self._pendf.keys())}). "
                    f"Falling back to raw MF3."
                )

        # Fall back to raw MF3 if not reconstructed or reconstruction unavailable
        if plot_data is None:
            plot_data = self.files[3].to_plot_data(mt=mt, **kwargs)

        # --- Uncertainty from MF33 ---
        if not uncertainty:
            return plot_data

        uncertainty_band = None

        if 33 in self.files and mt in self.files[33].mt:
            try:
                mf33_mt = self.files[33].mt[mt]

                # Build sibling and MF3 section maps for NC LTY=0 resolution
                mf33_siblings = {
                    int(k): v for k, v in self.files[33].sections.items()
                    if int(k) != mt
                } if hasattr(self.files[33], 'sections') else None

                # No auto-reconstruction: if the caller set endf.pendf it is
                # overlaid below, otherwise the raw MF3 background is used.
                # This used to call reconstruct_xs() inside a bare
                # `except Exception: pass`, so a failure of a reconstructor
                # already known to be wrong was invisible twice over.

                # Build cross-section map: start with raw MF3, then overlay
                # reconstructed (PENDF) data for MTs that need resonance
                # contributions (MT1, MT2, MT102, ...).
                mf3_secs = None
                if 3 in self.files and hasattr(self.files[3], 'sections'):
                    mf3_secs = {int(k): v for k, v in
                                self.files[3].sections.items()}
                if self._pendf:
                    if mf3_secs is None:
                        mf3_secs = {}
                    for k, v in self._pendf.items():
                        mf3_secs[int(k)] = v

                mf33_covmat = mf33_mt.to_xs_covmat(
                    sibling_sections=mf33_siblings,
                    mf3_sections=mf3_secs,
                )

                isotope_id = self.zaid if self.zaid is not None else int(mf33_mt._za)

                # Prepare styling kwargs (remove MF3-specific params)
                styling_kwargs = {k: v for k, v in kwargs.items()
                                  if k not in ['mt', 'nuclide', 'uncertainty_type']}

                _, unc_native = mf33_covmat.to_plot_data(
                    nuclide=isotope_id,
                    mt=mt,
                    sigma=sigma,
                    color=plot_data.color,
                    **styling_kwargs
                )

                if unc_native is not None:
                    if sigma != 1.0 and unc_native.label:
                        unc_native.label = unc_native.label.replace(
                            '(σ %)', f'({sigma}σ %)')

                    unc_native.color = plot_data.color
                    uncertainty_band = unc_native

                    if plot_data.label is not None:
                        sigma_suffix = f" (±{sigma}σ)" if sigma != 1.0 else " (±1σ)"
                        plot_data.label += sigma_suffix

            except Exception as e:
                print(f"Warning: Could not create MF33 uncertainty band: {e}")
                uncertainty_band = None

        return plot_data, uncertainty_band
    
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
