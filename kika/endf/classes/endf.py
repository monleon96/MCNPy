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
    
    def to_plot_data(self, mf: int, mt: int, uncertainty: bool = True, 
                     sigma: float = 1.0, **kwargs):
        """
        Create a PlotData object from the specified MF and MT sections.
        
        This is a convenience method that delegates to the MF file's to_plot_data method.
        For MF4 (angular distributions), this requires an 'order' parameter.
        
        For MF4, this method automatically checks MF34 for uncertainty data and returns
        it along with the nominal data, enabling easy plotting with uncertainty bands.
        This matches the unified API pattern used by multigroup covariance classes.
        
        Parameters
        ----------
        mf : int
            MF file number to extract data from
        mt : int
            MT section number to extract data from
        uncertainty : bool, optional
            If True and mf=4, extract uncertainty bands from MF34. 
            If None (default), automatically set to True for MF4, False for other MF files.
            Only works for MF4 since it combines data from MF4 (nominal) and MF34 (uncertainties).
        sigma : float, optional
            Number of sigma levels for uncertainty bands (default: 1.0 for 1σ).
            Only used when uncertainty=True.
        **kwargs
            Additional parameters passed to the underlying to_plot_data method.
            For MF4, this should include 'order' (Legendre polynomial order).
            May also include styling parameters (label, color, linestyle, etc.)
            
        Returns
        -------
        PlotData or tuple of (PlotData, UncertaintyBand or None)
            - For MF4 with uncertainty=True (default): Returns tuple of 
              (PlotData, UncertaintyBand or None). The UncertaintyBand will be None 
              if MF34 data is not available.
            - For other MF files or uncertainty=False: Returns PlotData object only.
            
        Raises
        ------
        KeyError
            If the MF file or MT section doesn't exist
        AttributeError
            If the MT section doesn't support to_plot_data
        ValueError
            If uncertainty=True for MF files other than MF4
            
        Examples
        --------
        >>> # For MF4 data - automatically includes uncertainties if MF34 present
        >>> endf = read_endf('fe56.endf')
        >>> data, unc_band = endf.to_plot_data(mf=4, mt=2, order=1)
        >>> 
        >>> # Explicitly control uncertainty extraction
        >>> data, unc_band = endf.to_plot_data(mf=4, mt=2, order=1, 
        ...                                      uncertainty=True, 
        ...                                      sigma=1.0)
        >>> 
        >>> # Disable uncertainty extraction for MF4
        >>> data = endf.to_plot_data(mf=4, mt=2, order=1, uncertainty=False)
        >>> 
        >>> # Use with PlotBuilder - unified API with multigroup classes
        >>> from kika.plotting import PlotBuilder
        >>> data, unc_band = endf.to_plot_data(mf=4, mt=2, order=1)
        >>> builder = PlotBuilder()
        >>> if unc_band is not None:
        ...     builder.add_data(data, uncertainty=unc_band)
        ... else:
        ...     builder.add_data(data)
        >>> fig = builder.build()
        """
        if mf not in self.files:
            raise KeyError(f"MF file {mf} not found in ENDF")
        
        # Auto-set uncertainty based on MF file type
        if uncertainty is None:
            # Default to True for MF4 (to extract MF34 uncertainties)
            # Default to False for MF34 (since MF34 IS the uncertainty data)
            # Default to False for all other MF files
            uncertainty = (mf == 4)
        
        # Force uncertainty=False for MF34 since it IS the uncertainty data
        if mf == 34 and uncertainty:
            uncertainty = False  # Silently override - MF34 doesn't have uncertainties of uncertainties
        
        # Validate that uncertainty is only used with MF4
        if uncertainty and mf != 4:
            raise ValueError(
                f"uncertainty is only supported for MF4 (angular distributions), not MF{mf}. "
                "This feature requires combining data from MF4 (nominal) and MF34 (uncertainties)."
            )
        
        # Get the main plot data
        plot_data = self.files[mf].to_plot_data(mt=mt, **kwargs)
        
        # If uncertainties are not requested, return just the plot data
        if not uncertainty:
            return plot_data
        
        # Try to create uncertainty band from MF34 data
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
                
                # Prepare kwargs for MF34CovMat.to_plot_data - remove parameters we're setting explicitly
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
