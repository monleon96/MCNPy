"""
Format-agnostic angular distribution for a single reaction.

Supports Legendre, tabulated, and isotropic representations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from kika.ace.classes.ace import Ace
    from kika.cov.legendre_covariance import LegendreCovariance
    from kika.endf.classes.mf4.base import MF4MT
    from kika.plotting.plot_data import LegendreCoeffPlotData, LegendreUncertaintyPlotData


@dataclass
class AngularDistribution:
    """Format-agnostic angular distribution for one reaction.

    Attributes
    ----------
    energies : np.ndarray
        Incident-neutron energies in eV, sorted ascending.
    coefficients : Dict[int, np.ndarray]
        Legendre coefficients ``{order: values[n_energies]}``.
        Order 0 is always 1.0 (ENDF normalised PDF convention).
        Present for ``"legendre"`` and ``"isotropic"`` representations;
        may also be populated for ``"tabulated"`` via projection.
    reaction : int
        MT number.
    nuclide_id : int
        ZA = 1000*Z + A.
    frame : str
        Reference frame: ``"CM"`` or ``"LAB"``.
    representation : str
        ``"legendre"``, ``"tabulated"``, ``"isotropic"``, or ``"mixed"``.
    tabulated_data : dict or None
        For tabulated representation:
        ``{"cosines": List[List[float]], "probabilities": List[List[float]],
          "angular_interpolation": List[List[Tuple[int,int]]]}``.
    metadata : dict
        Format-specific extras for lossless round-trips.
        ENDF keys: ``"mat"``, ``"awr"``, ``"ltt"``, ``"li"``, ``"lct"``,
        ``"energy_interpolation"``.
    uncertainties : Dict[int, np.ndarray] or None
        Optional diagonal standard deviations ``{order: std_dev[n_bins]}``.
        Populated via :meth:`attach_uncertainties`.
    uncertainty_energies : np.ndarray or None
        Energy bin edges (N+1 points for N bins) in eV for the uncertainty
        data.  Different from the pointwise ``energies`` grid.
    uncertainty_type : str
        ``"relative"`` (δA_ℓ/A_ℓ, fractional) or ``"absolute"`` (δA_ℓ).
    """

    energies: np.ndarray
    coefficients: Dict[int, np.ndarray] = field(default_factory=dict)
    reaction: int = 0
    nuclide_id: int = 0
    frame: str = "CM"
    representation: str = "legendre"
    tabulated_data: Optional[Dict] = None
    metadata: Dict = field(default_factory=dict)

    # Optional uncertainty data (diagonal of covariance matrix)
    uncertainties: Optional[Dict[int, np.ndarray]] = None
    uncertainty_energies: Optional[np.ndarray] = None
    uncertainty_type: str = "relative"

    # ------------------------------------------------------------------
    # ENDF adapter
    # ------------------------------------------------------------------

    @classmethod
    def from_endf(cls, mf4mt: "MF4MT") -> "AngularDistribution":
        """Create from an ENDF ``MF4MT`` (any subclass).

        Parameters
        ----------
        mf4mt : MF4MT
            Parsed MF4/MT section (Isotropic, Legendre, Tabulated, or Mixed).

        Returns
        -------
        AngularDistribution
        """
        from kika.endf.classes.mf4.isotropic import MF4MTIsotropic
        from kika.endf.classes.mf4.polynomial import MF4MTLegendre
        from kika.endf.classes.mf4.tabulated import MF4MTTabulated

        za = int(mf4mt.zaid) if mf4mt.zaid is not None else 0
        frame = mf4mt.frame  # "CM" or "LAB"
        ltt = mf4mt._ltt
        li = mf4mt._li
        lct = mf4mt._lct
        mat = getattr(mf4mt, "_mat", None)
        awr = mf4mt.atomic_weight_ratio

        common_meta = {
            "mat": mat,
            "awr": awr,
            "ltt": ltt,
            "li": li,
            "lct": lct,
        }

        if isinstance(mf4mt, MF4MTIsotropic):
            return cls(
                energies=np.array([], dtype=float),
                coefficients={0: np.array([1.0])},
                reaction=mf4mt.number,
                nuclide_id=za,
                frame=frame,
                representation="isotropic",
                metadata=common_meta,
            )

        if isinstance(mf4mt, MF4MTLegendre):
            energies = np.asarray(mf4mt.energies, dtype=float)
            n_e = len(energies)

            # Determine max order across all energy points
            max_order = 0
            for row in mf4mt.legendre_coefficients:
                if row:
                    max_order = max(max_order, len(row))

            # Build coefficients dict  {order: array[n_e]}
            coefficients: Dict[int, np.ndarray] = {
                0: np.ones(n_e, dtype=float),
            }
            for l in range(1, max_order + 1):
                arr = np.zeros(n_e, dtype=float)
                for i, row in enumerate(mf4mt.legendre_coefficients):
                    if row and (l - 1) < len(row):
                        arr[i] = row[l - 1]
                coefficients[l] = arr

            common_meta["energy_interpolation"] = list(mf4mt.energy_interpolation)

            return cls(
                energies=energies,
                coefficients=coefficients,
                reaction=mf4mt.number,
                nuclide_id=za,
                frame=frame,
                representation="legendre",
                metadata=common_meta,
            )

        if isinstance(mf4mt, MF4MTTabulated):
            energies = np.asarray(mf4mt.energies, dtype=float)

            common_meta["energy_interpolation"] = list(mf4mt.energy_interpolation)

            tab_data = {
                "cosines": [list(c) for c in mf4mt.cosines],
                "probabilities": [list(p) for p in mf4mt.probabilities],
                "angular_interpolation": [
                    list(ai) for ai in mf4mt.cosine_interpolation
                ],
            }

            return cls(
                energies=energies,
                coefficients={},
                reaction=mf4mt.number,
                nuclide_id=za,
                frame=frame,
                representation="tabulated",
                tabulated_data=tab_data,
                metadata=common_meta,
            )

        # Mixed (LTT=3): Legendre for low E, tabulated for high E
        try:
            from kika.endf.classes.mf4.mixed import MF4MTMixed

            if isinstance(mf4mt, MF4MTMixed):
                # Legendre part
                leg_energies = np.asarray(mf4mt.legendre_energies, dtype=float)
                n_leg = len(leg_energies)

                max_order = 0
                for row in mf4mt.legendre_coefficients:
                    if row:
                        max_order = max(max_order, len(row))

                coefficients: Dict[int, np.ndarray] = {
                    0: np.ones(n_leg, dtype=float)
                }
                for l in range(1, max_order + 1):
                    arr = np.zeros(n_leg, dtype=float)
                    for i, row in enumerate(mf4mt.legendre_coefficients):
                        if row and (l - 1) < len(row):
                            arr[i] = row[l - 1]
                    coefficients[l] = arr

                # Combined energy grid (Legendre + tabulated)
                tab_energies = np.asarray(mf4mt.tabulated_energies, dtype=float)
                all_energies = np.concatenate([leg_energies, tab_energies])

                tab_data = {
                    "cosines": [list(c) for c in mf4mt.tabulated_cosines],
                    "probabilities": [list(p) for p in mf4mt.tabulated_probabilities],
                    "angular_interpolation": [
                        list(ai)
                        for ai in getattr(mf4mt, "_angular_interpolation", [])
                    ],
                    "tabulated_energies": list(tab_energies),
                    "legendre_energies": list(leg_energies),
                }

                common_meta["energy_interpolation"] = list(
                    getattr(mf4mt, "_interpolation", [])
                )
                common_meta["tab_interpolation"] = list(
                    getattr(mf4mt, "_tab_interpolation", [])
                )

                return cls(
                    energies=all_energies,
                    coefficients=coefficients,
                    reaction=mf4mt.number,
                    nuclide_id=za,
                    frame=frame,
                    representation="mixed",
                    tabulated_data=tab_data,
                    metadata=common_meta,
                )
        except ImportError:
            pass

        # Fallback for unknown subtypes
        energies_list = getattr(mf4mt, "_energies", [])
        energies = (
            np.asarray(energies_list, dtype=float)
            if energies_list
            else np.array([], dtype=float)
        )

        return cls(
            energies=energies,
            coefficients={},
            reaction=mf4mt.number,
            nuclide_id=za,
            frame=frame,
            representation="mixed",
            metadata=common_meta,
        )

    # ------------------------------------------------------------------
    # ACE adapter
    # ------------------------------------------------------------------

    @classmethod
    def from_ace(cls, ace: "Ace", mt: int) -> "AngularDistribution":
        """Create from an ACE object for a given MT.

        Parameters
        ----------
        ace : Ace
            Parsed ACE file.
        mt : int
            Reaction MT number.

        Returns
        -------
        AngularDistribution

        Raises
        ------
        ValueError
            If the MT is not available or uses Kalbach-Mann representation.
        """
        import logging
        from kika._constants import BOLTZMANN_CONSTANT
        from kika.ace.classes.angular_distribution.distributions.isotropic import (
            IsotropicAngularDistribution,
        )
        from kika.ace.classes.angular_distribution.distributions.tabulated import (
            TabulatedAngularDistribution,
        )
        from kika.ace.classes.angular_distribution.distributions.equiprobable import (
            EquiprobableAngularDistribution,
        )
        from kika.ace.classes.angular_distribution.distributions.kalbach_mann import (
            KalbachMannAngularDistribution,
        )

        logger = logging.getLogger(__name__)

        # Look up the ACE angular distribution
        if mt == 2:
            dist = ace.angular_distributions.elastic
        else:
            dist = ace.angular_distributions.incident_neutron.get(mt)

        if dist is None:
            raise ValueError(
                f"No angular distribution for MT={mt} in ACE file "
                f"(ZAID={ace.header.zaid})"
            )

        # Kalbach-Mann: not cleanly separable
        if isinstance(dist, KalbachMannAngularDistribution):
            logger.warning(
                "MT=%d uses Kalbach-Mann angular-energy distribution — "
                "cannot extract angular distribution alone", mt
            )
            raise ValueError(
                f"MT={mt} uses Kalbach-Mann representation which couples "
                f"energy and angle; cannot extract angular distribution alone"
            )

        zaid = ace.header.zaid or 0
        ace_energies = np.asarray(dist.energies, dtype=float) * 1e6  # MeV → eV
        ace_dist_type = type(dist).__name__

        meta = {
            "source_format": "ace",
            "ace_distribution_type": ace_dist_type,
            "ace_zaid": zaid,
        }

        # Isotropic
        if isinstance(dist, IsotropicAngularDistribution):
            n_e = len(ace_energies)
            return cls(
                energies=ace_energies,
                coefficients={0: np.ones(n_e, dtype=float)},
                reaction=mt,
                nuclide_id=zaid,
                frame="LAB",
                representation="isotropic",
                metadata=meta,
            )

        # Tabulated
        if isinstance(dist, TabulatedAngularDistribution):
            tab_data = {
                "cosines": [list(c) for c in dist.cosine_grid],
                "probabilities": [list(p) for p in dist.pdf],
                "angular_interpolation": [],
            }
            return cls(
                energies=ace_energies,
                coefficients={},
                reaction=mt,
                nuclide_id=zaid,
                frame="LAB",
                representation="tabulated",
                tabulated_data=tab_data,
                metadata=meta,
            )

        # Equiprobable → convert to tabulated
        if isinstance(dist, EquiprobableAngularDistribution):
            cosines_list = []
            pdf_list = []
            for energy_bins in dist.cosine_bins:
                # 33 bin edges → 32 bins
                edges = np.asarray(energy_bins, dtype=float)
                centers = 0.5 * (edges[:-1] + edges[1:])
                widths = np.diff(edges)
                pdf = np.zeros_like(widths)
                nonzero = widths > 0
                pdf[nonzero] = (1.0 / 32.0) / widths[nonzero]
                cosines_list.append(centers.tolist())
                pdf_list.append(pdf.tolist())

            tab_data = {
                "cosines": cosines_list,
                "probabilities": pdf_list,
                "angular_interpolation": [],
            }
            return cls(
                energies=ace_energies,
                coefficients={},
                reaction=mt,
                nuclide_id=zaid,
                frame="LAB",
                representation="tabulated",
                tabulated_data=tab_data,
                metadata=meta,
            )

        raise ValueError(
            f"Unsupported ACE angular distribution type: {ace_dist_type}"
        )

    @classmethod
    def all_from_ace(cls, ace: "Ace") -> Dict[int, "AngularDistribution"]:
        """Extract all available angular distributions from an ACE file.

        Parameters
        ----------
        ace : Ace
            Parsed ACE file.

        Returns
        -------
        Dict[int, AngularDistribution]
            Mapping of MT number to ``AngularDistribution``.
        """
        result: Dict[int, "AngularDistribution"] = {}

        # Elastic (MT=2)
        if ace.angular_distributions.has_elastic_data:
            try:
                result[2] = cls.from_ace(ace, mt=2)
            except ValueError:
                pass

        # Other neutron reactions
        for mt in ace.angular_distributions.get_neutron_reaction_mt_numbers():
            try:
                result[mt] = cls.from_ace(ace, mt=mt)
            except ValueError:
                continue

        return result

    # ------------------------------------------------------------------
    # Analysis methods
    # ------------------------------------------------------------------

    def project_to_legendre(self, max_order: int = 6) -> None:
        """Fit Legendre coefficients from tabulated PDF data.

        Uses Gauss-Legendre quadrature to compute:
            a_l = (2l+1)/2 * integral f(mu) P_l(mu) dmu

        Coefficients are normalised so a_0 = 1.0 (ENDF convention).
        The original tabulated data is preserved; only ``coefficients``
        is populated.

        Parameters
        ----------
        max_order : int
            Maximum Legendre order to fit (default 6).

        Raises
        ------
        ValueError
            If no tabulated data is available.
        """
        if self.tabulated_data is None:
            raise ValueError(
                "No tabulated data to project — representation is "
                f"'{self.representation}'"
            )

        cosines_list = self.tabulated_data["cosines"]
        pdf_list = self.tabulated_data["probabilities"]
        n_energies = len(cosines_list)

        # Gauss-Legendre quadrature nodes and weights
        n_quad = max(64, 2 * max_order + 1)
        quad_nodes, quad_weights = np.polynomial.legendre.leggauss(n_quad)

        # Pre-compute Legendre polynomials at quadrature nodes
        leg_at_nodes = np.zeros((max_order + 1, n_quad))
        for l in range(max_order + 1):
            c = np.zeros(l + 1)
            c[l] = 1.0
            leg_at_nodes[l] = np.polynomial.legendre.legval(quad_nodes, c)

        coefficients: Dict[int, np.ndarray] = {
            l: np.zeros(n_energies, dtype=float) for l in range(max_order + 1)
        }

        for i in range(n_energies):
            cos_pts = np.asarray(cosines_list[i], dtype=float)
            pdf_pts = np.asarray(pdf_list[i], dtype=float)

            # Interpolate tabulated PDF onto quadrature nodes
            pdf_at_nodes = np.interp(quad_nodes, cos_pts, pdf_pts)

            # Compute raw coefficients
            raw = np.zeros(max_order + 1)
            for l in range(max_order + 1):
                raw[l] = (2 * l + 1) / 2.0 * np.sum(
                    quad_weights * pdf_at_nodes * leg_at_nodes[l]
                )

            # Normalise so a_0 = 1.0
            if abs(raw[0]) > 1e-30:
                raw /= raw[0]
            raw[0] = 1.0

            for l in range(max_order + 1):
                coefficients[l][i] = raw[l]

        self.coefficients = coefficients

    def evaluate_pdf(
        self,
        energy: float,
        cosines: Optional[np.ndarray] = None,
        num_points: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate the angular PDF at a specific incident energy.

        Works for any representation (Legendre, tabulated, isotropic, mixed).

        Parameters
        ----------
        energy : float
            Incident neutron energy in eV.
        cosines : np.ndarray, optional
            Cosine values at which to evaluate. If None, uses
            ``np.linspace(-1, 1, num_points)``.
        num_points : int
            Number of points if ``cosines`` is None (default 100).

        Returns
        -------
        (cosines, pdf_values) : tuple of np.ndarray
        """
        if cosines is None:
            cosines = np.linspace(-1, 1, num_points)
        cosines = np.asarray(cosines, dtype=float)

        if self.representation == "isotropic":
            return cosines, np.full_like(cosines, 0.5)

        # Determine which sub-representation to use for mixed
        use_legendre = False
        use_tabulated = False

        if self.representation == "legendre":
            use_legendre = True
        elif self.representation == "tabulated":
            use_tabulated = True
        elif self.representation == "mixed" and self.tabulated_data is not None:
            leg_energies = self.tabulated_data.get("legendre_energies")
            tab_energies = self.tabulated_data.get("tabulated_energies")
            if leg_energies is not None and len(leg_energies) > 0:
                if energy <= max(leg_energies):
                    use_legendre = True
                else:
                    use_tabulated = True
            elif tab_energies is not None:
                use_tabulated = True
            else:
                use_legendre = bool(self.coefficients)
        else:
            # Fallback: use coefficients if available, otherwise tabulated
            if self.coefficients:
                use_legendre = True
            elif self.tabulated_data is not None:
                use_tabulated = True

        if use_legendre and self.coefficients:
            return self._evaluate_pdf_legendre(energy, cosines)
        if use_tabulated and self.tabulated_data is not None:
            return self._evaluate_pdf_tabulated(energy, cosines)

        raise ValueError(
            f"Cannot evaluate PDF: representation={self.representation!r}, "
            f"coefficients={'yes' if self.coefficients else 'no'}, "
            f"tabulated={'yes' if self.tabulated_data else 'no'}"
        )

    def _evaluate_pdf_legendre(
        self, energy: float, cosines: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate PDF from Legendre coefficients at a given energy."""
        # For mixed, use legendre_energies if available
        if (
            self.representation == "mixed"
            and self.tabulated_data
            and "legendre_energies" in self.tabulated_data
        ):
            energies = np.asarray(self.tabulated_data["legendre_energies"], dtype=float)
        else:
            energies = self.energies

        max_order = max(self.coefficients.keys())

        # Interpolate coefficients to requested energy
        interp_coeffs = np.zeros(max_order + 1)
        for l in range(max_order + 1):
            if l in self.coefficients:
                arr = self.coefficients[l]
                e_grid = energies[:len(arr)]
                interp_coeffs[l] = np.interp(energy, e_grid, arr)
            else:
                interp_coeffs[l] = 0.0

        # Evaluate: f(mu) = sum_l (2l+1)/2 * a_l * P_l(mu)
        pdf = np.zeros_like(cosines)
        for l in range(max_order + 1):
            c = np.zeros(l + 1)
            c[l] = 1.0
            pl = np.polynomial.legendre.legval(cosines, c)
            pdf += (2 * l + 1) / 2.0 * interp_coeffs[l] * pl

        return cosines, pdf

    def _evaluate_pdf_tabulated(
        self, energy: float, cosines: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate PDF from tabulated data at a given energy."""
        tab = self.tabulated_data
        cos_list = tab["cosines"]
        pdf_list = tab["probabilities"]

        # Use tabulated_energies for mixed, otherwise self.energies
        if "tabulated_energies" in tab:
            energies = np.asarray(tab["tabulated_energies"], dtype=float)
        else:
            energies = self.energies

        n_tab = len(cos_list)
        if n_tab == 0:
            return cosines, np.full_like(cosines, 0.5)

        # Find bounding energy indices
        if energy <= energies[0]:
            idx_lo = idx_hi = 0
            frac = 0.0
        elif energy >= energies[n_tab - 1]:
            idx_lo = idx_hi = n_tab - 1
            frac = 0.0
        else:
            idx_hi = int(np.searchsorted(energies[:n_tab], energy))
            idx_lo = idx_hi - 1
            e_lo, e_hi = energies[idx_lo], energies[idx_hi]
            frac = (energy - e_lo) / (e_hi - e_lo) if e_hi != e_lo else 0.0

        # Interpolate PDF at the requested cosine points
        pdf_lo = np.interp(
            cosines,
            np.asarray(cos_list[idx_lo], dtype=float),
            np.asarray(pdf_list[idx_lo], dtype=float),
        )
        if idx_lo == idx_hi:
            return cosines, pdf_lo

        pdf_hi = np.interp(
            cosines,
            np.asarray(cos_list[idx_hi], dtype=float),
            np.asarray(pdf_list[idx_hi], dtype=float),
        )

        pdf = (1.0 - frac) * pdf_lo + frac * pdf_hi
        return cosines, pdf

    # ------------------------------------------------------------------
    # ENDF export
    # ------------------------------------------------------------------

    def to_endf(self, mat: Optional[int] = None) -> "MF4MT":
        """Convert back to an ENDF ``MF4MT`` subclass.

        Parameters
        ----------
        mat : int, optional
            MAT number.  Falls back to ``metadata["mat"]``.

        Returns
        -------
        MF4MT subclass
        """
        mat = mat if mat is not None else self.metadata.get("mat")
        awr = self.metadata.get("awr", 0.0)
        li = self.metadata.get("li", 0)
        lct = self.metadata.get("lct", 2)
        interp = self.metadata.get("energy_interpolation", [])

        if self.representation == "isotropic":
            from kika.endf.classes.mf4.isotropic import MF4MTIsotropic

            obj = MF4MTIsotropic(number=self.reaction)
            obj._za = float(self.nuclide_id)
            obj._awr = awr
            obj._mat = mat
            obj._li = 1
            obj._lct = lct
            return obj

        if self.representation == "legendre":
            from kika.endf.classes.mf4.polynomial import MF4MTLegendre

            obj = MF4MTLegendre(number=self.reaction)
            obj._za = float(self.nuclide_id)
            obj._awr = awr
            obj._mat = mat
            obj._li = li
            obj._lct = lct
            obj._energies = list(self.energies)
            obj._ne = len(self.energies)
            obj._interpolation = list(interp) if interp else [(len(self.energies), 2)]
            obj._nr = len(obj._interpolation)

            # Reconstruct coefficient rows (a_1, a_2, ...) per energy
            max_order = max(self.coefficients.keys()) if self.coefficients else 0
            n_e = len(self.energies)
            rows = []
            for i in range(n_e):
                row = []
                for l in range(1, max_order + 1):
                    if l in self.coefficients:
                        row.append(float(self.coefficients[l][i]))
                    else:
                        row.append(0.0)
                # Trim trailing zeros
                while row and abs(row[-1]) < 1e-30:
                    row.pop()
                rows.append(row)
            obj._legendre_coeffs = rows

            return obj

        if self.representation == "tabulated" and self.tabulated_data is not None:
            from kika.endf.classes.mf4.tabulated import MF4MTTabulated

            obj = MF4MTTabulated(number=self.reaction)
            obj._za = float(self.nuclide_id)
            obj._awr = awr
            obj._mat = mat
            obj._li = li
            obj._lct = lct
            obj._energies = list(self.energies)
            obj._ne = len(self.energies)
            obj._interpolation = list(interp) if interp else [(len(self.energies), 2)]
            obj._nr = len(obj._interpolation)
            obj._cosines = self.tabulated_data["cosines"]
            obj._probabilities = self.tabulated_data["probabilities"]
            obj._angular_interpolation = self.tabulated_data.get(
                "angular_interpolation", []
            )
            return obj

        if self.representation == "mixed" and self.tabulated_data is not None:
            from kika.endf.classes.mf4.mixed import MF4MTMixed

            obj = MF4MTMixed(number=self.reaction)
            obj._za = float(self.nuclide_id)
            obj._awr = awr
            obj._mat = mat
            obj._li = li
            obj._lct = lct

            # Legendre part
            leg_energies = self.tabulated_data.get("legendre_energies", [])
            obj._energies = list(leg_energies)
            obj._ne1 = len(leg_energies)
            obj._interpolation = list(interp) if interp else []
            obj._nr = len(obj._interpolation)

            # Reconstruct coefficient rows
            max_order = max(self.coefficients.keys()) if self.coefficients else 0
            n_leg = len(leg_energies)
            rows = []
            for i in range(n_leg):
                row = []
                for l in range(1, max_order + 1):
                    if l in self.coefficients:
                        row.append(float(self.coefficients[l][i]))
                    else:
                        row.append(0.0)
                while row and abs(row[-1]) < 1e-30:
                    row.pop()
                rows.append(row)
            obj._legendre_coeffs = rows

            # Tabulated part
            tab_energies = self.tabulated_data.get("tabulated_energies", [])
            obj._tabulated_energies = list(tab_energies)
            obj._ne2 = len(tab_energies)
            obj._tab_interpolation = list(
                self.metadata.get("tab_interpolation", [])
            )
            obj._nr_tab = len(obj._tab_interpolation)
            obj._tabulated_cosines = self.tabulated_data.get("cosines", [])
            obj._tabulated_probabilities = self.tabulated_data.get(
                "probabilities", []
            )
            obj._angular_interpolation = self.tabulated_data.get(
                "angular_interpolation", []
            )

            return obj

        raise ValueError(
            f"Cannot convert representation={self.representation!r} to ENDF MF4"
        )

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def to_plot_data(
        self, order: int, label: Optional[str] = None, **styling_kwargs
    ) -> "LegendreCoeffPlotData":
        """Create a ``LegendreCoeffPlotData`` for a given Legendre order.

        Parameters
        ----------
        order : int
            Legendre polynomial order to extract.
        label : str, optional
            Plot label.
        **styling_kwargs
            Passed to ``LegendreCoeffPlotData`` (color, linestyle, ...).

        Returns
        -------
        LegendreCoeffPlotData

        Raises
        ------
        ValueError
            If no Legendre coefficients are available, the requested order
            is invalid, or order > 0 for isotropic distributions.
        """
        from kika.plotting import LegendreCoeffPlotData
        from kika._utils import zaid_to_symbol

        if not self.coefficients:
            raise ValueError(
                "No Legendre coefficients available. "
                "Representation is '{}' — project to Legendre first.".format(
                    self.representation
                )
            )

        if self.representation == "isotropic" and order > 0:
            raise ValueError(
                "Isotropic distribution: only order=0 is meaningful (a_l=0 for l>0)."
            )

        if order not in self.coefficients:
            raise ValueError(
                f"Order {order} not available. Max order is {self.max_order}."
            )

        coeff_values = self.coefficients[order]
        # For mixed representation, coefficients only cover Legendre-region energies
        energies = self.energies[: len(coeff_values)]

        isotope = zaid_to_symbol(self.nuclide_id) if self.nuclide_id else None

        if label is None:
            label = f"L={order}"
            if isotope:
                label = f"{isotope} {label}"

        return LegendreCoeffPlotData(
            x=energies,
            y=coeff_values,
            order=order,
            isotope=isotope,
            mt=self.reaction,
            energy_range=(
                (float(energies.min()), float(energies.max()))
                if len(energies) > 0
                else None
            ),
            label=label,
            **styling_kwargs,
        )

    # ------------------------------------------------------------------
    # Uncertainties
    # ------------------------------------------------------------------

    def attach_uncertainties(self, covariance: "LegendreCovariance") -> None:
        """Populate uncertainty fields from a ``LegendreCovariance``.

        Extracts diagonal standard deviations for each available Legendre
        order from the covariance object.

        Parameters
        ----------
        covariance : LegendreCovariance
            Parsed MF34 covariance data.
        """
        orders = list(self.coefficients.keys())
        if not orders:
            return

        unc_dict: Dict[int, np.ndarray] = {}
        unc_energies = None
        unc_type = "relative"

        for order in orders:
            result = covariance.get_uncertainties_for_legendre_coefficient(
                isotope=self.nuclide_id, mt=self.reaction, l_coefficient=order
            )
            if result is not None:
                unc_dict[order] = result["uncertainties"]
                if unc_energies is None:
                    unc_energies = result["energies"]
                    unc_type = "relative" if result["is_relative"] else "absolute"

        if unc_dict:
            self.uncertainties = unc_dict
            self.uncertainty_energies = unc_energies
            self.uncertainty_type = unc_type

    def to_uncertainty_plot_data(
        self, order: int, sigma: float = 1.0,
        label: Optional[str] = None, **styling_kwargs
    ) -> "LegendreUncertaintyPlotData":
        """Create a ``LegendreUncertaintyPlotData`` for a given Legendre order.

        Parameters
        ----------
        order : int
            Legendre polynomial order.
        sigma : float
            Number of standard deviations (default 1.0).
        label : str, optional
            Plot label.
        **styling_kwargs
            Passed to ``LegendreUncertaintyPlotData``.

        Returns
        -------
        LegendreUncertaintyPlotData

        Raises
        ------
        ValueError
            If no uncertainties are attached or the requested order
            is unavailable.
        """
        from kika.plotting import LegendreUncertaintyPlotData
        from kika._utils import zaid_to_symbol

        if self.uncertainties is None:
            raise ValueError(
                "No uncertainties attached. Call attach_uncertainties() first."
            )

        if order not in self.uncertainties:
            available = sorted(self.uncertainties.keys())
            raise ValueError(
                f"No uncertainty for order {order}. Available: {available}"
            )

        unc_values = self.uncertainties[order]
        energy_bins = self.uncertainty_energies

        # For relative uncertainties, convert to percentage for display
        if self.uncertainty_type == "relative":
            y_values = unc_values * 100.0 * sigma
        else:
            y_values = unc_values * sigma

        # Extend for step plot (G values → G+1 points)
        y_extended = np.append(y_values, y_values[-1])

        isotope = zaid_to_symbol(self.nuclide_id) if self.nuclide_id else None

        if label is None:
            label = f"L={order} unc. ({sigma}σ)"
            if isotope:
                label = f"{isotope} {label}"

        return LegendreUncertaintyPlotData(
            x=energy_bins,
            y=y_extended,
            order=order,
            isotope=isotope,
            mt=self.reaction,
            uncertainty_type=self.uncertainty_type,
            sigma=sigma,
            energy_bins=energy_bins,
            label=label,
            **styling_kwargs,
        )

    @property
    def has_uncertainties(self) -> bool:
        """Whether uncertainty data is attached."""
        return self.uncertainties is not None and len(self.uncertainties) > 0

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def max_order(self) -> int:
        """Highest Legendre order with data."""
        return max(self.coefficients.keys()) if self.coefficients else 0

    def __repr__(self) -> str:
        n = len(self.energies) if len(self.energies) > 0 else 0
        return (
            f"AngularDistribution(MT={self.reaction}, ZA={self.nuclide_id}, "
            f"repr={self.representation!r}, frame={self.frame!r}, "
            f"max_L={self.max_order}, n_energies={n})"
        )
