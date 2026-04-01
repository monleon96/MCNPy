"""Optional PyVista-based visualization for WWINP data."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from kika.wwinp.classes.wwinp import WWINP


class WWPlotter:
    """Interactive 3-D weight window visualiser (requires PyVista)."""

    def __init__(self, wwinp: WWINP) -> None:
        try:
            import pyvista as pv  # noqa: F401
        except ImportError:
            raise ImportError(
                "PyVista is required for WWINP visualisation. "
                "Install it with: pip install pyvista"
            )
        self.wwinp = wwinp
        self._current_grid = None
        self._energy_idx = 0
        self.mesh_opacity = 0.7

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_mesh_grid(self, particle: int, time_idx: int = 0):
        import pyvista as pv

        fm = self.wwinp.geometry.fine_mesh
        labels = self.wwinp.geometry.axis_labels
        x = fm[labels[0]]
        y = fm[labels[1]]
        z = fm[labels[2]]
        grid = pv.RectilinearGrid(x, y, z)

        ww = self.wwinp.values.ww_values[particle]  # (nt, ne, nfz, nfy, nfx)
        vals_te = ww[time_idx]  # (ne, nfz, nfy, nfx)

        for e in range(vals_te.shape[0]):
            grid.cell_data[f"energy_{e}"] = vals_te[e].flatten(order="F")

        return grid

    def _get_value_range(self) -> tuple[float, float]:
        values = self._current_grid.cell_data.values()
        all_vals = np.concatenate([a.flatten() for a in values])
        return float(np.min(all_vals)), float(np.max(all_vals))

    def _get_energy_labels(self, particle: int) -> list[str]:
        e_bins = self.wwinp.energy_bins.get(particle, np.array([]))
        return [f"{e:.2e}" for e in e_bins]

    def _update_energy(self, energy_idx: int) -> None:
        if self._current_grid is None:
            return
        import pyvista as pv  # noqa: F401

        self._plotter.remove_actor("mesh")
        self._plotter.add_mesh(
            self._current_grid,
            scalars=f"energy_{int(energy_idx)}",
            opacity=self.mesh_opacity,
            name="mesh",
            show_edges=True,
            cmap="viridis",
            clim=self._get_value_range(),
        )
        self._energy_idx = int(energy_idx)
        self._plotter.render()

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def plot_3d(self, particle: int = 0, time_idx: int = 0) -> None:
        """Interactive 3-D visualisation with energy slider."""
        import pyvista as pv

        self._current_grid = self._create_mesh_grid(particle, time_idx)
        self._plotter = pv.Plotter()
        self._plotter.add_mesh(
            self._current_grid,
            scalars="energy_0",
            opacity=self.mesh_opacity,
            name="mesh",
            show_edges=True,
            cmap="viridis",
            clim=self._get_value_range(),
        )

        labels = self._get_energy_labels(particle)
        if len(labels) > 1:
            self._plotter.add_slider_widget(
                callback=self._update_energy,
                rng=[0, len(labels) - 1],
                value=0,
                title="Energy (MeV)",
                pointa=(0.025, 0.1),
                pointb=(0.31, 0.1),
                style="modern",
            )

        self._plotter.add_text(
            f"Particle: {particle}  Energy: {labels[0] if labels else '?'} MeV",
            position="upper_left",
        )
        self._plotter.show()

    def plot_slices(self, particle: int = 0, time_idx: int = 0) -> None:
        """Interactive orthogonal-slice visualisation."""
        import pyvista as pv

        self._current_grid = self._create_mesh_grid(particle, time_idx)
        self._plotter = pv.Plotter()
        self._plotter.add_mesh_slice_orthogonal(
            self._current_grid,
            scalars="energy_0",
            cmap="viridis",
            clim=self._get_value_range(),
        )

        labels = self._get_energy_labels(particle)
        if len(labels) > 1:
            self._plotter.add_slider_widget(
                callback=self._update_energy,
                rng=[0, len(labels) - 1],
                value=0,
                title="Energy (MeV)",
                pointa=(0.025, 0.1),
                pointb=(0.31, 0.1),
                style="modern",
            )

        self._plotter.add_text(
            f"Particle: {particle}  Energy: {labels[0] if labels else '?'} MeV",
            position="upper_left",
        )
        self._plotter.show()
