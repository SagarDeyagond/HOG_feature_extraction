"""Visualisation helpers for HOG cell gradients."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np

__all__ = ["HOGVisualizer"]


class HOGVisualizer:
    """Render per-cell HOG gradient maps with matplotlib."""

    def plot_bin(
        self,
        gradients: np.ndarray,
        bin_index: int,
        *,
        bin_span_degrees: float = 180.0,
        title: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True,
    ) -> plt.Figure:
        """Plot the gradient magnitude of one orientation bin per cell.

        Args:
            gradients: Cell gradients of shape ``(rows, cols, bins)``,
                as returned by
                :meth:`hog_features.HOGFeatureExtractor.compute_cell_gradients`.
            bin_index: Orientation bin to display (``0 <= bin < bins``).
            bin_span_degrees: Angular range covered by all bins,
                used only for the default title (OpenCV uses 0-180).
            title: Custom plot title; auto-generated when ``None``.
            save_path: If given, the figure is written to this path.
            show: Whether to display the figure interactively.

        Returns:
            The created matplotlib figure.

        Raises:
            IndexError: If ``bin_index`` is out of range.
        """
        num_bins = gradients.shape[2]
        if not 0 <= bin_index < num_bins:
            raise IndexError(
                f"bin_index {bin_index} out of range for {num_bins} bins "
                f"(valid: 0..{num_bins - 1})"
            )

        if title is None:
            bin_width = bin_span_degrees / num_bins
            lo, hi = bin_index * bin_width, (bin_index + 1) * bin_width
            title = f"HOG bin {bin_index} ({lo:.0f}°–{hi:.0f}°)"

        fig, ax = plt.subplots()
        mesh = ax.pcolor(gradients[:, :, bin_index])
        # Image row 0 is at the top, so flip the y axis to match.
        ax.invert_yaxis()
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        fig.colorbar(mesh, ax=ax)

        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
        if show:
            plt.show()
        return fig

    def plot_magnitude(
        self,
        gradients: np.ndarray,
        *,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True,
    ) -> plt.Figure:
        """Plot the total gradient energy (sum over all bins) per cell.

        Args:
            gradients: Cell gradients of shape ``(rows, cols, bins)``.
            save_path: If given, the figure is written to this path.
            show: Whether to display the figure interactively.

        Returns:
            The created matplotlib figure.
        """
        fig, ax = plt.subplots()
        mesh = ax.pcolor(gradients.sum(axis=2))
        ax.invert_yaxis()
        ax.set_aspect("equal", adjustable="box")
        ax.set_title("HOG gradient magnitude (all bins)")
        fig.colorbar(mesh, ax=ax)

        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
        if show:
            plt.show()
        return fig
