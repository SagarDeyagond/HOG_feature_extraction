"""HOG feature extraction built on OpenCV's ``cv2.HOGDescriptor``.

The Histogram of Oriented Gradients (Dalal & Triggs, 2005) describes an
image by the distribution of local gradient orientations:

1. The image is divided into small cells (e.g. 8x8 pixels) and a
   gradient vector is computed at every pixel.
2. Each cell's gradients are binned into an orientation histogram
   (typically 9 bins covering 0-180 degrees), reducing 64 gradient
   vectors to just 9 values per cell.
3. Cells are grouped into overlapping blocks (e.g. 2x2 cells) and each
   block is contrast-normalised, making the descriptor robust to
   illumination changes.

OpenCV returns one histogram per cell *per block*, so cells covered by
several overlapping blocks appear multiple times with different
normalisation.  :meth:`HOGFeatureExtractor.compute_cell_gradients`
averages those duplicates back into a single ``(rows, cols, bins)``
grid, which is convenient for visualisation and downstream features.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np

__all__ = ["HOGConfig", "HOGFeatureExtractor"]


@dataclass(frozen=True)
class HOGConfig:
    """Configuration for HOG feature extraction.

    Attributes:
        cell_size: Cell size in pixels as ``(height, width)``.
        block_size: Block size in cells as ``(height, width)``.
        num_bins: Number of orientation histogram bins. Dalal & Triggs
            used 9 bins over 0-180 degrees (20 degrees per bin).
    """

    cell_size: Tuple[int, int] = (8, 8)
    block_size: Tuple[int, int] = (2, 2)
    num_bins: int = 9

    def __post_init__(self) -> None:
        if any(v <= 0 for v in (*self.cell_size, *self.block_size)):
            raise ValueError("cell_size and block_size must be positive")
        if self.num_bins <= 0:
            raise ValueError("num_bins must be positive")


class HOGFeatureExtractor:
    """Compute HOG features for grayscale or BGR images.

    Example:
        >>> extractor = HOGFeatureExtractor(HOGConfig(num_bins=9))
        >>> gradients = extractor.compute_cell_gradients(image)
        >>> gradients.shape  # (cells_y, cells_x, 9)
    """

    def __init__(self, config: HOGConfig | None = None) -> None:
        self.config = config or HOGConfig()

    @staticmethod
    def load_grayscale(image_path: Union[str, Path]) -> np.ndarray:
        """Load an image from disk and convert it to grayscale.

        Args:
            image_path: Path to the image file.

        Returns:
            The image as a 2-D ``uint8`` array.

        Raises:
            FileNotFoundError: If the file does not exist or cannot be
                decoded as an image.
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def grid_shape(self, image: np.ndarray) -> Tuple[int, int]:
        """Return the cell grid dimensions ``(rows, cols)`` for ``image``."""
        cell_h, cell_w = self.config.cell_size
        return image.shape[0] // cell_h, image.shape[1] // cell_w

    def _build_descriptor(self, gray: np.ndarray) -> cv2.HOGDescriptor:
        """Create a ``cv2.HOGDescriptor`` sized to ``gray``.

        The window is the image cropped to a multiple of the cell size,
        and the block stride equals one cell so blocks overlap by all
        but one cell in each direction.
        """
        cell_h, cell_w = self.config.cell_size
        block_h, block_w = self.config.block_size
        # OpenCV expects (width, height) ordering for all size arguments.
        return cv2.HOGDescriptor(
            _winSize=(
                gray.shape[1] // cell_w * cell_w,
                gray.shape[0] // cell_h * cell_h,
            ),
            _blockSize=(block_w * cell_w, block_h * cell_h),
            _blockStride=(cell_w, cell_h),
            _cellSize=(cell_w, cell_h),
            _nbins=self.config.num_bins,
        )

    def compute_raw_features(self, gray: np.ndarray) -> np.ndarray:
        """Compute block-normalised HOG features as a 5-D array.

        Args:
            gray: Grayscale input image.

        Returns:
            Array of shape ``(blocks_y, blocks_x, block_h, block_w,
            num_bins)`` holding, for every block position, the
            normalised histogram of each cell inside that block.
        """
        block_h, block_w = self.config.block_size
        n_cells_y, n_cells_x = self.grid_shape(gray)
        hog = self._build_descriptor(gray)
        # OpenCV flattens column-major over block positions, so reshape
        # with x first and transpose to row-major (y, x) indexing.
        return (
            hog.compute(gray)
            .reshape(
                n_cells_x - block_w + 1,
                n_cells_y - block_h + 1,
                block_h,
                block_w,
                self.config.num_bins,
            )
            .transpose((1, 0, 2, 3, 4))
        )

    def compute_cell_gradients(self, image: np.ndarray) -> np.ndarray:
        """Compute one averaged orientation histogram per cell.

        Overlapping blocks normalise each cell several times; this
        averages those copies into a single histogram per cell.

        Args:
            image: Grayscale (2-D) or BGR (3-D) input image.

        Returns:
            Array of shape ``(cells_y, cells_x, num_bins)``.
        """
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        block_h, block_w = self.config.block_size
        n_cells_y, n_cells_x = self.grid_shape(image)
        hog_feats = self.compute_raw_features(image)

        gradients = np.zeros((n_cells_y, n_cells_x, self.config.num_bins))
        cell_count = np.zeros((n_cells_y, n_cells_x, 1), dtype=int)

        # Accumulate every block-normalised copy of each cell, then
        # divide by how many blocks covered it.
        for off_y in range(block_h):
            for off_x in range(block_w):
                y_slice = slice(off_y, n_cells_y - block_h + off_y + 1)
                x_slice = slice(off_x, n_cells_x - block_w + off_x + 1)
                gradients[y_slice, x_slice] += hog_feats[:, :, off_y, off_x, :]
                cell_count[y_slice, x_slice] += 1

        gradients /= cell_count
        return gradients
