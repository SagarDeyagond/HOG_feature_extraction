"""Unit tests for hog_features.extractor."""

import numpy as np
import pytest

from hog_features import HOGConfig, HOGFeatureExtractor


@pytest.fixture()
def extractor() -> HOGFeatureExtractor:
    return HOGFeatureExtractor(HOGConfig(cell_size=(8, 8), block_size=(2, 2), num_bins=9))


@pytest.fixture()
def vertical_edge_image() -> np.ndarray:
    """A 64x64 image: black left half, white right half."""
    image = np.zeros((64, 64), dtype=np.uint8)
    image[:, 32:] = 255
    return image


class TestHOGConfig:
    def test_defaults(self) -> None:
        config = HOGConfig()
        assert config.cell_size == (8, 8)
        assert config.block_size == (2, 2)
        assert config.num_bins == 9

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"cell_size": (0, 8)},
            {"block_size": (2, -1)},
            {"num_bins": 0},
        ],
    )
    def test_rejects_invalid_values(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            HOGConfig(**kwargs)


class TestHOGFeatureExtractor:
    def test_grid_shape(self, extractor: HOGFeatureExtractor) -> None:
        image = np.zeros((64, 128), dtype=np.uint8)
        assert extractor.grid_shape(image) == (8, 16)

    def test_cell_gradients_shape(
        self, extractor: HOGFeatureExtractor, vertical_edge_image: np.ndarray
    ) -> None:
        gradients = extractor.compute_cell_gradients(vertical_edge_image)
        assert gradients.shape == (8, 8, 9)

    def test_accepts_bgr_input(
        self, extractor: HOGFeatureExtractor, vertical_edge_image: np.ndarray
    ) -> None:
        bgr = np.stack([vertical_edge_image] * 3, axis=-1)
        gradients = extractor.compute_cell_gradients(bgr)
        assert gradients.shape == (8, 8, 9)

    def test_gradients_are_normalised(
        self, extractor: HOGFeatureExtractor, vertical_edge_image: np.ndarray
    ) -> None:
        gradients = extractor.compute_cell_gradients(vertical_edge_image)
        assert np.all(gradients >= 0)
        assert np.all(gradients <= 1)

    def test_vertical_edge_energy_concentrated_at_edge(
        self, extractor: HOGFeatureExtractor, vertical_edge_image: np.ndarray
    ) -> None:
        """All gradient energy should lie in the cell columns at the edge.

        The black/white boundary sits at pixel 32 (cell columns 3-4);
        OpenCV's spatial interpolation also spreads some energy into
        the neighbouring columns 2 and 5.
        """
        gradients = extractor.compute_cell_gradients(vertical_edge_image)
        energy_per_column = gradients.sum(axis=(0, 2))
        edge_columns = {2, 3, 4, 5}
        for col, energy in enumerate(energy_per_column):
            if col in edge_columns:
                assert energy > 0
            else:
                assert energy == pytest.approx(0)

    def test_uniform_image_has_no_gradients(
        self, extractor: HOGFeatureExtractor
    ) -> None:
        flat = np.full((64, 64), 128, dtype=np.uint8)
        gradients = extractor.compute_cell_gradients(flat)
        assert np.allclose(gradients, 0)

    def test_load_grayscale_missing_file(
        self, extractor: HOGFeatureExtractor
    ) -> None:
        with pytest.raises(FileNotFoundError):
            extractor.load_grayscale("does/not/exist.jpg")
