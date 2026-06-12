# HOG Feature Extraction

Histogram of Oriented Gradients (HOG) feature extraction and visualisation
built on OpenCV's `cv2.HOGDescriptor`, with a clean object-oriented API.

HOG (Dalal & Triggs, 2005) describes an image by the distribution of local
gradient orientations and is a classic feature descriptor for object
detection, typically paired with an SVM classifier. See
[docs/HOG_THEORY.md](docs/HOG_THEORY.md) for a step-by-step explanation.

## Project structure

```
.
├── main.py                  # CLI entry point
├── hog_features/            # Core package
│   ├── extractor.py         # HOGConfig + HOGFeatureExtractor
│   └── visualizer.py        # HOGVisualizer (matplotlib plots)
├── tests/                   # Pytest unit tests
├── images/                  # Sample input images
├── docs/                    # Theory notes
├── notebooks/               # Original exploratory Jupyter notebook
├── requirements.txt
└── requirements-dev.txt
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt        # runtime only
pip install -r requirements-dev.txt    # + pytest, for development
```

## Usage

### Command line

```bash
# Visualise orientation bin 0 of the bundled sample image
python main.py

# Custom image, bin 4, save the plot without opening a window
python main.py path/to/image.jpg --bin 4 --save-plot hog_bin4.png --no-show

# Total gradient energy across all bins, also showing the original image
python main.py path/to/image.jpg --magnitude --show-image
```

Run `python main.py --help` for all options (cell size, block size,
number of bins, …).

### As a library

```python
from hog_features import HOGConfig, HOGFeatureExtractor, HOGVisualizer

extractor = HOGFeatureExtractor(HOGConfig(cell_size=(8, 8),
                                          block_size=(2, 2),
                                          num_bins=9))
gray = extractor.load_grayscale("images/sample.jpeg")

# (cells_y, cells_x, num_bins) — one averaged histogram per cell
gradients = extractor.compute_cell_gradients(gray)

HOGVisualizer().plot_bin(gradients, bin_index=0)
```

## Running the tests

```bash
pytest
```

## How it works

1. The image is divided into cells (default 8×8 px) and a gradient
   vector is computed at every pixel.
2. Each cell's 64 gradient vectors are binned into a 9-bin orientation
   histogram (0–180°, 20° per bin).
3. Cells are grouped into overlapping 2×2-cell blocks and each block is
   contrast-normalised for illumination invariance.
4. Because blocks overlap, each cell is normalised several times;
   `HOGFeatureExtractor.compute_cell_gradients` averages those copies
   into a single `(rows, cols, bins)` grid for visualisation.

## References

- N. Dalal and B. Triggs, *Histograms of Oriented Gradients for Human
  Detection*, CVPR 2005.
- [OpenCV HOGDescriptor documentation](https://docs.opencv.org/4.x/d5/d33/structcv_1_1HOGDescriptor.html)
- [python-hog](https://github.com/JeanKossaifi/python-hog) — a vectorised
  pure-NumPy HOG implementation, useful for study.
