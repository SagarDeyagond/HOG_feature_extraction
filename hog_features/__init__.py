"""Histogram of Oriented Gradients (HOG) feature extraction toolkit.

This package wraps OpenCV's ``cv2.HOGDescriptor`` with a clean,
object-oriented API for computing per-cell HOG features and
visualising the resulting gradient histograms.
"""

from hog_features.extractor import HOGConfig, HOGFeatureExtractor
from hog_features.visualizer import HOGVisualizer

__all__ = ["HOGConfig", "HOGFeatureExtractor", "HOGVisualizer"]

__version__ = "1.0.0"
