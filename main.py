#!/usr/bin/env python3
"""Command-line entry point for HOG feature extraction.

Computes per-cell Histogram of Oriented Gradients features for an
image and visualises either a single orientation bin or the total
gradient magnitude.

Examples:
    python main.py images/sample.jpeg
    python main.py images/sample.jpeg --bin 4 --save-plot hog_bin4.png
    python main.py images/sample.jpeg --magnitude --show-image
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from hog_features import HOGConfig, HOGFeatureExtractor, HOGVisualizer

DEFAULT_IMAGE = Path(__file__).parent / "images" / "sample.jpeg"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract and visualise HOG features from an image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "image",
        nargs="?",
        default=str(DEFAULT_IMAGE),
        help="Path to the input image",
    )
    parser.add_argument(
        "--cell-size", type=int, default=8, help="Cell size in pixels (square)"
    )
    parser.add_argument(
        "--block-size", type=int, default=2, help="Block size in cells (square)"
    )
    parser.add_argument(
        "--bins", type=int, default=9, help="Number of orientation bins"
    )
    parser.add_argument(
        "--bin",
        type=int,
        default=0,
        dest="bin_index",
        help="Orientation bin to visualise",
    )
    parser.add_argument(
        "--magnitude",
        action="store_true",
        help="Plot total gradient magnitude instead of a single bin",
    )
    parser.add_argument(
        "--save-plot",
        type=Path,
        default=None,
        metavar="PATH",
        help="Save the plot to this file instead of only displaying it",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive plot window",
    )
    parser.add_argument(
        "--show-image",
        action="store_true",
        help="Also display the original image in an OpenCV window",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run HOG extraction and visualisation; return a process exit code."""
    args = parse_args(argv)

    config = HOGConfig(
        cell_size=(args.cell_size, args.cell_size),
        block_size=(args.block_size, args.block_size),
        num_bins=args.bins,
    )
    extractor = HOGFeatureExtractor(config)

    try:
        gray = extractor.load_grayscale(args.image)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.show_image:
        cv2.imshow("Original Image", cv2.imread(args.image))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    gradients = extractor.compute_cell_gradients(gray)
    print(
        f"Image: {args.image}\n"
        f"Cell grid: {gradients.shape[0]} x {gradients.shape[1]} cells, "
        f"{gradients.shape[2]} orientation bins per cell"
    )

    visualizer = HOGVisualizer()
    if args.magnitude:
        visualizer.plot_magnitude(
            gradients, save_path=args.save_plot, show=not args.no_show
        )
    else:
        visualizer.plot_bin(
            gradients,
            args.bin_index,
            save_path=args.save_plot,
            show=not args.no_show,
        )

    if args.save_plot is not None:
        print(f"Plot saved to {args.save_plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
