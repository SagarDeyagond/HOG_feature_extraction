# Histogram of Oriented Gradients (HOG) — Theory Notes

HOG is a feature descriptor introduced by Dalal & Triggs (CVPR 2005)
that is widely and successfully used for object detection. It
represents an object as a **single feature vector**, as opposed to a
set of vectors each describing a segment of the image. In a detection
pipeline it is computed over a sliding window at multiple scales
(image pyramid), and each descriptor is typically fed to an SVM
classifier to decide whether the object is present.

## Step by step

1. **Cells.** Divide the image into small cells, e.g. 8×8 pixels, and
   compute the gradient vector (edge orientation and magnitude) at
   each pixel.

2. **Orientation histograms.** Each cell's 64 gradient vectors are
   accumulated into a histogram of orientations. In the original
   paper, 9 bins cover 0–180° (20° per bin), reducing 64 vectors to
   just 9 values per cell.

3. **Robustness to deformation.** Because the descriptor stores
   coarse, binned gradient statistics rather than exact pixel values,
   it is relatively immune to small shape deformations.

4. **Illumination invariance.** Brightening an image increases all
   pixel intensities — and hence gradient magnitudes — roughly
   uniformly. Dividing the histogram vectors by an overall gradient
   magnitude (normalisation) therefore yields the same descriptor
   regardless of brightness and contrast changes.

5. **Block normalisation.** Rather than normalising each cell in
   isolation, cells are grouped into overlapping blocks (e.g. 2×2
   cells) and normalisation is performed per block, taking the
   neighbouring cells into account. The block provides the
   normalisation constant over a larger segment of the image, which
   is more robust than per-cell normalisation.

6. **Descriptor assembly.** The normalised block histograms are
   concatenated into the final feature vector. Because blocks overlap
   (the block stride is one cell), each cell contributes to several
   blocks, each time with a different normalisation constant.

## What OpenCV returns

`cv2.HOGDescriptor.compute` returns a flat vector ordered by block
position (column-major), then by cell within the block, then by bin.
This project reshapes it into a 5-D array
`(blocks_y, blocks_x, block_h, block_w, bins)` and averages the
multiple normalised copies of each cell into a single
`(cells_y, cells_x, bins)` grid — convenient for visualisation and for
downstream per-cell features.

## Glossary

| Term | Meaning |
|---|---|
| Cell | Small pixel patch (e.g. 8×8 px) over which one histogram is built |
| Block | Group of cells (e.g. 2×2) used as the normalisation unit |
| Block stride | Step between successive block positions (here: one cell) |
| Bin | One orientation interval of the histogram (e.g. 20° wide) |
| Descriptor | The concatenation of all normalised block histograms |

## References

- N. Dalal and B. Triggs, *Histograms of Oriented Gradients for Human
  Detection*, CVPR 2005.
- OpenCV documentation: `cv2.HOGDescriptor`.
