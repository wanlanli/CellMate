#!/usr/bin/env python
"""Command-line demo: Step 3 - measure cell morphology from a labeled mask.

``cellmate.image_measure.ImageMeasure`` turns a single labeled instance mask
(one 2D image where every cell has a unique integer label, e.g. the output of
step 2's tracker, or of any instance-segmentation model) into a table of
per-cell morphology: area, major/minor axis length, eccentricity, skeleton
("medial axis") length, and more. It also answers spatial questions between
pairs of cells, such as "who is this cell's neighbor?" and "what is the
distance/angle between these two cells?" -- the kind of features used
upstream of mating classification.

See ``03_measure_image.ipynb`` in this same folder for an annotated
walkthrough with visualizations (skeleton/contour overlays, neighbor graphs)
of the same data.

Example
-------
Measure a single frame from the bundled example movie::

    python 03_measure_image.py

Measure every frame of a movie and concatenate into one CSV (one row per
cell per frame)::

    python 03_measure_image.py --batch --output measurements.csv

Measure a single, already-2D mask file (e.g. a PNG/TIFF from your own
segmentation) instead of picking a frame out of a movie::

    python 03_measure_image.py --input my_mask.png --mask-channel none
"""
import argparse
from pathlib import Path

import pandas as pd
from skimage.io import imread
from tqdm import trange

REPO_ROOT = Path(__file__).resolve().parents[2]
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cellmate.image_measure import ImageMeasure  # noqa: E402

DEFAULT_INPUT = REPO_ROOT / "demo" / "YAV2104xYAV2105_20220826_exp011_05_R3D_rigid_003002.tif"
DEFAULT_OUTPUT = REPO_ROOT / "demo" / "quickstart" / "outputs" / "03_measurements_demo.csv"

# A small, readable subset of the ~20 columns ImageMeasure computes.
SUMMARY_COLUMNS = ["label", "semantic", "instance", "area",
                  "axis_major_length", "axis_minor_length",
                  "eccentricity", "skeleton_major_length"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure per-cell morphology from a labeled mask with cellmate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="Path to a movie (.tif) or a single 2D mask image.")
    parser.add_argument("--mask-channel", default="-1",
                        help="Channel index holding the instance-label mask in a "
                             "[T, C, H, W] movie, or 'none' if --input is already a "
                             "single 2D mask image.")
    parser.add_argument("--frame", type=int, default=0,
                        help="Which frame to measure when not using --batch.")
    parser.add_argument("--batch", action="store_true",
                        help="Measure every frame of the movie instead of just one, "
                             "and concatenate the results (adds a 'frame' column).")
    parser.add_argument("--pixel-size", type=float, default=1.0,
                        help="Physical size of one pixel (e.g. microns/pixel). "
                             "Length/area columns are scaled by this.")
    parser.add_argument("--sampling-interval", type=int, default=5,
                        help="Spacing (in pixels) used when re-sampling each cell's "
                             "contour/skeleton -- lower is finer but slower.")
    parser.add_argument("--equidistant", action="store_true", default=True,
                        help="Re-sample contour points at equal arc-length spacing.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Where to write the per-cell measurements (.csv).")
    return parser.parse_args()


def measure_frame(mask, args, frame_index=None):
    measure = ImageMeasure(mask, pixel_size=args.pixel_size,
                           sampling_interval=args.sampling_interval,
                           equidistant=args.equidistant)
    table = measure.property_table[SUMMARY_COLUMNS].copy()
    if frame_index is not None:
        table["frame"] = frame_index
    return table


def main():
    args = parse_args()

    print(f"Loading {args.input}")
    image = imread(str(args.input))

    if args.mask_channel.lower() == "none":
        masks = image
    else:
        masks = image[:, int(args.mask_channel)]

    if args.batch:
        if masks.ndim != 3:
            raise ValueError(f"--batch expects a movie [T, H, W], got shape {masks.shape}")
        print(f"Measuring all {masks.shape[0]} frames ...")
        tables = []
        for t in trange(masks.shape[0]):
            tables.append(measure_frame(masks[t], args, frame_index=t))
        table = pd.concat(tables, ignore_index=True)
    else:
        mask = masks[args.frame] if masks.ndim == 3 else masks
        print(f"Measuring frame shape {mask.shape}, {len(set(mask.flatten())) - 1} cell(s) found")
        table = measure_frame(mask, args)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(f"Saved {len(table)} row(s) to {args.output}")

    print("\nSummary by semantic class:")
    print(table.groupby("semantic")["area"].agg(["count", "mean"]))


if __name__ == "__main__":
    main()
