#!/usr/bin/env python
"""Command-line demo: Step 1 - rigid registration of a time-lapse movie.

Yeast cells drift slightly on the microscope stage between frames. Rigid
registration removes that drift by translating every frame (and every
channel of every frame) so that a chosen *reference channel* stays aligned
with frame 0. This is normally the very first step of the pipeline, run
before segmentation/tracking/measurement.

This script wraps ``cellmate.registration.rigid`` -- see
``01_rigid_registration.ipynb`` in this same folder for an annotated,
step-by-step walkthrough with visualizations of the same data.

Example
-------
Run on the bundled example movie (a small crop, so it finishes in seconds)::

    python 01_rigid_registration.py

Run on your own data::

    python 01_rigid_registration.py \\
        --input /path/to/movie.tif \\
        --output /path/to/movie_rigid.tif \\
        --reference-channel 0

The input array is expected as ``[Time, Channel, Height, Width]``. If your
file was saved with channels last (``[Time, Height, Width, Channel]``, which
is how the bundled example is stored), pass ``--channel-last`` and it will be
moved into place automatically.
"""
import argparse
import time
from pathlib import Path

import numpy as np
from skimage.io import imread, imsave

REPO_ROOT = Path(__file__).resolve().parents[2]
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cellmate.registration import rigid  # noqa: E402

DEFAULT_INPUT = REPO_ROOT / "demo" / "yav2104yav2105_mating_example_01.tif"
DEFAULT_OUTPUT = REPO_ROOT / "demo" / "quickstart" / "outputs" / "01_rigid_registration_demo.tif"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rigid-register a multi-channel time-lapse movie with cellmate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="Path to the raw movie (.tif).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Where to write the registered movie (.tif).")
    parser.add_argument("--channel-last", action="store_true",
                        help="Set this if the input is stored as [T, H, W, C] "
                             "instead of [T, C, H, W] (the bundled example needs this).")
    parser.add_argument("--frames", type=int, default=None,
                        help="Only register the first N frames (useful for a quick test run).")
    parser.add_argument("--reference-channel", type=int, default=0,
                        help="Channel used to estimate the transform between consecutive frames.")
    parser.add_argument("--max-step", type=float, default=1.0,
                        help="Maximum optimizer step size (SimpleITK RegularStepGradientDescent).")
    parser.add_argument("--min-step", type=float, default=0.01,
                        help="Minimum optimizer step size; smaller -> more precise but slower.")
    parser.add_argument("--iterations", type=int, default=1000,
                        help="Maximum number of optimizer iterations per frame.")
    parser.add_argument("--relaxation-factor", type=float, default=0.05,
                        help="How quickly the optimizer shrinks its step size.")
    parser.add_argument("--tolerance", type=float, default=1e-8,
                        help="Gradient-magnitude tolerance used as a stopping criterion.")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading {args.input}")
    image = imread(str(args.input))

    if args.channel_last:
        # [T, H, W, C] -> [T, C, H, W]
        image = np.moveaxis(image, -1, 1)

    if args.frames is not None:
        image = image[: args.frames]

    print(f"Movie shape (T, C, H, W): {image.shape}, dtype={image.dtype}")
    print("Running rigid registration ...")

    start = time.time()
    moved = rigid(
        image,
        reference_channel=args.reference_channel,
        maxStep=args.max_step,
        minStep=args.min_step,
        numberOfIterations=args.iterations,
        relaxationFactor=args.relaxation_factor,
        tolerance=args.tolerance,
    )
    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s ({elapsed / image.shape[0]:.2f}s/frame)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    imsave(str(args.output), moved.astype(np.uint16), imagej=True, check_contrast=False)
    print(f"Saved registered movie to {args.output}")


if __name__ == "__main__":
    main()
