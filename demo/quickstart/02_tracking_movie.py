#!/usr/bin/env python
"""Command-line demo: Step 2 - track segmented cells through a movie.

Once a movie has been registered (step 1) and each frame has been segmented
into a labeled instance mask, ``cellmate.tracking`` links the same physical
cell across frames by matching instance masks with the highest IoU
(intersection-over-union) between consecutive frames. Two special events are
also detected automatically:

* **division** - one tracked object splits into two (e.g. mitosis/budding).
* **fusion** - two tracked objects merge into one (e.g. mating of two yeast
  cells), which is the event of primary interest for CellMate.

These relationships are recorded in a small directed graph
(``tracker.network``) alongside the per-object frame history
(``tracker.trackers`` / ``tracker.tracker_end``).

See ``02_tracking_movie.ipynb`` in this same folder for an annotated
walkthrough with visualizations (colored overlays + lineage graph) of the
same data.

Example
-------
Run on the bundled example movie (already registered + segmented)::

    python 02_tracking_movie.py

Run on your own segmented movie, where the mask lives in some other channel::

    python 02_tracking_movie.py --input /path/to/movie.tif --mask-channel -1

If your input is already a single-channel label movie ``[T, H, W]``, pass
``--mask-channel none``.
"""
import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: only saving figures, not showing them
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread, imsave

REPO_ROOT = Path(__file__).resolve().parents[2]
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cellmate.tracking._iou_tracker import Tracker  # noqa: E402
from cellmate.visualize import draw_graph_by_layer  # noqa: E402
from cellmate.utils import dump_to_pkl  # noqa: E402

DEFAULT_INPUT = REPO_ROOT / "demo" / "YAV2104xYAV2105_20220826_exp011_05_R3D_rigid_003002.tif"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "demo" / "quickstart" / "outputs"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Track segmented cells through a movie with cellmate, "
                    "detecting divisions and fusions (matings).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="Path to a segmented movie (.tif).")
    parser.add_argument("--mask-channel", default="-1",
                        help="Channel index holding the instance-label mask in a "
                             "[T, C, H, W] movie, or 'none' if the movie is already "
                             "a single-channel label movie [T, H, W].")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to write the tracked movie / graph / pickles to.")
    parser.add_argument("--threshold", type=float, default=0.25,
                        help="Minimum IoU for two masks to be considered the same object.")
    parser.add_argument("--min-hist", type=int, default=1,
                        help="Drop tracks shorter than this many frames from the output.")
    parser.add_argument("--max-miss", type=int, default=10,
                        help="Max consecutive frames an object may go undetected "
                             "before its track is closed.")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading {args.input}")
    image = imread(str(args.input))

    if args.mask_channel.lower() == "none":
        masks = image
    else:
        masks = image[:, int(args.mask_channel)]

    print(f"Mask movie shape (T, H, W): {masks.shape}, dtype={masks.dtype}")
    print("Running IoU tracking ...")

    tracker = Tracker(masks, threshold=args.threshold, min_hist=args.min_hist,
                      max_miss=args.max_miss)
    tracker()

    tracked_image = tracker.to_image()
    n_objects = len(tracker.all_trackers())
    n_division = sum(1 for *_, d in tracker.network.edges(data="weight") if d == 1) // 2
    n_fusion = sum(1 for *_, d in tracker.network.edges(data="weight") if d == 2) // 2
    print(f"Tracked {n_objects} objects over {masks.shape[0]} frames "
         f"({n_division} division(s), {n_fusion} fusion/mating(s)).")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    tracked_path = args.output_dir / "02_tracked_movie_demo.tif"
    imsave(str(tracked_path), tracked_image.astype(np.uint16), imagej=True, check_contrast=False)
    print(f"Saved tracked label movie to {tracked_path}")

    network_path = args.output_dir / "02_tracking_network_demo.pkl"
    dump_to_pkl(tracker.network, network_path)
    print(f"Saved lineage graph to {network_path}")

    trackers_path = args.output_dir / "02_tracking_trackers_demo.pkl"
    dump_to_pkl(tracker.save_trackers(), trackers_path)
    print(f"Saved per-object frame history to {trackers_path}")

    if tracker.network.number_of_nodes() > 0:
        draw_graph_by_layer(tracker.network)
        figure_path = args.output_dir / "02_tracking_lineage_demo.png"
        plt.savefig(figure_path, dpi=150, bbox_inches="tight")
        print(f"Saved lineage figure to {figure_path}")


if __name__ == "__main__":
    main()
