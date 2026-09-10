"""Tests for cellmate.tracking.retrace and CellNetwork.from_tracked_movie --
building a CellNetwork directly from an already-tracked movie by
re-tracking it, instead of needing the original Tracker or its pickled
network/trackers kept around (see the pipeline discussion this implements:
"just save the tracked movie" vs. "save tracker state as pkl side-cars" --
we went with the former).
"""
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cellmate.tracking._iou_tracker import Tracker
from cellmate.tracking import retrace
from cellmate.mating import CellNetwork, CellNetwork90


TOP = slice(1, 6)
BOTTOM = slice(6, 11)
ALL_ROWS = slice(1, 11)
ALL_COLS = slice(1, 11)


def division_movie():
    image = np.zeros((2, 12, 12), dtype=np.uint16)
    image[0, ALL_ROWS, ALL_COLS] = 7
    image[1, TOP, ALL_COLS] = 11
    image[1, BOTTOM, ALL_COLS] = 12
    return image


class RetraceTests(unittest.TestCase):
    def test_retrace_reconstructs_the_same_division(self):
        tracker = Tracker(division_movie(), threshold=0.25, min_hist=1, max_miss=1)
        tracker()
        _, filled = tracker.to_image_auto_fill_miss()

        retraced = retrace(filled, threshold=0.25)
        weights = sorted(w for *_, w in retraced.network.edges(data="weight"))
        self.assertEqual(weights, [1, 1])  # same division structure, fresh ids


class CellNetworkFromTrackedMovieTests(unittest.TestCase):
    def test_matches_manual_construction(self):
        tracker = Tracker(division_movie(), threshold=0.25, min_hist=1, max_miss=1)
        tracker()
        _, filled = tracker.to_image_auto_fill_miss()

        via_helper = CellNetwork.from_tracked_movie(filled, tracking_threshold=0.25, threshold=50)

        manual_tracker = retrace(filled, threshold=0.25)
        via_manual = CellNetwork(image=filled, time_network=manual_tracker.network,
                                 tracker=manual_tracker.save_trackers(), threshold=50)

        self.assertEqual(set(via_helper.cells.keys()), set(via_manual.cells.keys()))
        for cid in via_helper.cells:
            self.assertEqual(via_helper.cells[cid].frames.tolist(), via_manual.cells[cid].frames.tolist())
            self.assertEqual(via_helper.cells[cid].label, via_manual.cells[cid].label)

    def test_cells_reflect_the_division(self):
        tracker = Tracker(division_movie(), threshold=0.25, min_hist=1, max_miss=1)
        tracker()
        _, filled = tracker.to_image_auto_fill_miss()

        cellnet = CellNetwork.from_tracked_movie(filled, tracking_threshold=0.25, threshold=50)

        self.assertEqual(len(cellnet.cells), 3)  # parent + 2 daughters
        frames_by_cell = {cid: cell.frames.tolist() for cid, cell in cellnet.cells.items()}
        self.assertEqual(sorted(frames_by_cell.values()), [[0], [1], [1]])

    def test_downstream_measurement_works(self):
        """aligned_coords_overtime/aligned_skeleton_overtime -- the methods
        03_create_data.ipynb actually calls -- need a cell tracked across
        more than one frame to be meaningful, so use a stable (undivided)
        object across a short movie instead of the division fixture."""
        T, H, W = 4, 30, 30
        masks = np.zeros((T, H, W), dtype=np.uint16)
        masks[:, 10:16, 10:16] = 5001

        tracker = Tracker(masks, threshold=0.25, min_hist=1, max_miss=1)
        tracker()
        _, filled = tracker.to_image_auto_fill_miss()

        cellnet = CellNetwork.from_tracked_movie(filled, tracking_threshold=0.25, threshold=50)
        cell_id = next(iter(cellnet.cells))

        coords = cellnet.aligned_coords_overtime(cell_id)
        skeleton = cellnet.aligned_skeleton_overtime(cell_id)
        self.assertEqual(coords.shape[0], T)
        self.assertEqual(skeleton.shape[0], T)

    def test_works_with_cellnetwork_subclasses(self):
        tracker = Tracker(division_movie(), threshold=0.25, min_hist=1, max_miss=1)
        tracker()
        _, filled = tracker.to_image_auto_fill_miss()

        cellnet90 = CellNetwork90.from_tracked_movie(filled, tracking_threshold=0.25, threshold=50)
        self.assertIsInstance(cellnet90, CellNetwork90)
        self.assertEqual(len(cellnet90.cells), 3)


if __name__ == "__main__":
    unittest.main()
