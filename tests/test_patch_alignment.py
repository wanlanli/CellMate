"""Tests for contour alignment (CellNetwork.aligned_coords_overtime and
friends) against the failure modes it guards: skeleton endpoints coming in
swapped order, contours with opposite winding, and a disk-filter patch
intensity that should match the exact per-point disk mean."""
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cellmate.tracking._iou_tracker import Tracker
from cellmate.mating import CellNetwork
from cellmate.patch import intensity_multiple_points, intensity_multiple_points_fast, move_inward
from cellmate.patch._patchcell import common_frames


def elongated_cell_network(T=4):
    masks = np.zeros((T, 40, 60), dtype=np.uint16)
    masks[:, 15:25, 10:50] = 5001
    tracker = Tracker(masks, threshold=0.25, min_hist=1, max_miss=1)
    tracker()
    _, filled = tracker.to_image_auto_fill_miss()
    cellnet = CellNetwork.from_tracked_movie(filled, tracking_threshold=0.25, threshold=50)
    return cellnet, next(iter(cellnet.cells))


class AlignmentTests(unittest.TestCase):
    def test_swapped_tips_are_reordered(self):
        cellnet, cell_id = elongated_cell_network()
        tips = cellnet.tips_overtime(cell_id)
        swapped = tips.copy()
        swapped[1::2] = swapped[1::2, ::-1]
        cellnet.tips_overtime = lambda _: swapped

        oriented = cellnet.oriented_tips_overtime(cell_id)
        np.testing.assert_allclose(oriented, np.repeat(tips[:1], len(tips), axis=0))
        center_1, center_2 = cellnet.center_tips(cell_id)
        np.testing.assert_allclose(center_1, tips[0, 0])
        np.testing.assert_allclose(center_2, tips[0, 1])

    def test_contour_winding_does_not_change_alignment(self):
        cellnet, cell_id = elongated_cell_network()
        reference = cellnet.aligned_coords_overtime(cell_id)

        coords = cellnet.coords_overtime(cell_id)
        reversed_some = np.array([c[::-1] if i % 2 else c for i, c in enumerate(coords)])
        cellnet.coords_overtime = lambda _: reversed_some
        np.testing.assert_allclose(cellnet.aligned_coords_overtime(cell_id), reference)

    def test_coincident_tips_still_split(self):
        cellnet, cell_id = elongated_cell_network()
        tips = cellnet.tips_overtime(cell_id)
        same = np.stack([tips[:, 0], tips[:, 0]], axis=1)
        cellnet.tips_overtime = lambda _: same
        coords = cellnet.aligned_coords_overtime(cell_id, num_samples=100)
        self.assertEqual(coords.shape, (len(tips), 100, 2))


class DiskFilterTests(unittest.TestCase):
    def test_matches_exact_disk_mean_on_grid_points(self):
        rng = np.random.default_rng(0)
        image = rng.uniform(100, 200, size=(80, 90))
        mask = np.zeros(image.shape, dtype=bool)
        mask[20:60, 15:75] = True
        # integer centres, including ones whose disk crosses the image edge
        centres = np.array([[30, 30], [40, 60], [55, 20], [3, 4], [78, 88]], dtype=float)
        mask[0:10, 0:10] = True
        mask[70:80, 80:90] = True

        exact, bg_exact = intensity_multiple_points(image, centres, 6, mask)
        fast, bg_fast = intensity_multiple_points_fast(image, centres, 6, mask)
        np.testing.assert_allclose(fast, exact)
        self.assertAlmostEqual(bg_fast, bg_exact)


class MoveInwardTests(unittest.TestCase):
    def rectangle(self):
        # 40 x 10 rectangle, rows 0..10, cols 0..40, traced counter-clockwise in (row, col)
        top = [(0, c) for c in range(0, 40)]
        right = [(r, 40) for r in range(0, 10)]
        bottom = [(10, c) for c in range(40, 0, -1)]
        left = [(r, 0) for r in range(10, 0, -1)]
        return np.array(top + right + bottom + left, dtype=float)

    def test_moves_straight_in_on_straight_edges_for_both_windings(self):
        contour = self.rectangle()
        for points in (contour, contour[::-1]):
            moved = move_inward(points, dist=2)
            side = (points[:, 0] == 0) & (points[:, 1] > 5) & (points[:, 1] < 35)
            np.testing.assert_allclose(moved[side, 0], 2)
            np.testing.assert_allclose(moved[side, 1], points[side, 1])
            self.assertTrue(((moved[:, 0] > 0) & (moved[:, 0] < 10)).all())


class CommonFramesTests(unittest.TestCase):
    def test_indices_point_at_shared_frames(self):
        f1 = np.array([2, 3, 4, 5, 7])
        f2 = np.array([4, 5, 6, 7, 8])
        self.assertEqual([tuple(int(v) for v in t) for t in common_frames(f1, f2)],
                         [(4, 2, 0), (5, 3, 1), (7, 4, 3)])


if __name__ == "__main__":
    unittest.main()
