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
from cellmate.patch import find_patch_peaks


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

    def test_excluded_pixels_do_not_leak_into_disk(self):
        image = np.full((60, 60), 100.0)
        mask = np.zeros(image.shape, dtype=bool)
        mask[10:50, 10:30] = True
        neighbour = np.zeros(image.shape, dtype=bool)
        neighbour[10:50, 32:50] = True
        image[neighbour] = 1000.0
        # centre 3 px inside the edge, disk of radius 6 reaches into the neighbour
        centres = np.array([[30, 26]], dtype=float)

        leaky, _ = intensity_multiple_points_fast(image, centres, 6, mask)
        clean, _ = intensity_multiple_points_fast(image, centres, 6, mask, exclude=neighbour)
        self.assertGreater(leaky[0], 100.0)
        self.assertAlmostEqual(clean[0], 100.0)


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


def bumps(centres, heights, length=200, width=4):
    x = np.arange(length)
    signal = np.zeros(length)
    for c, h in zip(centres, heights):
        d = np.abs(x - c)
        d = np.minimum(d, length - d)
        signal += h * np.exp(-d ** 2 / (2 * width ** 2))
    return signal


class FindPatchPeaksTests(unittest.TestCase):
    def test_weak_and_strong_patches_are_both_found(self):
        signal = bumps([25, 95, 135], [8, 17, 94])
        positions, intensities = find_patch_peaks(signal)
        np.testing.assert_array_equal(positions, [25, 95, 135])
        np.testing.assert_allclose(intensities, signal[[25, 95, 135]])

    def test_peak_across_the_wrap_point(self):
        for centre in (0, 1, 198, 199):
            positions, _ = find_patch_peaks(bumps([centre, 100], [10, 10]))
            np.testing.assert_array_equal(positions, sorted([centre, 100]))

    def test_small_ripple_on_a_slope_is_not_a_peak(self):
        signal = bumps([100], [100], width=10)
        signal[115] += 6  # just above its neighbour at 114, a tiny local max on the slope
        self.assertGreater(len(find_patch_peaks(signal, min_prominence=0)[0]), 1)
        np.testing.assert_array_equal(find_patch_peaks(signal)[0], [100])

    def test_min_height_and_flat_signal(self):
        signal = bumps([50, 150], [5, 50])
        np.testing.assert_array_equal(find_patch_peaks(signal, min_height=10)[0], [150])
        self.assertEqual(len(find_patch_peaks(np.full(200, 3.0))[0]), 0)

    def test_near_zero_ripple_is_noise(self):
        signal = bumps([50, 150], [100, 10])
        signal[100] = 0.5  # isolated bump where there is no patch
        np.testing.assert_array_equal(find_patch_peaks(signal)[0], [50, 150])
        np.testing.assert_array_equal(find_patch_peaks(signal, noise_floor=0)[0], [50, 100, 150])

    def test_plateau_gives_one_peak(self):
        signal = np.zeros(200)
        signal[40:46] = 5
        positions, intensities = find_patch_peaks(signal)
        self.assertEqual(len(positions), 1)
        self.assertTrue(40 <= positions[0] < 46)
        self.assertEqual(intensities[0], 5)


class CommonFramesTests(unittest.TestCase):
    def test_indices_point_at_shared_frames(self):
        f1 = np.array([2, 3, 4, 5, 7])
        f2 = np.array([4, 5, 6, 7, 8])
        self.assertEqual([tuple(int(v) for v in t) for t in common_frames(f1, f2)],
                         [(4, 2, 0), (5, 3, 1), (7, 4, 3)])


if __name__ == "__main__":
    unittest.main()
