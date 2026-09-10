"""Tests for cellmate.tracking: the IoU tracker, its gap-filling, and the
tracking-quality-control check in cellmate.tracking._qc.

Uses small synthetic label movies (12x12, a handful of frames) rather than
real microscopy data, so these run anywhere with just numpy installed and
stay fast. Geometry (which pixels belong to which blob, in which frame) is
chosen deliberately so the IoU/IoA/IoB values driving the tracker's
matching logic (see cellmate/tracking/_perfect_match.py) land on the
correct side of `threshold=0.25` for the event under test -- see the
comments on each movie for the intended intersection-over-{union,a,b}.
"""
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cellmate.tracking._iou_tracker import Tracker
from cellmate.tracking._qc import check_tracking_quality


def movie(*frames):
    """Build a (T, 12, 12) uint16 label movie from per-frame blob lists.

    Each frame is a list of (label, row_slice, col_slice) placed on a
    12x12 canvas with a 1px border left as background (label 0) --
    `init_first_frame`/`get_image_feature` assume 0 = background and strip
    it via `np.unique(...)[1:]`, which silently drops a real label instead
    if a frame has no background pixel at all (e.g. a blob filling the
    whole canvas), so every frame here keeps that border.
    """
    out = np.zeros((len(frames), 12, 12), dtype=np.uint16)
    for t, blobs in enumerate(frames):
        for label, rows, cols in blobs:
            out[t, rows, cols] = label
    return out


def run_qc(tracker, tracked_image=None, threshold=None):
    """Run `to_image_auto_fill_miss` (unless a `tracked_image` is already
    given) and `check_tracking_quality`, returning (ok, printed_lines)."""
    if tracked_image is None:
        _, tracked_image, gap_report = tracker.to_image_auto_fill_miss(return_report=True)
    else:
        _, _, gap_report = tracker.to_image_auto_fill_miss(return_report=True)
    buf = io.StringIO()
    with redirect_stdout(buf):
        ok = check_tracking_quality(tracker, gap_report, tracked_image, threshold=threshold)
    return ok, buf.getvalue().splitlines()


TOP = slice(1, 6)      # rows 1-5: top half of the blob area (row 0 stays background)
BOTTOM = slice(6, 11)  # rows 6-10: bottom half (row 11 stays background)
ALL_ROWS = slice(1, 11)
ALL_COLS = slice(1, 11)


class BasicTrackingTests(unittest.TestCase):
    def test_stationary_object_keeps_one_id_across_frames(self):
        image = movie(
            [(1, TOP, ALL_COLS)],
            [(1, TOP, ALL_COLS)],
            [(1, TOP, ALL_COLS)],
        )
        tracker = Tracker(image, threshold=0.25, min_hist=1, max_miss=1)
        tracker()

        all_boxes = tracker.all_trackers()
        self.assertEqual(len(all_boxes), 1)
        self.assertEqual(all_boxes[0].frame, [0, 1, 2])

    def test_no_events_means_clean_qc(self):
        image = movie(
            [(1, TOP, ALL_COLS)],
            [(1, TOP, ALL_COLS)],
        )
        tracker = Tracker(image, threshold=0.25, min_hist=1, max_miss=1)
        tracker()

        self.assertEqual(list(tracker.network.nodes), [])
        ok, lines = run_qc(tracker)
        self.assertTrue(ok)
        self.assertEqual(lines, ["tracking QC: no issues found"])


class DivisionFusionTests(unittest.TestCase):
    def test_division_is_detected(self):
        # frame 0: one blob filling the canvas.
        # frame 1: two blobs that exactly tile it -- each is fully inside
        # the parent's footprint (IoB=1.0 per child), so this should read
        # as one tracker splitting into two, not two independent objects.
        image = movie(
            [(7, ALL_ROWS, ALL_COLS)],
            [(11, TOP, ALL_COLS), (12, BOTTOM, ALL_COLS)],
        )
        tracker = Tracker(image, threshold=0.25, min_hist=1, max_miss=1)
        tracker()

        weights = sorted(w for *_, w in tracker.network.edges(data="weight"))
        self.assertEqual(weights, [1, 1])  # two division edges, no fusion

    def test_fusion_is_detected(self):
        # The mirror image of the division case: two blobs merge into one
        # that covers both their footprints (IoA=1.0 per parent).
        image = movie(
            [(11, TOP, ALL_COLS), (12, BOTTOM, ALL_COLS)],
            [(20, ALL_ROWS, ALL_COLS)],
        )
        tracker = Tracker(image, threshold=0.25, min_hist=1, max_miss=1)
        tracker()

        weights = sorted(w for *_, w in tracker.network.edges(data="weight"))
        self.assertEqual(weights, [2, 2])  # one fusion, recorded as 2 edges in


class GapFillingTests(unittest.TestCase):
    def test_default_call_still_returns_a_2_tuple(self):
        """Regression guard: many existing notebooks unpack this as
        `_, tracked_image = tracker.to_image_auto_fill_miss()` -- adding the
        opt-in `return_report` must not change that."""
        image = movie(
            [(1, TOP, ALL_COLS)],
            [(1, TOP, ALL_COLS)],
        )
        tracker = Tracker(image, threshold=0.25, min_hist=1, max_miss=1)
        tracker()
        result = tracker.to_image_auto_fill_miss()
        self.assertEqual(len(result), 2)

    def test_no_trackers_returns_empty_result_instead_of_crashing(self):
        """Previously `if len(tracker) < 1: return None` was unreachable --
        `traced_image.copy()` (on a None from to_image()) raised first."""
        image = movie([])  # single blank frame, nothing to track
        tracker = Tracker(image, threshold=0.25, min_hist=1, max_miss=1)
        tracker()
        traced_image, filled = tracker.to_image_auto_fill_miss()
        self.assertIsNone(traced_image)
        self.assertIsNone(filled)

    def test_gap_with_matching_masks_is_filled_and_reported(self):
        # Same object at frames 0, 1, and 3 (frame 2 missing -- e.g. a
        # dropped detection); same footprint throughout, so the gap should
        # be filled (IoU(start, end) = 1.0 > 0.8) using that footprint.
        image = movie(
            [(1, TOP, ALL_COLS)],
            [(1, TOP, ALL_COLS)],
            [],
            [(1, TOP, ALL_COLS)],
        )
        tracker = Tracker(image, threshold=0.25, min_hist=1, max_miss=2)
        tracker()

        raw, filled, report = tracker.to_image_auto_fill_miss(return_report=True)
        self.assertTrue((filled[2][TOP, ALL_COLS] > 0).all())
        self.assertTrue((raw[2] == 0).all())  # to_image() alone leaves the gap empty
        self.assertEqual(len(report), 1)
        self.assertTrue(report[0]["filled"])
        self.assertIsNone(report[0]["reason"])

        ok, lines = run_qc(tracker, tracked_image=filled)
        self.assertTrue(ok)

    def test_gap_with_mismatched_masks_is_left_unfilled_and_reported(self):
        # Same object at frames 0, 1, and 3 (frame 2 missing), but it has
        # drifted by the time it reappears: rows 1-5 -> rows 3-7, a ~43%
        # IoU. That's still enough to match it as a continuation of the
        # same track (above the tracker's own 0.25 threshold), but well
        # under to_image_auto_fill_miss's own, stricter 0.8 fill gate --
        # too different a shape to trust guessing the missing frame from.
        SHIFTED = slice(3, 8)
        image = movie(
            [(1, TOP, ALL_COLS)],
            [(1, TOP, ALL_COLS)],
            [],
            [(1, SHIFTED, ALL_COLS)],
        )
        tracker = Tracker(image, threshold=0.25, min_hist=1, max_miss=2)
        tracker()

        _, filled, report = tracker.to_image_auto_fill_miss(return_report=True)
        self.assertEqual(len(report), 1)
        self.assertFalse(report[0]["filled"])

        ok, lines = run_qc(tracker, tracked_image=filled)
        self.assertFalse(ok)
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith("missing label: id=1 frames 1-3"))


class QualityControlTests(unittest.TestCase):
    def test_short_lived_daughter_is_reported_as_a_missing_node(self):
        # Object divides at frame 1 into two daughters; only one of them
        # (the top half) survives to frame 2 -- the other just vanishes
        # (e.g. a segmentation drop), never reaching the end of the movie
        # and never resolved by a later fusion/division either.
        image = movie(
            [(7, ALL_ROWS, ALL_COLS)],
            [(11, TOP, ALL_COLS), (12, BOTTOM, ALL_COLS)],
            [(11, TOP, ALL_COLS)],
        )
        tracker = Tracker(image, threshold=0.25, min_hist=1, max_miss=2)
        tracker()

        saved_ids = set((tracker.save_trackers() or {}).keys())
        network_ids = set(tracker.network.nodes)
        missing = network_ids - saved_ids
        self.assertEqual(len(missing), 1)

        ok, lines = run_qc(tracker)
        self.assertFalse(ok)
        self.assertTrue(any(f"missing node: id={next(iter(missing))}" in line for line in lines))


class RetraceVerificationTests(unittest.TestCase):
    """The "mistake event" check re-tracks the movie about to be saved and
    cross-checks its division/fusion events against the original tracking
    pass. The original tracker's `box.label` history is its raw per-frame
    segmentation label (e.g. 7, 11, 12 below), which never appears in the
    delivered movie -- what's actually rendered there is the composite
    `id + category*DIVISION` -- so these tests also guard that translation,
    not just the comparison logic."""

    def _division_movie_and_tracker(self):
        image = movie(
            [(7, ALL_ROWS, ALL_COLS)],
            [(11, TOP, ALL_COLS), (12, BOTTOM, ALL_COLS)],
        )
        tracker = Tracker(image, threshold=0.25, min_hist=1, max_miss=1)
        tracker()
        return tracker

    def test_clean_delivery_matches_on_retrace(self):
        tracker = self._division_movie_and_tracker()
        ok, lines = run_qc(tracker)
        self.assertTrue(ok)
        self.assertEqual(lines, ["tracking QC: no issues found"])

    def test_corrupted_delivery_is_caught_as_a_mistake_event(self):
        tracker = self._division_movie_and_tracker()
        _, filled = tracker.to_image_auto_fill_miss()

        # Simulate gap-filling (or any other post-processing) accidentally
        # merging the two daughters back into one blob in the delivered
        # movie -- the division the original tracking found is no longer
        # visible in what's actually being handed off.
        corrupted = filled.copy()
        corrupted[1][corrupted[1] > 0] = 2

        ok, lines = run_qc(tracker, tracked_image=corrupted)
        self.assertFalse(ok)
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith("mistake event: ('division', 1, (2, 3))"))
        self.assertIn("retracing the saved movie didn't", lines[0])


if __name__ == "__main__":
    unittest.main()
