"""Tests for CellNetwork.create_cell_type (cellmate/mating/_mating.py) and the
underlying cellmate.mating.prediction_cell_type_snr (cellmate/mating/_classification.py)
-- classifies each tracked cell's mating type (h+ / h- / unclassified) from
its fluorescence intensity relative to background, per channel, and stashes
the result on Cell.strain_type (1 = channel 0 marker, 2 = channel 1 marker,
3 = both, 0 = neither).

`prediction_cell_type_snr` (via `create_cell_type`) is the current default:
each channel is decided on/off independently by comparing directly against
that frame's background (mean + std) -- on if more than `z_threshold`
background standard deviations above the background mean. See
SnrPredictionTests for the rule and `create_cell_type_snr` for why it's
tunable that way.

`prediction_cell_type` (via `create_cell_type_legacy`) is the original
classifier -- 2-means clustering a log-ratio normalization against a fixed
`high_val` -- kept only as a reference to double-check the SNR method
against on other datasets; see PredictionCellTypeTests and
CreateCellTypeLegacyTests.

For eyeballing the classification on real data, see `plot_cell_types` in
cellmate.visualize and the "Verify create_cell_type" section added to
hpm/data_process/04_overview_cellnetwork_local.ipynb.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cellmate.mating import CellNetwork, prediction_cell_type, prediction_cell_type_snr
from cellmate.visualize import plot_cell_types


def synthetic_two_type_movie(frames=3, size=60, box=8, high=200.0, low=5.0):
    """Stable (no division/fusion), 4-cell tracked-label movie plus a matching
    2-channel fluorescent movie: cells 1-2 are bright in channel 0 (the "h+"
    marker), cells 3-4 are bright in channel 1 (the "h-" marker), everything
    else (including background) sits at `low`.

    Flat background (no pixel-to-pixel variance) -- fine for the legacy
    log-ratio method, but not for prediction_cell_type_snr (bg_std=0); use
    synthetic_two_type_movie_with_noise for that.

    Returns (masks [T,H,W], fluorescent [T,2,H,W], boxes {label: (row, col)}).
    """
    boxes = {1: (5, 5), 2: (5, size - 5 - box),
             3: (size - 5 - box, 5), 4: (size - 5 - box, size - 5 - box)}
    masks = np.zeros((frames, size, size), dtype=np.uint16)
    for label, (r, c) in boxes.items():
        masks[:, r:r + box, c:c + box] = label

    fluorescent = np.full((frames, 2, size, size), low)
    for label in (1, 2):
        r, c = boxes[label]
        fluorescent[:, 0, r:r + box, c:c + box] = high
    for label in (3, 4):
        r, c = boxes[label]
        fluorescent[:, 1, r:r + box, c:c + box] = high

    return masks, fluorescent, boxes


def synthetic_two_type_movie_with_noise(frames=3, size=60, box=8, high=200.0, bg_mean=5.0, bg_std=1.0, seed=0):
    """Like synthetic_two_type_movie, but with actual per-pixel background
    noise instead of a flat value -- needed for prediction_cell_type_snr,
    which measures on/off relative to the background's own variance (a
    perfectly flat background has bg_std=0)."""
    rng = np.random.default_rng(seed)
    boxes = {1: (5, 5), 2: (5, size - 5 - box),
             3: (size - 5 - box, 5), 4: (size - 5 - box, size - 5 - box)}
    masks = np.zeros((frames, size, size), dtype=np.uint16)
    for label, (r, c) in boxes.items():
        masks[:, r:r + box, c:c + box] = label

    fluorescent = rng.normal(bg_mean, bg_std, size=(frames, 2, size, size)).clip(0.1, None)
    for label in (1, 2):
        r, c = boxes[label]
        fluorescent[:, 0, r:r + box, c:c + box] = high
    for label in (3, 4):
        r, c = boxes[label]
        fluorescent[:, 1, r:r + box, c:c + box] = high

    return masks, fluorescent, boxes


class SnrPredictionTests(unittest.TestCase):
    """prediction_cell_type_snr (the current default) decides each channel's
    on/off independently, replacing the legacy method's 2-means clustering
    of an odd log-ratio with a plain, tunable threshold on signal-to-noise
    relative to the frame's own measured background."""

    def test_classifies_by_channel_independently(self):
        masks, fluorescent, _ = synthetic_two_type_movie_with_noise()

        cell_types, _ = prediction_cell_type_snr(fluorescent, masks, channel_number=2,
                                                   bg_threshold=10, fc_threshold=50, z_threshold=3.0)

        self.assertEqual(cell_types.to_dict(), {1: 1, 2: 1, 3: 2, 4: 2})

    def test_cell_dim_in_every_channel_is_unclassified(self):
        masks, fluorescent, _ = synthetic_two_type_movie_with_noise()
        masks[:, 25:33, 25:33] = 5  # 5th cell, left at background level in both channels

        cell_types, _ = prediction_cell_type_snr(fluorescent, masks, channel_number=2,
                                                   bg_threshold=10, fc_threshold=50, z_threshold=3.0)

        self.assertEqual(cell_types.to_dict()[5], 0)

    def test_works_with_a_single_channel(self):
        """Each channel is decided on its own (no cross-channel comparison),
        so a single-channel movie works unchanged."""
        masks, fluorescent, _ = synthetic_two_type_movie_with_noise()
        single_channel = fluorescent[:, 0:1]  # keep only the channel-0 marker

        cell_types, _ = prediction_cell_type_snr(single_channel, masks, channel_number=1,
                                                   bg_threshold=10, fc_threshold=50, z_threshold=3.0)

        self.assertEqual(cell_types.to_dict(), {1: 1, 2: 1, 3: 0, 4: 0})

    def test_z_threshold_trades_off_sensitivity_to_bleed_through(self):
        """A purely per-channel test can't tell real crosstalk from a
        genuine double-positive by itself -- but since the decision is a
        plain SNR threshold (not an opaque clustering result), z_threshold
        can be raised to reject bleed-through that sits well below true
        signal's SNR, which is the point of exposing it as a tunable knob."""
        masks, fluorescent, _ = synthetic_two_type_movie_with_noise()
        r, c = 25, 25
        masks[:, r:r + 8, c:c + 8] = 5
        fluorescent[:, 0, r:r + 8, c:c + 8] = 200.0  # dominant channel 0 (SNR ~195)
        fluorescent[:, 1, r:r + 8, c:c + 8] = 20.0   # modest channel-1 bleed-through (SNR ~15)

        loose, _ = prediction_cell_type_snr(fluorescent, masks, channel_number=2,
                                             bg_threshold=10, fc_threshold=50, z_threshold=3.0)
        strict, _ = prediction_cell_type_snr(fluorescent, masks, channel_number=2,
                                              bg_threshold=10, fc_threshold=50, z_threshold=16.0)

        self.assertEqual(loose.to_dict()[5], 3)    # bleed-through alone clears a loose threshold
        self.assertEqual(strict.to_dict()[5], 1)   # a stricter one correctly rejects it


class PredictionCellTypeTests(unittest.TestCase):
    """Directly exercises the legacy prediction_cell_type -- kept only as a
    reference to double-check prediction_cell_type_snr against on other
    datasets, no CellNetwork/tracking involved."""

    def test_classifies_by_brightest_channel(self):
        masks, fluorescent, _ = synthetic_two_type_movie()

        cell_types, _ = prediction_cell_type(fluorescent, masks, channel_number=2,
                                              bg_threshold=10, fc_threshold=50, high_val=0.8)

        self.assertEqual(cell_types.to_dict(), {1: 1, 2: 1, 3: 2, 4: 2})

    def test_cell_dim_in_every_channel_is_unclassified(self):
        """A cell with no marker signal above background in either channel
        should come out as type 0 (neither the ch0 nor ch1 bit set)."""
        masks, fluorescent, _ = synthetic_two_type_movie()
        masks[:, 25:33, 25:33] = 5  # 5th cell, left at background level in both channels

        cell_types, _ = prediction_cell_type(fluorescent, masks, channel_number=2,
                                              bg_threshold=10, fc_threshold=50, high_val=0.8)

        self.assertEqual(cell_types.to_dict()[5], 0)


class CreateCellTypeTests(unittest.TestCase):
    """End-to-end test of CellNetwork.create_cell_type -- the current
    default, SNR-based -- the way 03_create_cellnetwork_local.ipynb (or a
    future step) would call it."""

    def test_strain_type_assigned_from_fluorescence(self):
        masks, fluorescent, _ = synthetic_two_type_movie_with_noise()
        cellnet = CellNetwork.from_tracked_movie(masks, tracking_threshold=0.25, threshold=50)

        cellnet.create_cell_type(fluorescent, mask=cellnet.image, channel_number=2,
                                  bg_threshold=10, fc_threshold=50, z_threshold=3.0)

        strain_types = {cid: cell.strain_type for cid, cell in cellnet.cells.items()}
        self.assertEqual(strain_types, {1: 1, 2: 1, 3: 2, 4: 2})

    def test_fluorescent_intensity_table_is_populated(self):
        masks, fluorescent, _ = synthetic_two_type_movie_with_noise()
        cellnet = CellNetwork.from_tracked_movie(masks, tracking_threshold=0.25, threshold=50)
        self.assertIsNone(cellnet.fluorescent_intensity)

        cellnet.create_cell_type(fluorescent, mask=cellnet.image, channel_number=2,
                                  bg_threshold=10, fc_threshold=50, z_threshold=3.0)

        self.assertIsNotNone(cellnet.fluorescent_intensity)
        self.assertEqual(set(cellnet.fluorescent_intensity["label"].unique()), {1, 2, 3, 4})

    def test_plot_cell_types_runs_on_classified_network(self):
        """Smoke test for cellmate.visualize.plot_cell_types -- the same
        helper used in 04_overview_cellnetwork_local.ipynb to eyeball
        create_cell_type's output on real data."""
        masks, fluorescent, _ = synthetic_two_type_movie_with_noise()
        cellnet = CellNetwork.from_tracked_movie(masks, tracking_threshold=0.25, threshold=50)
        cellnet.create_cell_type(fluorescent, mask=cellnet.image, channel_number=2,
                                  bg_threshold=10, fc_threshold=50, z_threshold=3.0)

        ax = plot_cell_types(cellnet, frame=0)
        self.assertIsNotNone(ax)


class CreateCellTypeLegacyTests(unittest.TestCase):
    """CellNetwork.create_cell_type_legacy -- the old KMeans-based method,
    reachable the same convenient way as create_cell_type, for double
    -checking on other datasets."""

    def test_strain_type_assigned_from_fluorescence(self):
        masks, fluorescent, _ = synthetic_two_type_movie()
        cellnet = CellNetwork.from_tracked_movie(masks, tracking_threshold=0.25, threshold=50)

        cellnet.create_cell_type_legacy(fluorescent, mask=cellnet.image, channel_number=2,
                                         bg_threshold=10, fc_threshold=50, high_val=0.8)

        strain_types = {cid: cell.strain_type for cid, cell in cellnet.cells.items()}
        self.assertEqual(strain_types, {1: 1, 2: 1, 3: 2, 4: 2})


if __name__ == "__main__":
    unittest.main()
