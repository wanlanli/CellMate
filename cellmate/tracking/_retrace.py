"""Re-run the IoU tracker on an already-tracked movie.

A tracked movie's ids are already stable across a track's lifetime, so
re-tracking it just re-derives the same lineage from geometry alone. Two
uses for that:

* verifying a tracked movie is self-consistent before saving it
  (`cellmate.tracking.check_tracking_quality`), and
* reconstructing the `network`/`tracker` a `CellNetwork` needs straight
  from a saved tracked movie (`CellNetwork.from_tracked_movie`), instead of
  keeping the original Tracker or its pickled output around.
"""
from ._iou_tracker import Tracker


def retrace(tracked_image, threshold, min_hist=1, max_miss=1):
    """Track `tracked_image` (an already-tracked label movie, [T, H, W])
    again, from scratch.

    `min_hist`/`max_miss` default to 1 because a properly delivered tracked
    movie already has stable ids and filled gaps -- there's nothing left to
    tolerate. Loosen them if `tracked_image` might still have unresolved
    gaps (e.g. you're retracing *before* filling, which is unusual).

    Returns the finished Tracker; use `.network` and `.save_trackers()`,
    e.g.:

        tracker = retrace(tracked_image, threshold=0.25)
        cellnet = CellNetwork(image=tracked_image, time_network=tracker.network,
                              tracker=tracker.save_trackers(), threshold=50)

    (`CellNetwork.from_tracked_movie` wraps exactly this.)
    """
    tracker = Tracker(tracked_image, threshold=threshold, min_hist=min_hist, max_miss=max_miss)
    tracker()
    return tracker
