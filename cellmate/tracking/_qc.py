"""Sanity-check a finished Tracker run before saving its output.

Three checks, each reported with its location if found:

* missing node -- a lineage-graph node that never made it into the final
  tracked output.
* missing label -- a frame gap `to_image_auto_fill_miss` couldn't fill.
* mistake event -- re-tracking the movie you're about to save finds a
  different set of divisions/fusions than the original tracking did.
"""
import contextlib
import io

from cellmate.configs import DIVISION

from ._iou_tracker import Tracker


def check_tracking_quality(tracker, gap_report, tracked_image, threshold=None):
    """
    Parameters:
    -----------
    tracker: the Tracker run on the original segmentation (already called).
    gap_report: the 3rd value of `tracker.to_image_auto_fill_miss(return_report=True)`.
    tracked_image: that same call's filled output -- the movie about to be saved.
    threshold: IoU threshold for the verification retrace; defaults to `tracker.threshold`.

    Prints the location of anything found. Returns True if nothing was.
    """
    ok = True

    saved_ids = set((tracker.save_trackers() or {}).keys())
    for node in sorted(set(tracker.network.nodes) - saved_ids):
        ok = False
        print(f"missing node: id={node} never made it into the final tracked movie")

    for g in gap_report:
        if not g["filled"]:
            ok = False
            print(f"missing label: id={g['id']} frames {g['start']}-{g['end']} ({g['reason']})")

    boxes = {b.id: b for b in tracker.all_trackers()}
    original_events = _events(
        tracker.network,
        label_of=lambda n: boxes[n].id + boxes[n].category() * DIVISION,
        frame_of=lambda n: boxes[n].frame[-1],
    )

    # Tracker() prints its own frame-by-frame progress and division/fusion
    # log -- fine for the original run (you're watching it happen), just
    # noise for this internal verification pass.
    retrace = Tracker(tracked_image, threshold=threshold or tracker.threshold, min_hist=1, max_miss=1)
    with contextlib.redirect_stdout(io.StringIO()):
        retrace()
    retrace_boxes = {b.id: b for b in retrace.all_trackers()}
    retrace_events = _events(
        retrace.network,
        label_of=lambda n: int(retrace_boxes[n].label[-1]),
        frame_of=lambda n: retrace_boxes[n].frame[0],
    )

    for key in sorted(set(original_events) - set(retrace_events)):
        ok = False
        print(f"mistake event: {key} around frame {original_events[key]} "
              f"-- original tracking found this, retracing the saved movie didn't")
    for key in sorted(set(retrace_events) - set(original_events)):
        ok = False
        print(f"mistake event: {key} around frame {retrace_events[key]} "
              f"-- retracing the saved movie found this, original tracking didn't")

    if ok:
        print("tracking QC: no issues found")
    return ok


def _events(network, label_of, frame_of):
    """{(kind, ...labels): frame} for every division/fusion in `network`,
    keyed by the movie's own pixel labels (via `label_of`) rather than
    internal track ids -- so two independent Tracker runs over the same
    movie can be compared even though their internal ids differ."""
    events = {}
    for node in network.nodes:
        daughters = network.daughter(node)
        if daughters:
            key = ("division", label_of(node), tuple(sorted(label_of(d) for d in daughters)))
            events[key] = frame_of(node)
        parents = network.parents(node)
        if parents:
            key = ("fusion", tuple(sorted(label_of(p) for p in parents)), label_of(node))
            events[key] = frame_of(node)
    return events
