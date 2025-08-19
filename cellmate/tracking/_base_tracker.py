from typing import Any

import numpy as np

from ..network._network import NetCell
from ..configs import DIVISION
from ._distance import compute_mask_iou


class BaseTracker():
    def __init__(self, image, min_hist=0, max_miss=None, is_generation=True,
                 is_allow_new_object=False
                 ) -> None:
        self.image = image
        self.tracker_end = []
        self.trackers = []
        self.count = 0
        self.distance = {}
        self.min_hist = min_hist
        self.max_miss = max_miss
        self.is_generation = is_generation
        if self.is_generation:
            self.network = NetCell()  # nx.DiGraph()
        else:
            self.network = None
        self.is_allow_new_object = is_allow_new_object
        self.match_method = "perfect"  # max, perfet

    def init_first_frame(self) -> None:
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass

    def tracker_features_exist(self):
        pass

    def to_image(self, is_keep_middle=False):
        """
        Save the tracked result into an image, where the new pixel values represent tracked information.
        The object ID in the tracked image is calculated as self.idx + tracker.category() * 1000.

        Parameters:
        -----------
        None

        Returns:
        -----------
        tracked_image: ndarray, same shape as self.image.
            The pixel values are replaced by new tracked IDs + category * 1000.
        """
        tracker = self.tracker_end + self.trackers
        if len(tracker) < 1:
            return None
        traced_image = np.zeros(self.image.shape, dtype=np.uint16)
        for i in range(len(tracker)):
            if tracker[i].life_time() < self.min_hist:
                continue

            if not is_keep_middle:
                if (~tracker[i].end) & (tracker[i].last_update() < self.image.shape[0]-1):
                    continue

            frame = tracker[i].frame
            label = np.array(tracker[i].label, dtype=np.int_).reshape([-1, 1, 1])
            class_id = tracker[i].category()
            masks = self.image[frame] == label
            for f, index in enumerate(frame):
                traced_image[index][masks[f]] = tracker[i].id + class_id*DIVISION
        return traced_image

    def all_trackers(self):
        return self.tracker_end + self.trackers

    def distance_matrix(self):
        pass

    def update_match(self):
        pass

    def update_fusion(self):
        pass

    def update_division(self):
        pass

    def save_trackers(self, is_keep_middle=False):
        tracker = self.tracker_end + self.trackers
        if len(tracker) < 1:
            return None
        tracker_saved = {}
        for i in range(len(tracker)):
            if tracker[i].life_time() < self.min_hist:
                continue

            if not is_keep_middle:
                if (~tracker[i].end) & (tracker[i].last_update() < self.image.shape[0]-1):
                    continue
            class_id = tracker[i].category()
            label = tracker[i].id + class_id*DIVISION
            tracker_saved[tracker[i].id] = {"label": label,
                                            "frame": tracker[i].frame}
        return tracker_saved

    def to_image_auto_fill_miss(self, is_keep_middle=False):
        traced_image = self.to_image()
        traced_image_filled = traced_image.copy()
        tracker = self.tracker_end + self.trackers
        if len(tracker) < 1:
            return None
        for i in range(len(tracker)):
            if tracker[i].life_time() < self.min_hist:
                continue

            if not is_keep_middle:
                if (~tracker[i].end) & (tracker[i].last_update() < self.image.shape[0]-1):
                    continue

            class_id = tracker[i].category()
            new_label = tracker[i].id + class_id*DIVISION

            frame = tracker[i].frame
            missed_frame = []
            for j in range(0, len(frame)-1):
                if frame[j+1] - frame[j] > 1:
                    missed_frame.append([frame[j], frame[j+1]])

            if len(missed_frame) <= 0:
                continue

            for k in range(0, len(missed_frame)):
                start = missed_frame[k][0]
                end = missed_frame[k][1]

                mask_start = traced_image[start] == new_label
                mask_end = traced_image[end] == new_label
                iou, _, _ = compute_mask_iou(mask_start, mask_end)
                if iou > 0.8:
                    overlap_mask = mask_start & mask_end
                    for m_f in range(start+1, end):
                        if (traced_image_filled[m_f][overlap_mask] > 0).sum() < 1000:
                            traced_image_filled[m_f][overlap_mask] = new_label
                        else:
                            print(new_label, start, "area > 1000")
                else:
                    print(new_label, m_f, "overlap < 0.9", iou)
        return traced_image, traced_image_filled
