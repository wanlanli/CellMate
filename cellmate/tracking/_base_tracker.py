from typing import Any

import numpy as np

from ..network._network import NetCell
from ..configs import DIVISION


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

    def to_image(self):
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
