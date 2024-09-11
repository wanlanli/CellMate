# from __future__ import absolute_import, division
from typing import Any

import numpy as np

from ._perfect_match import one_to_one_match, match_cells_generation_iou, wash_distance_with_golden
from ._tracked_box import IoUTrackedBox
from ._base_tracker import BaseTracker
from ._distance import compute_mask_iou


class Tracker(BaseTracker):
    def __init__(self, image, threshold=0.7, min_hist=0, max_miss=None, is_generation=True, is_allow_new_object=False) -> None:
        super().__init__(image, min_hist, max_miss, is_generation, is_allow_new_object)
    # def __init__(self, image, threshold=0.7, min_hist=0, max_age=0) -> None:
    #     super().__init__(image, min_hist, max_age)
        self.threshold = threshold
        self.count = 0
        self.distance = {}

    def distance_matrix(self, last_track, instances):
        number_of_instances = len(instances)
        number_of_tracks = len(last_track)
        iou_matrix = np.zeros((number_of_tracks, number_of_instances, 3))
        for i, track in enumerate(last_track):
            for j, instance in enumerate(instances):
                instance_mask = self.image[instance[1]] == instance[0]
                track_mask = self.image[track[1]] == track[0]
                iou = compute_mask_iou(track_mask, instance_mask)
                iou_matrix[i, j] = iou
        return iou_matrix

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """
        Execute object tracking.

        This method performs object tracking based on the provided input arguments.

        Returns:
        ----------
        self.trackers: list
            A list of tracked objects after performing object tracking.
        """
        # init the 0 frame
        if self.image.shape[0] > 0:
            self.init_first_frame()

        # update from 1st to end
        for frame in range(1, self.image.shape[0]):
            print(frame)
            instances = get_image_feature(self.image[frame], frame)
            if (len(self.trackers) > 0):
                trackers = self.traker_features_exist(frame)
                distance_matrix = self.distance_matrix(trackers, instances)
                self.distance[frame] = distance_matrix
                golden_distance = wash_distance_with_golden(distance_matrix[:, :, 0])
                distance_matrix[:, :, 0] = golden_distance
                match_dict = one_to_one_match(golden_distance >= self.threshold)
                unmatched_trackers = self.update_matched(match_dict, instances)
                if frame == 98:
                    print(match_dict)
                matched_genteration_dict = match_cells_generation_iou(distance_matrix,
                                                                      self.threshold,
                                                                      unmatched_trackers,
                                                                      match_dict[2])

                self.update_fusion(matched_genteration_dict[0], instances)

                self.update_divison(matched_genteration_dict[1], instances)
                if ((len(matched_genteration_dict[2]) > 0) | (len(matched_genteration_dict[3]) > 0)):
                    print("frame=", frame, matched_genteration_dict[2], matched_genteration_dict[3])

            else:
                pass
        return self.trackers

    def init_first_frame(self):
        """
        Add all objects from the first frame to the tracker.
        Generate resampled coordinate points and initialize the features for the first frame.

        Returns:
        ----------
        sample_index: ndarray
            Resampled feature points' index from the coordinate list of the object cortex.
        columns: list
            Threshold and label index in the ImageMeasure object.
        """
        labels = np.unique(self.image[0])[1:]
        for i in range(0, len(labels)):
            self.count += 1
            trk = IoUTrackedBox(idx=self.count,
                                   label=labels[i],
                                   frame=0,
                                   feature=0)
            self.trackers.append(trk)

    def traker_features_exist(self, frame):
        """
        Retrieve the features and thresholds of existing trackers.

        Returns:
        ----------
        thres: ndarray
            An array containing thresholds with shape [self.count, 2].
        coords: ndarray
            An array containing features with shape [self.count, self.dimension].
        """
        data = []
        # if self.max_miss is not None:
        #     self.trackers = [obj for obj in self.trackers if (frame - obj.last_update() <= self.max_miss)]
        if self.max_miss is None:
            self.max_miss = self.image.shape[0]
        # if self.max_miss is not None:
        new_tracker = []
        for obj in self.trackers:
            if (frame - obj.last_update() <= self.max_miss) & (~obj.end):
                new_tracker.append(obj)
            else:
                self.tracker_end.append(obj)
        self.trackers = new_tracker

        for t in range(0, len(self.trackers)):
            feature = self.trackers[t].predict()
            data.append(feature)
        return data

    def update_matched(self, match_dict, instances):
        """
        Update object information in matched pairs and delete unmatched but related objects from the unmatched list.

        Parameters:
        -----------
        match_pairs: list of tuples
            A list of tuples representing matched pairs of indices (tracker_index, detection_index).
            Pairs are accepted if the distance between them is within the specified ratio.

        unmatched_trackers: list
            A list of unmatched trackers that were not accepted.

        input_feature: ndarray
            New detected object's features in matched pairs, to be updated to the tracker.

        input_threshold: ndarray
            New detected object's thresholds in matched pairs, to be updated to the tracker.

        frame: int
            Frame number for new detected objects from matched pairs, to be updated to the tracker.

        data: ndarray
            Data from new detected objects from matched pairs, containing features to be saved for the tracker.
            To be updated to the tracker.

        Returns:
        -----------
        unmatched_trks: list
            A list of unmatched trackers and unrelated objects.
        """
        matched_pairs = match_dict[0]
        unmatched_trks = match_dict[1]
        for m in matched_pairs:
            matched_inst = instances[m[1]]
            self.trackers[m[0]].update(matched_inst[0], matched_inst[1])
            node = self.trackers[m[0]].id
            if self.network.has_node(node):
                upstream = self.network.upstream(node)  # [n for n in nx.traversal.bfs_tree(self.network, node, reverse=True) if n != node]
                if len(upstream):
                    up = [i for i in unmatched_trks if self.trackers[i].id in upstream]
                    if len(up):
                        print(f"{up} in {node} upstream")
                    unmatched_trks = [i for i in unmatched_trks if self.trackers[i].id not in upstream]
                downstream = self.network.downstream(node)  # [n for n in nx.traversal.dfs_tree(self.network, node) if n != node]
                if len(downstream):
                    down = [i for i in unmatched_trks if self.trackers[i].id in downstream]
                    if len(down):
                        print(f"{down} in {node} upstream")
                    unmatched_trks = [i for i in unmatched_trks if self.trackers[i].id not in downstream]
        return unmatched_trks

    def update_fusion(self, matched_fusion, instance):
        """
        Update object information in matched fusion pairs and create a new tracker for the fused objects.
        Update generation relationships in self.network.

        Parameters:
        -----------
        matched_fusion: list of tuples
            A list of tuples representing matched fusion pairs of indices (tracker_index1, tracker_index2, detection_index1).

        input_feature: ndarray
            New fused object's features in matched pairs, used to create a new tracker.

        input_threshold: ndarray
            New fused object's thresholds in matched pairs, used to create a new tracker.

        frame: int
            Frame number for the new fused objects from matched pairs, used to create a new tracker.

        data: ndarray
            Data from the new fused objects from matched pairs, containing features to be saved for the new tracker.

        Returns:
        -----------
        """
        for fus in matched_fusion:
            print("upate fusion")
            self.count += 1
            label = instance[fus[2]]
            trk = IoUTrackedBox(
                idx=self.count,
                # feature=input_feature[fus[2]].reshape([-1, self.dimention]),
                # threshold=input_threshold[fus[2], 1:],
                label=label[0],
                frame=label[1],
                feature=0,
                )
            
            self.trackers.append(trk)
            self.network.add_weighted_edges_from([[self.trackers[fus[0]].id, self.count, 2],
                                                  [self.trackers[fus[1]].id, self.count, 2]])
            self.trackers[fus[0]].end = True
            self.trackers[fus[1]].end = True

            # self.trackers[fus[0]].time_since_update = 0
            # self.trackers[fus[1]].time_since_update = 0

    def update_divison(self, matched_division, instances):
        """
        Update object information in matched division pairs and create a new tracker for the divided objects.
        Update generation relationships in self.network.

        Parameters:
        -----------
        matched_division: list of tuples
            A list of tuples representing matched division pairs of indices (tracker_index1, detection_index1, detection_index2).

        input_feature: ndarray
            New divided object's features in matched pairs, used to create a new tracker.

        input_threshold: ndarray
            New divided object's thresholds in matched pairs, used to create a new tracker.

        frame: int
            Frame number for the new divided objects from matched pairs, used to create a new tracker.

        data: ndarray
            Data from the new divided objects from matched pairs, containing features to be saved for the new tracker.

        Returns:
        -----------
        """
        for div in matched_division:
            for div_i in div[1:]:
                self.count += 1
                trk = IoUTrackedBox(
                    idx=self.count,
                    label=instances[div_i][0],
                    frame=instances[div_i][1],
                    feature=0,
                    )
                self.trackers.append(trk)

            self.network.add_weighted_edges_from([[self.trackers[div[0]].id, self.count, 1],
                                                  [self.trackers[div[0]].id, self.count-1, 1]])
            self.trackers[div[0]].end = True


def get_image_feature(image, frame):
    labels = np.unique(image)
    labels = labels[labels != 0]
    return [[i, frame] for i in labels]


if __name__ == "__main__":
    from ..io import imread, imsave
    images = imread("./cellmating/data/example_for_tracking.tif").astype(np.uint16)
    trace = Tracker(images)
    _ = trace()
    traced_image = trace.to_image()
    imsave("./cellmating/data/example_for_traced.tif",
           traced_image.astype(np.uint16),
           imagej=True)
# python -m python -m cellmating.sort._sort_cell
