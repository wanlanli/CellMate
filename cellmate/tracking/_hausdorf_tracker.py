# from __future__ import absolute_import, division
from typing import Any
import logging

import numpy as np


from ..image_measure import ImageMeasure
from ..configs import h_img_col, TRACK_FEATURE_DIMENSION
from ._tracked_box import HausdorffTrackedBox
from ._base_tracker import BaseTracker
from ._distance import hausdorff_distance
from ._perfect_match import one_to_one_match, match_cells_generation_hausdorff


class Tracker(BaseTracker):
    """
    Track objects over time based on their morphology and dynamic information.

    Properties:
    -----------
    image: ndarray
        A 3D array of shape [Time x Width x Height] containing segmented images with uint16 data type.
        Pixels with the same label represent a single object.

    dimension: int
        The number of features used to measure the distance between objects.

    trackers: list
        A list of tracked objects.

    network: DiGraph
        A directed graph representing the generation (e.g., fusion, division) network between trackers.

    low_ratio: float
        The threshold used to measure matching between objects.

    high_ratio: float
        The threshold used to measure generation relationships between objects.

    count: int
        The number of tracked objects.
    """
    def __init__(self, image, dimension=TRACK_FEATURE_DIMENSION, low_ratio=0.8, high_ratio=1.5, min_hist=0,
                 max_miss=None, is_generation=True, is_allow_new_object=False, is_acc=False) -> None:
        super().__init__(image, min_hist, max_miss, is_generation, is_allow_new_object)
        self.dimension = dimension
        self.low_ratio = low_ratio  # threshold for measure match
        self.high_ratio = high_ratio  # threshold for measure division & fusion
        self.is_acc = is_acc

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
            coord_idex, columns = self.init_first_frame()

        # update from 1st to end
        for frame in range(1, self.image.shape[0]):
            instance_threshold, instance_feature = get_image_feature(
                self.image[frame], columns, coord_idex)

            if (len(self.trackers) > 0):
                tracker_threshold, tracker_feature, tracker_label = self.tracker_features_exist(frame)

                distance_matrix = hausdorff_distance(
                    tracker_feature.reshape([-1, self.dimension//2, 2]), instance_feature, is_acc=self.is_acc)

                distance_binary = (distance_matrix[0] < (tracker_threshold[:, 0, None]*self.low_ratio)) & \
                                  (distance_matrix[0] < (instance_threshold[:, 0:][None, :, 0])*self.low_ratio)

                match_dict = one_to_one_match(distance_binary)

                unmatched_trackers = self.update_matched(match_dict[0], match_dict[1], instance_feature, instance_threshold, frame)

                matched_generation_dict = match_cells_generation_hausdorff(
                    distance_matrix[1], unmatched_trackers, match_dict[2],
                    tracker_threshold, instance_threshold[:, 1:], self.high_ratio)

                self.update_fusion(matched_generation_dict[0], instance_feature,
                                   instance_threshold, frame)

                self.update_division(matched_generation_dict[1], instance_feature,
                                     instance_threshold, frame)

            # create and initialise new trackers for unmatched detections
            if ((len(matched_generation_dict[2]) > 0) | (len(matched_generation_dict[3]) > 0)):
                unmatched_tracker_label = [int(tracker_label[i]) for i in matched_generation_dict[2]]
                unmatched_instance_label = [int(instance_threshold[i][0]) for i in matched_generation_dict[3]]
                print("frame=", frame, unmatched_tracker_label, unmatched_instance_label)
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
        # init measure objects
        maskobj = ImageMeasure(self.image[0])
        length = maskobj.coordinate.shape[1]
        sample_index = np.linspace(0, length, self.dimension//2+1, dtype=np.int8)[:-1]
        columns = [h_img_col["label"], h_img_col["axis_minor_length"], h_img_col["axis_major_length"]]

        # take 4 points as tracking feature
        valid_obj = ~maskobj.is_border.astype(np.bool_)
        data = maskobj._properties[valid_obj]
        features = (data[:, columns]).astype(float)

        coords = maskobj.coordinate[:, sample_index, :]
        coords = coords[valid_obj]
        # update all objects into trackers
        for i in range(0, coords.shape[0]):
            self.count += 1
            trk = HausdorffTrackedBox(
                idx=self.count,
                label=int(features[i, 0]),
                frame=0,
                feature=coords[i].reshape([-1, self.dimension]),
                threshold=features[i, 1:],
                # data=features[i, 1:]
                )  # data save the threshold for tracking
            self.trackers.append(trk)
        return sample_index, columns

    def tracker_features_exist(self, frame):
        """
        Retrieve the features and thresholds of existing trackers.

        Returns:
        ----------
        thresholds: ndarray
            An array containing thresholds with shape [self.count, 2].
        coords: ndarray
            An array containing features with shape [self.count, self.dimension].
        """
        if self.max_miss is not None:
            new_tracker = []
            for obj in self.trackers:
                if (frame - obj.last_update() <= self.max_miss):
                    new_tracker.append(obj)
                else:
                    self.tracker_end.append(obj)
            self.trackers = new_tracker
        thresholds = np.zeros((len(self.trackers), 2))
        coords = np.zeros((len(self.trackers), self.dimension))
        labels = np.zeros(len(self.trackers))

        for t in range(0, len(self.trackers)):
            pos, threshold, label = self.trackers[t].predict()
            coords[t] = pos
            thresholds[t] = threshold
            labels[t] = label
        return thresholds, coords, labels

    def update_matched(self, matched_pairs, unmatched_trks,
                       input_feature, input_threshold, frame, data=None):
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
        # if data is None:
        #     data = [None]*len(matched_pairs)
        for m in matched_pairs:
            self.trackers[m[0]].update(
                label=int(input_threshold[m[1], 0]),
                frame=frame,
                feature=input_feature[m[1]].reshape(-1, self.dimension),
                data=input_threshold[m[1], 1:],
                )
            node = self.trackers[m[0]].id
            if self.network.has_node(node):
                upstream = self.network.upstream(node)  # [n for n in nx.traversal.bfs_tree(self.network, node, reverse=True) if n != node]
                if len(upstream):
                    unmatched_trks = [i for i in unmatched_trks if self.trackers[i].id not in upstream]

                downstream = self.network.downstream(node)  # [n for n in nx.traversal.dfs_tree(self.network, node) if n != node]
                if len(downstream):
                    unmatched_trks = [i for i in unmatched_trks if self.trackers[i].id not in downstream]
        return unmatched_trks

    def update_fusion(self, matched_fusion, input_feature, input_threshold, frame, data=None):
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
            # print(frame-1, "fusion:", fus)
            self.count += 1
            trk = HausdorffTrackedBox(
                idx=self.count,
                label=int(input_threshold[fus[2], 0]),
                frame=frame,
                feature=input_feature[fus[2]].reshape([-1, self.dimension]),
                threshold=input_threshold[fus[2], 1:],
                # data=input_threshold[fus[2], 1:]
                )
            self.trackers.append(trk)
            self.network.add_weighted_edges_from([[self.trackers[fus[0]].id, self.count, 2],
                                                  [self.trackers[fus[1]].id, self.count, 2]])
            # self.trackers[fus[0]].time_since_update = 0
            # self.trackers[fus[1]].time_since_update = 0

    def update_division(self, matched_division, input_feature, input_threshold, frame, data=None):
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
                trk = HausdorffTrackedBox(
                    idx=self.count,
                    label=input_threshold[div_i, 0],
                    frame=frame,
                    feature=input_feature[div_i].reshape([-1, self.dimension]),
                    threshold=input_threshold[div_i, 1:],
                    # data=input_threshold[div_i, 1:]
                    )
                # print(self.count, frame, data[div_i])
                self.trackers.append(trk)
                # print("trackers:", trk.id, trk.history)

            self.network.add_weighted_edges_from([[self.trackers[div[0]].id, self.count, 1],
                                                  [self.trackers[div[0]].id, self.count-1, 1]])


def get_image_feature(image, columns, sample_index):
    """
    From a single object, retrieve tracking information:
    - Label
    - Major and minor axis length (used as measures for threshold)
    - 4 tip points at the cortex as key features.

    Parameters:
    -----------
    image: ndarray
        Source image with measurements.

    columns: list
        Indices of features used in tracker to save object information.

    sample_index: list
        Indices of coordinates to pick out N tip points from the cortex.

    Returns:
    -----------
    feature: list
        Information including label and axis lengths used for setting thresholds.

    coords: list
        Tip points of the cortex used in tracking.

    data: list
        All object properties within the image.
    """

    maskobj = ImageMeasure(image)
    data = maskobj._properties
    valid_obj = ~maskobj.is_border.astype(np.bool_)
    data = data[valid_obj]
    threshold = (data[:, columns]).astype(float)
    coords = maskobj.coordinate[:, sample_index, :]
    coords = coords[valid_obj]
    return threshold, coords


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
