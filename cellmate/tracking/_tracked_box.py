import statistics
import numpy as np

from ..configs import DIVISION


class TrackedBox(object):
    """
    Represents the internal state of an individually tracked object observed as a feature array.

    This class is used to make predictions based on the feature array.

    Parameters:
    -----------
    idx: int
        The unique ID for each tracker.

    label: list
        A list to store the history of tracked labels.

    frame: list
        A list to store historical frame numbers related to tracking.

    feature: ndarray
        An array representing the features traced from the last tracker,
        which are used for prediction.

    data:
        Any others feature want to save.
    """
    def __init__(self,
                 idx,
                 label,
                 frame,
                 feature=None,
                 data=None):
        self.id = idx  # unique id for each tracked object
        self.label = [label]  # keep history label & frame information
        self.frame = [frame]
        self.feature = [feature]  # feature vector for calculate overlap.
        self.data = [data]  # other features want to save
        self.end = False

    def update(self, label, frame, feature=None, data=None):
        """
        Updates the state vector with observed bbox.
        """
        self.label.append(label)
        self.frame.append(frame)
        self.feature.append(feature)
        self.data.append(data)

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        return [self.label[-1], self.frame[-1]]

    def category(self):
        mode = statistics.mode(np.array(self.label) // DIVISION)
        return mode

    def last_update(self):
        return self.frame[-1]

    def life_time(self):
        return len(self.frame)


class HausdorffTrackedBox(TrackedBox):
    def __init__(self, idx, label, frame, threshold=None, feature=None, data=None):
        super().__init__(idx, label, frame, feature, data)
        self.threshold = [threshold]

    def predict(self):
        return [self.feature[-1], self.threshold[-1], self.label[-1]]


class IoUTrackedBox(TrackedBox):
    def __init__(self, idx, label, frame, feature=None, data=None):
        super().__init__(idx, label, frame, feature, data)


class KalmanTrackedBox(TrackedBox):
    def __init__(self, idx, label, frame, feature=None, data=None):
        super().__init__(idx, label, frame, feature, data)
