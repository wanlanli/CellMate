"""Original mating-type classifier: 2-means clustering a log-ratio
normalization against a fixed `high_val`, per channel. Superseded by the
SNR-based default in `._classification` (`prediction_cell_type_snr`, via
`CellNetwork.create_cell_type`) -- kept here only as a reference/fallback to
double-check that default against on datasets where its background-noise
estimate doesn't behave (e.g. a channel with unusually structured,
non-Gaussian background). Reachable via `CellNetwork.create_cell_type_legacy`.
"""
import itertools

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from ._classification import background, instance_fluorescent_intensity


class FluorescentClassification():
    """
    Classifies cells into mating types from per-instance fluorescence
    intensity, one channel at a time, via a log-ratio normalization
    2-means-clustered against a fixed `high_val`.

    Parameters:
    -----------
    data: pd.DataFrame
        Output of `instance_fluorescent_intensity` -- one row per (cell,
        frame), with `ch_{i}`, `bg_{i}` columns per channel.
    channel_number: int
        The number of fluorescent channels in the data.
    """

    def __init__(self, data: np.array, channel_number) -> None:
        self.data = data.copy()
        self.channel_number = channel_number
        self.model = None

    def normalize_intensity(self):
        for i in range(0, self.channel_number):
            x = self.data['ch_%d' % i] - self.data['bg_%d' % i]
            x[x <= 0] = 1e-6
            x = np.log(x)/np.log(self.data['bg_%d' % i])
            x[x <= 0] = 0
            self.data['ch_%d_norm' % i] = x
        return self.data

    def prediction_data_type(self, high_val=0.8):
        """Normalizes to a log-ratio and 2-means clusters each channel
        independently against a fixed `high_val`."""
        self.data = self.normalize_intensity()
        self.data = self.data.dropna()
        label_chs = 0
        for i in range(0, self.channel_number):
            label_i = single_channel_prediction(self.data['ch_%d_norm' % i], high_val=high_val)
            self.data["ch_%d_prediction" % i] = label_i
            label_chs += label_i*2**i
        self.data["channel_prediction"] = label_chs
        return self.data

    def prediction_by_label(self, high_val=0.8):
        _ = self.prediction_data_type(high_val=high_val)
        pred = (
                self.data
                .groupby('label')['channel_prediction']
                .agg(lambda x: pd.Series.mode(x).iloc[0])
            )
        return pred


def rename_classes(data, cluster_centers):
    coords = __get_bounding_points(data)
    class_map = {}
    for i, c in enumerate(cluster_centers):
        dist = np.sqrt(np.square(coords - c).sum(axis=1))
        label = np.argmin(dist)
        class_map[i] = label
    return class_map


def __get_bounding_points(data):
    box_min = data.min()
    box_max = data.max()
    coords = np.array([box_min, box_max]).T
    coords_index = np.flip(np.array(list(itertools.product([0, 1], repeat=coords.shape[0]))), axis=1)
    out = []
    for i in range(coords.shape[0]):
        out.append(coords[i, coords_index[:, i]])
    out = np.array(out).T
    return out


def prediction_cell_type(fluorescent_image, masks, channel_number=2, bg_threshold=10, bg_region="background", fc_threshold=50, high_val=0.8):
    """Original classifier (KMeans on a log-ratio normalization). Superseded
    by `cellmate.mating.prediction_cell_type_snr` as the default -- kept
    here as a reference method to double-check against on other datasets,
    not called by `CellNetwork.create_cell_type` by default anymore.
    """
    bg = background(fluorescent_image, masks, threshold=bg_threshold, region=bg_region)
    data = instance_fluorescent_intensity(fluorescent_image, masks, bg, measure_line=fc_threshold)
    fc = FluorescentClassification(data, channel_number=channel_number)
    cell_types = fc.prediction_by_label(high_val=high_val)
    return cell_types, fc.data


def single_channel_prediction(X, high_val=0.8):
    X = np.array(X)
    X = X.reshape((X.shape[0], 1))

    init_centroids = np.array([[np.min(X)], [high_val]])  # e.g., [low_val], [high_val]
    kmeans = KMeans(n_clusters=2, init=init_centroids, n_init=1)
    labels = kmeans.fit_predict(X)
    return labels
