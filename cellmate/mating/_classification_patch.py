import numpy as np
import pandas as pd
# from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans


class FluorescentClassificationPatch():
    """
    Classifies cell types based on fluorescence intensity using a GaussianMixture model.

    This function takes a dataframe of fluorescence intensity data and uses a GaussianMixture 
    model to classify cells into different types. The classification is based on the intensity 
    data across multiple fluorescent channels.

    Parameters:
    -----------
    data: np.array
        A numpy array obtained from cells.fluorescence_intensity. Each row represents a cell 
        with the format [cell label, frame, intensity at channel n, background intensity at channel n].

    channel_number: int
        The number of fluorescent channels in the data.

    model: GaussianMixture
        An instance of GaussianMixture model, which is fit with the data for classification.

    Returns:
    --------
    This function does not return a value but updates internal properties based on the classification.

    Example:
    --------
    >>> gaussian_classifier = FluorescentClassification(data)
    >>> gaussian_classifier.predition_data_type(n_components=n_components)
    >>> pred = gaussian_classifier.data.groupby('label')['channel_prediction'].agg(pd.Series.mode)

    Notes:
    ------
    Ensure that the data passed to the function is preprocessed correctly and matches the expected format.

    """

    def __init__(self, data: np.array, channel_number) -> None:
        self.data = data.copy()
        self.channel_number = channel_number
        self.model = None

    def normalize_intensity(self):
        for i in range(0, self.channel_number):
            # self.data['ch%d_norm' % i] = (np.log(self.data['ch_%d' % i])-np.log(self.data['bg_%d' %i ]))/np.log(self.data['bg_%d' %i ])
            x = self.data['ch_%d' % i] - self.data['bg_%d' % i]
            x[x <= 0] = 1e-6
            x = np.log(x)/np.log(self.data['bg_%d' % i])
            # x[x <= 1] = 1
            # x
            self.data['ch_%d_norm' % i] = x #np.log(x)/np.log(self.data['bg_%d' % i])
        return self.data

    def prediction_data_type(self):
        self.data = self.normalize_intensity()
        self.data = self.data.dropna()
        # data = self.data[['ch%d_norm' % i for i in range(0, self.channel_number)]]
        # clustering = GaussianMixture(**args).fit(data)
        # data_pred = clustering.predict(data)
        # class_map = rename_classes(data, clustering.means_)
        # data_pred = [class_map[x] for x in data_pred]
        # self.model = clustering
        # self.data["channel_prediction"] = data_pred
        # return data_pred, clustering
        label_chs = 0
        for i in range(0, self.channel_number):
            label_i = single_channel_prediction(self.data['ch_%d_norm' % i])
            self.data["ch_%d_prediction" % i] = label_i
            label_chs += label_i*2**i
        self.data["channel_prediction"] = label_chs
        return self.data

    def prediction_by_label(self):
        _ = self.prediction_data_type()
        # pred = self.data.groupby('label')['channel_prediction'].agg(pd.Series.mode)
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
    import itertools
    coords_index = np.flip(np.array(list(itertools.product([0, 1], repeat=coords.shape[0]))), axis=1)
    out = []
    for i in range(coords.shape[0]):
        out.append(coords[i, coords_index[:, i]])
    out = np.array(out).T
    return out


def background(fluorescent_image, masks, threshold: float = 10) -> np.array:
    """
    Get the background threshold for every channel.

    Parameters:
    ----------
    threshold: float
        The percentile of data not taken into account.

    Returns:
    -------
    bg_threshold: np.array with shape [frame x number of channels]
        An array containing background thresholds for each frame and channel.
    """
    if fluorescent_image.ndim == 3:
        masked = fluorescent_image*((masks == 0)[:, :, None])
    else:
        masked = fluorescent_image*((masks == 0)[:, None, :, :])
    channel_number = fluorescent_image.shape[1]
    bg_threshold = np.zeros((fluorescent_image.shape[0],  channel_number))
    for i in range(0, channel_number):
        for f in range(0, fluorescent_image.shape[0]):
            value = flatten_nonzero_value(masked[f, i])
            if value.sum() == 0:
                bg_threshold[f, i] = 0
            else:
                floor = np.percentile(value, threshold)
                celling = np.percentile(value, 100-threshold)
                value = value[(value >= floor) & (value <= celling)]
                bg_threshold[f, i] = np.mean(value)
    return bg_threshold


def instance_fluorescent_intensity(fluorescent_image, masks, bg=None, measure_line=70):
    data_sheet = []
    label_list = np.unique(masks)[1:]
    for label in label_list:
        mask = masks == label
        index = mask.sum(axis=(1, 2))
        index = np.where(index)[0]
        intensity = mask[:, None, :, :] * fluorescent_image

        for f in index:
            data = [label, f]
            for ch in range(0, fluorescent_image.shape[1]):
                ch_v = np.percentile(flatten_nonzero_value(intensity[f][ch]), measure_line)
                data.append(ch_v)
                if bg is not None:
                    data.append(bg[f, ch])
            data_sheet.append(data)
    data_sheet = pd.DataFrame(data_sheet)
    column = ["label", "frame"]
    if bg is not None:
        for i in range(0, fluorescent_image.shape[1]):
            column += [f"ch_{i}", f"bg_{i}"]
    else:
        for i in range(0, fluorescent_image.shape[1]):
            column.append(f"ch_{i}")
    data_sheet.columns = column
    return data_sheet


def flatten_nonzero_value(data):
    """flatten all non-zero values in data
    data: array_like
    """
    flatten = data.flatten()
    flatten = flatten[flatten > 0]
    return flatten


def prediction_cell_type_patch(fluorescent_image, masks, channel_number=2, bg_threshold=10, fc_threshold=50):
    bg = background(fluorescent_image, masks, threshold=bg_threshold)
    data = instance_fluorescent_intensity(fluorescent_image, masks, bg, measure_line=fc_threshold)
    fc = FluorescentClassificationPatch(data, channel_number=channel_number)
    cell_types = fc.prediction_by_label()
    return cell_types, fc.data


def single_channel_prediction(X, high_val=0.8):
    # Example data: rows are samples, columns are features
    # X = np.array(np.log(a.ch_0 - a.bg_0) / np.log(a.bg_0))
    X = np.array(X)
    X = X.reshape((X.shape[0], 1))

    init_centroids = np.array([[np.min(X)], [high_val]])  # e.g., [low_val], [high_val]
    kmeans = KMeans(n_clusters=2, init=init_centroids, n_init=1)
    labels = kmeans.fit_predict(X)
    return labels
