import numpy as np
import pandas as pd
# from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans


class FluorescentClassification():
    """
    Classifies cells into mating types from per-instance fluorescence
    intensity, one channel at a time, relative to that frame's background.

    Parameters:
    -----------
    data: pd.DataFrame
        Output of `instance_fluorescent_intensity` -- one row per (cell,
        frame), with `ch_{i}`, `bg_{i}` (and `bg_std_{i}` if computed)
        columns per channel.

    channel_number: int
        The number of fluorescent channels in the data.

    Current default: `prediction_by_label_snr` / `prediction_cell_type_snr`
    (SNR relative to background). `prediction_by_label` /
    `prediction_cell_type` (KMeans on a log-ratio normalization) is the
    original method, kept as a reference to double-check against on other
    datasets -- see the docstrings below and `prediction_cell_type_snr`'s
    docstring for why the default changed.
    """

    def __init__(self, data: np.array, channel_number) -> None:
        self.data = data.copy()
        self.channel_number = channel_number
        self.model = None

    # ---- current default: SNR relative to background -----------------

    def snr_intensity(self, min_noise=1.0):
        """Signal-to-noise ratio per channel: how many background standard
        deviations a cell's intensity sits above that frame's background
        mean, for that channel -- `(ch_i - bg_i) / bg_std_i`. Needs `bg_std_i`
        columns (pass `bg_std=...` into `instance_fluorescent_intensity`,
        i.e. call `background(..., return_std=True)`).

        `min_noise` floors `bg_std_i` before dividing, so a frame/channel
        with (near-)zero background variance doesn't blow up into a huge or
        infinite ratio.
        """
        for i in range(0, self.channel_number):
            signal = self.data['ch_%d' % i] - self.data['bg_%d' % i]
            noise = self.data['bg_std_%d' % i].clip(lower=min_noise)
            self.data['ch_%d_snr' % i] = signal / noise
        return self.data

    def prediction_data_type_snr(self, z_threshold=3.0, min_noise=1.0):
        """Decides each channel "on" or "off" by directly comparing that
        cell's intensity to its frame's background -- on if it's more than
        `z_threshold` background standard deviations above the background
        mean -- instead of normalizing to an odd log-ratio and
        2-means-clustering it (`single_channel_prediction`) against a
        fixed, data-independent `high_val` (see `prediction_data_type`,
        kept as a reference method below). Each channel is decided
        independently, so type 0/1/2/3 (for 2 channels) all stay possible,
        and it works unchanged for a single channel.
        """
        self.data = self.snr_intensity(min_noise=min_noise)
        self.data = self.data.dropna()
        label_chs = 0
        for i in range(0, self.channel_number):
            on_i = (self.data['ch_%d_snr' % i] > z_threshold).astype(int)
            self.data["ch_%d_prediction_snr" % i] = on_i
            label_chs += on_i * 2**i
        self.data["channel_prediction_snr"] = label_chs
        return self.data

    def prediction_by_label_snr(self, z_threshold=3.0, min_noise=1.0):
        _ = self.prediction_data_type_snr(z_threshold=z_threshold, min_noise=min_noise)
        pred = (
                self.data
                .groupby('label')['channel_prediction_snr']
                .agg(lambda x: pd.Series.mode(x).iloc[0])
            )
        return pred

    # ---- reference method (kept for double-checking other datasets) --

    def normalize_intensity(self):
        for i in range(0, self.channel_number):
            # self.data['ch%d_norm' % i] = (np.log(self.data['ch_%d' % i])-np.log(self.data['bg_%d' %i ]))/np.log(self.data['bg_%d' %i ])
            x = self.data['ch_%d' % i] - self.data['bg_%d' % i]
            x[x <= 0] = 1e-6
            x = np.log(x)/np.log(self.data['bg_%d' % i])
            x[x <= 0] = 0
            # x
            self.data['ch_%d_norm' % i] = x #np.log(x)/np.log(self.data['bg_%d' % i])
        return self.data

    def prediction_data_type(self, high_val=0.8):
        """Original classifier: normalizes to a log-ratio and 2-means
        clusters each channel independently against a fixed `high_val`.
        Superseded by `prediction_data_type_snr` (see its docstring), kept
        here as a reference/fallback to double-check against on datasets
        where the SNR method's background-noise estimate doesn't behave
        (e.g. a channel with unusually structured, non-Gaussian background).
        """
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
            label_i = single_channel_prediction(self.data['ch_%d_norm' % i], high_val=high_val)
            self.data["ch_%d_prediction" % i] = label_i
            label_chs += label_i*2**i
        self.data["channel_prediction"] = label_chs
        return self.data

    def prediction_by_label(self, high_val=0.8):
        _ = self.prediction_data_type(high_val=high_val)
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


def background(fluorescent_image, masks, threshold: float = 10, region="background", return_std: bool = False):
    """
    Get the background threshold for every channel.

    Parameters:
    ----------
    threshold: float
        The percentile of data not taken into account.
    return_std: bool
        If True, also return the (same trimmed-percentile) background
        standard deviation per frame/channel -- the noise level used by
        `prediction_cell_type_snr` to decide on/off relative to background.

    Returns:
    -------
    bg_mean: np.array with shape [frame x number of channels]
        An array containing background thresholds for each frame and channel.
    bg_std: np.array with shape [frame x number of channels], only if return_std
    """
    if region not in ["background", "cell"]:
        raise ValueError("region must be 'background' or 'cell'")

    # choose mask condition
    if region == "background":
        mask_condition = (masks == 0)
    else:
        mask_condition = (masks != 0)

    if fluorescent_image.ndim == 3:
        masked = fluorescent_image*(mask_condition[:, :, None])
    else:
        masked = fluorescent_image*(mask_condition[:, None, :, :])
    channel_number = fluorescent_image.shape[1]
    bg_mean = np.zeros((fluorescent_image.shape[0],  channel_number))
    bg_std = np.zeros((fluorescent_image.shape[0],  channel_number))
    for i in range(0, channel_number):
        for f in range(0, fluorescent_image.shape[0]):
            value = flatten_nonzero_value(masked[f, i])
            if value.sum() == 0:
                bg_mean[f, i] = 0
                bg_std[f, i] = 0
            else:
                floor = np.percentile(value, threshold)
                celling = np.percentile(value, 100-threshold)
                value = value[(value >= floor) & (value <= celling)]
                bg_mean[f, i] = np.mean(value)
                bg_std[f, i] = np.std(value)
    if return_std:
        return bg_mean, bg_std
    return bg_mean


def instance_fluorescent_intensity(fluorescent_image, masks, bg=None, bg_std=None, measure_line=70):
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
                if bg_std is not None:
                    data.append(bg_std[f, ch])
            data_sheet.append(data)
    data_sheet = pd.DataFrame(data_sheet)
    column = ["label", "frame"]
    for i in range(0, fluorescent_image.shape[1]):
        column.append(f"ch_{i}")
        if bg is not None:
            column.append(f"bg_{i}")
        if bg_std is not None:
            column.append(f"bg_std_{i}")
    data_sheet.columns = column
    return data_sheet


def flatten_nonzero_value(data):
    """flatten all non-zero values in data
    data: array_like
    """
    flatten = data.flatten()
    flatten = flatten[flatten > 0]
    return flatten


def prediction_cell_type_snr(fluorescent_image, masks, channel_number=2, bg_threshold=10, bg_region="background", fc_threshold=50, z_threshold=3.0, min_noise=1.0):
    """Current default classifier: decides each channel on/off independently
    by comparing directly against that frame's background (mean + std),
    rather than 2-means-clustering an odd log-ratio normalization against a
    fixed `high_val` -- see `FluorescentClassification.prediction_data_type_snr`
    for the rule. Every channel stays an independent on/off call, so type
    0/1/2/3 (for 2 channels) are all still possible outcomes, and it works
    unchanged for `channel_number=1` movies.

    `z_threshold` (background standard deviations) is the one knob to tune
    per dataset -- watch where the real clusters sit in a `ch_i_snr` scatter
    and set it above bleed-through's SNR, below real signal's. It can't
    tell genuine crosstalk from a real double-positive by itself (no
    per-channel-only test can) -- see `prediction_cell_type` below if you
    need a second opinion on an unfamiliar dataset.
    """
    bg, bg_std = background(fluorescent_image, masks, threshold=bg_threshold, region=bg_region, return_std=True)
    data = instance_fluorescent_intensity(fluorescent_image, masks, bg=bg, bg_std=bg_std, measure_line=fc_threshold)
    fc = FluorescentClassification(data, channel_number=channel_number)
    cell_types = fc.prediction_by_label_snr(z_threshold=z_threshold, min_noise=min_noise)
    return cell_types, fc.data


def prediction_cell_type(fluorescent_image, masks, channel_number=2, bg_threshold=10, bg_region="background", fc_threshold=50, high_val=0.8):
    """Original classifier (KMeans on a log-ratio normalization). Superseded
    by `prediction_cell_type_snr` as the default -- kept here as a reference
    method to double-check against on other datasets, not called by
    `CellNetwork.create_cell_type` by default anymore.
    """
    print(high_val)
    bg = background(fluorescent_image, masks, threshold=bg_threshold, region=bg_region)
    data = instance_fluorescent_intensity(fluorescent_image, masks, bg, measure_line=fc_threshold)
    fc = FluorescentClassification(data, channel_number=channel_number)
    cell_types = fc.prediction_by_label(high_val=high_val)
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
