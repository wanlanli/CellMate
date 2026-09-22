import numpy as np
import pandas as pd


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


# ---- SNR classification pipeline (current default) --------------------
#
# One row per (cell, frame) flows through four small steps, each adding
# columns to the same DataFrame -- no shared mutable state, no class:
#
#   instance_fluorescent_intensity  ch_i, bg_i, bg_std_i (raw measurements)
#   -> _add_channel_snr             + ch_i_snr           (relative to background)
#   -> _add_channel_calls           + ch_i_prediction_snr, channel_prediction_snr (per-frame on/off + bitmask)
#   -> _mode_per_cell                 one call per cell (mode across its frames)

def _add_channel_snr(data, channel_number, min_noise=1.0):
    """Add a `ch_{i}_snr` column per channel: how many background standard
    deviations a cell's intensity sits above that frame's background mean,
    `(ch_i - bg_i) / bg_std_i`. Needs `bg_i` / `bg_std_i` columns (i.e.
    `instance_fluorescent_intensity` called with `bg=`, `bg_std=` from
    `background(..., return_std=True)`).

    `min_noise` floors `bg_std_i` before dividing, so a frame/channel with
    (near-)zero background variance doesn't blow up into a huge or infinite
    ratio. Drops any row left with missing values (e.g. no bg_std).
    """
    data = data.copy()
    for i in range(0, channel_number):
        signal = data[f"ch_{i}"] - data[f"bg_{i}"]
        noise = data[f"bg_std_{i}"].clip(lower=min_noise)
        data[f"ch_{i}_snr"] = signal / noise
    return data.dropna()


def _add_channel_calls(data, channel_number, z_threshold=3.0):
    """Decide each channel "on" or "off" per (cell, frame) by directly
    comparing `ch_{i}_snr` to `z_threshold` -- on once it's more than
    `z_threshold` background standard deviations above the background mean.
    Each channel is decided independently (`ch_{i}_prediction_snr`), so
    type 0/1/2/3 (for 2 channels) all stay possible; the per-channel calls
    are also combined into one bitmask column, `channel_prediction_snr`
    (bit i = channel i on), and it works unchanged for a single channel.
    """
    data = data.copy()
    channel_prediction = 0
    for i in range(0, channel_number):
        on_i = (data[f"ch_{i}_snr"] > z_threshold).astype(int)
        data[f"ch_{i}_prediction_snr"] = on_i
        channel_prediction = channel_prediction + on_i * 2**i
    data["channel_prediction_snr"] = channel_prediction
    return data


def _mode_per_cell(data, column):
    """Collapse per-frame calls down to one call per cell: the most common
    value of `column` across that cell's frames."""
    return data.groupby("label")[column].agg(lambda x: pd.Series.mode(x).iloc[0])


def prediction_cell_type_snr(fluorescent_image, masks, channel_number=2, bg_threshold=10, bg_region="background", fc_threshold=50, z_threshold=3.0, min_noise=1.0):
    """Current default classifier: decides each channel on/off independently
    by comparing directly against that frame's background (mean + std),
    rather than 2-means-clustering an odd log-ratio normalization against a
    fixed `high_val` -- see `_add_channel_calls` for the rule. Every channel
    stays an independent on/off call, so type 0/1/2/3 (for 2 channels) are
    all still possible outcomes, and it works unchanged for
    `channel_number=1` movies.

    `z_threshold` (background standard deviations) is the one knob to tune
    per dataset -- watch where the real clusters sit in a `ch_i_snr` scatter
    and set it above bleed-through's SNR, below real signal's. It can't
    tell genuine crosstalk from a real double-positive by itself (no
    per-channel-only test can) -- see `cellmate.mating._classification_legacy`
    if you need a second opinion on an unfamiliar dataset.

    Returns
    -------
    cell_types: pd.Series, index = label, value = channel_prediction_snr
        (mode across that cell's frames).
    data: pd.DataFrame, one row per (cell, frame) with every intermediate
        column (ch_i, bg_i, bg_std_i, ch_i_snr, ch_i_prediction_snr,
        channel_prediction_snr) -- e.g. for the SNR scatter plot in
        `04_overview_cellnetwork_local.ipynb`.
    """
    bg_mean, bg_std = background(fluorescent_image, masks, threshold=bg_threshold, region=bg_region, return_std=True)
    data = instance_fluorescent_intensity(fluorescent_image, masks, bg=bg_mean, bg_std=bg_std, measure_line=fc_threshold)
    data = _add_channel_snr(data, channel_number, min_noise=min_noise)
    data = _add_channel_calls(data, channel_number, z_threshold=z_threshold)
    cell_types = _mode_per_cell(data, "channel_prediction_snr")
    return cell_types, data
