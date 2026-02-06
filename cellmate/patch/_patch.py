# Copyright 2024 wlli
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
from scipy.signal import savgol_filter, find_peaks, correlate
from sklearn.cluster import MeanShift
import scipy.ndimage as ndimage
from scipy.ndimage import gaussian_filter1d, uniform_filter1d


class DynamicPatch():
    def __init__(self, data, background, threshold=2):
        """
        Initialize the DynamicPatch object with data, background, threshold, and padding length.

        Parameters:
        - data: numpy array representing the signal data.
        - background: numpy array representing the background levels to be subtracted from data.
        - threshold: float, threshold value for normalization to highlight significant signals.
        """
        self.data = data
        self.background = background
        self.threshold = threshold
        self.patch_length = 20
        self._peaks = None
        self._properties = None

    @property
    def normalize(self):
        """
        Normalize the signal data based on the background and threshold.
        - Subtract the background from each data point.
        - Normalize by dividing by the median of positive values, highlighting significant signals.
        - Subtract the threshold value to further isolate high signals.
        - Clip values below zero to ensure only positive signals are retained.
        Returns:
        - data_norm: numpy array, the normalized data with only significant values.
        """
        data_norm = (self.data - self.background[:, None])
        # data_norm = data_norm / np.median(data_norm[data_norm > 0])
        # data_norm = data_norm - self.threshold
        data_norm = data_norm.clip(0, None)
        return data_norm

    @property
    def binarized(self):
        peaks, properties = self.activation()
        binary_data = np.zeros(self.data.shape)
        for i in range(0, binary_data.shape[0]):
            peak = peaks[i]
            property = properties[i]
            for j in range(0, len(peak)):
                x_min = property["left_ips"][j]
                x_max = property["right_ips"][j]
                y_i = property["width_heights"][j]
                if x_min < x_max:
                    binary_data[i][int(x_min):int(x_max)] = y_i
                else:
                    binary_data[i][int(x_min):] = y_i
                    binary_data[i][:int(x_max)] = y_i

        return binary_data

    def label(self, bandwidth=15):
        index = np.where(self.binarized.sum(axis=0) == 0)[0][0]
        rolled_map = np.roll(self.binarized, -index, 1)
        labeled, num = ndimage.label(rolled_map > 0)
        labeled_feature = []
        for i in range(1, num+1):
            data = labeled == i
            coord_index = np.where(data.sum(axis=0))[0]
            labeled_feature.append([coord_index.min(), coord_index.max(), coord_index.mean(), np.median(coord_index)])
        cluster = MeanShift(bandwidth=bandwidth).fit(labeled_feature)

        cluster.cluster_centers_  = (cluster.cluster_centers_ + index) % labeled.shape[1]
        cluster.labels_  += 1

        mapped_array_take = np.zeros_like(labeled)
        # mapped_array_take = np.take([0]+list(cluster.labels_+1), labeled)
        non_zero_mask = labeled > 0
        mapped_array_take[non_zero_mask] = cluster.labels_[labeled[non_zero_mask] - 1]
        mapped_array_take = np.roll(mapped_array_take, index, 1)
        return mapped_array_take, cluster.cluster_centers_

    def instant_activation(self, time, *args, **kwargs):
        """
        Detects peaks in the normalized signal at a specific time step.

        Parameters:
        - time: int, the time step at which to analyze the signal for peaks.
        - *args, **kwargs: additional arguments passed to the peak detection function.

        Returns:
        - peaks: list or array, indices of detected peaks in the signal.
        - peaks_info: dict, additional information about each detected peak.
        """
        peaks, peaks_info = detect_peaks(self.normalize[time], *args, **kwargs)
        return peaks, peaks_info

    def activation(self):
        """
        Analyzes the signal across all time steps to detect peaks and gather their properties.

        - Iterates through each time step in the data.
        - For each time step, calls the `instant_activation` method to detect peaks.
        - Collects and stores the detected peaks and their properties for each time step.

        Returns:
        - peaks: list of arrays, each array contains the indices of detected peaks at each time step.
        - properties: list of dictionaries, each dictionary contains additional properties for the detected peaks at each time step.
        """
        properties = []
        peaks = []
        for i in range(0, self.data.shape[0]):
            peak, property = self.instant_activation(time=i)
            properties.append(property)
            peaks.append(peak)
        return peaks, properties

    def norm_dff(self):
        data_norm = (self.data - self.background[:, None])/(self.background[:, None] + 1e-6)
        data_norm = data_norm.clip(0, None)
        return data_norm

    def post_process(self):
        # 1 remove bg:
        data = self.data - self.background[:, None] # /(self.background[:, None] + 1e-6)
        # 2 top k mean


def smooth(data, sigma=2):
    sig = gaussian_filter1d(data, sigma=sigma)
    sig = sig - sig.min()
    return sig/sig.max()


def post_process(path, space_range, sigma):
    norm = path.norm_dff()
    norm = norm[:, space_range].sum(axis=1)
    smoothed = smooth(norm, sigma)
    return smoothed


# def estimate_delay(x, y, dt=1, max_lag=30):
#     x = np.tanh((x - x.mean()) / (x.std() + 1e-9))
#     y = np.tanh((y - y.mean()) / (y.std() + 1e-9))

#     x0 = x - np.mean(x)
#     y0 = y - np.mean(y)

#     corr = correlate(y0, x0, mode="full")
#     lags = np.arange(-len(x)+1, len(x))

#     mask = np.abs(lags) <= max_lag
#     corr_w = corr[mask]
#     lags_w = lags[mask]

#     idx = np.argmax(corr_w)
#     best_lag = lags_w[idx]
#     best_corr = corr_w[idx]

#     # normalize correlation
#     best_corr /= (np.std(x0) * np.std(y0) * len(x0))

#     delay_time = best_lag * dt

#     y = corr_w  # your 1D signal
#     # find all peaks
#     peaks, props = find_peaks(y)
#     # take the 3 highest peaks
#     top3 = peaks[np.argsort(y[peaks])[-3:]]
#     top3 = np.sort(top3)

#     return best_lag, delay_time, best_corr, corr_w, lags_w, lags_w[top3]


def detect_peaks(signal, window_length=21, polyorder=5, prominence=0.4, width=3):
    """
    Detect peaks in a signal with optional smoothing.

    Parameters:
    - signal (array-like): Input 1D signal array.
    - window_length (int): Length of the filter window (must be odd).
    - polyorder (int): Polynomial order for Savitzky-Golay filter.
    - prominence (float): Minimum prominence of peaks.
    - width (float): Minimum width of peaks.

    Returns:window_length
    - peaks (ndarray): Indices of detected peaks in the original signal.
    - properties (dict): Properties of detected peaks.
    """
    non_zero_start = np.argmin(signal)  # Find first non-zero element
    signal_rolled = np.roll(signal, -non_zero_start)
    smoothed_signal_rolled = savgol_filter(signal_rolled, window_length=window_length, polyorder=polyorder)
    peaks_rolled, properties_rolled = find_peaks(smoothed_signal_rolled, prominence=prominence, width=width)

    peaks = (peaks_rolled + non_zero_start) % len(signal)  # Adjust peaks back to original positions
    properties = properties_rolled
    properties['left_bases'] = (properties['left_bases'] + non_zero_start) % len(signal)
    properties['right_bases'] = (properties['right_bases'] + non_zero_start) % len(signal)
    properties['left_ips'] = (properties['left_ips'] + non_zero_start) % len(signal)
    properties['right_ips'] = (properties['right_ips'] + non_zero_start) % len(signal)

    return peaks, properties


def recurrent_index(start, end, maxlength):
    if end < start:
        end = end+maxlength
    indices = np.arange(start, end) % maxlength  # end + 10 to cover the wrap-around
    return indices


def patch_activity_picker(patch, key_range, win=50, sigma=2):
    raw_data  = patch.data.copy()
    rm_bg = raw_data - patch.background[:, None]
    # rm_bg = rm_bg.clip(0, None)
    data_range = rm_bg[:, key_range]
    # topk = np.percentile(data_range, 1000/len(key_range), axis=1)
    # mask = data_range > topk[:, None]
    # topk_data = np.mean(data_range, axis=1, where=mask)
    max_data = data_range.max(axis=1)
    # mean_data = data_range.mean(axis=1)
    # data_list = [max_data, mean_data, topk_data]
    dff_list = local_zscore(max_data, win=win)
    smooth_list = gaussian_filter1d(dff_list, sigma=sigma)
    zscore_list = zscore(smooth_list)
    tanh_list = np.tanh(zscore_list)
    # labels = ["max", "mean", "topk"]
    final = tanh_list.clip(0, None)
    return final


def dff(trace, win=30):
    # if f0_mode == "percentile":
    #     f0 = np.nanpercentile(trace, f0_percentile)
    # elif f0_mode == "first_n":
    #     f0 = np.nanmean(trace[:f0_first_n])
    # elif f0_mode == "median":
    #     f0 = np.nanmedian(trace)
    # else:
    #     raise ValueError("invalid f0_mode")
    f0 = uniform_filter1d(trace, size=win, mode="nearest")
    trace = (trace - f0) / (abs(f0) + 1e-6)
    return trace


def local_zscore(x, win=80, eps=1e-6, std_floor=None):
    """
    Local z-score: (x - local_mean) / local_std
    std_floor: optional minimum std to avoid blow-up, e.g., 0.05 * np.std(x)
    """
    x = np.asarray(x, dtype=float)
    mu = uniform_filter1d(x, size=win, mode="nearest")
    mu2 = uniform_filter1d(x**2, size=win, mode="nearest")
    var = np.maximum(mu2 - mu**2, 0.0)
    sigma = np.sqrt(var)

    if std_floor is not None:
        sigma = np.maximum(sigma, std_floor)

    z = (x - mu) / (sigma + eps)
    return z


def zscore(data):
    return (data-data.mean()) / (data.std() + 1e-6)


def estimate_delay(
    x, y, dt=1.0, eps=1e-12,
    use_abs_peaks=False,
    min_peak_dist=1
):
    """
    Full-length cross-correlation, choose the PEAK whose |lag| is minimal.

    Returns
    -------
    peak_lag : int
    peak_delay_time : float
    peak_corr_norm : float
    corr_norm : np.ndarray
    lags : np.ndarray
    """
    # x = np.asarray(x, dtype=float)
    # y = np.asarray(y, dtype=float)

    # demean
    x0 = x - np.mean(x)
    y0 = y - np.mean(y)

    # full cross-correlation
    corr = correlate(y0, x0, mode="full")
    lags = np.arange(-len(x0) + 1, len(x0))

    # normalize
    denom = (np.std(x0) * np.std(y0) * len(x0)) + eps
    corr_norm = corr / denom

    # peak finding
    score = np.abs(corr_norm) if use_abs_peaks else corr_norm
    peaks, _ = find_peaks(score, distance=max(1, int(min_peak_dist)))

    if len(peaks) == 0:
        return None, None, None, corr_norm, lags

    # 🔑 choose peak with minimal |lag|
    best_peak_idx = peaks[np.argmin(np.abs(lags[peaks]))]

    peak_lag = int(lags[best_peak_idx])
    peak_corr_norm = float(corr_norm[best_peak_idx])
    peak_delay_time = peak_lag * dt

    return peak_lag, peak_delay_time, peak_corr_norm, corr_norm, lags, peaks
