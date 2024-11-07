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
from scipy.signal import savgol_filter, find_peaks


class DynamicPatch():
    def __init__(self, data, background, threshold=2, padding_length: int = 5):
        self.data = data
        self.background = background
        self.threshold = threshold
        self.padding_length = padding_length  # Make the signal periodic by padding it
        self.patch_length = 20

    @property
    def normalize(self):
        data_norm = (self.data - self.background[:, None])
        # data_norm = data_norm.clip(0, None)
        data_norm = data_norm / np.median(data_norm[data_norm > 0])
        data_norm = data_norm - self.threshold
        data_norm = data_norm.clip(0, None)
        return data_norm

    @property
    def binarized(self):
        pass

    def instant_activation(self, time, *args, **kwargs):
        peaks, peaks_info = detect_peaks(self.normalize[time], *args, **kwargs)
        return peaks, peaks_info


def detect_peaks(signal, window_length=21, polyorder=5, prominence=0.4, width=3):
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
