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
import pandas as pd
from scipy.signal import savgol_filter, find_peaks, peak_widths, correlate
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import MeanShift
import scipy.ndimage as ndimage
from scipy.ndimage import gaussian_filter1d, uniform_filter1d


class DynamicPatch():
    def __init__(self, data, background):
        """
        Initialize the DynamicPatch object with data and background.

        Parameters:
        - data: numpy array representing the signal data.
        - background: numpy array representing the background levels to be subtracted from data.
        """
        self.data = data
        self.background = background
        self.patch_length = 20
        self._peaks = None
        self._properties = None

    @property
    def normalize(self):
        """
        Background-subtract the signal data, clipping away everything that
        doesn't exceed its own frame's background.
        - Subtract the background from each data point.
        - Clip values below zero to ensure only positive signals are retained.

        Stays in `data`'s original (raw intensity) units -- not rescaled or
        divided by anything, so an absolute cutoff (e.g. `min_height` in
        `detect_single_patches()`/`track_patches()`, or `low_threshold`/
        `high_threshold` in `detect_patches_hysteresis()`) means a different
        thing for data from a different camera/laser setting or channel, and
        should be re-checked rather than assumed to carry over.

        Returns:
        - data_norm: numpy array, the normalized data with only significant values.
        """
        data_norm = (self.data - self.background[:, None])
        data_norm = data_norm.clip(0, None)
        return data_norm

    @property
    def noise_scale(self):
        """
        Robust estimate of this cell's own background-subtracted noise level,
        in the same (raw) units as `normalize` -- meant as a per-cell
        reference scale for an absolute detection threshold (e.g.
        `min_height` in `detect_single_patches()`/`track_patches()`, or
        `low_threshold`/`high_threshold` in `detect_patches_hysteresis()`),
        since one fixed number doesn't necessarily transfer across cells or
        imaging sessions of different absolute brightness (see `normalize`'s
        own docstring) but a real patch's amplitude shouldn't be rescaled
        away either (it may itself be the thing being compared, e.g. between
        strains) -- so this gives a per-cell multiplier to scale a threshold
        by, instead of rescaling the signal itself.

        Computed as the median absolute deviation (MAD, scaled by 1.4826 to
        be directly comparable to a standard deviation for normally
        distributed noise) of `data - background`, *not* clipped to positive
        the way `normalize` is -- using the unclipped, signed residual keeps
        the estimate centred on zero. A patch occupies only a small fraction
        of all (time, position) pixels for a cell (a handful of frames out of
        the whole movie, a handful of positions out of the full contour), so
        the bulk of this distribution -- and with it the median and MAD -- is
        governed by the "off" baseline noise, barely moved by the rare real
        patches; unlike e.g. `0.2 * self.normalize.max()`, this doesn't
        define the threshold circularly in terms of the very peak it's
        meant to help detect, and isn't thrown off by that one peak alone.

        Returns:
        - float, this cell's noise scale (a small constant is added so a
          perfectly flat, noiseless background returns a tiny positive
          number rather than exactly 0).

        Example:
        >>> patch = DynamicPatch(data, background)
        >>> tracks_df = patch.track_patches(min_height=2 * patch.noise_scale)
        """
        residual = self.data - self.background[:, None]
        mad = np.median(np.abs(residual - np.median(residual)))
        return 1.4826 * mad + 1e-6

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

    def binarized_mask(self, **kwargs):
        """
        Same per-frame peak-width fill as `binarized` above, but forwards
        `**kwargs` (`prominence`, `width`, `window_length`, `polyorder`) to
        each frame's `detect_peaks` call instead of `activation()`'s fixed
        defaults -- so how strict "is this frame's local mountain real"
        is can be tuned, e.g. to make a weak fade in the middle of what looks
        like one long patch actually fail detection there and split it:
        `patch.detect_patches(mask=patch.binarized_mask(prominence=1.0, width=5) > 0)`.
        `binarized` itself is left as-is since other notebooks call it with
        no arguments.
        """
        binary_data = np.zeros(self.data.shape)
        for i in range(0, binary_data.shape[0]):
            peak, property = self.instant_activation(time=i, **kwargs)
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
        empty_cols = np.where(self.binarized.sum(axis=0) == 0)[0]
        # Normally rolled to a column that's never active, so no cluster can straddle the
        # array's own seam; falls back to not rolling on the rare cell whose signal covers
        # every contour position at some point, same fallback detect_patches() uses -- this
        # can then in that rare case double-count a cluster that straddles position 0,
        # instead of crashing the way a bare index [0] into an empty `np.where()` would.
        index = int(empty_cols[0]) if len(empty_cols) else 0
        rolled_map = np.roll(self.binarized, -index, 1)
        labeled, num = ndimage.label(rolled_map > 0)
        if num == 0:
            # No peaks detected anywhere (flat/no signal) -- nothing to cluster.
            return np.zeros_like(labeled), np.empty((0, 4))
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

    def detect_patches(self, data=None, threshold=0.0, mask=None, structure=None, min_size=1):
        """
        Detect individual patches as connected regions in the time x position
        map, where "connected" means 4-connected (sharing a time- or space-edge,
        not just a corner) in both the time axis and the circular position axis
        (position 0 and position -1 are neighbours, since the aligned contour is
        a closed loop around the cell). One patch = one such maximal connected
        region of active pixels.

        By default "active" means `data > threshold` -- a single global cutoff.
        That has an inherent tension: raised to split two real patches sitting
        close together (so the dip between them actually drops out), it can
        also drop a real but weaker patch elsewhere; lowered to keep that weak
        patch, the close pair stays merged into one long blob. Pass `mask`
        (any boolean time x position array of the same shape as `data`) to
        decide "active" some other way and skip `threshold` entirely -- in
        particular, `mask=self.binarized > 0` reuses the per-frame peak-width
        detection `binarized`/`activation()` already do above (`detect_peaks`,
        judged by *relative* prominence against each frame's own neighbouring
        signal, same as `label()` relies on), rather than one absolute level
        shared across every frame and every patch. That makes the two
        decisions independent: a frame where the signal fades back toward
        background no longer finds a peak there at all, which breaks the
        connected region in time right where it should (a real fade "back to
        black" becomes two patches); a frame with only a weak signal can still
        register its own (locally-relative) peak, so it isn't missed just
        because some other frame needed a higher bar. `data` itself is still
        used for the intensity features below (duration/extent come from the
        mask; intensity values are read from `data` at the same pixels), so
        pass `data=self.normalize` (the default) alongside a `binarized` mask
        to get real intensity units rather than `binarized`'s own plateau
        values.

        Parameters:
        - data: time x position array providing the intensity values for
          duration/extent-independent features (peak/mean/integrated
          intensity, centroids). Defaults to `self.normalize`.
        - threshold: float, a pixel is "active" when `data > threshold`.
          Ignored when `mask` is given.
        - mask: optional boolean time x position array marking "active"
          pixels directly, overriding `threshold`. See `self.binarized > 0`
          above.
        - structure: connectivity structure forwarded to `scipy.ndimage.label`.
          Defaults (None) to 4-connectivity (face neighbours only); pass
          `np.ones((3, 3))` for 8-connectivity if diagonal touches should merge.
        - min_size: minimum number of pixels (time-points x positions) for a
          connected region to be kept as a patch; smaller ones are discarded.

        To handle the circular position axis without double-counting, this
        rolls the active-pixel mask so a position column that is inactive at
        every time point sits at the roll boundary before labelling (same
        trick `label()` uses above), then rolls the result back. That boundary
        column is guaranteed not to split a real patch. If no column is ever
        fully inactive (the signal covers the whole circle at some point),
        it falls back to not rolling, which can in that rare case double-count
        a patch that straddles position 0.

        Returns:
        - pandas.DataFrame, one row per detected patch, with columns:
          - patch_id: 1-based label id (arbitrary, not stable across calls)
          - n_pixels: footprint size -- number of (time, position) pixels in
            the patch
          - time_start, time_end, duration: first/last frame and frame count
            the patch is active for
          - position_start, position_end: first/last aligned-contour position,
            in real (not rolled) coordinates -- position_start > position_end
            indicates the patch wraps through position 0
          - spatial_width: number of contour positions spanned (wrap-aware,
            unlike `position_end - position_start`)
          - time_center, position_center: intensity-weighted centroid, i.e.
            where the patch's mass sits in time/space (position_center is
            circular-aware)
          - peak_intensity, peak_time, peak_position: the patch's maximum
            value and where it occurs
          - mean_intensity: average intensity over the patch's footprint
          - integrated_intensity: sum of intensity over the patch's footprint
            ("volume" -- combines how strong and how large/long the patch is)

        Example:
        >>> patch = DynamicPatch(data, background)
        >>> patches_df = patch.detect_patches()
        >>> patches_df[["duration", "spatial_width", "peak_intensity", "integrated_intensity"]]
        >>> # local-mountain version, reusing the existing per-frame peak detection:
        >>> patches_df2 = patch.detect_patches(mask=patch.binarized > 0)
        """
        data = self.normalize if data is None else np.asarray(data)
        mask = np.asarray(mask, dtype=bool) if mask is not None else data > threshold
        n_time, n_pos = mask.shape

        empty_cols = np.where(mask.sum(axis=0) == 0)[0]
        roll_index = int(empty_cols[0]) if len(empty_cols) else 0
        rolled_mask = np.roll(mask, -roll_index, axis=1)
        labeled, num_labels = ndimage.label(rolled_mask, structure=structure)

        records = []
        for patch_id in range(1, num_labels + 1):
            t_idx, pos_idx_rolled = np.where(labeled == patch_id)
            if len(t_idx) < min_size:
                continue
            pos_idx = (pos_idx_rolled + roll_index) % n_pos
            values = data[t_idx, pos_idx]
            weights = values / values.sum()

            t_start, t_end = int(t_idx.min()), int(t_idx.max())
            pos_start_rolled, pos_end_rolled = int(pos_idx_rolled.min()), int(pos_idx_rolled.max())
            peak_local = int(np.argmax(values))

            records.append({
                "patch_id": patch_id,
                "n_pixels": int(len(t_idx)),
                "time_start": t_start,
                "time_end": t_end,
                "duration": t_end - t_start + 1,
                "position_start": (pos_start_rolled + roll_index) % n_pos,
                "position_end": (pos_end_rolled + roll_index) % n_pos,
                "spatial_width": pos_end_rolled - pos_start_rolled + 1,
                "time_center": float(np.sum(t_idx * weights)),
                "position_center": float((np.sum(pos_idx_rolled * weights) + roll_index) % n_pos),
                "peak_intensity": float(values[peak_local]),
                "peak_time": int(t_idx[peak_local]),
                "peak_position": int(pos_idx[peak_local]),
                "mean_intensity": float(values.mean()),
                "integrated_intensity": float(values.sum()),
            })
        return pd.DataFrame.from_records(records)

    def track_patches(self, data=None, min_prominence=0.1, noise_floor=0.05, min_height=None,
                       peak_min_distance=3, max_shift=5, max_gap=1, min_duration=2,
                       return_tracks=False):
        """
        Detect patches as local intensity "mountains" in space, tracked across
        time -- an alternative to `detect_patches()` for when a single absolute
        intensity cutoff can't win both ways: raised to split two genuinely
        separate patches that sit close together (so the valley between them
        doesn't stay connected), it also buries a real but weaker patch
        elsewhere; lowered to keep that weak patch, it merges the close pair
        into one long blob.

        Each frame is searched independently with `find_patch_peaks()` for
        local maxima along the circular contour, judged by *relative*
        prominence against their own neighbourhood -- not a global level -- so
        a weak real patch is found on its own terms. Peaks are then linked
        frame-to-frame into tracks (nearest-neighbour matching, circular-aware,
        solved with `scipy.optimize.linear_sum_assignment` since several peaks
        can be present at once): a peak extends an existing track if it is
        within `max_shift` contour points of that track's last position,
        which is what lets one patch be followed through a bit of drift
        instead of being cut into pieces every frame it moves. A track that
        finds no matching peak for more than `max_gap` consecutive frames is
        closed -- including when a patch genuinely fades out ("goes back to
        black") in the middle and a new one starts later at the same spot;
        set `min_height` (forwarded to `find_patch_peaks`, e.g.
        `2 * self.noise_scale` to scale with this cell's own noise level) if
        a dip should count as "gone" even while technically still the
        strongest point in its own frame, which `min_prominence`/`noise_floor`
        alone can't express since they're relative to each frame individually.

        Parameters:
        - data: time x position array to detect on. Defaults to `self.normalize`.
        - min_prominence, noise_floor, min_height: forwarded to `find_patch_peaks()`
          for the per-frame spatial peak detection -- see its docstring.
        - peak_min_distance: minimum spacing (contour points) between two
          peaks in the same frame for them to count as separate mountains.
        - max_shift: maximum per-frame circular drift (contour points)
          allowed when linking a peak to an existing track.
        - max_gap: maximum consecutive frames a track may go undetected and
          still resume on reappearance, rather than starting a new track.
        - min_duration: drop tracks detected in fewer than this many frames.
        - return_tracks: if True, also return the raw per-detection tracks
          (each track's own list of times/positions/intensities, not just the
          summary row) -- useful for plotting an actual trajectory rather
          than just a start/end point.

        Returns:
        - pandas.DataFrame, one row per track, with columns:
          - track_id: arbitrary id, not stable across calls
          - time_start, time_end, duration: first/last frame detected and the
            elapsed span (duration can exceed n_detections across a `max_gap`)
          - n_detections: number of frames a peak was actually found
          - position_start, position_end: first/last detected position
          - path_length: total circular drift summed step-to-step (how much
            it wandered, as opposed to net displacement)
          - peak_intensity, peak_time, peak_position: the track's maximum and
            where it occurs
          - mean_intensity, integrated_intensity: average / summed intensity
            over the track's detections

        - list of dict (only when `return_tracks=True`, as `(df, tracks)`):
          one dict per *kept* track (same filter/order as the DataFrame rows),
          each with "track_id", "times", "positions", "intensities" (lists,
          one entry per detection).

        Example:
        >>> patch = DynamicPatch(data, background)
        >>> tracks_df = patch.track_patches(min_height=2 * patch.noise_scale)
        >>> tracks_df[["duration", "path_length", "peak_intensity"]]
        """
        data = self.normalize if data is None else np.asarray(data)
        n_time, n_pos = data.shape

        def circ_dist(a, b):
            d = abs(a - b)
            return min(d, n_pos - d)

        frame_peaks = [find_patch_peaks(data[t], min_prominence=min_prominence,
                                         noise_floor=noise_floor, min_height=min_height,
                                         min_distance=peak_min_distance)
                       for t in range(n_time)]

        active, finished = [], []
        next_track_id = 1
        for t in range(n_time):
            positions, intensities = frame_peaks[t]
            n_peaks, n_active = len(positions), len(active)
            matched_tracks, matched_peaks = set(), set()

            if n_active and n_peaks:
                cost = np.full((n_active, n_peaks), max_shift + 1.0)
                for i, trk in enumerate(active):
                    for j, pos in enumerate(positions):
                        d = circ_dist(trk["last_position"], pos)
                        if d <= max_shift:
                            cost[i, j] = d
                row_idx, col_idx = linear_sum_assignment(cost)
                for i, j in zip(row_idx, col_idx):
                    if cost[i, j] <= max_shift:
                        trk = active[i]
                        trk["times"].append(t)
                        trk["positions"].append(int(positions[j]))
                        trk["intensities"].append(float(intensities[j]))
                        trk["last_position"] = int(positions[j])
                        trk["last_time"] = t
                        matched_tracks.add(i)
                        matched_peaks.add(j)

            still_active = []
            for i, trk in enumerate(active):
                if i in matched_tracks or (t - trk["last_time"] <= max_gap):
                    still_active.append(trk)
                else:
                    finished.append(trk)
            active = still_active

            for j in range(n_peaks):
                if j not in matched_peaks:
                    active.append({
                        "track_id": next_track_id,
                        "times": [t],
                        "positions": [int(positions[j])],
                        "intensities": [float(intensities[j])],
                        "last_position": int(positions[j]),
                        "last_time": t,
                    })
                    next_track_id += 1
        finished.extend(active)

        records = []
        kept_tracks = []
        for trk in finished:
            if len(trk["times"]) < min_duration:
                continue
            times = np.array(trk["times"])
            positions_arr = np.array(trk["positions"])
            intensities_arr = np.array(trk["intensities"])
            peak_local = int(np.argmax(intensities_arr))
            path_length = sum(circ_dist(positions_arr[k], positions_arr[k + 1])
                               for k in range(len(positions_arr) - 1))

            records.append({
                "track_id": trk["track_id"],
                "time_start": int(times.min()),
                "time_end": int(times.max()),
                "duration": int(times.max() - times.min() + 1),
                "n_detections": int(len(times)),
                "position_start": int(positions_arr[0]),
                "position_end": int(positions_arr[-1]),
                "path_length": float(path_length),
                "peak_intensity": float(intensities_arr[peak_local]),
                "peak_time": int(times[peak_local]),
                "peak_position": int(positions_arr[peak_local]),
                "mean_intensity": float(intensities_arr.mean()),
                "integrated_intensity": float(intensities_arr.sum()),
            })
            kept_tracks.append(trk)
        patches_df = pd.DataFrame.from_records(records)
        return (patches_df, kept_tracks) if return_tracks else patches_df

    def track_wide_patches(self, data=None, frac=0.3, min_iou=0.3, max_gap=0, min_duration=2,
                            return_tracks=False):
        """
        Detect patches as a wide, roughly contiguous high-value region along the
        circular contour, tracked frame by frame across time -- for the shape
        `detect_patches()`/`track_patches()` aren't aimed at: a region that's
        wide in *space* but short in *time* (a handful of frames, rarely more
        than ~20), rather than a narrow point-like peak that may drift slowly
        over many frames.

        Each frame is segmented independently: the region active in that frame
        is whatever exceeds `frac` of THAT FRAME's own peak (so a dim frame and
        a bright frame are each judged on their own scale, not one shared
        absolute level), taken as one or more contiguous circular runs via
        `circular_intervals()`. Frames are then linked frame-to-frame by
        spatial overlap: a run in frame t+1 extends a track from frame t only
        if their intersection-over-union is at least `min_iou` (solved as an
        assignment problem with `scipy.optimize.linear_sum_assignment` when
        several runs are present at once) -- requiring *overlap proportional
        to size*, unlike plain space-time connectivity (`detect_patches()`,
        even on a `binarized_mask`), which merges on any touch at all and kept
        producing long spurious merges on this data. `max_gap=0` by default
        also requires a continuing track to be detected in literally every
        frame; allowing a gap lets runs that aren't really the same event
        rejoin through a quiet frame, which reintroduces those long merges.

        Parameters:
        - data: time x position array to detect on. Defaults to `self.normalize`.
        - frac: a position is "active" in a frame when its value exceeds `frac`
          times that frame's own maximum (a frame with no signal contributes no
          runs at all).
        - min_iou: minimum intersection-over-union between a track's last run
          and a candidate run in the next frame for them to count as the same
          patch continuing.
        - max_gap: maximum consecutive frames a track may go undetected and
          still resume on reappearance. 0 (the default) requires unbroken
          detection every frame.
        - min_duration: drop tracks detected in fewer than this many frames.
        - return_tracks: if True, also return the raw per-frame runs behind
          each kept track (times + position sets), e.g. for plotting.

        Returns:
        - pandas.DataFrame, one row per track, with columns:
          - track_id, time_start, time_end, duration, n_detections (as in
            `track_patches()`)
          - width_mean, width_max: mean / max run width (number of contour
            positions) over the track's detections
          - peak_intensity, peak_time: the track's maximum value and when it
            occurs
          - mean_intensity: average of each frame's own run-maximum over the
            track
        - list of dict (only when `return_tracks=True`, as `(df, tracks)`):
          one dict per kept track, each with "track_id", "times",
          "intervals" ((start, end) per detection -- start > end means that
          detection's run wraps through position 0) and "position_sets" (the
          same run, expanded to a set of contour positions).

        Example:
        >>> patch = DynamicPatch(data, background)
        >>> wide_df = patch.track_wide_patches()
        >>> wide_df[["duration", "width_max", "peak_intensity"]]
        """
        data = self.normalize if data is None else np.asarray(data)
        n_time, n_pos = data.shape
        row_max = data.max(axis=1)

        frame_runs = []
        for t in range(n_time):
            if row_max[t] <= 0:
                frame_runs.append([])
                continue
            active_row = data[t] > frac * row_max[t]
            frame_runs.append(circular_intervals(active_row))

        def run_positions(run):
            start, end = run
            return set(recurrent_index(start, end + 1, n_pos).tolist())

        active, finished = [], []
        next_track_id = 1
        for t in range(n_time):
            runs = frame_runs[t]
            pos_sets = [run_positions(r) for r in runs]
            n_runs, n_active = len(runs), len(active)
            matched_tracks, matched_runs = set(), set()

            if n_active and n_runs:
                cost = np.ones((n_active, n_runs))
                for i, trk in enumerate(active):
                    last_set = trk["position_sets"][-1]
                    for j, pos_set in enumerate(pos_sets):
                        union = len(last_set | pos_set)
                        iou = len(last_set & pos_set) / union if union else 0.0
                        cost[i, j] = 1.0 - iou
                row_idx, col_idx = linear_sum_assignment(cost)
                for i, j in zip(row_idx, col_idx):
                    iou = 1.0 - cost[i, j]
                    if iou >= min_iou:
                        trk = active[i]
                        trk["times"].append(t)
                        trk["intervals"].append(runs[j])
                        trk["position_sets"].append(pos_sets[j])
                        trk["values"].append(float(data[t, list(pos_sets[j])].max()))
                        trk["last_time"] = t
                        matched_tracks.add(i)
                        matched_runs.add(j)

            still_active = []
            for i, trk in enumerate(active):
                if i in matched_tracks or (t - trk["last_time"] <= max_gap):
                    still_active.append(trk)
                else:
                    finished.append(trk)
            active = still_active

            for j in range(n_runs):
                if j not in matched_runs:
                    active.append({
                        "track_id": next_track_id,
                        "times": [t],
                        "intervals": [runs[j]],
                        "position_sets": [pos_sets[j]],
                        "values": [float(data[t, list(pos_sets[j])].max())],
                        "last_time": t,
                    })
                    next_track_id += 1
        finished.extend(active)

        records = []
        kept_tracks = []
        for trk in finished:
            if len(trk["times"]) < min_duration:
                continue
            times = np.array(trk["times"])
            widths = np.array([len(s) for s in trk["position_sets"]])
            values = np.array(trk["values"])
            peak_local = int(np.argmax(values))

            records.append({
                "track_id": trk["track_id"],
                "time_start": int(times.min()),
                "time_end": int(times.max()),
                "duration": int(times.max() - times.min() + 1),
                "n_detections": int(len(times)),
                "width_mean": float(widths.mean()),
                "width_max": int(widths.max()),
                "peak_intensity": float(values[peak_local]),
                "peak_time": int(times[peak_local]),
                "mean_intensity": float(values.mean()),
            })
            kept_tracks.append(trk)
        patches_df = pd.DataFrame.from_records(records)
        return (patches_df, kept_tracks) if return_tracks else patches_df

    def detect_patches_hysteresis(self, data=None, low_threshold=0.25, high_threshold=0.50,
                                   sigma_time=0, sigma_space=1.0, min_duration=2, max_duration=19,
                                   min_width=3, max_width=None, min_area=6, connectivity=4,
                                   time_interval=1.0, space_interval=1.0, return_label_map=False):
        """
        Detect patches with hysteresis thresholding: a patch's extent is traced
        at the permissive `low_threshold`, but it's only kept if some pixel in
        it also clears the stricter `high_threshold` (a "strong seed") --
        so a weak-but-real patch isn't lost just because a cutoff tuned to
        separate two strong patches elsewhere would have missed it, without
        `detect_patches()`'s single cutoff having to do both jobs at once.
        Smooths with a Gaussian first (`sigma_time`/`sigma_space`, wrapping
        across the circular space axis) to reduce single-pixel noise blips.

        Connectivity across the circular seam (position 0 / position -1) is
        handled exactly, by union-find over the two boundary columns
        (`periodic_label()`), rather than `detect_patches()`'s roll-to-an-
        empty-column trick -- so there's no edge case to fall back on when a
        cell's signal happens to touch every position at some point.

        Patches are filtered by shape as well as amplitude: `max_duration`
        rejects (not splits) anything that still ends up spanning more than
        that many frames -- the blunt but explicit way to enforce "these are
        supposed to be short-lived, wide regions" when hysteresis alone
        doesn't prevent every long merge. `min_width`/`max_width` filter by a
        patch's own largest single-frame width, i.e. how wide it looks at its
        widest, not how wide its time-collapsed bounding box is.

        Parameters: see `hysteresis_patch_detect()` (module-level function
        this wraps) for the full parameter and output-column documentation;
        `data` defaults to `self.normalize` here instead of being required.
        `return_label_map`: if True, also return the instance-segmentation
        map (same shape as `data`, 0 = background, else patch id).

        Example:
        >>> patch = DynamicPatch(data, background)
        >>> patches_df = patch.detect_patches_hysteresis()
        >>> patches_df[["duration_frames", "max_width", "peak_intensity"]]
        """
        data = self.normalize if data is None else np.asarray(data)
        patches_df, label_map = hysteresis_patch_detect(
            data, low_threshold=low_threshold, high_threshold=high_threshold,
            sigma_time=sigma_time, sigma_space=sigma_space, min_duration=min_duration,
            max_duration=max_duration, min_width=min_width, max_width=max_width,
            min_area=min_area, connectivity=connectivity, time_interval=time_interval,
            space_interval=space_interval)
        return (patches_df, label_map) if return_label_map else patches_df

    def detect_single_patches(self, data=None, sigma_space=1.5, min_prominence=0.15,
                               noise_floor=0.05, min_height=None, peak_min_distance=5,
                               max_peaks_per_frame=3, width_rel_height=0.5, min_iou=0.2,
                               max_gap=0, min_duration=2, max_duration=None,
                               min_width=None, max_width=None, return_label_map=False):
        """
        Detect single patches as local "mountains" in space that are tracked
        across a few frames, following the definition: one patch is one
        connected high-value region in space (roughly 15 contour points wide)
        that stays on for a short run of frames (typically 2-5, rarely up to
        ~30) -- as opposed to `detect_patches()`/`detect_patches_hysteresis()`,
        which threshold the whole time x position map at once and then take
        connected components, with no notion of "at most a handful of patches
        active in any one frame" or "a gap means a new patch, not a
        continuation."

        The steps, in order:

        1. Smooth each frame's `normalize`d profile along the circular contour
           (`sigma_space`, Gaussian, wrapping at the position-0 seam) to reduce
           single-point noise before peak-finding.
        2. Find positive peaks independently in each smoothed frame
           (`find_patch_peaks_width()`, the same relative-prominence /
           `noise_floor` logic as `find_patch_peaks()` used by
           `track_patches()`, plus each peak's own spatial footprint -- the
           contiguous span around it still above `width_rel_height` of its
           prominence). A frame usually has one peak, sometimes two, rarely
           three; `max_peaks_per_frame` enforces that cap by keeping only the
           tallest ones. A peak too weak to clear `min_prominence`/
           `noise_floor`/`min_height` isn't reported at all.
        3. Link each frame's peaks to the previous frame's open tracks by
           footprint overlap: a peak extends a track only if its footprint's
           IoU with that track's last footprint is at least `min_iou` (solved
           as an assignment problem via `scipy.optimize.linear_sum_assignment`
           when several peaks/tracks are live at once, same scheme
           `track_wide_patches()` uses) -- i.e. peaks are "merged" into one
           patch across frames only when their positions actually overlap,
           not just when they're nearby.
        4. `max_gap` (0 by default) is how many consecutive frames a track may
           go undetected and still resume as the *same* patch. At 0, a track
           that finds no overlapping peak in the very next frame is closed
           immediately -- a patch that switches off and later back on at the
           same position is always two patches, never one, which is the
           behaviour `track_patches()`/`track_wide_patches()` only get via a
           caller-supplied `min_height`/`max_gap`.
        5. Each closed track is summarized into one row (spatial/temporal
           extent + intensity features, see Returns below); pass
           `return_label_map=True` to also get an instance-segmentation map
           (same shape as `data`, 0 = background, else each kept patch's own
           id) built directly from each detection's footprint.
        6. Every tunable above is a keyword argument, so detection can be
           re-run with different settings and compared.

        Parameters:
        - data: time x position array to detect on. Defaults to `self.normalize`.
        - sigma_space: Gaussian smoothing sigma (contour points) applied to
          each frame before peak-finding, wrapping across the circular seam.
          0 disables smoothing.
        - min_prominence, noise_floor, peak_min_distance: forwarded to
          `find_patch_peaks_width()` for the per-frame peak detection -- see
          its docstring. These are the only bar a single frame must clear to
          register as part of a track.
        - min_height: hysteresis-style "strong seed" requirement, applied once
          per finished track rather than to every frame (None disables it):
          a track is kept only if its own `peak_intensity` clears this
          absolute floor (same units as `data`), even though some of its
          frames -- typically its rise and decay -- may sit below it. This is
          what lets a patch's weak shoulder frames link up with its strong
          peak into one multi-frame patch instead of each shoulder frame
          going undetected on its own and leaving only a lone, too-short
          peak frame for `min_duration` to then discard. A fixed number here
          doesn't necessarily transfer to a different cell/session of
          different absolute brightness -- consider `k * self.noise_scale`
          (e.g. `k=2`) instead of a hand-picked constant.
        - max_peaks_per_frame: keep at most this many peaks per frame (the
          tallest ones), matching "never more than ~3 patches active at once."
        - width_rel_height: the fraction of each peak's own prominence used to
          define its spatial footprint (`scipy.signal.peak_widths`'
          `rel_height`) -- the span used both for the reported width and for
          frame-to-frame overlap matching.
        - min_iou: minimum intersection-over-union between a track's last
          footprint and a candidate peak's footprint in the next frame for
          them to count as the same patch continuing.
        - max_gap: maximum consecutive undetected frames before a track is
          closed rather than resumed. 0 (the default) means any gap starts a
          new patch -- see step 4 above.
        - min_duration: drop patches detected in fewer than this many frames.
        - max_duration: drop patches spanning more than this many frames
          (`time_end - time_start + 1`), e.g. to enforce "these are supposed
          to be short-lived" when something unexpectedly long slips through.
          None (default) applies no cap.
        - min_width, max_width: drop patches whose widest single-frame
          footprint falls outside this range (contour points). None (default)
          applies no bound.
        - return_label_map: if True, also return the instance-segmentation map.

        Returns:
        - pandas.DataFrame, one row per detected patch, with columns:
          - patch_id: 1-based id (arbitrary, not stable across calls)
          - time_start, time_end, duration: first/last frame and frame span
          - n_detections: number of frames a peak was actually linked into
            this patch (== duration unless a gap-tolerant max_gap > 0 was used)
          - position_start, position_end: first/last detected peak position
          - position_center, time_center: intensity-weighted centroid
            (position_center is circular-aware)
          - spatial_width_mean, spatial_width_max: mean / max footprint width
            over the patch's detections
          - peak_intensity, peak_time, peak_position: the patch's maximum
            value and where/when it occurs
          - mean_intensity, integrated_intensity: average / summed intensity
            over the patch's detections
        - numpy.ndarray (only when `return_label_map=True`, as `(df, label_map)`):
          time x position instance-segmentation map, 0 = background.

        Example:
        >>> patch = DynamicPatch(data, background)
        >>> patches_df = patch.detect_single_patches()
        >>> patches_df[["duration", "spatial_width_max", "peak_intensity"]]
        """
        raw = self.normalize if data is None else np.asarray(data)
        n_time, n_pos = raw.shape
        smoothed = (gaussian_filter1d(raw, sigma=sigma_space, axis=1, mode="wrap")
                    if sigma_space > 0 else raw)

        def footprint_positions(footprint):
            lo, hi = footprint
            lo_i, hi_i = int(np.floor(lo)) % n_pos, int(np.ceil(hi)) % n_pos
            return set(recurrent_index(lo_i, hi_i + 1, n_pos).tolist())

        frame_peaks = []
        for t in range(n_time):
            # min_height is *not* forwarded here: it's applied once per
            # finished track below (the patch's peak_intensity must clear
            # it), not to every individual frame. A patch's early rise and
            # late decay are often still below min_height while clearly the
            # same event as the strong frame next to them -- requiring every
            # frame to independently clear an absolute floor cuts exactly
            # those shoulder frames off from their peak, leaving a lone
            # one-frame detection that min_duration then discards (the patch
            # never gets detected at all). Hysteresis-style instead: a frame
            # only needs to clear the relative prominence/noise_floor bar to
            # register as part of a track; whether the track as a whole is
            # real is judged once, after linking, by its strongest frame.
            positions, intensities, footprints = find_patch_peaks_width(
                smoothed[t], min_prominence=min_prominence, noise_floor=noise_floor,
                min_height=None, min_distance=peak_min_distance,
                max_peaks=max_peaks_per_frame, rel_height=width_rel_height)
            pos_sets = [footprint_positions(fp) for fp in footprints]
            frame_peaks.append((positions, intensities, footprints, pos_sets))

        active, finished = [], []
        next_track_id = 1
        for t in range(n_time):
            positions, intensities, footprints, pos_sets = frame_peaks[t]
            n_peaks, n_active = len(positions), len(active)
            matched_tracks, matched_peaks = set(), set()

            if n_active and n_peaks:
                cost = np.ones((n_active, n_peaks))
                for i, trk in enumerate(active):
                    last_set = trk["position_sets"][-1]
                    for j, pos_set in enumerate(pos_sets):
                        union = len(last_set | pos_set)
                        iou = len(last_set & pos_set) / union if union else 0.0
                        cost[i, j] = 1.0 - iou
                row_idx, col_idx = linear_sum_assignment(cost)
                for i, j in zip(row_idx, col_idx):
                    iou = 1.0 - cost[i, j]
                    if iou >= min_iou:
                        trk = active[i]
                        trk["times"].append(t)
                        trk["positions"].append(int(positions[j]))
                        trk["intensities"].append(float(intensities[j]))
                        trk["position_sets"].append(pos_sets[j])
                        trk["last_time"] = t
                        matched_tracks.add(i)
                        matched_peaks.add(j)

            still_active = []
            for i, trk in enumerate(active):
                if i in matched_tracks or (t - trk["last_time"] <= max_gap):
                    still_active.append(trk)
                else:
                    finished.append(trk)
            active = still_active

            for j in range(n_peaks):
                if j not in matched_peaks:
                    active.append({
                        "track_id": next_track_id,
                        "times": [t],
                        "positions": [int(positions[j])],
                        "intensities": [float(intensities[j])],
                        "position_sets": [pos_sets[j]],
                        "last_time": t,
                    })
                    next_track_id += 1
        finished.extend(active)

        records = []
        patch_id = 0
        label_map = np.zeros((n_time, n_pos), dtype=np.int32) if return_label_map else None
        for trk in finished:
            if len(trk["times"]) < min_duration:
                continue
            times = np.array(trk["times"])
            positions_arr = np.array(trk["positions"])
            intensities_arr = np.array(trk["intensities"])
            widths = np.array([len(s) for s in trk["position_sets"]])
            duration = int(times.max() - times.min() + 1)

            if max_duration is not None and duration > max_duration:
                continue
            if min_width is not None and widths.max() < min_width:
                continue
            if max_width is not None and widths.max() > max_width:
                continue
            if min_height is not None and intensities_arr.max() < min_height:
                continue

            patch_id += 1
            peak_local = int(np.argmax(intensities_arr))
            position_center = circular_center(positions_arr, intensities_arr, n_pos)
            time_center = float(np.average(times, weights=intensities_arr))

            if label_map is not None:
                for t_i, pos_set in zip(trk["times"], trk["position_sets"]):
                    for p in pos_set:
                        label_map[t_i, p] = patch_id

            records.append({
                "patch_id": patch_id,
                "time_start": int(times.min()),
                "time_end": int(times.max()),
                "duration": duration,
                "n_detections": int(len(times)),
                "position_start": int(positions_arr[0]),
                "position_end": int(positions_arr[-1]),
                "position_center": position_center,
                "time_center": time_center,
                "spatial_width_mean": float(widths.mean()),
                "spatial_width_max": int(widths.max()),
                "peak_intensity": float(intensities_arr[peak_local]),
                "peak_time": int(times[peak_local]),
                "peak_position": int(positions_arr[peak_local]),
                "mean_intensity": float(intensities_arr.mean()),
                "integrated_intensity": float(intensities_arr.sum()),
            })

        patches_df = pd.DataFrame.from_records(records)
        return (patches_df, label_map) if return_label_map else patches_df

    def plot_patches(self, label_map, ax=None, edgecolor="cyan", title=None, cmap="magma"):
        """Draw `self.normalize` as a time x position heatmap with each patch
        in `label_map` outlined as a (wrap-aware) box. `label_map` is any
        instance-segmentation map matching `self.data`'s shape -- typically
        from `detect_single_patches(return_label_map=True)`, but
        `detect_patches_hysteresis(return_label_map=True)`'s works too.

        Parameters:
        - label_map: time x position int array, 0 = background, else each
          patch's own id (see `label_map_boxes`).
        - ax: matplotlib Axes to draw on; a new figure+axes if None.
        - edgecolor, title, cmap: passed to the box outlines / heatmap / title.

        Returns the Axes drawn on.

        Example:
        >>> patches_df, label_map = patch.detect_single_patches(return_label_map=True)
        >>> patch.plot_patches(label_map, title=f"{len(patches_df)} patches")
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(9, 4))
        n_pos = self.data.shape[1]
        ax.imshow(self.normalize.T, aspect="auto", cmap=cmap, origin="lower")
        box_kwargs = {"fill": False, "edgecolor": edgecolor, "linewidth": 1}
        for b in label_map_boxes(label_map, n_pos):
            t0, dur, p0, p1 = b["time_start"], b["duration"], b["position_start"], b["position_end"]
            if p0 <= p1:
                ax.add_patch(plt.Rectangle((t0 - 0.5, p0 - 0.5), dur, p1 - p0 + 1, **box_kwargs))
            else:
                ax.add_patch(plt.Rectangle((t0 - 0.5, p0 - 0.5), dur, n_pos - p0, **box_kwargs))
                ax.add_patch(plt.Rectangle((t0 - 0.5, -0.5), dur, p1 + 1, **box_kwargs))
        ax.set_xlabel("Time (frame)")
        ax.set_ylabel("Aligned contour position")
        if title:
            ax.set_title(title)
        return ax

    def plot_label_map(self, label_map, ax=None):
        """Instance-segmentation view of `label_map` (0 = background, each
        patch id its own flat color) -- useful to confirm two patches close
        in time/space really do get different ids, and that a single
        patch's footprint looks like one coherent blob rather than
        fragments. Returns the Axes drawn on (a new figure+axes if None)."""
        import matplotlib.pyplot as plt
        from matplotlib import colors as mcolors

        if ax is None:
            _, ax = plt.subplots(figsize=(9, 4))
        n_ids = int(label_map.max())
        cmap = mcolors.ListedColormap(["black"] + list(plt.cm.tab20(np.linspace(0, 1, max(n_ids, 1)))))
        ax.imshow(label_map.T, aspect="auto", cmap=cmap, origin="lower", interpolation="nearest",
                  vmin=-0.5, vmax=max(n_ids, 1) + 0.5)
        ax.set_xlabel("Time (frame)")
        ax.set_ylabel("Aligned contour position")
        ax.set_title(f"instance label map -- {n_ids} patches")
        return ax

    def plot_frame(self, label_map, patches_df, frame, ax=None, sigma_space=1.5):
        """Single-frame inspector: that one frame's raw vs. smoothed profile,
        with whichever patch(es) are present at it (per `label_map`) shaded
        by patch id -- a quick way to confirm a patch really is/isn't active
        at a given frame, reading the actual per-pixel detection rather than
        re-deriving it from `patches_df`'s own start/end summary.

        Parameters:
        - label_map, patches_df: as returned by
          `detect_single_patches(return_label_map=True)`.
        - frame: which frame (row) to inspect.
        - ax: matplotlib Axes to draw on; a new figure+axes if None.
        - sigma_space: smoothing sigma to overlay (should match whatever was
          used for detection, so the overlay reflects what detection saw).

        Returns the Axes drawn on.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 3.5))
        row_raw = self.normalize[frame]
        row_smooth = gaussian_filter1d(row_raw, sigma=sigma_space, mode="wrap") if sigma_space > 0 else row_raw
        frame_labels = label_map[frame]
        active_ids = [pid for pid in np.unique(frame_labels) if pid != 0]

        ax.plot(row_raw, color="0.7", label="normalize (raw)")
        ax.plot(row_smooth, color="crimson", linewidth=1.5, label="smoothed")

        cmap = plt.cm.tab20(np.linspace(0, 1, 20))
        for pid in active_ids:
            positions = np.where(frame_labels == pid)[0]
            color = cmap[(pid - 1) % 20]
            for p in positions:
                ax.axvspan(p - 0.5, p + 0.5, color=color, alpha=0.3, linewidth=0)
            row = patches_df[patches_df.patch_id == pid]
            span = f" (t{int(row.time_start.iloc[0])}-{int(row.time_end.iloc[0])})" if len(row) else ""
            ax.plot([], [], color=color, linewidth=6, alpha=0.4, label=f"patch {pid}{span}")

        ax.set_xlabel("Aligned contour position")
        ax.set_ylabel("Intensity")
        n_active = len(active_ids)
        status = "no patch detected" if n_active == 0 else f"{n_active} patch(es) detected"
        ax.set_title(f"frame {frame} -- {status}")
        ax.legend(fontsize=8, loc="upper right")
        return ax


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


def circular_intervals(active_row):
    """
    Contiguous runs of True in a circular boolean array, as (start, end) pairs
    (both inclusive); start > end means a run that wraps through index 0. A
    fully-True row is one run spanning the whole array; a fully-False row has
    none.
    """
    n = len(active_row)
    if active_row.all():
        return [(0, n - 1)]
    if not active_row.any():
        return []
    false_idx = int(np.where(~active_row)[0][0])
    rolled = np.roll(active_row, -false_idx)
    padded = np.concatenate([[False], rolled, [False]])
    diffs = np.diff(padded.astype(int))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0] - 1
    return [(int((s + false_idx) % n), int((e + false_idx) % n)) for s, e in zip(starts, ends)]


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


def find_patch_peaks(signal, min_prominence=0.1, noise_floor=0.05, min_height=None, min_distance=1):
    """
    Detect patches on a smoothed, closed (circular) 1D profile, e.g. a patch
    profile along the aligned contour. Every local high point is a peak unless
    it is filtered out by the options below. The signal is not smoothed here.

    Parameters:
    - signal (array-like): smoothed 1D profile; the last point neighbours the first.
    - min_prominence (float): minimum prominence of a peak, as a fraction of its
      own height above the signal's minimum. Removes small ripples on the slope
      of a larger peak while keeping weak but separate patches. 0 keeps every
      local maximum.
    - noise_floor (float): a peak must rise above the signal's minimum by at
      least this fraction of the signal's range, which drops near-zero ripples
      where there is no patch. It is a floor only; weak patches above it are kept.
    - min_height (float or None): minimum peak value, in the signal's units.
    - min_distance (int): minimum spacing between peaks (in points).

    Returns:
    - positions (ndarray of int): peak positions, in increasing order.
    - intensities (ndarray): signal value at each peak.
    """
    signal = np.asarray(signal, dtype=float)
    n = len(signal)
    floor = signal.min()
    value_range = signal.max() - floor
    if n < 3 or value_range == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    # tile three times and keep the middle copy, so peaks at the ends and
    # prominences across the wrap point are handled correctly
    tiled = np.concatenate([signal, signal, signal])
    peaks, props = find_peaks(tiled, prominence=0, distance=max(1, int(min_distance)))
    height = tiled[peaks] - floor
    keep = (props["prominences"] >= min_prominence * height) & (height >= noise_floor * value_range)
    if min_height is not None:
        keep &= tiled[peaks] >= min_height
    peaks = peaks[keep]
    positions = np.unique(peaks[(peaks >= n) & (peaks < 2 * n)] - n)
    return positions, signal[positions]


def find_patch_peaks_width(signal, min_prominence=0.1, noise_floor=0.05, min_height=None,
                            min_distance=1, max_peaks=None, rel_height=0.5):
    """
    Like `find_patch_peaks()`, but also returns each kept peak's spatial
    footprint -- the contiguous span around it still above `rel_height` of
    its own prominence (`scipy.signal.peak_widths`), used by
    `DynamicPatch.detect_single_patches()` to link peaks across frames by
    overlap rather than by raw distance. Computed on the same signal-tiled-
    three-times trick `find_patch_peaks()` uses, so a footprint that crosses
    the position-0 seam comes out correct, same as the peak position itself.

    Parameters: see `find_patch_peaks()`, plus:
    - max_peaks (int or None): keep at most this many peaks (the tallest),
      e.g. to enforce "never more than ~3 patches active in one frame."
    - rel_height (float): fraction of each peak's own prominence used to
      define its footprint -- 0.5 (the default) is each peak's own
      half-prominence width, 1.0 reaches down to its prominence base.

    Returns:
    - positions (ndarray of int): peak positions, in increasing order.
    - intensities (ndarray): signal value at each peak.
    - footprints (list of (float, float)): each peak's (left, right) span, in
      the same order as `positions`; right < left means the span wraps
      through position 0.
    """
    signal = np.asarray(signal, dtype=float)
    n = len(signal)
    floor = signal.min()
    value_range = signal.max() - floor
    if n < 3 or value_range == 0:
        return np.array([], dtype=int), np.array([], dtype=float), []

    tiled = np.concatenate([signal, signal, signal])
    peaks, props = find_peaks(tiled, prominence=0, distance=max(1, int(min_distance)))
    height = tiled[peaks] - floor
    keep = (props["prominences"] >= min_prominence * height) & (height >= noise_floor * value_range)
    if min_height is not None:
        keep &= tiled[peaks] >= min_height

    peaks = peaks[keep]
    prominences = props["prominences"][keep]
    left_bases = props["left_bases"][keep]
    right_bases = props["right_bases"][keep]

    # Only peaks whose position falls in the middle copy are kept (to avoid
    # triple-counting), but widths are computed on the tiled signal (correct
    # across the wrap) before mapping back to real coordinates.
    middle = (peaks >= n) & (peaks < 2 * n)
    peaks, prominences = peaks[middle], prominences[middle]
    left_bases, right_bases = left_bases[middle], right_bases[middle]

    if len(peaks) == 0:
        return np.array([], dtype=int), np.array([], dtype=float), []

    if max_peaks is not None and len(peaks) > max_peaks:
        keep_idx = np.argsort(tiled[peaks])[::-1][:max_peaks]
        peaks, prominences = peaks[keep_idx], prominences[keep_idx]
        left_bases, right_bases = left_bases[keep_idx], right_bases[keep_idx]

    order = np.argsort(peaks)
    peaks, prominences = peaks[order], prominences[order]
    left_bases, right_bases = left_bases[order], right_bases[order]

    widths, width_heights, left_ips, right_ips = peak_widths(
        tiled, peaks, rel_height=rel_height,
        prominence_data=(prominences, left_bases, right_bases))

    positions = (peaks - n) % n
    intensities = tiled[peaks]
    footprints = [(float((lo - n) % n), float((hi - n) % n)) for lo, hi in zip(left_ips, right_ips)]
    return positions, intensities, footprints


def circular_bbox(positions, n_pos):
    """
    Smallest circular bounding span covering the given position indices.
    Returns (start, end, width) in real coordinates; start > end means the
    span wraps through position 0.
    """
    positions = np.unique(positions)
    if len(positions) == n_pos:
        return 0, n_pos - 1, n_pos
    present = np.zeros(n_pos, dtype=bool)
    present[positions] = True
    roll_index = int(np.where(~present)[0][0])
    rolled = np.sort((positions - roll_index) % n_pos)
    start = int((rolled[0] + roll_index) % n_pos)
    end = int((rolled[-1] + roll_index) % n_pos)
    width = int(rolled[-1] - rolled[0] + 1)
    return start, end, width


def label_map_boxes(label_map, n_pos):
    """
    One box per id in an instance-segmentation `label_map` (0 = background),
    from each id's own pixels -- not a summary DataFrame's single start/end
    position, which can understate the footprint if the patch drifts. Works
    for any method's label map (`detect_single_patches`'s or
    `detect_patches_hysteresis`'s). Used by `DynamicPatch.plot_patches`.

    Returns a list of dict, each {"time_start", "duration", "position_start",
    "position_end"} (`draw_patch_boxes`-style; `position_start > position_end`
    means that patch wraps through position 0).
    """
    boxes = []
    for pid in np.unique(label_map):
        if pid == 0:
            continue
        t_idx, pos_idx = np.where(label_map == pid)
        p0, p1, _ = circular_bbox(pos_idx, n_pos)
        boxes.append({"time_start": int(t_idx.min()), "duration": int(t_idx.max() - t_idx.min() + 1),
                       "position_start": p0, "position_end": p1})
    return boxes


def circular_width(positions, n_space):
    """
    Minimum circular interval containing all positions.
    Correctly handles patches crossing the spatial boundary.
    """
    positions = np.unique(positions)

    if len(positions) == 0:
        return 0

    gaps = np.diff(
        np.r_[positions, positions[0] + n_space]
    )

    return int(n_space - gaps.max() + 1)


def circular_center(positions, weights, n_space):
    """Intensity-weighted center on a circular spatial axis."""
    angles = positions * 2 * np.pi / n_space

    z = np.sum(weights * np.exp(1j * angles))

    if np.abs(z) < 1e-12:
        return np.nan

    center = np.angle(z) % (2 * np.pi)

    return float(center * n_space / (2 * np.pi)) % n_space


def periodic_label(mask, connectivity=4):
    """
    Connected-component labeling with periodic space.

    Time axis: non-periodic
    Space axis: periodic
    """
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")

    structure = ndimage.generate_binary_structure(
        2, 1 if connectivity == 4 else 2
    )

    labels, n_labels = ndimage.label(
        mask, structure=structure
    )

    parent = np.arange(n_labels + 1)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        if a > 0 and b > 0:
            parent[find(b)] = find(a)

    n_time, n_space = mask.shape

    # Connect the left and right spatial boundaries
    offsets = [0] if connectivity == 4 else [-1, 0, 1]

    for t in range(n_time):
        for dt in offsets:
            t2 = t + dt

            if 0 <= t2 < n_time:
                union(
                    labels[t, 0],
                    labels[t2, -1]
                )

    # Resolve merged labels
    lookup = np.array([
        find(i) for i in range(n_labels + 1)
    ])

    labels = lookup[labels]

    # Relabel consecutively
    _, inverse = np.unique(
        labels, return_inverse=True
    )

    # Preserve background as zero, even if no background exists
    if not np.any(labels == 0):
        inverse += 1

    return inverse.reshape(mask.shape)


def hysteresis_patch_detect(
    intensity_map,
    low_threshold=0.25,
    high_threshold=0.50,
    sigma_time=0,
    sigma_space=1.0,
    min_duration=2,
    max_duration=19,
    min_width=3,
    max_width=None,
    min_area=6,
    connectivity=4,
    time_interval=1.0,
    space_interval=1.0,
):
    """
    Detect dynamic patches in a periodic time-space map.

    Parameters
    ----------
    intensity_map : ndarray
        Normalized 2D intensity map (time, space).

    low_threshold : float
        Threshold defining the patch boundary.

    high_threshold : float
        Threshold defining reliable patch seeds.

    sigma_time, sigma_space : float
        Gaussian smoothing sigma along each axis.

    min_duration, max_duration : int
        Allowed patch duration in frames.

    min_width, max_width : int or None
        Allowed maximum instantaneous spatial width.

    min_area : int
        Minimum number of pixels in a patch.

    connectivity : int
        4: strict spatial and temporal connectivity.
        8: also allows diagonal connections.

    time_interval : float
        Time interval between frames.

    space_interval : float
        Physical distance between spatial positions.

    Returns
    -------
    patch_df : pandas.DataFrame
        Quantitative information for each patch.

    label_map : ndarray
        Instance segmentation map.
    """

    data = np.asarray(intensity_map, dtype=float)

    if data.ndim != 2 or min(data.shape) == 0:
        raise ValueError("Input must be a non-empty 2D array")

    if not np.all(np.isfinite(data)):
        raise ValueError("Input contains NaN or Inf")

    if low_threshold > high_threshold:
        raise ValueError(
            "low_threshold must be <= high_threshold"
        )

    n_time, n_space = data.shape

    # ==========================================
    # 1. Gaussian smoothing
    # ==========================================
    # Space is periodic, but time is not.

    smooth_map = ndimage.gaussian_filter(
        data,
        sigma=(sigma_time, sigma_space),
        mode=("nearest", "wrap")
    )

    # ==========================================
    # 2. Hysteresis thresholding
    # ==========================================

    low_mask = smooth_map >= low_threshold
    high_mask = smooth_map >= high_threshold

    # ==========================================
    # 3. Periodic connected components
    # ==========================================

    labels = periodic_label(
        low_mask,
        connectivity=connectivity
    )

    # Only retain regions containing a strong seed
    valid_labels = np.unique(labels[high_mask])
    valid_labels = valid_labels[valid_labels > 0]

    label_map = np.zeros(data.shape, dtype=np.int32)

    records = []
    patch_id = 0

    # ==========================================
    # 4. Analyze each patch
    # ==========================================

    for label_id in valid_labels:

        mask = labels == label_id

        t, x = np.where(mask)
        values = data[t, x]

        # ------------------------------
        # Temporal features
        # ------------------------------

        start_frame = int(t.min())
        end_frame = int(t.max())

        duration = end_frame - start_frame + 1

        # ------------------------------
        # Spatial features
        # ------------------------------

        # Total spatial coverage over the patch lifetime
        total_width = circular_width(x, n_space)

        # Instantaneous spatial width for each frame
        frames = np.unique(t)

        frame_widths = np.array([
            circular_width(x[t == frame], n_space)
            for frame in frames
        ])

        mean_width = frame_widths.mean()
        max_frame_width = frame_widths.max()

        area = len(values)

        # ------------------------------
        # Filter invalid regions
        # ------------------------------

        if duration < min_duration:
            continue

        if max_duration is not None:
            if duration > max_duration:
                continue

        if max_frame_width < min_width:
            continue

        if max_width is not None:
            if max_frame_width > max_width:
                continue

        if area < min_area:
            continue

        # ------------------------------
        # Intensity features
        # ------------------------------

        peak_idx = np.argmax(values)

        peak_frame = int(t[peak_idx])
        peak_position = int(x[peak_idx])

        peak_intensity = float(values.max())
        mean_intensity = float(values.mean())

        # Integrated intensity
        volume = float(values.sum())

        # Intensity above the detection threshold
        excess_volume = float(
            np.maximum(values - low_threshold, 0).sum()
        )

        # Intensity-weighted center
        weights = np.maximum(
            values - low_threshold, 0
        )

        if weights.sum() == 0:
            weights = np.ones_like(values)

        center_frame = np.average(
            t, weights=weights
        )

        center_position = circular_center(
            x, weights, n_space
        )

        # ------------------------------
        # Dynamic features
        # ------------------------------

        frame_volumes = np.array([
            values[t == frame].sum()
            for frame in frames
        ])

        frame_peaks = np.array([
            values[t == frame].max()
            for frame in frames
        ])

        frame_centers = np.array([
            circular_center(
                x[t == frame],
                weights[t == frame]
                if weights[t == frame].sum() > 0
                else np.ones(np.sum(t == frame)),
                n_space
            )
            for frame in frames
        ])

        # Spatial movement, accounting for periodicity
        displacement = np.diff(frame_centers)

        displacement = (
            (displacement + n_space / 2) % n_space
            - n_space / 2
        )

        total_movement = np.abs(displacement).sum()

        # ------------------------------
        # Save results
        # ------------------------------

        patch_id += 1
        label_map[mask] = patch_id

        records.append({
            "patch_id": patch_id,

            # Time
            "start_frame": start_frame,
            "end_frame": end_frame,
            "duration_frames": duration,
            "duration_time": duration * time_interval,
            "peak_frame": peak_frame,
            "center_frame": center_frame,
            "rise_frames": peak_frame - start_frame,
            "decay_frames": end_frame - peak_frame,

            # Space
            "peak_position": peak_position,
            "center_position": center_position,
            "total_width": total_width,
            "mean_width": mean_width,
            "max_width": max_frame_width,
            "mean_width_physical": (
                mean_width * space_interval
            ),
            "peak_frame_width": circular_width(
                x[t == peak_frame], n_space
            ),

            # Intensity
            "peak_intensity": peak_intensity,
            "mean_intensity": mean_intensity,
            "median_intensity": np.median(values),
            "std_intensity": np.std(values),
            "volume": volume,
            "excess_volume": excess_volume,
            "max_frame_volume": frame_volumes.max(),

            # Shape and movement
            "area": area,
            "total_movement": total_movement,
            "mean_speed": (
                total_movement * space_interval
                / ((duration - 1) * time_interval)
                if duration > 1 else 0
            ),

            # Boundary information
            "cross_spatial_boundary": bool(
                np.any(x == 0)
                and np.any(x == n_space - 1)
            ),
            "touch_time_boundary": bool(
                start_frame == 0
                or end_frame == n_time - 1
            ),

            # Dynamic profiles
            "frames": frames.tolist(),
            "frame_widths": frame_widths.tolist(),
            "frame_volumes": frame_volumes.tolist(),
            "frame_peaks": frame_peaks.tolist(),
            "frame_centers": frame_centers.tolist(),
        })

    patch_df = pd.DataFrame(records)

    return patch_df, label_map
