import re

import pandas as pd
import networkx as nx
from tqdm import trange

from cellmate.image_measure import ImageMeasure
from ._cell import Cell
from ._classification import compute_snr_table, classify_snr_table
from ._classification_legacy import prediction_cell_type
from ._classification90 import prediction_cell_type_h90switch
from cellmate.configs import DIVISION
import numpy as np


class CellNetwork():
    def __init__(self, image, time_network, tracker, threshold, *args, **kwargs) -> None:
        self.image = image
        self.frame_number = self.image.shape[0]
        self.time_network = time_network
        self.neighbor_threshold = threshold
        self.space_net = None

        self.classification_params = None
        self._intensity_cache_key = None

        self.measure = []
        self.space_network = []
        self.space_net_map = {}
        self.cells = {}
        self.fluorescent_intensity = None
        last_labels = set([])
        for i in trange(0, self.image.shape[0]):
            mask = self.image[i]
            measure = ImageMeasure(mask, *args, **kwargs)
            self.measure.append(measure)
            labels = measure.labels % DIVISION
            if set(labels) == last_labels:
                self.space_net_map[i] = len(self.space_network) - 1
                continue
            else:
                adj = measure.adjacent_matrix(threshold=threshold)

                sorted_indices = np.argsort(labels)
                adj = adj[np.ix_(sorted_indices, sorted_indices)]

                pandas = pd.DataFrame(adj, index=labels[sorted_indices], columns=labels[sorted_indices])
                space_network = nx.from_pandas_adjacency(pandas)
                self.space_network.append(space_network)
                self.space_net_map[i] = len(self.space_network) - 1

                last_labels = set(labels)

        # for c in tracker:
        #     if time_network.has_node(c.id):
        #         gen_feature = time_network.feature(c.id)
        #     else:
        #         gen_feature = [0, None, None, [], [], None, None]
        #     frames = np.array(c.frame)
        #     frames = frames[frames < self.frame_number]
        #     self.cells[c.id] = Cell(id=c.id, frames=frames, generation_tree=gen_feature)

        for c in tracker.keys():
            if time_network.has_node(c):
                gen_feature = time_network.feature(c)
            else:
                gen_feature = [0, None, None, [], [], None, None]
            frames = np.array(tracker[c]['frame'])
            frames = frames[frames < self.frame_number]
            label = tracker[c]['label']
            self.cells[c] = Cell(id=c, label=label, frames=frames, generation_tree=gen_feature)

        self.label_map = []
        for t in range(0, self.image.shape[0]):
            self.label_map.append(self.label_trans(t))

    @classmethod
    def from_tracked_movie(cls, tracked_image, tracking_threshold, threshold, min_hist=1, max_miss=1, *args, **kwargs):
        """Build a CellNetwork straight from an already-tracked label movie
        (e.g. the mask channel of a saved tracked `.tif`), by re-tracking it
        to reconstruct the division/fusion relationships instead of
        requiring the original Tracker or its pickled network/trackers to
        still be around -- see `cellmate.tracking.retrace`.

        Parameters:
        -----------
        tracked_image: the tracked label movie, [T, H, W].
        tracking_threshold: IoU threshold for the retrace -- use the same
            value the tracking pass that produced `tracked_image` used.
        threshold: neighbor/adjacency threshold, passed straight through to
            `__init__` (same meaning as calling it directly).
        min_hist, max_miss: passed to the retrace; defaults (1, 1) assume
            `tracked_image` is already a clean, gap-filled delivery.
        """
        from cellmate.tracking import retrace

        tracker = retrace(tracked_image, threshold=tracking_threshold, min_hist=min_hist, max_miss=max_miss)
        return cls(image=tracked_image, time_network=tracker.network,
                  tracker=tracker.save_trackers(), threshold=threshold, *args, **kwargs)

    def space_network_t(self, time):
        index = self.space_net_map[time]
        return self.space_network[index]

    def create_cells(self):
        pass

    def create_cell_type(self, fluorescent_image=None, mask=None, channel_number=2, bg_threshold=10,
                         bg_region="background", fc_threshold=50, z_threshold=3.0, min_noise=1.0,
                         force_recompute=False):
        """Classify each cell's mating type from its fluorescence, per
        channel relative to background (see `prediction_cell_type_snr`).
        `z_threshold` (background std devs to call a channel "on", default
        3.0) is the knob to tune per dataset.

        Computing the per-(cell, frame) intensity/SNR table (one pass over
        the whole image stack, via `compute_snr_table`) is expensive;
        turning that table into type calls for a given `z_threshold` (via
        `classify_snr_table`) is cheap. So this caches the table in
        `self.fluorescent_intensity` and, as long as `fluorescent_image`/
        `mask`/`channel_number`/`bg_threshold`/`bg_region`/`fc_threshold`/
        `min_noise` haven't changed since it was built, reuses it -- pass
        just a new `z_threshold` to re-run classification only. Omit
        `fluorescent_image` entirely to force that reuse (e.g. after
        unpickling a `CellNetwork` that already has `fluorescent_intensity`
        set). Pass `force_recompute=True` to rebuild the table regardless.

        `classification_params`/the cache key are plain attributes set here,
        so a `CellNetwork` pickled *before* this caching existed won't have
        them -- `getattr(..., None)` below just treats that as "no cache
        info", and `channel_number` gets recovered from the old table's own
        `ch_i_snr` columns (unchanged schema) instead of raising.

        The cache is tagged `method="full"` so this won't be confused with a
        table built by `create_cell_type_by_lineage` (which only measures
        root cells, at their own start frame) -- reusing that as if it were
        this method's full per-(cell, frame) table would silently give every
        other cell whatever its lineage ancestor's summary happened to be,
        instead of running this method's own per-cell classification.
        """
        cached_params = getattr(self, "classification_params", None)
        cached_key = getattr(self, "_intensity_cache_key", None)

        if fluorescent_image is None:
            if force_recompute:
                raise ValueError("force_recompute=True needs fluorescent_image to recompute from")
            if self.fluorescent_intensity is None:
                raise ValueError("fluorescent_image is required the first time create_cell_type is "
                                  "called; omit it on later calls to just re-threshold the cached "
                                  "intensity table")
            if cached_params is None:
                # raw `ch_{i}_snr` columns only -- not `ch_{i}_prediction_snr`,
                # which is also present once classify_snr_table has run. A
                # pkl this old predates create_cell_type_by_lineage too, so
                # it can only be a `create_cell_type` table.
                snr_cols = [c for c in self.fluorescent_intensity.columns
                            if re.fullmatch(r"ch_\d+_snr", c)]
                cached_params = {"channel_number": len(snr_cols), "method": "full"}
            elif cached_params.get("method", "full") != "full":
                raise ValueError(
                    "self.fluorescent_intensity was built by create_cell_type_by_lineage "
                    "(method=%r), not create_cell_type -- it only has rows for root cells, "
                    "not every cell/frame. Pass fluorescent_image to (re)build a full table "
                    "here." % cached_params.get("method"))
            recompute = False
        else:
            if mask is None:
                mask = self.image
            cache_key = (id(fluorescent_image), id(mask), channel_number, bg_threshold,
                         bg_region, fc_threshold, min_noise)
            recompute = force_recompute or cache_key != cached_key

        if recompute:
            self.fluorescent_intensity = compute_snr_table(
                fluorescent_image, mask, channel_number=channel_number, bg_threshold=bg_threshold,
                bg_region=bg_region, fc_threshold=fc_threshold, min_noise=min_noise)
            cached_key = cache_key
            cached_params = dict(channel_number=channel_number, bg_threshold=bg_threshold,
                                 bg_region=bg_region, fc_threshold=fc_threshold,
                                 min_noise=min_noise, method="full")

        channel_number = cached_params["channel_number"]
        cell_pred, data = classify_snr_table(self.fluorescent_intensity, channel_number, z_threshold=z_threshold)
        type_maps = cell_pred.to_dict()
        for k, v in type_maps.items():
            self.cells[k % DIVISION].strain_type = v
        self.fluorescent_intensity = data
        cached_params["z_threshold"] = z_threshold
        self.classification_params = cached_params
        self._intensity_cache_key = cached_key

    def create_cell_type_by_lineage(self, fluorescent_image=None, mask=None, channel_number=2, bg_threshold=10,
                                    bg_region="background", fc_threshold=50, z_threshold=3.0, min_noise=1.0,
                                    force_recompute=False):
        """Classify cell types from the tracked lineage instead of every
        cell's every frame -- only valid for movies whose division/fusion
        tracking (`self.time_network`) is trustworthy throughout.

        Only "root" cells -- present from the start with no recorded parent
        (`cell.ancient is None and not cell.parents`, i.e. no division or
        fusion produced them within this movie) -- are measured directly,
        each on the single frame it starts in (`compute_snr_table` +
        `classify_snr_table`, same SNR rule as `create_cell_type`). Every
        other cell's type is then propagated down `self.time_network`:
        - division (`cell.ancient` set): the daughter copies its parent's
          type.
        - fusion (`cell.parents` set, two of them): the child's type is the
          bitwise OR of both parents' types -- so two complementary
          single-marker parents (e.g. 1 and 2) fuse into a double-positive
          zygote (3), matching real mating biology; two same-type parents
          just stay that type.

        Measuring one frame per root instead of every frame of every cell
        is far cheaper on long movies, but it will silently misclassify a
        lineage if a division/fusion was missed or mis-tracked, or if a
        cell's marker isn't stable across generations -- `create_cell_type`
        is the one to fall back on if that's a risk for this dataset.

        Caches the per-root intensity table the same way `create_cell_type`
        does: as long as `fluorescent_image`/`mask`/`channel_number`/
        `bg_threshold`/`bg_region`/`fc_threshold`/`min_noise` haven't
        changed, a later call with just a different `z_threshold` reuses it
        (and just re-propagates) instead of re-measuring -- omit
        `fluorescent_image` entirely to force that reuse. Pass
        `force_recompute=True` to rebuild it regardless.

        The cache is tagged `method="lineage"` -- if it's already tagged
        that way, a later no-image call just reuses it as-is (rows already
        one per root, at its own start frame). If instead `fluorescent_intensity`
        holds some other table (e.g. a `create_cell_type` one, with a row
        for every (cell, frame)), a no-image call takes that table's `frame
        == 0` rows as the root measurements instead of raising -- cheap
        (just a filter, no re-measuring) and correct as long as every root
        this movie has actually starts at frame 0; a root that first
        appears later (tracking gap, cell entering view mid-movie) won't be
        in that slice and is left with whatever `strain_type` it already
        had. Pass `fluorescent_image` (or `force_recompute=True` with one)
        to measure every root at its own real start frame instead.
        """
        cached_params = getattr(self, "classification_params", None)
        cached_key = getattr(self, "_intensity_cache_key", None)

        if fluorescent_image is None:
            if force_recompute:
                raise ValueError("force_recompute=True needs fluorescent_image to recompute from")
            if self.fluorescent_intensity is None:
                raise ValueError("fluorescent_image is required the first time "
                                  "create_cell_type_by_lineage is called; omit it on later calls to "
                                  "just re-threshold the cached intensity table")
            if cached_params is not None and cached_params.get("method") == "lineage":
                recompute = False
            else:
                # Not our own root-only table (e.g. a create_cell_type one,
                # with every cell at every frame) -- take its frame-0 rows
                # as the root measurements instead of requiring a recompute.
                if cached_params is not None:
                    channel_number = cached_params["channel_number"]
                else:
                    snr_cols = [c for c in self.fluorescent_intensity.columns
                                if re.fullmatch(r"ch_\d+_snr", c)]
                    channel_number = len(snr_cols)
                self.fluorescent_intensity = (
                    self.fluorescent_intensity[self.fluorescent_intensity["frame"] == 0].copy())
                cached_key = None
                cached_params = dict(channel_number=channel_number, method="lineage")
                recompute = False
        else:
            if mask is None:
                mask = self.image
            cache_key = (id(fluorescent_image), id(mask), channel_number, bg_threshold,
                         bg_region, fc_threshold, min_noise, "lineage")
            recompute = force_recompute or cache_key != cached_key

        if recompute:
            root_cells = [cell for cell in self.cells.values()
                         if cell.ancient is None and not cell.parents]

            frames_needed = {}
            for cell in root_cells:
                frames_needed.setdefault(cell.start, []).append(cell.id)

            tables = []
            for frame in frames_needed:
                table = compute_snr_table(
                    fluorescent_image[frame:frame + 1], mask[frame:frame + 1],
                    channel_number=channel_number, bg_threshold=bg_threshold, bg_region=bg_region,
                    fc_threshold=fc_threshold, min_noise=min_noise)
                table["frame"] = frame
                tables.append(table)
            self.fluorescent_intensity = (pd.concat(tables, ignore_index=True) if tables
                                          else pd.DataFrame())
            cached_key = cache_key
            cached_params = dict(channel_number=channel_number, bg_threshold=bg_threshold,
                                 bg_region=bg_region, fc_threshold=fc_threshold,
                                 min_noise=min_noise, method="lineage")

        channel_number = cached_params["channel_number"]
        root_pred, data = classify_snr_table(self.fluorescent_intensity, channel_number, z_threshold=z_threshold)
        for k, v in root_pred.to_dict().items():
            self.cells[k % DIVISION].strain_type = v

        for node in nx.topological_sort(self.time_network):
            if node not in self.cells:
                continue
            cell = self.cells[node]
            if cell.ancient is not None:
                cell.strain_type = self.cells[cell.ancient].strain_type
            elif cell.parents:
                parent_1, parent_2 = cell.parents
                cell.strain_type = self.cells[parent_1].strain_type | self.cells[parent_2].strain_type

        self.fluorescent_intensity = data
        cached_params["z_threshold"] = z_threshold
        self.classification_params = cached_params
        self._intensity_cache_key = cached_key

    def create_cell_type_legacy(self, fluorescent_image, mask=None, *arg, **kwargs):
        """Original classifier (KMeans on a log-ratio normalization, tuned
        via `high_val`) -- kept as a reference/fallback to double-check
        `create_cell_type`'s SNR-based result against on other datasets."""
        if mask is None:
            mask = self.image
        cell_pred, data = prediction_cell_type(fluorescent_image, mask, *arg, **kwargs)
        type_maps = cell_pred.to_dict()
        for k, v in type_maps.items():
            self.cells[k % DIVISION].strain_type = v
        self.fluorescent_intensity = data

    def fusion_cells(self):
        return [node[0] for node in self.time_network.in_degree if node[1] == 2]

    def pair_feature(self, p_label, m_label, time):
        """
        columns = ['p_id', 'm_id',
                   'p_start', 'p_area', 'p_major, p_minor', 'p_eccentricity', 'p_neighbor_same', 'p_neighbor_diff',
                   'm_start', 'm_area', 'm_major, m_minor', 'm_eccentricity', 'm_neighbor_same', 'm_neighbor_diff',
                   'p_angle', 'm_angle', 'p_angle_index', 'm_angle_index', 'p_angle_norm', 'm_angle_norm',
                   'center_dist', 'nearest_dist', 'tip_distance','p_in_new_tip', 'm_in_new_tip', 'time_stamp',]
        """
        cell_p = self.cells[p_label]
        cell_m = self.cells[m_label]
        measure = self.measure[time]
        id_label_map = dict(zip(measure.labels % DIVISION, measure.labels))
        f_1 = self.__cell_feature(cell_p, id_label_map[cell_p.id], measure, time)
        f_2 = self.__cell_feature(cell_m, id_label_map[cell_m.id], measure, time)
        f_pair = self.__pair_feature(measure, id_label_map[cell_p.id], id_label_map[cell_m.id])

        tips_ditance, tip_1, tip_2 = measure.tips_distance(id_label_map[cell_p.id],
                                                           id_label_map[cell_m.id],
                                                           ptype="label")

        cell1_tip = self.check_meet_tips(cell_p.id, tip_1, time)
        cell2_tip = self.check_meet_tips(cell_m.id, tip_2, time)

        pair_features = [cell_p.id, cell_m.id] + f_1 + f_2 + f_pair + [tips_ditance, cell1_tip, cell2_tip] + [time]
        return pair_features

    def __cell_feature(self, cell, label, measure, time):
        feature = [cell.start, measure.area(label=label),
                   measure.skeleton_length(label=label),
                   measure.medial_minor_length(label=label),
                   measure.eccentricity(label=label),
                   len(self.neighbor_same(node=cell.id, time=time)),
                   len(self.neighbor_diff(node=cell.id, time=time)),
                   ]
        return feature

    def __pair_feature(self, measure, label1, label2):
        feature = list(measure.between_angle(label1, label2, ptype="label")) +\
                  list(measure.between_angle_index(label1, label2, ptype="label", norm=False)) +\
                  list(measure.between_angle_index(label1, label2, ptype="label", norm=True)) +\
                  list(measure.distance(label1, label2, ptype="label")[0, 0, :2]) # +\
                   #list(measure.tips_distance(label1, label2, ptype="label")])
        return feature

    def neighbor(self, node, time):
        net = self.space_network_t(time)
        if node in net.nodes:
            return list(net.neighbors(node))
        else:
            return []

    def neighbor_diff(self, node, time):
        nei = self.neighbor(node, time)
        diff_nei = []
        target_type = self.cells[node].strain_type
        for c in nei:
            if self.cells[c].mating_competent():
                if target_type != self.cells[c].strain_type:
                    diff_nei.append(c)
        return diff_nei

    def neighbor_same(self, node, time):
        nei = self.neighbor(node, time)
        same_nei = []
        target_type = self.cells[node].strain_type
        for c in nei:
            if self.cells[c].mating_competent():
                if target_type == self.cells[c].strain_type:
                    same_nei.append(c)
        return same_nei

    def potential_mating_feature(self, parents, time_step: int = 10):
        columns = ['ref_id', 'ref_type', 'flag', 'p_id', 'm_id',
                   'p_start', 'p_area', 'p_major', 'p_minor', 'p_eccentricity', 'p_neighbor_same', 'p_neighbor_diff',
                   'm_start', 'm_area', 'm_major', 'm_minor', 'm_eccentricity', 'm_neighbor_same', 'm_neighbor_diff',
                   'p_angle', 'm_angle', 'p_angle_index', 'm_angle_index', 'p_angle_norm', 'm_angle_norm', 
                   'center_dist', 'nearest_dist', 'tip_distance', 'p_in_new_tip', 'm_in_new_tip', 
                   'time_stamp']
        if self.cells[parents[0]].strain_type == self.cells[parents[1]].strain_type:
            print("same type")
            return None
        if self.cells[parents[0]].strain_type > self.cells[parents[1]].strain_type:
            parents.reverse()
        data = pd.DataFrame(None, columns=columns)
        index = 0
        for i, ref in enumerate(parents):
            cell_ref = self.cells[ref]
            start_time = cell_ref.start
            end_time = cell_ref.end
            time_table = list(range(start_time, end_time, time_step)) + [end_time]
            for t in time_table:
                mating_competent = self.neighbor_diff(node=ref, time=t)
                for n in mating_competent:
                    if i == 0:
                        feature = self.pair_feature(ref, n, t)
                    else:
                        feature = self.pair_feature(n, ref, t)
                    if n == parents[1-i]:
                        flag = True
                    else:
                        flag = False
                    data.loc[index] = [ref, cell_ref.strain_type, flag]+feature
                    index += 1
        return data

    def label_trans(self, time):
        """
        Translate a global label into a time-specific label in the measure.

        Parameters:
        time (int): The time point for which to translate the label.

        Returns:
        dict: A dictionary mapping the global label (modulo division) to the original label.
        """
        measure = self.measure[time]
        return dict(zip(measure.labels % DIVISION, measure.labels))

    def bbox_overtime(self, cell_id):
        bbox = []
        frames = self.cells[cell_id].frames
        for time in frames:
            cell_label_t = self.label_map[time][cell_id]
            bbox.append(self.measure[time].bbox(label=cell_label_t))
        bbox = np.array(bbox)
        return bbox

    def tips_overtime(self, cell_id):
        tips = []
        frames = self.cells[cell_id].frames
        for time in frames:
            cell_label_t = self.label_map[time][cell_id]
            tips.append(self.measure[time].tip(label=cell_label_t))
        tips = np.array(tips)
        return tips

    def coords_overtime(self, cell_id):
        coords = []
        frames = self.cells[cell_id].frames
        for time in frames:
            cell_label_t = self.label_map[time][cell_id]
            coords.append(self.measure[time].coordinate(label=cell_label_t))
        coords = np.array(coords)
        # print(coords.shape)
        return coords

    def center_overtime(self, cell_id):
        coords = []
        frames = self.cells[cell_id].frames
        for time in frames:
            cell_label_t = self.label_map[time][cell_id]
            coords.append(self.measure[time].center(label=cell_label_t))
        coords = np.array(coords)
        # print(coords.shape)
        return coords

    def skeleton_overtime(self, cell_id):
        skeletons = []
        frames = self.cells[cell_id].frames
        for time in frames:
            cell_label_t = self.label_map[time][cell_id]
            skeletons.append(self.measure[time].skeleton(label=cell_label_t))
        skeletons = np.array(skeletons)
        return skeletons

    def center_tips(self, cell_id):
        """The two tip locations (skeleton endpoints), each centered
        (median position) over every frame the cell is tracked in -- a
        stable reference for "tip 1"/"tip 2" even though a single frame's
        raw tip positions drift/rotate and can flip order frame to frame."""
        from cellmate.patch._utils import centre_points

        tips = self.tips_overtime(cell_id)
        center_1 = centre_points(tips[:, 0])
        center_2 = centre_points(tips[:, 1])
        return center_1, center_2

    def aligned_coords_overtime(self, cell_id, num_samples=None):
        """Like coords_overtime, but the contour is split at the cell's two
        (time-stabilized) tips and each half resampled to a fixed point
        count, so point index N refers to roughly the same physical location
        on the cell in every frame -- see CellNetworkPatch.aligned_coords,
        which this generalizes to any CellNetwork (not just mating pairs)."""
        from cellmate.configs import CONTOURS_LENGTH
        from cellmate.patch._utils import circular_sequence, resample_curve

        num_samples = num_samples or CONTOURS_LENGTH
        half = num_samples // 2 + 1
        center_tip_1, center_tip_2 = self.center_tips(cell_id)
        coords = self.coords_overtime(cell_id)
        new_coords = []
        for i, time in enumerate(self.cells[cell_id].frames):
            coord_t = coords[i]
            cell_label_t = self.label_map[time][cell_id]
            _, tip_1_index = self.measure[time].nearest_coordinate(cell_label_t, [center_tip_1], ptype="label")
            _, tip_2_index = self.measure[time].nearest_coordinate(cell_label_t, [center_tip_2], ptype="label")
            tip_1_index = tip_1_index[0][0]
            tip_2_index = tip_2_index[0][0]
            max_id = len(coord_t)
            split_1 = circular_sequence(tip_1_index, tip_2_index, max_id)
            split_2 = circular_sequence(tip_2_index, tip_1_index, max_id)
            new_split1 = resample_curve(coord_t[split_1], half)
            new_split2 = resample_curve(coord_t[split_2], half)
            new_coord = np.vstack((new_split1, new_split2[1:-1]))
            new_coords.append(new_coord)
        return np.array(new_coords)

    def aligned_skeleton_overtime(self, cell_id, num_samples=None):
        """Like skeleton_overtime, but each frame's centerline is oriented to
        start from the same (time-stabilized) tip -- reversed when needed --
        and resampled to a fixed point count, so it can be compared point by
        point across frames the same way aligned_coords_overtime does for the
        contour."""
        from cellmate.configs import SKELETON_LENGTH
        from cellmate.patch._utils import resample_curve

        num_samples = num_samples or SKELETON_LENGTH
        center_tip_1, _ = self.center_tips(cell_id)
        skeletons = self.skeleton_overtime(cell_id)
        aligned = []
        for skeleton in skeletons:
            skeleton = np.asarray(skeleton)
            start_dist = np.linalg.norm(skeleton[0] - center_tip_1)
            end_dist = np.linalg.norm(skeleton[-1] - center_tip_1)
            if end_dist < start_dist:
                skeleton = skeleton[::-1]
            aligned.append(resample_curve(skeleton, num_samples))
        return np.array(aligned)

    def check_fusion_tips(self, cell_id):
        if self.cells[cell_id].start == 0:
            return 2
        if not self.cells[cell_id].fusion:
            return 2
        tips = self.tips_overtime(cell_id)
        division_point = self.get_division_point(cell_id)
        fusion_point = self.get_fusion_point(cell_id)
        is_old_tip = is_mated_in_new_tip(tips[0], tips[-1], division_point, fusion_point)
        return is_old_tip

    def check_meet_tips(self, cell_id, meet_point, time):
        if self.cells[cell_id].start == 0:
            return 2
        if self.cells[cell_id].sister is None:
            return 2
        cell_label = self.label_map[time][cell_id]
        start_tips = np.array(self.measure[time].tip(label=cell_label))
        # tips = self.tips_overtime(cell_id)

        division_point = self.get_division_point(cell_id)

        # fixed_time = np.where(self.cells[cell_id].frames == time)[0]
        # if len(fixed_time) == 0:
        #     return 2
        # else:
        #     fixed_time = fixed_time[0]

        cell_label_t = self.label_map[time][cell_id]
        end_tips = np.array(self.measure[time].tip(label=cell_label_t))
        # print(cell_id, time, cell_label_t, cell_label)
        is_old_tip = is_mated_in_new_tip(start_tips, end_tips, division_point, meet_point)
        return is_old_tip

    def get_division_point(self, cell_id):
        sister_0 = self.cells[cell_id].sister
        time = self.cells[cell_id].start

        cell_label_t0 = self.label_map[time][cell_id]
        cell_label_t1 = self.label_map[time][sister_0]

        division_points_0 = self.measure[time].nearest_point(source=cell_label_t0,
                                                             target=cell_label_t1,
                                                             ptype="label")
        return division_points_0[0]

    def get_fusion_point(self, cell_id):
        spouse_0 = self.cells[cell_id].spouse
        time = self.cells[cell_id].end

        cell_label_t0 = self.label_map[time][cell_id]
        cell_label_t1 = self.label_map[time][spouse_0]
        fusion_points = self.measure[time].nearest_point(source=cell_label_t0,
                                                         target=cell_label_t1, ptype="label")
        return fusion_points[0]



from scipy.spatial.distance import cdist
def is_mated_in_new_tip(tips_start, tips_end, point_start, point_end):
    dist_matrix = cdist(tips_start, tips_end)
    mapping = dist_matrix.argmin(axis=1)
    mappted_tips_end = tips_end[mapping]

    d0 = np.linalg.norm(tips_start - point_start, axis=1)
    idx0 = np.argmin(d0)  # index of closest tip
    d_end = np.linalg.norm(mappted_tips_end - point_end, axis=1)
    idx_end = np.argmin(d_end)

    return (idx0 == idx_end)*1


class CellNetwork90(CellNetwork):
    def __init__(self, image, time_network, tracker, threshold,  *args, **kwargs) -> None:
        super().__init__(image, time_network, tracker, threshold,  *args, **kwargs)

    def potential_mating_feature(self, parents, time_step: int = 10):
        columns = ['ref_id', 'ref_type', 'flag',
                   'p_id', 'm_id',
                   'p_start', 'p_area', 'p_major', 'p_minor', 'p_eccentricity', 'p_neighbor_same', 'p_neighbor_diff',
                   'm_start', 'm_area', 'm_major', 'm_minor', 'm_eccentricity', 'm_neighbor_same', 'm_neighbor_diff',
                   'p_angle', 'm_angle', 'p_angle_index', 'm_angle_index', 'p_angle_norm', 'm_angle_norm',
                   'center_dist', 'nearest_dist', 'tip_distance', 'p_in_new_tip', 'm_in_new_tip',
                   'time_stamp']
        data = pd.DataFrame(None, columns=columns)
        index = 0
        for i, ref in enumerate(parents):
            cell_ref = self.cells[ref]
            start_time = cell_ref.start
            end_time = cell_ref.end
            time_table = list(range(start_time, end_time, time_step)) + [end_time]
            for t in time_table:
                mating_competent = self.neighbor(node=ref, time=t)
                for n in mating_competent:
                    if i == 0:
                        feature = self.pair_feature(ref, n, t)
                    else:
                        feature = self.pair_feature(n, ref, t)
                    # feature = self.pair_feature(ref, n, t)
                    if n == parents[1-i]:
                        flag = True
                    else:
                        flag = False
                    data.loc[index] = [ref, cell_ref.strain_type, flag]+feature
                    index += 1
        return data
