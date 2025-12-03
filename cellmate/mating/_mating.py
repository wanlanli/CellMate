import pandas as pd
import networkx as nx
from tqdm import trange

from cellmate.image_measure import ImageMeasure
from ._cell import Cell
from ._classification import prediction_cell_type
from cellmate.configs import DIVISION
import numpy as np


class CellNetwork():
    def __init__(self, image, time_network, tracker, threshold, *args, **kwargs) -> None:
        self.image = image
        self.frame_number = self.image.shape[0]
        self.time_network = time_network
        self.neighbor_threshold = threshold
        self.space_net = None

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

    def space_network_t(self, time):
        index = self.space_net_map[time]
        return self.space_network[index]

    def create_cells(self):
        pass

    def create_cell_type(self, fluorescent_image, mask=None, *arg, **kwargs):
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
                   'center_dist', 'nearest_dist', 'tip_distance', 'p_in_new_tip', 'm_in_new_tip','time_stamp']
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
        print(coords.shape)
        return coords

    def center_overtime(self, cell_id):
        coords = []
        frames = self.cells[cell_id].frames
        for time in frames:
            cell_label_t = self.label_map[time][cell_id]
            coords.append(self.measure[time].center(label=cell_label_t))
        coords = np.array(coords)
        print(coords.shape)
        return coords

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
                   'center_dist', 'nearest_dist', 'time_stamp']
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
                    feature = self.pair_feature(ref, n, t)
                    if n == parents[1-i]:
                        flag = True
                    else:
                        flag = False
                    data.loc[index] = [ref, cell_ref.strain_type, flag]+feature
                    index += 1
        return data
