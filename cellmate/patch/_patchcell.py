import numpy as np

from cellmate.mating import CellNetwork90
from ._utils import centre_points, circular_sequence, resample_curve, move_to_center, intensity_multiple_points
from ..configs import CONTOURS_LENGTH
from ._classification_patch import prediction_cell_type_patch


DIVISION = 1000


class CellNetworkPatch(CellNetwork90):
    def __init__(self, image, time_network, tracker, threshold, *args, **kwargs):
        super().__init__(image, time_network, tracker, threshold, *args, **kwargs)
        self._aligned_coords = {}

    def raw_patch(self, cell_id, image, channel, dist=5, radius=9):
        data_overtime = []
        bg_overtime = []
        frames = self.cells[cell_id].frames
        coords = self.aligned_coords(cell_id)
        centers = self.center_overtime(cell_id)
        for i, time in enumerate(frames):
            coord_t = move_to_center(coords[i], centers[i], dist=dist)
            data, bg = intensity_multiple_points(image[time, channel],
                                                 coord_t, radius,
                                                 (image[time, -1] % 1000 == cell_id),
                                                 method="mean",
                                                 background_percentile=50)
            data_overtime.append(data)
            bg_overtime.append(bg)
        data_overtime = np.array(data_overtime)
        bg_overtime = np.array(bg_overtime)
        return data_overtime, bg_overtime

    def center_tips(self, cell_id):
        tips = self.tips_overtime(cell_id)
        center_1 = centre_points(tips[:, 0])
        center_2 = centre_points(tips[:, 1])
        return center_1, center_2

    def cal_aligned_coords(self, cell_id):
        center_tip_1, center_tip_2 = self.center_tips(cell_id)
        coords = self.coords_overtime(cell_id)
        new_coords = []
        for i, time in enumerate(self.cells[cell_id].frames):
            coord_t = coords[i]
            cell_label_t = self.label_trans(time)[cell_id]
            _, tip_1_index = self.measure[time].nearest_coordinate(cell_label_t, [center_tip_1], ptype="label")
            tip_1_index = tip_1_index[0][0]
            _, tip_2_index = self.measure[time].nearest_coordinate(cell_label_t, [center_tip_2], ptype="label")
            tip_2_index = tip_2_index[0][0]
            split_1 = circular_sequence(tip_1_index, tip_2_index, CONTOURS_LENGTH)
            split_2 = circular_sequence(tip_2_index, tip_1_index, CONTOURS_LENGTH)

            new_split1 = resample_curve(coord_t[split_1], CONTOURS_LENGTH//2+1)
            new_split2 = resample_curve(coord_t[split_2], CONTOURS_LENGTH//2+1)
            new_coord = np.vstack((new_split1, new_split2[1:-1]))
            new_coords.append(new_coord)
        return np.array(new_coords)

    def aligned_coords(self, cell_id):
        if cell_id not in self._aligned_coords.keys():
            self._aligned_coords[cell_id] = self.cal_aligned_coords(cell_id)
        return self._aligned_coords[cell_id]

    def nearest_points(self, cell_id1, cell_id2):
        frames = common_frames(self.cells[cell_id1].frames, self.cells[cell_id2].frames)

        aligned_coords_1 = self.aligned_coords(cell_id1)
        aligned_coords_2 = self.aligned_coords(cell_id2)

        aligned_index = {}
        for frame in frames:
            f, idx1, idx2 = frame
            measure = self.measure[f]
            cell_id_1_f = self.label_map[f][cell_id1]
            cell_id_2_f = self.label_map[f][cell_id2]

            index_no_aligned = measure.distance(cell_id_1_f, cell_id_2_f, ptype="label")[0, 0, 2:]
            point1 = measure.coordinate(label=cell_id_1_f)[index_no_aligned[0]]
            point2 = measure.coordinate(label=cell_id_2_f)[index_no_aligned[1]]

            index_aligned_1 = np.argmin(np.linalg.norm(aligned_coords_1[idx1] - point1, axis=1))
            index_aligned_2 = np.argmin(np.linalg.norm(aligned_coords_2[idx2] - point2, axis=1))
            aligned_index[f] = [index_aligned_1, index_aligned_2]
        return aligned_index

    def create_cell_type(self, fluorescent_image, mask=None, *arg, **kwargs):
        if mask is None:
            mask = self.image
        cell_pred, data = prediction_cell_type_patch(fluorescent_image, mask, *arg, **kwargs)
        type_maps = cell_pred.to_dict()
        for k, v in type_maps.items():
            self.cells[k % DIVISION].strain_type = v
        self.fluorescent_intensity = data


def common_frames(frames_1, frames_2):
    """
    Find common elements between two arrays and their indices in both arrays.

    Parameters:
    - frames_1 (np.ndarray): First array of elements.
    - frames_2 (np.ndarray): Second array of elements.

    Returns:
    - frames (list of tuples): Each tuple contains:
        - The common element
        - Its index in frames_1
        - Its index in frames_2
    """
    common_elements = np.intersect1d(frames_1, frames_2)
    indexes_array1 = np.nonzero(np.isin(frames_1, common_elements))[0]
    indexes_array2 = np.nonzero(np.isin(frames_2, common_elements))[0]
    frames = [(element, idx1, idx2) for element, idx1, idx2 in zip(common_elements, indexes_array1, indexes_array2)]
    return frames
