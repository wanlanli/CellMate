import numpy as np

from cellmate.mating import CellNetwork90
from ._utils import centre_points, circular_sequence, resample_curve
from ..configs import CONTOURS_LENGTH


class CellNetworkPatch(CellNetwork90):
    def __init__(self, image, time_network, tracker, threshold, *args, **kwargs):
        super().__init__(image, time_network, tracker, threshold, *args, **kwargs)
        self._aligned_coords = {}

    def raw_patch(self, cell_id):
        pass

    def center_tips(self, cell_id):
        tips = self.tips_overtime(cell_id)
        center_1 = centre_points(tips[:, 0])
        center_2 = centre_points(tips[:, 1])
        return center_1, center_2

    def cal_aligned_coords(self, cell_id):
        center_tip_1, center_tip_2 = self.center_tips(cell_id)
        coords = self.coords_overtime(cell_id)
        new_coords = []
        for time in self.cells[cell_id].frames:
            coord_t = coords[time]
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
