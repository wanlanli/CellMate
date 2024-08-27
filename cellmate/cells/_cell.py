from collections.abc import Iterable

import numpy as np

from cellmate.configs import (CELL_IMAGE_PARAM,
                              CellImgParamMap,
                              GENERATION_TREE,
                              CellGenParamMap,)
from cellmate.utils import hash_func, angle_of_vectors


class Cell(object):
    """Single cell object. Use to represent a single cell.
    """
    def __init__(self,
                 id,
                 frames=[],
                 reference_image=None,
                 fluorescence_image=None,
                 segmentation_mask=None,
                 tracking_mask=None,
                 offsets=None,
                 features=None,
                 generation_tree=[0, None, None, [], [], None, None],):
        # ID
        self.id = id
        self.frames = frames
        # Images
        self.reference_image = reference_image
        self.fluorescence_image = fluorescence_image
        self.segmentation_mask = segmentation_mask
        self.tracking_mask = tracking_mask
        # crop
        if offsets is None:
            self.offsets = np.zeros((len(self.frames), 2))
        else:
            self.offsets = offsets
        # Morphology Features
        self._features = features
        self._generation_trIMAGE_MEASURE_COLUMNSee = generation_tree
        # generation tree
        # Class
        self.type = None  # h+, h-, h90
        self.status = None  # from tracing [0: None, 1: cell, 2: paris, 3: fusion, 4: spores, 5: lysis,]

        self._hash_frame = hash_func(self.frames)

    # generation tree
    @property
    def start(self):
        return min(self.frames)

    @property
    def end(self):
        return max(self.frames)

    @property
    def life(self):
        return self.end - self.start + 1

    @property
    def semantic(self):
        return np.media(self._features[:, CellImgParamMap.get(CELL_IMAGE_PARAM.SEMANTIC_LABEL)].astype(np.uint16))

    @semantic.setter
    def semantic(self, v):
        self._features[:, CellImgParamMap.get(CELL_IMAGE_PARAM.SEMANTIC_LABEL)] = v

    @property
    def generation(self):
        return self._generation_tree[0]

    @property
    def division(self):
        if len(self.daughter_vg):
            return True
        else:
            return False

    @property
    def daughter_vg(self):
        return self._generation_tree[CellGenParamMap.get(GENERATION_TREE.DAUGHTER_VG)]

    @property
    def ancient(self):
        return self._generation_tree[CellGenParamMap.get(GENERATION_TREE.ANCIENT_VG)]

    @property
    def sister(self):
        return self._generation_tree[CellGenParamMap.get(GENERATION_TREE.SISTER)]

    @property
    def parents(self):
        return self._generation_tree[CellGenParamMap.get(GENERATION_TREE.PARENTS)]

    @property
    def fusion(self):
        if self.daughter_sex is None:
            return False
        else:
            return True

    @property
    def daughter_sex(self):
        return self._generation_tree[CellGenParamMap.get(GENERATION_TREE.DAUGHTER_SEX)]

    @property
    def spouse(self):
        return self._generation_tree[CellGenParamMap.get(GENERATION_TREE.SPOUSE)]

    def __index(self, frame, index):
        if index is None:
            if frame is None:
                raise ValueError("None input")
            else:
                if isinstance(frame, Iterable):
                    return [self._hash_frame.get(i) for i in frame if self._hash_frame.get(i) is not None]
                else:
                    return [self._hash_frame.get(frame)]
        else:
            if isinstance(index, Iterable):
                return [i for i in index if i < len(self.frames)]
            else:
                return [index]

    @property
    def feature_(self):
        return self._features

    @feature_.setter
    def feature_(self, v):
        self._features = v

    @property
    def tree_(self):
        return self._generation_tree

    @tree_.setter
    def tree_(self, v):
        self._generation_tree = v

    def feature(self, frame=None, index=None):
        index = self.__index(frame, index)
        return self._features[index]

    @property
    def label_(self):
        return self._features[:, CellImgParamMap.get(CELL_IMAGE_PARAM.LABEL)]

    def label(self, frame=None, index=None):
        index = self.__index(frame, index)
        return self.label_[index]

    @property
    def label_seg_(self):
        pass

    def label_seg(self, frame=None, index=None):
        index = self.__index(frame, index)
        pass

    @property
    def center_(self):
        col = [CellImgParamMap.get(i) for i in CELL_IMAGE_PARAM.CENTER_LIST]
        return self._features[:, col]

    def center(self, frame=None, index=None):
        index = self.__index(frame, index)
        return self.center_[index]

    @property
    def local_center_(self):
        return self.center_ - self.offsets

    def local_center(self, frame=None, index=None):
        index = self.__index(frame, index)
        return self.local_center_[index]

    @property
    def orientation_(self):
        return self._features[:, CellImgParamMap.get(CELL_IMAGE_PARAM.ORIENTATION)]

    def orientation(self, frame=None, index=None):
        index = self.__index(frame, index)
        return self.orientation_[index]

    @property
    def axis_major_length_(self):
        return self._features[:, CellImgParamMap.get(CELL_IMAGE_PARAM.MAJOR_AXIS)]

    def axis_major_length(self, frame=None, index=None):
        index = self.__index(frame, index)
        return self.axis_major_length_[index]

    @property
    def axis_minor_length_(self):
        return self._features[:, CellImgParamMap.get(CELL_IMAGE_PARAM.MINOR_AXIS)]

    def axis_minor_length(self, frame=None, index=None):
        index = self.__index(frame, index)
        return self.axis_minor_length_[index]

    @property
    def area_(self):
        return self._features[:, CellImgParamMap.get(CELL_IMAGE_PARAM.AREA)]

    def area(self, frame=None, index=None):
        index = self.__index(frame, index)
        return self.area_[index]

    @property
    def bbox_(self):
        col = [CellImgParamMap.get(i) for i in CELL_IMAGE_PARAM.BOUNDING_BOX_LIST]
        return self._features[:, col]

    def bbox(self, frame=None, index=None):
        index = self.__index(frame, index)
        return self.bbox_[index]

    @property
    def eccentricity_(self):
        return self._features[:, CellImgParamMap.get(CELL_IMAGE_PARAM.ECCENTRICITY)]

    def eccentricities(self, frame=None, index=None):
        index = self.__index(frame, index)
        return self.eccentricity_[index]

    @property
    def coordinates_(self):
        data = self._features[:, CellImgParamMap.get(CELL_IMAGE_PARAM.COORDINATE)]
        return np.array(list(data))

    def coordinates(self, frame=None, index=None):
        index = self.__index(frame, index)
        return self.coordinates_[index]

    @property
    def local_coordinates_(self):
        return self.coordinates_ - self.offsets[:, None, :]

    def local_coordinates(self, frame=None, index=None):
        index = self.__index(frame, index)
        return self.coordinates_[index] - self.offsets[index, None, :]

    @property
    def is_border_(self):
        return self._features[:, CellImgParamMap.get(CELL_IMAGE_PARAM.IS_BORDER)]

    def is_border(self, frame=None, index=None):
        index = self.__index(frame, index)
        return self.is_border_[index]

    @property
    def angle_(self):
        coords = self.coordinates_ - self.center_[:, None, :]
        angles = []
        for i in range(0, len(coords)):
            data = coords[i]
            angles.append([angle_of_vectors(data[j], data[0]) for j in range(0, data.shape[0])])
        return np.array(angles)

    @property
    def angle_index_(self):
        coords = self.coordinates_ - self.center_[:, None, :]
        angles_index = []
        for i in range(0, len(coords)):
            data = coords[i]
            angles_index.append([j for j in range(0, data.shape[0])])
        return np.array(angles_index)

    def angle(self, frame=None, index=None):
        index = self.__index(frame, index)
        return self.angle_[index]

    def key_feature_(self, keys: list, frame=None, index=None):
        index = self.__index(frame, index)
        key_index = [CellImgParamMap.get(k) for k in keys]
        data = self._features[index]
        data = data[:, key_index]
        return np.array(list(data))

    @property
    def tips(self):
        pass
