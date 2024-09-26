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
                 generation_tree=[0, None, None, [], [], None, None],):
        # ID
        self.id = id
        self.frames = frames
        #  self._features = features
        self._generation_tree = generation_tree
        # generation tree
        # Class
        self.strain_type = None  # h+, h-, h90
        # self.status = None  # from tracing [0: None, 1: cell, 2: paris, 3: fusion, 4: spores, 5: lysis,]

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

    def status(self):
        if self.division:
            return 1  # go to divided
        if self.fusion:
            return 2  # go to fusion
        if len(self.parents):
            return 3  # fusioned
        else:
            return 4  # last division

    def mating_competent(self):
        if self.division:
            return False
        if len(self.parents):
            return False
        return True


def flatten_nonzero_value(data):
    """flatten all non-zero values in data
    data: array_like
    """
    flatten = data.flatten()
    flatten = flatten[flatten > 0]
    return flatten
