from typing import Sequence, Union
from collections.abc import Iterable
from functools import cached_property

import numpy as np
import pandas as pd

from cellmate.configs import (IMAGE_MEASURE_PARAM, CELL_IMAGE_PARAM, DIVISION)
from .measure._regionprops import regionprops_table
from .measure import find_nearest_points
from cellmate.utils import create_line, angle_of_vectors, included_angle


class ImageMeasure(np.ndarray):
    """Extract segemented regions' information from mask, åsuch as area,
    center, boundingbox ect. al..
    Parameters
    ----------
    input_array : 2D matrix,dtype:int
        mask is a int type 2d mask array. stored the labels of segementation.
    """
    def __new__(cls, mask: np.ndarray):
        # Input array is an already formed ndarray instance first cast to
        # be our class type
        obj = np.asarray(mask).view(cls)
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return
        #  instance properties in numpy array,
        #  self._columns is the name of numpy each columns
        self._init_instance_properties()
        #  create hash map for columns "name->index"
        self.__hash_col = hash_func(self._columns)
        #  create hash map for objects "label->index"
        self.__hash_obj = hash_func(self._properties[:, 0])
        self.__cost = self._init_cost_matrix()
        self.pixel_resolution = 1

    def __index(self,
                index: Union[int, Sequence] = None,
                label: Union[int, Sequence] = None):
        """Inner function that retrieve rows by index or label arbitrarily.
        Note: Two parameters can and can only specify one of them.
        """
        if index is not None:
            if label is None:
                return self.__index_check(index)
            else:
                Warning("`index` and `label` cannot be specified at the same time," +
                        "the calculation is based on `index` only")
                return index
        else:
            if label is None:
                raise (ValueError("`index` and `label` cannot be None at the same time"))
            else:
                return self.label2index(label)

    def __index_trans(self, source, ptype="index"):
        if ptype == "label":
            source = self.label2index(source)
        elif ptype == "index":
            source = self.__index_check(source)
        else:
            raise (ValueError("ptype can only be index or label"))
        return source

    def __index_check(self, index: Union[int, Sequence[int]]):
        if isinstance(index, Iterable):
            index_n = [i for i in index if i < self._properties.shape[0]]
            if len(index_n) != len(index):
                Warning("Some `index` not exist!")
            return index_n
        else:
            if index < self._properties.shape[0]:
                return index
            else:
                raise (ValueError("`index` not exist! %d" % index))

    def label2index(self, label: Union[int, Sequence]):
        """image label to arg
        """
        if isinstance(label, int):
            return self.__hash_obj.get(label)
        else:
            return [self.__hash_obj.get(k) for k in label if self.__hash_obj.get(k)]

    # set properties
    def _init_instance_properties(self):
        """Calculate the attribute value of each instance of the generated mask.
        index: int, the order, from 0 to len(instances)
        label: int, the identify, equal with image values
        """
        props, columns = regionprops_table(self.__array__(),
                                           properties=IMAGE_MEASURE_PARAM)
        props = props.T
        data = np.empty((props.shape[0], 3), dtype=np.int_)
        # semantic
        data[:, 0] = props[:, columns.index(CELL_IMAGE_PARAM.LABEL)] // DIVISION
        # instance
        data[:, 1] = props[:, columns.index(CELL_IMAGE_PARAM.LABEL)] % DIVISION
        # is_border
        col = [columns.index(i) for i in CELL_IMAGE_PARAM.BOUNDING_BOX_LIST]
        data[:, 2] = _cal_is_border(props[:, col], self.shape)
        columns += [CELL_IMAGE_PARAM.SEMANTIC_LABEL,
                    CELL_IMAGE_PARAM.INSTANCE_LABEL,
                    CELL_IMAGE_PARAM.IS_BORDER]
        self._columns = columns
        self._properties = np.concatenate((props, data), axis=1)

    # get properties
    @cached_property
    def property_table(self):
        return pd.DataFrame(self._properties, columns=self._columns)

    def properties(self,  index=None, label=None):
        index = self.__index(index, label)
        return self._properties[index]

    @property
    def label(self):
        return self._properties[:, self.__hash_col.get(CELL_IMAGE_PARAM.LABEL)]

    def labels(self, index=None, label=None):
        index = self.__index(index, label)
        return self.label[index]

    @property
    def center(self):
        colum = [self.__hash_col.get(i) for i in CELL_IMAGE_PARAM.CENTER_LIST]
        return self._properties[:, colum]

    def centers(self, index=None, label=None):
        index = self.__index(index, label)
        return self.center[index]

    @property
    def orientation(self):
        return self._properties[:, self.__hash_col.get(CELL_IMAGE_PARAM.ORIENTATION)]

    def orientations(self, index=None, label=None):
        index = self.__index(index, label)
        return self.orientation[index]

    @property
    def axis_major_length(self):
        return self._properties[:, self.__hash_col.get(CELL_IMAGE_PARAM.MAJOR_AXIS)]

    def axis_major_lengths(self, index=None, label=None):
        index = self.__index(index, label)
        return self.axis_major_length[index]

    @property
    def axis_minor_length(self):
        return self._properties[:, self.__hash_col.get(CELL_IMAGE_PARAM.MINOR_AXIS)]

    def axis_minor_lengths(self, index=None, label=None):
        index = self.__index(index, label)
        return self.axis_minor_length[index]

    @property
    def area(self):
        return self._properties[:, self.__hash_col.get(CELL_IMAGE_PARAM.AREA)]

    def areas(self, index=None, label=None):
        index = self.__index(index, label)
        return self.area[index]

    @property
    def bbox(self):
        colum = [self.__hash_col.get(i) for i in CELL_IMAGE_PARAM.BOUNDING_BOX_LIST]
        return self._properties[:, colum]

    def bboxs(self, index=None, label=None):
        index = self.__index(index, label)
        return self.bbox[index]

    @property
    def eccentricity(self):
        return self._properties[:, self.__hash_col.get(CELL_IMAGE_PARAM.ECCENTRICITY)]

    def eccentricities(self, index=None, label=None):
        index = self.__index(index, label)
        return self.eccentricity[index]

    @property
    def coordinate(self):
        return np.array(list(self._properties[:, self.__hash_col.get(CELL_IMAGE_PARAM.COORDINATE)]),
                        dtype=np.float_)

    def coordinates(self, index=None, label=None):
        index = self.__index(index, label)
        return self.coordinate[index]

    @property
    def semantic(self):
        return self._properties[:, self.__hash_col.get(CELL_IMAGE_PARAM.SEMANTIC_LABEL)]

    @semantic.setter
    def semantic(self, v):
        self._properties[:, self.__hash_col.get(CELL_IMAGE_PARAM.SEMANTIC_LABEL)] = v

    def semantics(self, index=None, label=None):
        index = self.__index(index, label)
        return self.semantic[index]

    @property
    def instance(self):
        return self._properties[:, self.__hash_col.get(CELL_IMAGE_PARAM.INSTANCE_LABEL)]

    def instances(self, index=None, label=None):
        index = self.__index(index, label)
        return self.instances[index]

    @property
    def is_border(self):
        return self._properties[:, self.__hash_col.get(CELL_IMAGE_PARAM.IS_BORDER)]

    def borders(self, index=None, label=None):
        index = self.__index(index, label)
        return self.is_border[index]

    # def features(self):

    # calculate distance
    def _init_cost_matrix(self):
        """Initialize the distance matrix as -1
        Return
        ----------
        cost: array_like, region * region * [center, nearneast,
        nearnest point index in x,  nearnest point index in y]
        """
        length = self._properties.shape[0]
        cost = np.zeros((length, length, 4), dtype=object)
        cost[:, :, 2:] = cost[:, :, 2:].astype(np.int_)
        cost[:, :, :] = -1
        self.__cost = cost

    def cost(self):
        return self.__cost

    def distance_idx(self, sources: Sequence[int], targets: Sequence[int]):
        """Given two regions' label, return 2 types distance between 2 regions.
        source & target should be index list
        """
        if self.__cost is None:
            self._init_cost_matrix()
        for index_x in sources:
            for index_y in targets:
                if self.__distance_exist(index_x, index_y):
                    continue
                else:
                    dist = self.__cal_two_regions_distance(index_x, index_y)
                    # dist order: center_dist, nearnest_dis, idx_tgt, idx_src
                    self.__cost[index_x, index_y, :] = dist
                    self.__cost[index_y, index_x, :] = [dist[0], dist[1],
                                                        dist[3], dist[2]]
        data = self.__cost[sources]
        data = data[:, targets]
        return data

    def __distance_exist(self, x, y) -> bool:
        if self.__cost is not None:
            if self.__cost[x, y, 0] > 0:
                return True
            else:
                return False
        else:
            return False

    def __cal_two_regions_distance(self, target: int, source: int):
        """Given two regions' label, return 2 types distance between 2 regions.
        Parameters
        ----------
        target :int, index of target point
        source :int, index of source point
        Notes
        ----------
        """
        nearnest_dis, idx_tgt, idx_src = find_nearest_points(
            self.coordinates(target),
            self.coordinates(source))
        center_dist = np.sqrt(np.sum(
            np.square(self.centers(target)-self.centers(source))))
        return [center_dist, nearnest_dis, idx_tgt, idx_src]

    def distance(self,
                 source: Union[int, Sequence[int]],
                 target: Union[int, Sequence[int]],
                 ptype="index"):
        """Return distance between source & target.
        Parameters
        ----------
        source: int or list, source point(s)' index or label (Mark with ptype)
        source: int or list, source target(s)' index or label (Mark with ptype)
        """
        source_index = self.__index_trans(source, ptype)
        target_index = self.__index_trans(target, ptype)
        if not isinstance(source_index, Iterable):
            if source_index is None:
                source_index = []
            else:
                source_index = [source_index]
        if not isinstance(target_index, Iterable):
            if target_index is None:
                target_index = []
            else:
                target_index = [target_index]
        return self.distance_idx(source_index, target_index)

    # two region relationship
    def between_angle(self, source: int, target: int, ptype="index"):
        """Return include angles between source & target instance based
        on nearest points and major axis.

        Parameters
        ----------
        source: int, index or label
        target: int, index or label

        Returns
        ----------
        target_angle: include angle in target
        source_angle: include angle in source
        """
        source_index = self.__index_trans(source, ptype)
        target_index = self.__index_trans(target, ptype)
        target_angle, source_angle = self.__two_regions_angle(source_index,
                                                              target_index)
        return target_angle, source_angle

    def between_angle_index(self, source: int, target: int, ptype="index"):
        """Return include angles between source & target instance based
        on nearest points and major axis.

        Parameters
        ----------
        source: int, index or label
        target: int, index or label

        Returns
        ----------
        target_angle: include angle in target
        source_angle: include angle in source
        """
        source_index = self.__index_trans(source, ptype)
        target_index = self.__index_trans(target, ptype)
        indexs = self.distance_idx([source_index], [target_index])[0, 0, 2:]
        # target_angle, source_angle = self.__two_regions_angle(source_index,
        #                                                       target_index)
        return indexs[0], indexs[1]

    def __two_regions_angle(self, target: int, source: int):
        """Use index to calculate the angles between two objects.
        Parameters
        ----------
        target: index
        source: index
        """
        target_point, source_point = self.__nearest_point(target, source)
        target_angle = _two_coordinate_point_angle(
            target_point,
            self.coordinates(target)[0],
            self.centers(target))
        source_angle = _two_coordinate_point_angle(
            source_point,
            self.coordinates(source)[0],
            self.centers(source))
        return target_angle, source_angle

    def nearest_point(self, source: int, target: int, ptype="index"):
        """Return the nearnest point of two objects.

        Parameters
        ----------
        source: int, angle or label
        target: int, angle or label

        Returns
        ----------
        target_point: the coordinate of nearnest point in target to source.
        source_point: the coordinate of nearnest point in source to target.
        """
        source_index = self.__index_trans(source, ptype)
        target_index = self.__index_trans(target, ptype)
        target_point, source_point = self.__nearest_point(
            source_index, target_index)
        return target_point, source_point

    def __nearest_point(self, target, source):
        """
        target: index
        source: index
        """
        indexs = self.distance_idx([target], [source])[0, 0, 2:]
        target_point = self.coordinates(index=target)[indexs[0]]
        source_point = self.coordinates(index=source)[indexs[1]]
        return target_point, source_point

    # neighbor nodes
    def neighbor(self, center: int,
                 targets: Union[int, Sequence[int]] = None,
                 ptype="index"):
        """Return first layer of neighbor for center object.

        Parameters
        ----------
        center: int, index or label
        targets: int or list, index or label for the target neighbor range.

        Returns
        ----------
        neibor: the Dataframe of neighbor regions.
        """
        center = self.__index_trans(center, ptype)
        if center is None:
            return None
        if targets is not None:
            targets = self.__index_trans(targets, ptype)
        neibor = self.__neighbor_node(center, targets)
        return neibor

    def __neighbor_node(self, center: int,
                        targets: Union[int, Sequence[int]] = None):
        """Return the first layer closed regions.
        center: index
        targets: index(s), set the region of neibor.
        """
        # 最近点的连线若有其他细胞，则不算第一层
        if targets is None:
            neiber = np.arange(self._properties.shape[0])
        else:
            neiber = targets
        selected = []
        for i in neiber:
            if i != center:
                near_points = self.__nearest_point(center, i)
                sample_points = create_line(near_points[0], near_points[1])
                sample_value = self.__array__()[sample_points[0],
                                                sample_points[1]]
                source_value = [self.labels(center),
                                self.labels(i), 0]
                flag = _isin_list(sample_value, source_value)
                if flag:
                    selected.append(i)
        return selected


def _isin_list(source: list, target: list):
    """Whether all the source list elements are in the target list,
    Parameters
    ----------
    source: list of source points
    target: list of target points

    Returns
    ----------
    False: not all points in the target, not the first layer neibor.
    True: all sample points in the target, should be the first layer neibor.
    """
    return len(set(source).difference(set(target))) <= 0


def _cal_is_border(bbox, shape):
    """
    """
    min_row = bbox[:, 0] == 0
    min_col = bbox[:, 1] == 0
    max_row = bbox[:, 2] == shape[0]
    max_col = bbox[:, 3] == shape[1]
    border = min_row | min_col | max_row | max_col
    return border


def _two_coordinate_point_angle(point,
                                target_point=np.array([1, 0]),
                                center=np.array([0, 0])):
    angle = angle_of_vectors(point - center,
                             target_point - center)
    angle = included_angle(angle)
    return angle


def hash_func(data):
    return {k: i for i, k in enumerate(data, 0)}
