from typing import Sequence, Union
from collections.abc import Iterable
from functools import cached_property

import numpy as np
import pandas as pd

from cellmate.configs import (IMAGE_MEASURE_PARAM, CELL_IMAGE_PARAM, DIVISION, CONTOURS_LENGTH, SKELETON_LENGTH)
from .measure._regionprops import regionprops_table
from .measure import CoordTree
from cellmate.utils import create_line, angle_of_vectors, included_angle, hash_func


class ImageMeasure():
    """Extract segemented regions' information from mask, åsuch as area,
    center, boundingbox ect. al..
    Parameters
    ----------
    input_array : 2D matrix,dtype:int
        mask is a int type 2d mask array. stored the labels of segementation.
    """
    def __init__(self, obj, pixel_size=1, sampling_interval=1, equidistant=False):
        self.data = obj
        self.pixel_size = pixel_size
        self.sampling_interval = sampling_interval
        self.equidistant = equidistant
        self._columns = None
        self._properties = None
        self._init_instance_properties()
        #  create hash map for columns "name->index"
        self.__hash_col = hash_func(self._columns)
        #  create hash map for objects "label->index"
        self.__hash_obj = hash_func(self._properties[:, 0])
        self.__cost = self._init_cost_matrix()
        self.trees = self.init_trees()

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
        props, columns = regionprops_table(self.data,  # self.__array__(),
                                           properties=IMAGE_MEASURE_PARAM,
                                           pixel_size=self.pixel_size,
                                           sampling_interval=self.sampling_interval,
                                           equidistant=self.equidistant,
                                           skeleton_length=SKELETON_LENGTH,
                                           coord_length=CONTOURS_LENGTH)
        props = props.T
        data = np.empty((props.shape[0], 3), dtype=np.int_)
        # semantic
        data[:, 0] = props[:, columns.index(CELL_IMAGE_PARAM.LABEL)] // DIVISION
        # instance
        data[:, 1] = props[:, columns.index(CELL_IMAGE_PARAM.LABEL)] % DIVISION
        # is_border
        col = [columns.index(i) for i in CELL_IMAGE_PARAM.BOUNDING_BOX_LIST]
        data[:, 2] = _cal_is_border(props[:, col], self.data.shape)
        columns += [CELL_IMAGE_PARAM.SEMANTIC_LABEL,
                    CELL_IMAGE_PARAM.INSTANCE_LABEL,
                    CELL_IMAGE_PARAM.IS_BORDER]
        self._columns = columns
        self._properties = np.concatenate((props, data), axis=1)
    # get properties

    def init_trees(self):
        trees = []
        for i in range(0, self._properties.shape[0]):
            trees.append(CoordTree(self.coordinate(index=i)))
        return trees

    @cached_property
    def property_table(self):
        return pd.DataFrame(self._properties, columns=self._columns)

    def properties(self,  index=None, label=None):
        index = self.__index(index, label)
        return self._properties[index]

    @property
    def labels(self):
        return self._properties[:, self.__hash_col.get(CELL_IMAGE_PARAM.LABEL)]

    def label(self, index=None, label=None):
        index = self.__index(index, label)
        return self.labels[index]

    @property
    def centers(self):
        return self._properties[:, self.__hash_col.get(CELL_IMAGE_PARAM.SKELETON_CENTER)]

    def center(self, index=None, label=None):
        index = self.__index(index, label)
        return self.centers[index]

    @property
    def geometry_centers(self):
        colum = [self.__hash_col.get(i) for i in CELL_IMAGE_PARAM.CENTER]
        return self._properties[:, colum]

    def geometry_center(self, index=None, label=None):
        index = self.__index(index, label)
        return self.centers[index]

    @property
    def orientations(self):
        return self._properties[:, self.__hash_col.get(CELL_IMAGE_PARAM.ORIENTATION)]

    def orientation(self, index=None, label=None):
        index = self.__index(index, label)
        return self.orientations[index]

    @property
    def axis_major_lengths(self):
        return self._properties[:, self.__hash_col.get(CELL_IMAGE_PARAM.MAJOR_AXIS)]

    def axis_major_length(self, index=None, label=None):
        index = self.__index(index, label)
        return self.axis_major_lengths[index]

    @property
    def axis_minor_lengths(self):
        return self._properties[:, self.__hash_col.get(CELL_IMAGE_PARAM.MINOR_AXIS)]

    def axis_minor_length(self, index=None, label=None):
        index = self.__index(index, label)
        return self.axis_minor_lengths[index]

    @property
    def areas(self):
        return self._properties[:, self.__hash_col.get(CELL_IMAGE_PARAM.AREA)]

    def area(self, index=None, label=None):
        index = self.__index(index, label)
        return self.areas[index]

    @property
    def bboxes(self):
        colum = [self.__hash_col.get(i) for i in CELL_IMAGE_PARAM.BOUNDING_BOX_LIST]
        return self._properties[:, colum]

    def bbox(self, index=None, label=None):
        index = self.__index(index, label)
        return self.bboxes[index]

    @property
    def eccentricities(self):
        return self._properties[:, self.__hash_col.get(CELL_IMAGE_PARAM.ECCENTRICITY)]

    def eccentricity(self, index=None, label=None):
        index = self.__index(index, label)
        return self.eccentricities[index]

    @property
    def coordinates(self):
        return list(self._properties[:, self.__hash_col.get(CELL_IMAGE_PARAM.COORDINATE)])

    def coordinate(self, index=None, label=None):
        index = self.__index(index, label)
        return self.coordinates[index]

    @property
    def skeleton_minor_grids(self):
        return self._properties[:, self.__hash_col.get(CELL_IMAGE_PARAM.SKELETON_MINOR_GRID)]

    def skeleton_minor_grid(self, index=None, label=None):
        index = self.__index(index, label)
        return self.skeleton_minor_grids[index]

    @property
    def skeletons(self):
        return list(self._properties[:, self.__hash_col.get(CELL_IMAGE_PARAM.SKELETON)])

    def skeleton(self, index=None, label=None):
        index = self.__index(index, label)
        return self.skeletons[index]

    @property
    def skeleton_lengths(self):
        return self._properties[:, self.__hash_col.get(CELL_IMAGE_PARAM.SKELETON_MAJOR_LENGTH)]

    def skeleton_length(self, index=None, label=None):
        index = self.__index(index, label)
        return self.skeleton_lengths[index]

    @property
    def medial_minor_lengths(self):
        return self._properties[:, self.__hash_col.get(CELL_IMAGE_PARAM.SKELETON_MINOR_LENGTH)]

    def medial_minor_length(self, index=None, label=None):
        index = self.__index(index, label)
        return self.medial_minor_lengths[index]

    @property
    def skeleton_minor_axises(self):
        return self._properties[:, self.__hash_col.get(CELL_IMAGE_PARAM.SKELETON_MINOR)]

    def skeleton_minor_axis(self, index=None, label=None):
        index = self.__index(index, label)
        return self.skeleton_minor_axises[index]

    @property
    def skeleton_grid_lengths(self):
        return self._properties[:, self.__hash_col.get(CELL_IMAGE_PARAM.SKELETON_GRID_LENGTH)]

    def skeleton_grid_length(self, index=None, label=None):
        index = self.__index(index, label)
        return self.skeleton_grid_lengths[index]

    @property
    def semantics(self):
        return self._properties[:, self.__hash_col.get(CELL_IMAGE_PARAM.SEMANTIC_LABEL)]

    @semantics.setter
    def semantics(self, v):
        self._properties[:, self.__hash_col.get(CELL_IMAGE_PARAM.SEMANTIC_LABEL)] = v

    def semantic(self, index=None, label=None):
        index = self.__index(index, label)
        return self.semantics[index]

    @property
    def instances(self):
        return self._properties[:, self.__hash_col.get(CELL_IMAGE_PARAM.INSTANCE_LABEL)]

    def instance(self, index=None, label=None):
        index = self.__index(index, label)
        return self.instances[index]

    @property
    def are_borders(self):
        return self._properties[:, self.__hash_col.get(CELL_IMAGE_PARAM.IS_BORDER)]

    def is_border(self, index=None, label=None):
        index = self.__index(index, label)
        return self.are_borders[index]

    @property
    def tips(self):
        tips = []
        for s in self.skeletons:
            if s is None:
                tips.append([None])
            else:
                tips.append([s[0], s[-1]])
        return tips

    def tip(self, index=None, label=None):
        index = self.__index(index, label)
        return self.tips[index]

    def tip_index(self, index=None, label=None):
        tips = self.tip(index, label)
        coord = self.coordinate(index, label)
        _, index = CoordTree(coord, top_n=1).topn(tips)
        return index

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
        # nearnest_dis, idx_tgt, idx_src = find_nearest_points(
        #     self.coordinate(target),
        #     self.coordinate(source))
        nearnest_dis, idx_src = self.trees[source].topn(self.coordinate(target))
        idx_tgt = np.argmin(nearnest_dis)
        idx_src = idx_src[idx_tgt][0]
        center_dist = np.sqrt(np.sum(
            np.square(self.center(target)-self.center(source))))
        return [center_dist, nearnest_dis[idx_tgt][0], idx_tgt, idx_src]

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
        source_angle, target_angle = self.__two_regions_angle(source_index,
                                                              target_index)
        return source_angle, target_angle

    def between_angle_index(self, source: int, target: int, ptype="index", norm=True):
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
        indexes = self.distance_idx([source_index], [target_index])[0, 0, 2:]
        # target_angle, source_angle = self.__two_regions_angle(source_index,
        #                                                       target_index)
        source_tips = self.tip_index(index=source_index)
        target_tips = self.tip_index(index=target_index)
        source_angle_index = tips_distance_index(indexes[0], source_tips, length=CONTOURS_LENGTH, norm=norm)
        target_angle_index = tips_distance_index(indexes[1], target_tips, length=CONTOURS_LENGTH, norm=norm)
        return source_angle_index, target_angle_index

    def __two_regions_angle(self, region_0: int, region_1: int):
        """Use index to calculate the angles between two objects.
        Parameters
        ----------
        target: index
        source: index
        """
        region_0_point, region_1_point = self.__nearest_point(region_0, region_1)
        region_0_angle = point2tips_angle(
            region_0_point,
            self.tip(region_0),
            self.center(region_0))
        region_1_angle = point2tips_angle(
            region_1_point,
            self.tip(region_1),
            self.center(region_1))
        return region_0_angle, region_1_angle

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
        target_point = self.coordinate(index=target)[indexs[0]]
        source_point = self.coordinate(index=source)[indexs[1]]
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
        if ptype == "label":
            neibor = self.labels[neibor]
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
                sample_value = self.data[sample_points[0],
                                                sample_points[1]]
                source_value = [self.label(center),
                                self.label(i), 0]
                flag = _isin_list(sample_value, source_value)
                if flag:
                    selected.append(i)
        return selected

    def __is_neighbor(self, obj1: int, obj2: int, threshold):
        """
        obj1: index 1
        obj2: index 2
        """
        if self.distance_idx([obj1], [obj2])[0, 0, 1] > threshold:
            return False
        else:
            p1, p2 = self.__nearest_point(obj1, obj2)
            lines = create_line(p1, p2)
            sample_value = self.data[lines[0], lines[1]]
            flag = _isin_list(sample_value, [0, self.label(obj1), self.label(obj2)])
            return flag

    def is_neighbor(self, obj1: int, obj2: int, threshold: int = 100, ptype="index"):
        obj1 = self.__index_trans(obj1, ptype)
        obj2 = self.__index_trans(obj2, ptype)
        return self.__is_neighbor(obj1, obj2, threshold)

    def adjacent_matrix(self, threshold: int = 100):
        """
        threshold: if > threshold, not neighbor
        """
        length = len(self.labels)
        connected_matrix = np.zeros((length, length))
        for i in range(0, length-1):
            for j in range(i+1, length):
                if self.__is_neighbor(i, j, threshold):
                    connected_matrix[i, j] = 1
                    connected_matrix[j, i] = 1
                else:
                    continue
        return connected_matrix


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
    Determines if any part of the bounding box touches the border of an image.

    Parameters:
    -----------
    bbox : numpy.ndarray
        A 2D array of shape (n, 4), where each row represents a bounding box as [min_row, min_col, max_row, max_col].

    shape : tuple of int
        A tuple representing the shape of the image as (height, width).

    Returns:
    --------
    numpy.ndarray
        A boolean array of length `n` where each element is True if the corresponding bounding box
        touches the image border, and False otherwise.
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
    """
    Calculates the angle formed by the vector from 'center' to 'point' and the vector
    from 'center' to 'target_point'.

    Parameters:
    -----------
    point : numpy.ndarray
        A 1D array representing the (x, y) coordinates of the point.

    target_point : numpy.ndarray, optional
        A 1D array representing the (x, y) coordinates of the target point (default is [1, 0]).

    center : numpy.ndarray, optional
        A 1D array representing the (x, y) coordinates of the center point (default is [0, 0]).

    Returns:
    --------
    float
        The included angle (in degrees or radians, depending on your `included_angle` function) between
        the vector from 'center' to 'point' and the vector from 'center' to 'target_point'.
    """
    angle = angle_of_vectors(point - center,
                             target_point - center)
    angle = included_angle(angle)
    return angle


def point2tips_angle(point, tips, center):
    """
    Calculates the angle between a given point, the closest tip from a set of tips, through a center.

    Parameters:
    -----------
    point : numpy.ndarray
        A 1D array representing the (x, y) coordinates of the point.

    tips : numpy.ndarray
        A 2D array of shape (n, 2), where each row represents the (x, y) coordinates of a tip.

    center : numpy.ndarray
        A 1D array representing the (x, y) coordinates of the center point.

    Returns:
    --------
    float
        The angle (in degrees or radians depending on your angle calculation function) between the point,
        the closest tip, and the center.
    """
    index = np.argmin(np.sqrt(np.square(point - tips).sum(axis=1)))
    angle = _two_coordinate_point_angle(point, tips[index], center)
    return angle


def tips_distance_index(point_index, tips_index, length, norm=True):
    """
    Calculates the index distance from a given point to the nearest tip in a list of points.

    Parameters:
    -----------
    point_index : int
        The index of the point for which the distance to the nearest tip is being calculated.

    tips_index : tuple of int
        A tuple containing the indices of the two tips, where `tips_index[0]` should be 0 
        (the start) and `tips_index[1]` is the index of the second tip.

    length : int
        The total length of the path (i.e., the number of points in the path).

    norm : bool, optional
        Whether to normalize the distance by the total path length. Default is True.

    Returns:
    --------
    float or int
        The distance from `point_index` to the nearest tip. If `norm` is True, the distance 
        is normalized to a range between 0 and 1.
    """
    if tips_index[0] != 0:
        print("Warning: coordinate not start from tip")
    if point_index > tips_index[1]:
        index_diff = int(min(point_index - tips_index[1], length - point_index))
        if norm:
            index_diff = float(index_diff/(length-tips_index[1]))
    else:
        index_diff = int(min(point_index, tips_index[1]-point_index))
        if norm:
            index_diff = float(index_diff/tips_index[1])
    return index_diff
