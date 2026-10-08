from typing import Sequence, Union
from collections.abc import Iterable
from functools import cached_property

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.graph import MCP_Geometric
from skimage.segmentation import expand_labels

from cellmate.configs import (IMAGE_MEASURE_PARAM, CELL_IMAGE_PARAM, DIVISION, CONTOURS_LENGTH, SKELETON_LENGTH,
                              SKELETON_ECC_THRESHOLD, NEIGHBOR_DISTANCE_UM, NEIGHBOR_DISTANCE_PX)
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
    def __init__(self, obj, pixel_size=1, sampling_interval=1, equidistant=False,
                 skeleton_ecc_threshold=SKELETON_ECC_THRESHOLD):
        self.data = obj
        self.pixel_size = pixel_size
        self.sampling_interval = sampling_interval
        self.equidistant = equidistant
        self.skeleton_ecc_threshold = skeleton_ecc_threshold
        self._columns = None
        self._properties = None
        self._init_instance_properties()
        #  create hash map for columns "name->index"
        self.__hash_col = hash_func(self._columns)
        #  create hash map for objects "label->index"
        self.__hash_obj = hash_func(self._properties[:, 0])
        self.set_pixel_size()
        self.__cost = self._init_cost_matrix()
        self.trees = self.init_trees()

    def __getstate__(self):
        # the Voronoi label images are full-size; rebuilt on demand
        state = self.__dict__.copy()
        state.pop("_voronoi_cache", None)
        return state

    def __index(self,
                index: Union[int, Sequence] = None,
                label: Union[int, Sequence] = None):
        """Inner function that retrieve rows by index or label arbitrarily.
        Note: Two parameters can and can only specify one of them.
        """
        if index is not None:
            if label is None:
                return self.__index_check(index)
            # both given: `index` wins
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
            return [i for i in index if i < self._properties.shape[0]]
        else:
            if index < self._properties.shape[0]:
                return index
            else:
                raise (ValueError("`index` not exist! %d" % index))

    def label2index(self, label: Union[int, Sequence]):
        """image label to arg
        """
        if isinstance(label, (int, np.integer)):
            return self.__hash_obj.get(int(label))
        else:
            return [self.__hash_obj.get(k) for k in label if self.__hash_obj.get(k) is not None]

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
                                           coord_length=CONTOURS_LENGTH,
                                           skeleton_ecc_threshold=self.skeleton_ecc_threshold)
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

    def set_pixel_size(self):
        if self.pixel_size != 1:
            self._properties[:, self.__hash_col.get(CELL_IMAGE_PARAM.AREA)] *= self.pixel_size**2
            self._properties[:, self.__hash_col.get(CELL_IMAGE_PARAM.MAJOR_AXIS)] *= self.pixel_size
            self._properties[:, self.__hash_col.get(CELL_IMAGE_PARAM.MINOR_AXIS)] *= self.pixel_size
            self._properties[:, self.__hash_col.get(CELL_IMAGE_PARAM.SKELETON_MAJOR_LENGTH)] *= self.pixel_size
            self._properties[:, self.__hash_col.get(CELL_IMAGE_PARAM.SKELETON_MINOR_LENGTH)] *= self.pixel_size
            self._properties[:, self.__hash_col.get(CELL_IMAGE_PARAM.SKELETON_GRID_LENGTH)] *= self.pixel_size

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
        colum = [self.__hash_col.get(i) for i in CELL_IMAGE_PARAM.CENTER_LIST]
        return self._properties[:, colum]

    def geometry_center(self, index=None, label=None):
        index = self.__index(index, label)
        return self.geometry_centers[index]

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
        column = [self.__hash_col.get(i) for i in CELL_IMAGE_PARAM.BOUNDING_BOX_LIST]
        return self._properties[:, column]

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
        # one cell's entry, without building the whole `coordinates` list
        return self._properties[index, self.__hash_col.get(CELL_IMAGE_PARAM.COORDINATE)]

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
        return self._properties[index, self.__hash_col.get(CELL_IMAGE_PARAM.SKELETON)]

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
        return [_skeleton_tips(s) for s in self.skeletons]

    def tip(self, index=None, label=None):
        index = self.__index(index, label)
        return _skeleton_tips(self.skeleton(index=index))

    def tip_index(self, index=None, label=None):
        index = self.__index(index, label)
        # nearest contour point to each tip, from the cell's prebuilt tree
        _, nearest = self.trees[index].topn(self.tip(index=index), top_n=1)
        return nearest

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
        return cost

    def cost(self):
        return self.__cost

    def distance_idx(self, sources: Sequence[int], targets: Sequence[int]):
        """Given two regions' label, return 2 types distance between 2 regions.
        source & target should be index list
        """
        if self.__cost is None:  # objects pickled before the matrix was built in __init__
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
        return self.__cost[x, y, 0] > 0

    def __cal_two_regions_distance(self, target: int, source: int):
        """Given two regions' label, return 2 types distance between 2 regions.
        Parameters
        ----------
        target :int, index of target object
        source :int, index of source object
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
    @property
    def neighbor_distance(self):
        """Default neighbor threshold in px: NEIGHBOR_DISTANCE_UM (12 um, about
        two cell lengths) converted with pixel_size, or NEIGHBOR_DISTANCE_PX
        (100 px) when pixel_size isn't set (1)."""
        if self.pixel_size and self.pixel_size != 1:
            return NEIGHBOR_DISTANCE_UM / self.pixel_size
        return NEIGHBOR_DISTANCE_PX

    def neighbor(self, center: int,
                 targets: Union[int, Sequence[int]] = None,
                 ptype="index", threshold=None, method="voronoi",
                 min_contact=1, line_margin=0.09):
        """Neighbors of `center` (see is_neighbor for the rule and parameters).

        center: int, index or label
        targets: int or list, index or label: only test these (default: all).
        threshold: max nearest distance in px; None: `neighbor_distance`.

        Returns indices (ptype="index") or labels (ptype="label") of neighbors.
        """
        threshold = self.neighbor_distance if threshold is None else threshold
        center = self.__index_trans(center, ptype)
        if center is None:
            return None
        if targets is None:
            targets = range(self._properties.shape[0])
        else:
            targets = self.__index_trans(targets, ptype)
            if not isinstance(targets, Iterable):
                targets = [targets]
        selected = [i for i in targets if i != center and np.isfinite(
            self.__neighbor_distance(center, i, threshold, method, min_contact, line_margin))]
        if ptype == "label":
            return self.labels[selected]
        return selected

    def voronoi(self, threshold):
        """Label image where every background pixel within threshold/2 of a
        region is given to its nearest region (skimage expand_labels). Used by
        the "voronoi" neighbor method: the gap between two first-layer cells
        belongs to those two cells, so a cell behind them can't reach through
        it, while cells facing each other across background meet halfway.
        Cached per threshold."""
        cache = self.__dict__.setdefault("_voronoi_cache", {})
        if threshold not in cache:
            cache[threshold] = expand_labels(self.data, distance=threshold / 2)
        return cache[threshold]

    def voronoi_contacts(self, threshold):
        """{(label_a, label_b): border length in pixels} for every pair of
        touching Voronoi regions (label_a < label_b). Cached per threshold."""
        cache = self.__dict__.setdefault("_contacts_cache", {})
        if threshold not in cache:
            cache[threshold] = _label_contacts(self.voronoi(threshold))
        return cache[threshold]

    def __neighbor_distance(self, obj1: int, obj2: int, threshold, method="voronoi",
                            min_contact=1, line_margin=0.09):
        """Reachable distance between two regions, np.inf if not neighbors.

        obj1: index 1
        obj2: index 2
        threshold: max straight-line nearest distance between the two contours.
        method:
            "voronoi": neighbors when their Voronoi regions (see `voronoi`)
                share at least `min_contact` border pixels, or (open space
                between them) when the straight line between the nearest
                points keeps more than `line_margin` x their distance away
                from every other region (a viewing cone: a narrow opening is
                fine at short range, not for a cell far behind it).
            "straight": neighbors when the straight line between the nearest
                points crosses only background (or the two regions).
        min_contact ("voronoi" only): min shared Voronoi border in pixels.
        line_margin ("voronoi" only): see method; None: Voronoi contact only.

        Returns the straight-line nearest distance when the line is clear;
        for "voronoi" when blocked, that distance scaled by the detour
        through the two Voronoi regions (dist * path / free).
        """
        if method not in ("voronoi", "straight"):
            raise ValueError(f"method must be 'voronoi' or 'straight', got {method!r}")
        # center dist, nearest dist, nearest point index in obj1, in obj2
        _, dist, k1, k2 = self.distance_idx([obj1], [obj2])[0, 0]
        if dist > threshold:
            return np.inf
        label1, label2 = self.label(obj1), self.label(obj2)
        lines = create_line(self.coordinate(index=obj1)[k1], self.coordinate(index=obj2)[k2])
        if method == "voronoi":
            key = (min(label1, label2), max(label1, label2))
            if self.voronoi_contacts(threshold).get(key, 0) < min_contact:
                if line_margin is not None and _line_margin(
                        self.data, lines, label1, label2, line_margin * dist) > line_margin * dist:
                    return dist
                return np.inf
        sample_value = self.data[lines[0], lines[1]]
        if _isin_list(sample_value, [0, label1, label2]):
            return dist
        if method == "voronoi":
            return dist * _voronoi_detour_ratio(self.data, self.voronoi(threshold), label1, label2)
        return np.inf

    def is_neighbor(self, obj1: int, obj2: int, threshold=None, ptype="index",
                    method="voronoi", min_contact=1, line_margin=0.09):
        """See __neighbor_distance for method / min_contact / line_margin;
        threshold defaults to `neighbor_distance`."""
        threshold = self.neighbor_distance if threshold is None else threshold
        obj1 = self.__index_trans(obj1, ptype)
        obj2 = self.__index_trans(obj2, ptype)
        return np.isfinite(self.__neighbor_distance(obj1, obj2, threshold, method, min_contact,
                                                    line_margin))

    def reachable_distance(self, obj1: int, obj2: int, threshold=None, ptype="index",
                           method="voronoi", min_contact=1, line_margin=0.09):
        """Nearest contour distance between two regions, accounting for cells
        in between: the straight-line distance when the line is clear, scaled
        by the detour when it is blocked. np.inf if not neighbors (see
        is_neighbor). For the plain straight-line distance use distance().
        """
        threshold = self.neighbor_distance if threshold is None else threshold
        obj1 = self.__index_trans(obj1, ptype)
        obj2 = self.__index_trans(obj2, ptype)
        return self.__neighbor_distance(obj1, obj2, threshold, method, min_contact, line_margin)

    def adjacent_matrix(self, threshold=None, method="voronoi", min_contact=1,
                        line_margin=0.09, return_distance=False):
        """
        threshold: if straight-line nearest distance (px) > threshold, not
            neighbor. None: `neighbor_distance`.
        method, min_contact, line_margin: see __neighbor_distance.
        return_distance: also return the reachable distance matrix (see
            __neighbor_distance), np.inf for non-neighbors and on the diagonal.
            Kept separate from the 0/1 matrix because touching cells can have
            distance 0, which would read as "no edge".
        """
        threshold = self.neighbor_distance if threshold is None else threshold
        length = len(self.labels)
        connected_matrix = np.zeros((length, length))
        distance_matrix = np.full((length, length), np.inf)
        if method == "voronoi" and line_margin is None:
            # only pairs whose Voronoi regions touch can be neighbors
            pairs = []
            for (label1, label2), contact in self.voronoi_contacts(threshold).items():
                i, j = self.label2index(label1), self.label2index(label2)
                if contact >= min_contact and i is not None and j is not None:
                    pairs.append((i, j))
        else:
            # Boxes around the contour coords themselves (the points the exact
            # distance uses), not the pixel bbox: the smoothed contour can
            # bulge outside the pixel bbox, which would make the pre-filter
            # below overestimate the gap and drop real neighbors.
            bboxes = _coord_bboxes(self.coordinates)
            # Cheap, exact lower bound first: if the two regions' bounding
            # boxes are already farther apart than `threshold` allows, the
            # real (expensive, per-pair nearest-neighbor search) check can
            # only agree they're not neighbors -- skip it.
            pairs = [(i, j) for i in range(length - 1) for j in range(i + 1, length)
                     if _bbox_min_distance(bboxes[i], bboxes[j]) <= threshold]
        for i, j in pairs:
            dist = self.__neighbor_distance(i, j, threshold, method, min_contact, line_margin)
            if np.isfinite(dist):
                connected_matrix[i, j] = 1
                connected_matrix[j, i] = 1
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
        if return_distance:
            return connected_matrix, distance_matrix
        return connected_matrix

    def nearest_coordinate(self,
                           source: Union[int, Sequence[int]],
                           points: np.array,
                           ptype="index"):
        """
        Calculates the nearest point and the distance from a source region to a set of target points.

        Parameters
        ----------
        source : int
            int or list, source point(s)' index or label (Mark with ptype).
        points : np.ndarray
            A 2D array where each row represents a point [x, y] in the target region.

        Returns
        -------
        list
            A list containing two elements:
            - nearnest_dis: The minimum distance between the source and the nearest target point.
            - idx_src: Index of the nearest point in the target region relative to the source.
        """
        source_index = self.__index_trans(source, ptype)
        nearnest_dis, idx_src = self.trees[source_index].topn(points)
        return [nearnest_dis, idx_src]

    def tips_distance(self, source: int, target: int, ptype="index"):
        source_index = self.__index_trans(source, ptype)
        target_index = self.__index_trans(target, ptype)
        tips_source = np.array(self.tip(source_index))
        tips_target = np.array(self.tip(target_index))

        dist_matrix = np.linalg.norm(tips_source[:, None, :] - tips_target[None, :, :], axis=2)
        # Minimum distance
        min_distance = dist_matrix.min()
        min_distance_arg = np.argmin(dist_matrix)
        return min_distance, tips_source[min_distance_arg//2], tips_target[min_distance_arg%2]


def _skeleton_tips(skeleton):
    """The two skeleton endpoints, or [None] without a skeleton."""
    if skeleton is None:
        return [None]
    return [skeleton[0], skeleton[-1]]


def _label_contacts(labels):
    """{(a, b): number of 4-connected pixel pairs where label a meets label
    b} over all pairs of different nonzero labels, a < b."""
    pairs = []
    for x, y in ((labels[:, :-1], labels[:, 1:]), (labels[:-1, :], labels[1:, :])):
        touch = (x != y) & (x > 0) & (y > 0)
        a, b = x[touch], y[touch]
        pairs.append(np.stack([np.minimum(a, b), np.maximum(a, b)], axis=1))
    pairs = np.concatenate(pairs)
    if len(pairs) == 0:
        return {}
    keys, counts = np.unique(pairs, axis=0, return_counts=True)
    return {(int(a), int(b)): int(n) for (a, b), n in zip(keys, counts)}


def _line_margin(data, line, label1, label2, max_margin):
    """Closest approach (px) of any region other than label1 / label2 to the
    pixels of `line`, capped at max_margin + 1 (only that range is searched)."""
    if len(line[0]) == 0:  # nearest points coincide: nothing in between
        return max_margin + 1
    pad = int(np.ceil(max_margin)) + 2
    r0, c0 = max(line[0].min() - pad, 0), max(line[1].min() - pad, 0)
    crop = data[r0:line[0].max() + pad + 1, c0:line[1].max() + pad + 1]
    third = (crop > 0) & (crop != label1) & (crop != label2)
    if not third.any():
        return max_margin + 1
    return min(ndi.distance_transform_edt(~third)[line[0] - r0, line[1] - c0].min(), max_margin + 1)


def _voronoi_detour_ratio(data, voronoi, label1, label2):
    """Shortest pixel path from region label1 to label2 staying inside their
    own Voronoi regions (so never squeezing past a third cell), divided by
    the unobstructed shortest pixel path (octile distance between the
    regions' nearest edge pixels). Both are 8-connected pixel paths, so grid
    discretization cancels out in the ratio. np.inf if no such path.
    """
    inside = (voronoi == label1) | (voronoi == label2)
    rows, cols = np.nonzero(inside)
    crop = (slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1))
    inside = inside[crop]
    mask1 = data[crop] == label1
    mask2 = data[crop] == label2
    eight = np.ones((3, 3), dtype=bool)
    # paths leave/enter a region through its edge pixels
    edge1 = mask1 & ~ndi.binary_erosion(mask1, structure=eight)
    edge2 = mask2 & ~ndi.binary_erosion(mask2, structure=eight)
    starts = np.argwhere(edge1)
    ends = np.argwhere(edge2)
    if len(starts) == 0 or len(ends) == 0:
        return np.inf
    gap = np.abs(starts[:, None, :] - ends[None, :, :])
    free = (np.abs(gap[..., 0] - gap[..., 1]) + np.sqrt(2) * gap.min(axis=2)).min()
    cumulative, _ = MCP_Geometric(np.where(inside, 1.0, np.inf)).find_costs(
        starts, ends, find_all_ends=False)
    return cumulative[edge2].min() / free


def _coord_bboxes(coords):
    """[min_row, min_col, max_row, max_col] of each region's coordinate set.
    Regions without coordinates get an infinite box, so the bbox pre-filter
    never skips them and the exact check decides.
    """
    bboxes = np.empty((len(coords), 4))
    for i, c in enumerate(coords):
        if c is None or len(c) == 0:
            bboxes[i] = [-np.inf, -np.inf, np.inf, np.inf]
        else:
            c = np.asarray(c)
            bboxes[i, :2] = c.min(axis=0)
            bboxes[i, 2:] = c.max(axis=0)
    return bboxes


def _bbox_min_distance(bbox1, bbox2):
    """Lower bound on the distance between ANY point in region 1 and ANY
    point in region 2, from their bounding boxes alone
    ([min_row, min_col, max_row, max_col]).

    The gap between the two boxes along one axis is 0 if they overlap on
    that axis, else the actual separation; combining both axes gives the
    closest the two boxes could possibly come. Since every point of a
    region lies within its own bounding box, the true nearest-point
    distance between the regions can never be smaller than this -- so if
    this already exceeds a neighbor threshold, the regions cannot be
    neighbors and the (much more expensive) exact nearest-point search can
    be skipped with no risk of a false negative.
    """
    row_gap = max(0.0, max(bbox1[0] - bbox2[2], bbox2[0] - bbox1[2]))
    col_gap = max(0.0, max(bbox1[1] - bbox2[3], bbox2[1] - bbox1[3]))
    return (row_gap ** 2 + col_gap ** 2) ** 0.5


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
