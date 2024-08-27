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

from ._cell_property_base import _PropertyBase, get_attr_items, hash_func


class _CellImageParamNameCommon:
    """
    _CellImageParamNameCommon is a centralized collection of standard parameter names
    used for attributes and measurements of cells extracted from images.
    These constants ensure consistency in naming conventions across the project.
    """
    LABEL = "label"
    ORIENTATION = "orientation"
    MAJOR_AXIS = "axis_major_length"
    MINOR_AXIS = "axis_minor_length"
    AREA = "area"
    ECCENTRICITY = "eccentricity"
    COORDINATE = "coords"
    SKELETON = "skeleton"
    MEDIAL_AXIS = "medial_axis_length"
    MEDIAL_MINOR_AXIS = "medial_minor_axis"
    MEDIAL_MINOR_LENGTH = "medial_minor_axis_length"


class _CellImageParamNameInput:
    """
    _CellImageParamNameInput stores the names of parameters that are provided as input
    for cell image processing, such as centroids and bounding boxes.
    """
    CENTER = "centroid"
    BOUNDING_BOX = "bbox"


class _CellImageParamNameOutput:
    """
    _CellImageParamNameOutput contains the names of parameters that are generated as output
    from cell image processing, such as lists of centroids and bounding boxes, and labels.
    """
    CENTER_LIST = ['centroid_0', 'centroid_1']
    BOUNDING_BOX_LIST = ['bbox_0', 'bbox_1', 'bbox_2', 'bbox_3']
    SEMANTIC_LABEL = "semantic"
    INSTANCE_LABEL = "instance"
    IS_BORDER = "is_out_of_border"


class _CellImageParam(_PropertyBase):
    def __init__(self) -> None:
        super().__init__()
        self.set_attr_items(_CellImageParamNameCommon)
        self.set_attr_items(_CellImageParamNameInput)
        self.set_attr_items(_CellImageParamNameOutput)

    def input(self):
        data = get_attr_items(_CellImageParamNameCommon)
        data.update(get_attr_items(_CellImageParamNameInput))
        data = self._flatten(list(data.values()))
        return data

    def output(self):
        data = get_attr_items(_CellImageParamNameInput)
        data.update(get_attr_items(_CellImageParamNameOutput))
        data = self._flatten(list(data.values()))
        return data

    def _flatten(self, data):
        data_flatten = []
        for x in data:
            if isinstance(x, list):
                data_flatten += x
            else:
                data_flatten.append(x)
        return data_flatten


CELL_IMAGE_PARAM = _CellImageParam()
IMAGE_MEASURE_PARAM = CELL_IMAGE_PARAM.input()
CellImgParamMap = hash_func(CELL_IMAGE_PARAM.output())
