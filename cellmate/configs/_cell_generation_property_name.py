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


class _CellGenerationParamName:
    GENERATION = "generation"
    # from division
    ANCIENT_VG = "ancient_vg"
    SISTER = "sister"
    # from fusion
    PARENTS = "parents"
    # to division
    DAUGHTER_VG = "daughter_vg"
    # to fusion
    SPOUSE = "spouse"
    DAUGHTER_SEX = "daughter_sex"


class _CellGenerationParamNameBase:
    LABEL = "label"
    DIVISION = "division"


class _GenerationTreeParam(_PropertyBase):
    def __init__(self) -> None:
        super().__init__()
        self.set_attr_items(_CellGenerationParamName)
        self.set_attr_items(_CellGenerationParamNameBase)

    def columns(self):
        data = get_attr_items(_CellGenerationParamName)
        data = list(data.values())
        return data[1:]


GENERATION_TREE = _GenerationTreeParam()
GENERATION_COLUMNS = GENERATION_TREE.columns()
CellGenParamMap = hash_func(GENERATION_COLUMNS)
