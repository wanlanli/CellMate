# Copyright 2026 wlli
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

import numpy as np
import pandas as pd


def get_mating_feature(cellnet):
    fusion_cells = cellnet.fusion_cells()
    fusion_data_table = None
    group_index = 0
    for key in fusion_cells:
        fcell = cellnet.cells[key]
        parents = fcell.parents
        print(parents)
        data = cellnet.potential_mating_feature(parents)
        if data is None:
            continue
        group_index += 1
        data["fusion_key"] = key
        data["fusion_time"] = fcell.start
        data["group_index"] = group_index
        data['near_dist_rank'] = data.groupby(['ref_type', 'time_stamp'])['nearest_dist'].rank(ascending=True, method='min').astype(int)
        data['center_dist_rank'] = data.groupby(['ref_type', 'time_stamp'])['center_dist'].rank(ascending=True, method='min').astype(int)
        data['p_star_time_rank'] = data.groupby(['ref_type', 'time_stamp'])['p_start'].rank(ascending=True, method='min').astype(int)
        data['m_star_time_rank'] = data.groupby(['ref_type', 'time_stamp'])['m_start'].rank(ascending=True, method='min').astype(int)
        data['tip_dist_rank'] = data.groupby(['ref_type', 'time_stamp'])['tip_distance'].rank(ascending=True, method='min').astype(int)
        data['time_diff'] = np.where(data['ref_type'] == 1, data['m_start'] - data['p_start'], data['p_start'] - data['m_start'])
        fusion_data_table = pd.concat([fusion_data_table, data])
    return fusion_data_table
