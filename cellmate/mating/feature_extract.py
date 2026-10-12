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


RANK_COLUMNS = {
    'nearest_dist': 'near_dist_rank',
    'center_dist': 'center_dist_rank',
    'p_start': 'p_star_time_rank',
    'm_start': 'm_star_time_rank',
    'tip_distance': 'tip_dist_rank',
}


def get_mating_feature(cellnet, time_step: int = 10):
    """One row per (reference parent, candidate partner, time snapshot).

    Two id columns tie rows together for downstream grouping/normalization:
    - group_index / fusion_key: one mating event (one fused cell, i.e. one
      pair of real parents and every candidate either of them considered).
    - decision_id: one classification decision -- a single reference cell
      at a single time_stamp, together with every candidate partner it had
      then. Exactly one row per decision_id should have flag == True (the
      real eventual partner, labeled True at every snapshot where it's
      already a neighbor -- even well before fusion); zero True rows in a
      decision_id's group signals a tracking/classification problem
      upstream, not a legitimate outcome. Rank features (RANK_COLUMNS) are
      computed within decision_id, since that's the actual 1-true-vs-n
      candidate set a classifier chooses among.
    """
    fusion_cells = cellnet.fusion_cells()
    frames = []
    group_index = 0
    for key in fusion_cells:
        fcell = cellnet.cells[key]
        # copy: potential_mating_feature reorders parents in place
        parents = list(fcell.parents)
        data = cellnet.potential_mating_feature(parents, time_step=time_step)
        if data is None:
            continue
        group_index += 1
        data["fusion_key"] = key
        data["fusion_time"] = fcell.start
        data["group_index"] = group_index
        frames.append(data)
    if not frames:
        return None
    data = pd.concat(frames, ignore_index=True)

    # decision_id: unique per (mating event, reference cell, time snapshot)
    # -- the atomic group ranks and future per-group normalization should
    # use, rather than re-deriving it from ref_type/time_stamp each time.
    data["decision_id"] = data.groupby(["fusion_key", "ref_id", "time_stamp"]).ngroup()

    ranks = data.groupby("decision_id")[list(RANK_COLUMNS)].rank(ascending=True, method="min")
    data[list(RANK_COLUMNS.values())] = ranks.rename(columns=RANK_COLUMNS).astype("Int64")
    # time_diff = partner start - ref start
    data["time_diff"] = np.where(data["ref_id"] == data["p_id"], data["m_start"] - data["p_start"], data["p_start"] - data["m_start"])
    return data
