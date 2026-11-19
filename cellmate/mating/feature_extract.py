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
    'tip_distance': 'tip_dist_rank',
    'candidate_start': 'candidate_start_rank',
    'candidate_area': 'candidate_area_rank',
    
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


# Rank features computed within one decision_id group (get_mating_feature_v2's
# equivalent of RANK_COLUMNS). Only candidate_* columns are ranked, unlike
# RANK_COLUMNS's p_start/m_start pair -- focal_start is the same physical
# reference cell on every row of a decision, so ranking it is always a
# constant rank 1 with zero discriminative signal (see FEATURE_COLUMNS's
# docstring in _svm_prediction.py for the real-data numbers behind that,
# measured on the old p_*/m_* columns this avoids needing in the first
# place). ascending=True: smallest raw value (closest candidate, earliest
# start) -> rank 1.
RANK_COLUMNS_V2 = {
    'nearest_dist': 'nearest_dist_rank',
    'center_dist': 'center_dist_rank',
    'tip_distance': 'tip_distance_rank',
    'candidate_start': 'candidate_start_rank',
}


def get_mating_feature_v2(cellnet, time_step: int = 10, image=None):
    """`get_mating_feature`, rebuilt around a clearer key hierarchy and a
    pixel_size fix -- a parallel implementation, not a replacement:
    `get_mating_feature` is kept as is (every existing caller --
    `_svm_prediction.py`, `_rank_prediction.py`, the hpm notebooks -- keys
    off its p_*/m_* columns).

    Two problems this fixes:
    - pixel_size wasn't applied uniformly. `ImageMeasure.set_pixel_size`
      already converts the per-cell shape features (area/major/minor) to
      physical units, but `pair_feature`'s center_dist/nearest_dist/
      contact_length/tip_distance are raw pixel distances -- `flag`-level
      geometry stayed in a different unit than the shape it's being
      compared against. Fixed in `CellNetwork.focal_candidate_feature`
      (called through `potential_mating_feature_v2`); the `pixel_size`
      actually used is kept as its own column for traceability.
    - the row/column structure needed a p/m -> reference/candidate
      canonicalization pass downstream (`canonicalize_candidate_features`)
      purely because `pair_feature` only knows "smaller strain_type" vs
      "larger", not which side is the reference for a given row. Rows here
      are already labeled focal_id/candidate_id, so every "focal_*"/
      "candidate_*" column means the same thing regardless of strain type
      -- no separate canonicalization step needed.

    Key hierarchy (each level narrows the one above -- "image, fusion
    event, p/m focal, frame" are the four things that together locate any
    row):
    - image: one CellNetwork/movie. `CellNetwork` has no notion of this on
      its own; pass `image` to tag every row of this call with it, the
      same convention `_svm_prediction`/`_rank_prediction` already use for
      concatenating several images' tables. Omitted by default -- no
      `"image"` column at all, rather than a column full of `None`.
    - fusion_id: one fusion event -- one fused cell, i.e. one pair of real
      parents and every candidate either of them ever considered.
    - view_id: one (fusion event, focal cell) -- one parent's whole
      candidate search over its life, kept apart from its mate's own
      search (a different focal cell, a different, opposite-strain set of
      candidates).
    - decision_id: one (view, time snapshot) -- the actual set of
      candidates a classifier picks among at one instant. Same one-true-
      partner-per-decision contract as `get_mating_feature`'s own
      `decision_id` (see its docstring for what a violation means).

    Columns are grouped in this order: key columns (image, fusion_id,
    focal_id, focal_type, time_stamp, view_id, decision_id, candidate_id);
    focal_* features (the reference cell's own measurements -- constant
    within a decision); candidate_* features (vary per candidate row);
    pair features (one physical quantity between the two, order-
    independent: distances, angle difference, contact length, tip checks,
    time_diff); then everything else (flag, fusion_time, pixel_size,
    candidate counts, ranks).

    `n_candidates_frame` / `n_candidates_view` answer "how many candidates
    was this search choosing among" at the two grains `get_mating_feature`
    needed two separate downstream functions for
    (`_svm_prediction.add_partner_columns`'s `N_candidates`,
    `_rank_prediction.add_total_candidates`) -- both included directly
    here, `n_candidates_frame` at decision_id grain and `n_candidates_view`
    at view_id grain. Rank features (RANK_COLUMNS_V2) are within
    decision_id only, same reasoning as `get_mating_feature`'s
    RANK_COLUMNS -- comparing a candidate across every frame of a view is
    a different question, answered by `select_view_candidates` +
    `add_group_rank_pct` instead, not just a wider version of this rank.
    """
    fusion_cells = cellnet.fusion_cells()
    frames = []
    for key in fusion_cells:
        fcell = cellnet.cells[key]
        parents = list(fcell.parents)
        data = cellnet.potential_mating_feature_v2(parents, time_step=time_step)
        if data is None:
            continue
        data["fusion_id"] = key
        data["fusion_time"] = fcell.start
        frames.append(data)
    if not frames:
        return None
    data = pd.concat(frames, ignore_index=True)
    if image is not None:
        data["image"] = image

    data["view_id"] = data.groupby(["fusion_id", "focal_id"]).ngroup()
    data["decision_id"] = data.groupby(["fusion_id", "focal_id", "time_stamp"]).ngroup()

    data["n_candidates_frame"] = data.groupby("decision_id")["candidate_id"].transform("nunique")
    data["n_candidates_view"] = data.groupby("view_id")["candidate_id"].transform("nunique")

    # view_first_frame: every row of a view's earliest decision (the frame
    # `select_view_candidates`/`split_by_fusion_event` would otherwise have
    # to re-derive by grouping again). candidate_first_frame: the one row
    # per (view_id, candidate_id) marking that candidate's own earliest
    # appearance within the view -- same selection `select_view_candidates`
    # (strategy="first") makes, kept as a flag instead of a filtered copy
    # so both grains are queryable without recomputation.
    data["view_first_frame"] = data["time_stamp"] == data.groupby("view_id")["time_stamp"].transform("min")
    data["candidate_first_frame"] = data["time_stamp"] == data.groupby(
        ["view_id", "candidate_id"])["time_stamp"].transform("min")

    ranks = data.groupby("decision_id")[list(RANK_COLUMNS_V2)].rank(ascending=True, method="min")
    data[list(RANK_COLUMNS_V2.values())] = ranks.rename(columns=RANK_COLUMNS_V2).astype("Int64")

    data["time_diff"] = data["candidate_start"] - data["focal_start"]

    key_cols = (["image"] if image is not None else []) + [
        "fusion_id", "focal_id", "focal_type", "time_stamp", "view_id", "decision_id", "candidate_id"]
    focal_cols = [
        "focal_start", "focal_area", "focal_major", "focal_minor", "focal_eccentricity",
        "focal_neighbor_same", "focal_neighbor_diff", "focal_angle", "focal_angle_index", "focal_angle_norm"]
    candidate_cols = [
        "candidate_start", "candidate_area", "candidate_major", "candidate_minor", "candidate_eccentricity",
        "candidate_neighbor_same", "candidate_neighbor_diff",
        "candidate_angle", "candidate_angle_index", "candidate_angle_norm"]
    pair_cols = [
        "center_dist", "nearest_dist", "orientation_diff", "contact_length",
        "tip_distance", "focal_in_new_tip", "candidate_in_new_tip", "time_diff"]
    rest_cols = (["flag", "fusion_time", "pixel_size", "n_candidates_frame", "n_candidates_view",
                  "view_first_frame", "candidate_first_frame"]
                 + list(RANK_COLUMNS_V2.values()))

    return data[key_cols + focal_cols + candidate_cols + pair_cols + rest_cols]


def select_view_candidates(data: pd.DataFrame, strategy: str = "first", random_state: int = 0) -> pd.DataFrame:
    """One row per distinct candidate per view (`view_id`) -- the view-level
    counterpart to `get_mating_feature_v2`'s per-decision (per-frame) rows.
    Lets a focal cell's candidates be compared/normalized against each
    other over its *whole* search instead of one snapshot at a time (pass
    the result to `add_group_rank_pct(..., group_col="view_id")`), the
    same "within group" idea `get_mating_feature_v2`'s own decision-level
    ranks use, one level up.

    `strategy`: which of a candidate's own rows (one per frame it was a
    candidate within this view) represents it here --
    - `"first"` (default): its earliest appearance (smallest `time_stamp`).
    - `"random"`: uniform over its own frames, seeded by `random_state`.
    """
    if strategy not in ("first", "random"):
        raise ValueError(f"strategy must be 'first' or 'random', got {strategy!r}")
    keyed = data.sort_values("time_stamp")
    if strategy == "first":
        return keyed.groupby(["view_id", "candidate_id"], as_index=False).first()
    rng = np.random.default_rng(random_state)
    return (keyed.groupby(["view_id", "candidate_id"], group_keys=False)
                 .apply(lambda g: g.sample(n=1, random_state=rng)))


def add_group_rank_pct(data: pd.DataFrame, columns, group_col, ascending: bool = True) -> pd.DataFrame:
    """Add `f"{col}_rank_pct"` (0 = best-ranked by `ascending`, 1 =
    worst-ranked, scaled to use the full 0..1 range regardless of group
    size) for every `columns`, within each `group_col` group. A 1-member
    group is pinned to 0 (the `(rank - 1) / (N - 1)` scaling would divide
    by zero there, and there's nothing to rank against anyway).

    `group_col` is a plain column name here, not a decision_id/view_id
    convention -- call with `group_col="decision_id"` for the same
    within-frame comparison `get_mating_feature_v2`'s own RANK_COLUMNS_V2
    does, or `group_col="view_id"` on `select_view_candidates`'s output
    for the within-view comparison across every frame of a view.
    """
    out = data.copy()
    columns = list(columns)
    ranks = out.groupby(group_col)[columns].rank(ascending=ascending, method="min")
    sizes = out.groupby(group_col)[columns[0]].transform("size")
    denom = (sizes - 1).where(sizes > 1, 1)
    for col in columns:
        pct = (ranks[col] - 1) / denom
        out[f"{col}_rank_pct"] = pct.where(sizes > 1, 0.0)
    return out
