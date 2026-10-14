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
"""Pick the true mating partner out of a reference cell's candidates.

Takes the per-(reference cell, candidate partner, time snapshot) table built
by `feature_extract.get_mating_feature` and does everything after that:
per-decision normalization, a group-aware train/test split, an SVC that
scores "is this candidate the true partner", and the top-k partner-pick
accuracy that's the metric that actually matters -- not row-level
accuracy, since a `decision_id` with 5 candidates and 1 true partner is
"guess the partner", not "classify this row".

Pulled out of the exploratory hpm/02_svm_prediction.ipynb notebook so it's
reusable and testable outside any one notebook; see that notebook (in the
Pipeline_Cell_Behavior_Prediction repo) for the end-to-end run on real data,
including the learning-curve datasize check built on `learning_curve_by_fusion_event`.

`decision_id` is only unique *within* one `get_mating_feature` call (one
image/CellNetwork) -- concatenate several images' tables and it collides
across them (on real data: 1503 distinct `decision_id` values cover 18730
actual (image, decision_id) decisions). Every grouped function here
resolves its `group_col` via `_resolve_group_col`, which defaults to
`["image", "decision_id"]` once an `image` column is present, specifically
so this isn't a silent footgun; pass `group_col="decision_id"` explicitly
only if you're deliberately working within a single image's table.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, f1_score


# Candidate-own features that only mean anything relative to the other
# candidates competing for the same decision (e.g. "area" is meaningless in
# isolation, but "largest of the 4 candidates" is a signal) -- min-max
# normalized within each decision_id by `normalize_candidates`.
NORMALIZE_COLUMNS = [
    'p_start', 'p_area', 'p_major', 'p_minor', 'p_eccentricity', 'p_neighbor_same', 'p_neighbor_diff',
    'm_start', 'm_area', 'm_major', 'm_minor', 'm_eccentricity', 'm_neighbor_same', 'm_neighbor_diff',
    'center_dist', 'nearest_dist', 'tip_distance',
]

# Full feature set the classifier trains on: raw shape/neighbor/angle/rank
# features plus the within-decision normalized versions above.
#
# `p_*`/`m_*` here are "the smaller-strain_type cell" / "the larger one" in
# a pair (see MATING_FEATURE_COLUMNS in _mating.py), not "reference" /
# "candidate" -- which one is the reference cell for a given decision
# depends on `ref_type`. For the shape/neighbor/start-time features below
# (area, major, minor, eccentricity, neighbor_same, neighbor_diff,
# *_star_time_rank), that means whichever side matches `ref_type` is the
# *same physical ref cell* repeated across every candidate row of that
# decision -- not a per-candidate measurement at all. Confirmed on real
# data: when ref_type==1 (ref is p), `p_area` is constant within
# 9275/9283 decisions and `p_star_time_rank` within all of them, while
# `m_area` (the actual candidate) varies as expected. So for any given row,
# half of these columns carry zero within-decision discriminative signal,
# and *which* half flips with `ref_type` -- with nothing in the feature
# vector itself telling the classifier which regime it's in. See
# `canonicalize_candidate_features` / `CANONICAL_FEATURE_COLUMNS` for the
# ref_type-agnostic alternative; kept here, unchanged, as the baseline it's
# compared against -- real numbers for both are in the hpm notebook, not
# repeated here since they go stale.
FEATURE_COLUMNS = [
    'p_area', 'p_major', 'p_minor', 'p_eccentricity', 'p_neighbor_same', 'p_neighbor_diff',
    'm_area', 'm_major', 'm_minor', 'm_eccentricity', 'm_neighbor_same', 'm_neighbor_diff',
    'p_angle', 'm_angle', 'p_angle_index', 'm_angle_index', 'p_angle_norm', 'm_angle_norm',
    'center_dist', 'nearest_dist', 'tip_distance', 'near_dist_rank', 'center_dist_rank', 'tip_dist_rank',
    'p_star_time_rank', 'm_star_time_rank', 'time_diff',
    'p_area_norm', 'p_major_norm', 'p_minor_norm', 'p_eccentricity_norm',
    'p_neighbor_same_norm', 'p_neighbor_diff_norm',
    'm_area_norm', 'm_major_norm', 'm_minor_norm', 'm_eccentricity_norm',
    'm_neighbor_same_norm', 'm_neighbor_diff_norm',
    'center_dist_norm', 'nearest_dist_norm', 'tip_distance_norm',
]

# Distance-only ablation: how much worse is a classifier that only sees
# geometry (no shape/neighbor/angle features)? Kept as a baseline to compare
# FEATURE_COLUMNS against, not because it's a model worth shipping on its own.
DISTANCE_ONLY_COLUMNS = [
    'center_dist', 'nearest_dist', 'tip_distance',
    'near_dist_rank', 'center_dist_rank', 'tip_dist_rank',
    'center_dist_norm', 'nearest_dist_norm', 'tip_distance_norm',
]

# name -> (p_column, m_column) for the single-cell features where exactly
# one side is "the reference cell's own measurement" (constant within a
# decision, see FEATURE_COLUMNS) and the other is "the candidate's own" --
# canonicalize_candidate_features keeps only the candidate side, as
# `candidate_<name>`.
_CANDIDATE_OWN_PAIRS = {
    "area": ("p_area", "m_area"), "major": ("p_major", "m_major"),
    "minor": ("p_minor", "m_minor"), "eccentricity": ("p_eccentricity", "m_eccentricity"),
    "neighbor_same": ("p_neighbor_same", "m_neighbor_same"),
    "neighbor_diff": ("p_neighbor_diff", "m_neighbor_diff"),
    "star_time_rank": ("p_star_time_rank", "m_star_time_rank"),
}

# name -> (p_column, m_column) for the pair-direction (angle) features,
# where *both* sides are informative -- each cell's own tip-direction
# toward the other -- but still p_*/m_* mislabeled the same way;
# canonicalize_candidate_features relabels (not drops) these, into
# `ref_<name>` / `candidate_<name>`.
_ANGLE_SIDE_PAIRS = {
    "angle": ("p_angle", "m_angle"), "angle_index": ("p_angle_index", "m_angle_index"),
    "angle_norm": ("p_angle_norm", "m_angle_norm"),
}


def canonicalize_candidate_features(data: pd.DataFrame, ref_type_col: str = "ref_type") -> pd.DataFrame:
    """Add a `ref_type`-agnostic view of the side-labeled columns: always
    "the candidate's own feature" (`candidate_<name>`, for `_CANDIDATE_OWN_PAIRS`)
    and "the reference's own" vs "the candidate's own" direction
    (`ref_<name>`/`candidate_<name>`, for `_ANGLE_SIDE_PAIRS`), regardless
    of whether the candidate happens to be p or m for that row. See
    `FEATURE_COLUMNS`'s docstring for why this distinction matters.

    Also adds `is_p_ref` (1 if the reference cell is p-type, 0 if m-type):
    unlike the measurement redundancy above, *which* strain is the
    reference isn't noise to throw away -- p and m cells are known to have
    different tip-formation/orientation bias, so this is kept as an
    explicit feature rather than only implicitly recoverable from which of
    `p_*`/`m_*` went into `candidate_*`/`ref_*` for a row.

    Original `p_*`/`m_*` columns are kept untouched; `ref_type not in (1, 2)`
    rows (rare -- 45/48875 on real data) fall back to treating the row as
    if `ref_type == 2`, same as `add_partner_columns`'s `partner_id`.
    """
    out = data.copy()
    is_p_ref = out[ref_type_col].eq(1)
    out["is_p_ref"] = is_p_ref.astype(int)
    for name, (p_col, m_col) in _CANDIDATE_OWN_PAIRS.items():
        out[f"candidate_{name}"] = np.where(is_p_ref, out[m_col], out[p_col])
    for name, (p_col, m_col) in _ANGLE_SIDE_PAIRS.items():
        out[f"ref_{name}"] = np.where(is_p_ref, out[p_col], out[m_col])
        out[f"candidate_{name}"] = np.where(is_p_ref, out[m_col], out[p_col])
    return out


# `canonicalize_candidate_features`'s candidate-own columns that are still
# only meaningful relative to the other candidates competing for the same
# decision -- pass to `normalize_candidates(columns=CANDIDATE_NORMALIZE_COLUMNS)`
# in place of `NORMALIZE_COLUMNS`. Excludes `candidate_star_time_rank`,
# already a rank (1..N_candidates), same as `NORMALIZE_COLUMNS` excludes
# the other rank columns.
CANDIDATE_NORMALIZE_COLUMNS = [
    "candidate_area", "candidate_major", "candidate_minor", "candidate_eccentricity",
    "candidate_neighbor_same", "candidate_neighbor_diff",
    "center_dist", "nearest_dist", "tip_distance",
]

# FEATURE_COLUMNS' ref_type-agnostic counterpart: every column means the
# same thing regardless of which side (p or m) happens to be the reference
# cell for that row, with no ref-own, zero-signal columns taking up space.
# `is_p_ref` keeps *which* side is the reference as an explicit feature
# (not redundant like the measurements above -- p/m have a known
# tip-formation/orientation bias asymmetry). Requires
# `canonicalize_candidate_features` then
# `normalize_candidates(columns=CANDIDATE_NORMALIZE_COLUMNS)` to have been
# run first.
CANONICAL_FEATURE_COLUMNS = [
    "is_p_ref",
    "candidate_area", "candidate_major", "candidate_minor", "candidate_eccentricity",
    "candidate_neighbor_same", "candidate_neighbor_diff", "candidate_star_time_rank",
    "ref_angle", "candidate_angle", "ref_angle_index", "candidate_angle_index",
    "ref_angle_norm", "candidate_angle_norm",
    "center_dist", "nearest_dist", "tip_distance", "near_dist_rank", "center_dist_rank", "tip_dist_rank",
    "time_diff",
    "candidate_area_norm", "candidate_major_norm", "candidate_minor_norm", "candidate_eccentricity_norm",
    "candidate_neighbor_same_norm", "candidate_neighbor_diff_norm",
    "center_dist_norm", "nearest_dist_norm", "tip_distance_norm",
]


def _resolve_group_col(data: pd.DataFrame, group_col):
    """`decision_id` (from `get_mating_feature`) is only unique *within* one
    call -- i.e. one image/CellNetwork; concatenate several images' tables
    (as the hpm pipeline's 01_generate_data.ipynb does, tagging each with an
    `image` column) and `decision_id` values collide across them, silently
    merging unrelated decisions into the same "group" for every function
    below. Default to `["image", "decision_id"]` once `image` is present so
    that mistake takes an explicit `group_col="decision_id"` override to
    make, rather than being the default.
    """
    if group_col is not None:
        return group_col
    return ["image", "decision_id"] if "image" in data.columns else "decision_id"


def _group_ids(data: pd.DataFrame, group_col) -> pd.Series:
    """Flat, row-aligned group label, usable with `.unique()`/`.isin()`
    (and `GroupShuffleSplit`, which rejects a multi-column `groups`) even
    when `group_col` is a list of columns.
    """
    if isinstance(group_col, (list, tuple)):
        return data.groupby(list(group_col)).ngroup()
    return data[group_col]


def _resolve_fusion_event_group_col(data: pd.DataFrame, group_col):
    """One (fusion event, ref_id) view is ~19 decisions on average (one per
    time snapshot, often just 10 frames apart) of the *same* reference cell
    choosing among the *same* kind of candidates -- highly correlated,
    nearly-duplicate examples. Grouping a train/test split (or a
    learning-curve subsample) by decision rather than by this unit lets
    snapshots of one cell's search land on both sides, which leaks
    near-duplicates across the split and inflates test accuracy relative to
    how the model would do on a genuinely new case never seen in any form.

    Defaults to `["image", "fusion_key", "ref_id"]` -- the p-view and the
    m-view of one event are two different cells' candidate-selection
    histories (different candidates, mostly different feature values), not
    duplicates of each other, so they're kept as separate splittable units
    rather than forced onto the same side. `fusion_key` is technically
    redundant once `ref_id` is included -- confirmed on real data, a given
    (image, ref_id) always has exactly one fusion_key, since a cell only
    fuses once -- but is kept in the key for clarity and as a defensive
    no-op if that ever isn't true for some dataset.

    Known, accepted tradeoff: at a decision's time_stamp shared between a
    (fusion_key, ref_id) group and its (fusion_key, other ref_id) partner
    group (confirmed common on real data: median 100%, mean 61% of
    time_stamps shared between the two views of an event), the
    true-partner row's pair-distance features (center_dist, nearest_dist,
    tip_distance -- the same physical measurement regardless of which cell
    is "ref") are numerically identical between the two groups' matching
    rows. Splitting those two groups across train/test leaks that one
    row's distance features -- narrower than the whole-decision-duplicate
    leakage this function exists to prevent (one row, a few features, not
    an entire near-duplicate decision), judged an acceptable tradeoff for
    the extra splittable granularity.

    Falls back to `_resolve_group_col`'s decision-level default when
    `fusion_key` or `ref_id` isn't in `data` (e.g. the synthetic fixtures
    in the test suite that don't model multiple events).
    """
    if group_col is not None:
        return group_col
    if "fusion_key" in data.columns and "ref_id" in data.columns:
        cols = ["fusion_key", "ref_id"]
        return ["image"] + cols if "image" in data.columns else cols
    return _resolve_group_col(data, None)


def normalize_candidates(data: pd.DataFrame, group_col=None,
                          columns=NORMALIZE_COLUMNS) -> pd.DataFrame:
    """Min-max normalize `columns` within each `group_col` group, appending
    `f"{column}_norm"` columns. Vectorized `groupby().transform()` equivalent
    of fitting a fresh `MinMaxScaler` per group -- including sklearn's
    zero-range behavior: a group where a column is constant gets 0 for that
    column's `_norm`, not NaN. `group_col` defaults per `_resolve_group_col`.

    Two things worth knowing about what that "_norm" value does and
    doesn't mean:
    - At a 1-candidate decision, every column hits the zero-range case and
      gets 0 regardless of its actual raw value -- a different "0" than
      the real, discriminating "smallest/closest of N" 0 at an N>1
      decision, folded into the same number. These decisions are also
      trivial for training: with one candidate, `get_mating_feature`'s
      one-true-partner-per-decision contract makes it deterministically
      the true partner, so there's no real discrimination to learn from
      it either way. See `drop_trivial_decisions`.
    - Min-max keeps whichever raw direction the data came in -- it does
      *not* reorient columns so that, say, 1 always means "closer" or
      "more likely the true partner". For an RBF `SVC` that's a non-issue
      (it can learn whatever relationship exists between a value and the
      label regardless of which direction the number happens to run), and
      imposing a "which direction is better" convention on a column whose
      true relationship to the label isn't already known from domain
      knowledge (unlike, say, the distance columns, where closer-is-more-
      likely is a real biological prior already reflected in the
      `*_rank` columns' unambiguous ascending-distance ordering) would be
      asserting an assumption, not discovering one.
    """
    group_col = _resolve_group_col(data, group_col)
    grouped = data.groupby(group_col)[columns]
    gmin = grouped.transform("min")
    gmax = grouped.transform("max")
    span = (gmax - gmin).replace(0, 1)
    normed = (data[columns] - gmin) / span
    normed.columns = [f"{c}_norm" for c in columns]
    return pd.concat([data, normed], axis=1)


def add_partner_columns(data: pd.DataFrame, group_cols=None) -> pd.DataFrame:
    """Add `partner_id` (the candidate on the other side of `ref_id`) and
    `N_candidates` (how many distinct partners `ref_id` was actually
    choosing among *at this specific decision* -- the thing that sets the
    random-guess baseline `top_k_hits`/`random_baseline_series` compare
    against, and what `min_candidates` there filters on).

    `group_cols` identifies one decision's candidates; defaults per
    `_resolve_group_col` (`["image", "decision_id"]` once `image` is
    present) -- i.e. one reference cell at one time snapshot, matching
    every other grouped function here.

    This used to default to `["image", "group_index", "ref_type", "ref_id"]`
    -- one whole fusion event's worth of candidates across *every* snapshot
    of its life, not one decision's. Confirmed on real data: the two
    disagree for 9618/18730 decisions (51%), always because the event-wide
    count is larger (candidates come and go as cells move/lose competence
    over the event's life) -- so the random baseline and the `N >= 3`
    filter used throughout this module had been comparing the model's
    per-decision pick against the wrong, inflated denominator.
    """
    group_cols = _resolve_group_col(data, group_cols)

    out = data.copy()
    out["partner_id"] = np.where(
        out["ref_type"].eq(1), out["m_id"],
        np.where(out["ref_type"].eq(2), out["p_id"], np.nan),
    )
    out["N_candidates"] = out.groupby(group_cols)["partner_id"].transform("nunique")
    return out


def split_by_fusion_event(data: pd.DataFrame, test_size: float = 0.3, random_state: int = 42,
                           group_col=None):
    """Train/test split grouped by fusion event, not by decision -- see
    `_resolve_fusion_event_group_col` for why a decision-level split leaks.
    `group_col` overrides the default (e.g. pass the `_resolve_group_col`
    decision-level grouping explicitly only if you specifically want that
    and accept the leakage it implies).
    """
    group_col = _resolve_fusion_event_group_col(data, group_col)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(data, groups=_group_ids(data, group_col)))
    return data.iloc[train_idx].copy(), data.iloc[test_idx].copy()


def drop_decisions_without_true_partner(data: pd.DataFrame, group_col=None,
                                         label_column: str = "flag"):
    """Remove every row belonging to a decision with zero `label_column`
    hits. `get_mating_feature` guarantees exactly one true partner per
    decision (see its docstring); zero signals an upstream tracking/
    labeling problem -- the true partner wasn't detected as a candidate at
    that snapshot -- not a legitimate "no match" outcome, so these rows are
    not safe to train on (every row in them is a guaranteed, spurious
    negative) any more than to evaluate on (`top_k_hits` already drops them
    there as a second line of defense if this step is skipped).

    Returns `(clean_data, n_dropped_decisions, n_dropped_rows)`.
    """
    group_col = _resolve_group_col(data, group_col)
    group_ids = _group_ids(data, group_col)
    has_true_partner = data.groupby(group_ids)[label_column].transform("sum") > 0
    n_dropped_rows = int((~has_true_partner).sum())
    n_dropped_decisions = int(group_ids[~has_true_partner].nunique())
    return data[has_true_partner].copy(), n_dropped_decisions, n_dropped_rows


def drop_trivial_decisions(data: pd.DataFrame, min_candidates: int = 2, group_col=None,
                            n_candidates_col: str = "N_candidates"):
    """Remove decisions with fewer than `min_candidates` candidates (default:
    drop 1-candidate decisions).

    At `N_candidates == 1` there's no real choice: that one candidate is
    deterministically the true partner (`get_mating_feature`'s one-true-
    partner-per-decision contract), so the row carries zero information
    about *discriminating* between competitors -- the thing this
    classifier exists to do. It's also where `normalize_candidates`'s
    min-max normalization is at its most degenerate (every `_norm` column
    forced to 0 regardless of the candidate's real value, conflated with
    the real "closest/smallest of N" 0 that same number means at N > 1 --
    see `normalize_candidates`'s docstring). Training on them risks the
    model partly learning the trivial shortcut "near-zero normalized
    features => True" instead of a genuine ranking rule. `top_k_hits`'s
    `min_candidates` already excludes the equivalent at evaluation; this
    is training's counterpart -- requires `N_candidates`
    (`add_partner_columns`).

    Returns `(clean_data, n_dropped_decisions, n_dropped_rows)`.
    """
    group_col = _resolve_group_col(data, group_col)
    group_ids = _group_ids(data, group_col)
    keep = data[n_candidates_col] >= min_candidates
    n_dropped_rows = int((~keep).sum())
    n_dropped_decisions = int(group_ids[~keep].nunique())
    return data[keep].copy(), n_dropped_decisions, n_dropped_rows


def train_partner_classifier(train_df: pd.DataFrame, feature_columns=FEATURE_COLUMNS,
                              label_column: str = "flag", shuffle: bool = True,
                              random_state: int = 0, **svc_kwargs):
    """Fit a `StandardScaler` + `SVC(probability=True)` pipeline on
    `feature_columns` to predict `label_column` (1 = true partner).

    `train_df` typically comes straight out of `split_by_fusion_event`, still in
    its original file order (grouped by image, then decision, then
    candidate) -- `SVC(probability=True)`'s internal Platt-scaling folds
    assume i.i.d. row order, and a block-sorted input can skew a fold's
    image/class mix. `shuffle` (default True) randomizes row order first;
    `random_state` seeds both that shuffle and the SVC. `svc_kwargs`
    override the SVC defaults (`gamma="auto"`).
    """
    if shuffle:
        train_df = train_df.sample(frac=1, random_state=random_state)
    X = train_df[feature_columns]
    y = train_df[label_column].astype(int)
    params = {"gamma": "auto", "probability": True, "random_state": random_state}
    params.update(svc_kwargs)
    clf = make_pipeline(StandardScaler(), SVC(**params))
    clf.fit(X, y)
    return clf


def score_candidates(clf, data: pd.DataFrame, feature_columns=FEATURE_COLUMNS) -> np.ndarray:
    """Predicted probability that each row's candidate is the true partner."""
    return clf.predict_proba(data[feature_columns])[:, 1]


def evaluate_row_level(clf, test_df: pd.DataFrame, feature_columns=FEATURE_COLUMNS,
                        label_column: str = "flag") -> dict:
    """Row-level accuracy/recall/f1 -- a sanity check on the classifier, not
    the partner-pick metric (see `top_k_accuracy` for that).
    """
    X_test = test_df[feature_columns]
    y_test = test_df[label_column].astype(int)
    pred = clf.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "recall": recall_score(y_test, pred, average="macro"),
        "f1": f1_score(y_test, pred, average="macro"),
    }


def top_k_hits(scored_df: pd.DataFrame, k: int = 1, group_col=None,
               score_col: str = "pred_prob", label_col: str = "flag",
               min_candidates: int = None, n_candidates_col: str = "N_candidates") -> pd.Series:
    """Per-decision 0/1: is the true partner among the top-`k` scored
    candidates? Indexed by `group_col` (defaults per `_resolve_group_col`);
    `top_k_accuracy` is this series' mean, and the per-decision values are
    what `bootstrap_ci` and `cellmate.visualize.plot_top_k_accuracy` want
    directly.

    `min_candidates` restricts to decisions with at least that many
    candidates (a harder, more informative subset -- with 2 candidates,
    top-1 is nearly guaranteed by construction).

    Decisions with zero true-partner rows are dropped: `get_mating_feature`
    guarantees exactly one true partner per decision, so zero signals an
    upstream tracking/labeling problem, not a legitimate outcome to average
    classifier performance over. Prefer catching these with
    `drop_decisions_without_true_partner` before training, not just here.
    """
    group_col = _resolve_group_col(scored_df, group_col)
    df = scored_df
    if min_candidates is not None:
        df = df[df[n_candidates_col] >= min_candidates]
    has_true_partner = df.groupby(group_col)[label_col].transform("sum") > 0
    df = df[has_true_partner]
    if df.empty:
        return pd.Series(dtype=float)
    topk = df.sort_values(score_col, ascending=False).groupby(group_col).head(k)
    return topk.groupby(group_col)[label_col].max()


def top_k_accuracy(scored_df: pd.DataFrame, k: int = 1, **kwargs) -> float:
    """Fraction of decisions where the true partner is among the top-`k`
    scored candidates -- the mean of `top_k_hits`; see there for `kwargs`.
    """
    hits = top_k_hits(scored_df, k=k, **kwargs)
    return hits.mean() if len(hits) else np.nan


def random_baseline_series(scored_df: pd.DataFrame, k: int = 1, group_col=None,
                            n_candidates_col: str = "N_candidates", min_candidates: int = None) -> pd.Series:
    """Per-decision top-k hit probability if the partner were picked
    uniformly at random among `N_candidates` (`k / N_candidates`, clipped to
    1) -- what `top_k_hits` on the same subset needs to beat. `group_col`
    defaults per `_resolve_group_col`.
    """
    group_col = _resolve_group_col(scored_df, group_col)
    df = scored_df
    if min_candidates is not None:
        df = df[df[n_candidates_col] >= min_candidates]
    n = df.drop_duplicates(group_col).set_index(group_col)[n_candidates_col]
    return np.minimum(k / n, 1.0)


def random_baseline_top_k(scored_df: pd.DataFrame, k: int = 1, **kwargs) -> float:
    """Expected top-k hit rate under random guessing -- the mean of
    `random_baseline_series`; see there for `kwargs`.
    """
    return random_baseline_series(scored_df, k=k, **kwargs).mean()


def top_k_accuracy_by_value(scored_df: pd.DataFrame, value_col: str, k: int = 1, group_col=None,
                             min_candidates: int = None, min_count: int = 1, **kwargs) -> pd.DataFrame:
    """Top-k accuracy (and the matching random baseline) broken out by each
    distinct value of `value_col` -- e.g. is a decision easier to call
    correctly the closer it is to fusion (`value_col="time_stamp"`)?

    Assumes one `value_col` value per decision (true for `time_stamp`:
    `get_mating_feature`'s `decision_id` is itself keyed by `time_stamp`, so
    it's constant within a decision's rows). `min_count` drops values backed
    by too few decisions to be a meaningful rate.

    Returns one row per `value_col` value: that value, `top_k_acc`,
    `random_baseline`, `n_decisions`; sorted by `value_col`.
    """
    group_col = _resolve_group_col(scored_df, group_col)
    hits = top_k_hits(scored_df, k=k, group_col=group_col, min_candidates=min_candidates, **kwargs)
    if hits.empty:
        return pd.DataFrame(columns=[value_col, "top_k_acc", "random_baseline", "n_decisions"])
    baseline = random_baseline_series(scored_df, k=k, group_col=group_col, min_candidates=min_candidates)
    value_per_group = scored_df.drop_duplicates(group_col).set_index(group_col)[value_col]
    combined = pd.DataFrame({
        "top_k_acc": hits,
        "random_baseline": baseline.reindex(hits.index),
        value_col: value_per_group.reindex(hits.index),
    })
    summary = (
        combined.groupby(value_col)
        .agg(top_k_acc=("top_k_acc", "mean"), random_baseline=("random_baseline", "mean"),
             n_decisions=("top_k_acc", "size"))
        .reset_index()
    )
    if min_count is not None:
        summary = summary[summary["n_decisions"] >= min_count]
    return summary.sort_values(value_col).reset_index(drop=True)


def pick_one_decision_per_view(scored_df: pd.DataFrame, strategy: str = "random",
                                group_col=None, eval_group_col=None,
                                value_col: str = "time_to_fusion", cutoff: float = None,
                                random_state: int = 0) -> pd.DataFrame:
    """Reduce each level-2 view (one reference cell's whole candidate search
    for one event) to exactly one decision, for a *per-view* accuracy
    metric -- `top_k_hits`/`top_k_accuracy` as normally used average over
    every decision (frame), so a view tracked for 50 frames counts 50x as
    much as one tracked for 2; this makes every view count once.

    `strategy`:
    - `"random"`: one decision chosen uniformly at random per view
      (default -- the unbiased headline number: "at a random moment in a
      cell's search, is the top pick correct").
    - `"closest_to_fusion"`: the decision with the smallest `value_col`
      (e.g. `time_to_fusion`) -- "right before fusion, do we get it
      right". Note this is also the easiest point per
      `top_k_accuracy_by_value`, so expect a higher number here than the
      other two strategies -- that's the point of comparing them.
    - `"cutoff"`: among a view's decisions with `value_col <= cutoff`, the
      one with the *largest* `value_col` -- the first decision
      (chronologically) once the view has come within `cutoff` of fusion.
      Views that never get within `cutoff` are dropped (can't honor the
      cutoff for them) -- requires `cutoff`.

    `group_col` is the level-2 (view) grouping (default per
    `_resolve_fusion_event_group_col`); `eval_group_col` is the level-3
    (decision) grouping picked *within* a view (default per
    `_resolve_group_col`).

    Returns the subset of `scored_df`'s rows belonging to the selected
    decisions (every candidate row of each, so `top_k_hits`/`top_k_accuracy`
    work on the result unchanged).
    """
    group_col = _resolve_fusion_event_group_col(scored_df, group_col)
    eval_group_col = _resolve_group_col(scored_df, eval_group_col)
    # `.loc[idxmin()/idxmax() result]` below needs a unique index to be
    # unambiguous -- `scored_df`'s own row index isn't guaranteed to be
    # (confirmed on real data: `fusion_data_table.csv`'s index is ~92%
    # duplicated, an unreset index surviving an upstream `pd.concat`), so
    # work on a reset copy throughout and return rows selected from it.
    data = scored_df.reset_index(drop=True)
    view_ids = _group_ids(data, group_col)
    decision_ids = _group_ids(data, eval_group_col)

    per_decision = data.assign(_view=view_ids, _decision=decision_ids).drop_duplicates("_decision")

    if strategy == "random":
        rng = np.random.default_rng(random_state)
        selected = per_decision.groupby("_view", group_keys=False).apply(
            lambda g: g.sample(n=1, random_state=rng))
    elif strategy == "closest_to_fusion":
        selected = per_decision.loc[per_decision.groupby("_view")[value_col].idxmin()]
    elif strategy == "cutoff":
        if cutoff is None:
            raise ValueError("strategy='cutoff' requires `cutoff`")
        eligible = per_decision[per_decision[value_col] <= cutoff]
        selected = eligible.loc[eligible.groupby("_view")[value_col].idxmax()] if not eligible.empty else eligible
    else:
        raise ValueError(f"unknown strategy {strategy!r}")

    selected_decisions = set(selected["_decision"])
    return data[decision_ids.isin(selected_decisions)].copy()


def true_partner_score_trajectory(scored_df: pd.DataFrame, group_col=None,
                                   value_col: str = "time_to_fusion",
                                   score_col: str = "pred_prob", label_col: str = "flag") -> pd.DataFrame:
    """One row per decision: the true partner's own predicted score at that
    decision, `value_col` (e.g. `time_to_fusion`), and the level-2 view it
    belongs to -- the individual-view complement to
    `top_k_accuracy_by_value`'s binned/averaged population view (does
    confidence in the correct candidate climb steadily for a given cell, or
    jump late, or never, rather than just the population's average shape).
    Feed to `cellmate.visualize.plot_score_trajectories`.

    Every decision has exactly one true-partner row once
    `drop_decisions_without_true_partner` has run (`get_mating_feature`'s
    one-true-partner-per-decision contract) -- this is just those rows,
    tagged with their view. Returns columns `view_id`, `value_col`,
    `score_col`, sorted by `(view_id, value_col)`.
    """
    group_col = _resolve_fusion_event_group_col(scored_df, group_col)
    # see pick_one_decision_per_view: scored_df's row index isn't
    # guaranteed unique, so cross-referencing by index label isn't safe --
    # reset first and align by boolean mask/position instead.
    data = scored_df.reset_index(drop=True)
    is_true = data[label_col].astype(bool)
    view_ids = _group_ids(data, group_col)
    out = pd.DataFrame({
        "view_id": view_ids[is_true].astype(str).to_numpy(),
        value_col: data.loc[is_true, value_col].to_numpy(),
        score_col: data.loc[is_true, score_col].to_numpy(),
    })
    return out.sort_values(["view_id", value_col]).reset_index(drop=True)


def bootstrap_ci(x, n_boot: int = 5000, seed: int = 0, ci=(2.5, 97.5)) -> np.ndarray:
    """Percentile bootstrap CI for the mean of `x` (e.g. a column of per-decision
    hits/misses -- `bootstrap_ci(topk.groupby("decision_id")["flag"].max())`).
    """
    x = np.asarray(x)
    rng = np.random.default_rng(seed)
    boots = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return np.percentile(boots, ci)


def learning_curve_by_fusion_event(train_df: pd.DataFrame, test_df: pd.DataFrame,
                                    fractions=(0.2, 0.4, 0.6, 0.8, 1.0), n_repeats: int = 5,
                                    feature_columns=FEATURE_COLUMNS, metric: str = "top_k",
                                    k_values=(1,), sample_group_col=None, eval_group_col=None,
                                    min_candidates: int = None, random_state: int = 0,
                                    **svc_kwargs) -> pd.DataFrame:
    """Does more labeled training data actually buy a better classifier, or
    has it plateaued? For each `fraction`, repeatedly subsample that many
    whole (fusion event, ref_id) views -- one reference cell's entire
    candidate search for one event (`sample_group_col`, default per
    `_resolve_fusion_event_group_col`) -- never individual decisions or
    rows -- retrain, and score on the *fixed* `test_df`.

    Subsampling at this granularity rather than by decision matters here
    beyond the leakage `split_by_fusion_event` guards against: real
    additional data arrives as whole new searches (each contributing ~19
    correlated time-snapshot decisions for free), so subsampling decisions
    *within* a retained search wouldn't track how many distinct
    cells'-worth of search the classifier has actually seen -- the
    quantity this curve is meant to answer "do we need more of".

    `metric`:
    - `"top_k"` (default): top-k partner-pick accuracy (`k_values`,
      `min_candidates`, grouped by decision -- `eval_group_col`, default
      per `_resolve_group_col`). One row per (fraction, repeat, k):
      `..., k, top_k_acc`. Plot with `cellmate.visualize.plot_learning_curve`.
    - `"row_level"`: plain classification accuracy/recall/f1
      (`evaluate_row_level`) -- cheaper, standard, and doesn't depend on
      `k`, but not the partner-pick metric that matters most (see this
      module's docstring). One row per (fraction, repeat): `..., accuracy,
      recall, f1`. Plot with `cellmate.visualize.plot_row_level_learning_curve`.

    Both share `fraction`, `repeat`, `n_train_events`, `n_train_decisions`.
    """
    if metric not in ("top_k", "row_level"):
        raise ValueError(f"metric must be 'top_k' or 'row_level', got {metric!r}")
    sample_group_col = _resolve_fusion_event_group_col(train_df, sample_group_col)
    eval_group_col = _resolve_group_col(train_df, eval_group_col)
    svc_random_state = svc_kwargs.pop("random_state", None)
    rng = np.random.default_rng(random_state)
    sample_ids = _group_ids(train_df, sample_group_col)
    unique_ids = sample_ids.unique()
    records = []
    for fraction in fractions:
        n_events = max(1, int(round(fraction * len(unique_ids))))
        for repeat in range(n_repeats):
            sampled_ids = rng.choice(unique_ids, size=n_events, replace=False)
            sub_train = train_df[sample_ids.isin(sampled_ids)]
            n_decisions = _group_ids(sub_train, eval_group_col).nunique()
            run_random_state = svc_random_state if svc_random_state is not None else random_state + repeat
            clf = train_partner_classifier(sub_train, feature_columns=feature_columns,
                                            random_state=run_random_state, **svc_kwargs)
            base = {"fraction": fraction, "repeat": repeat,
                    "n_train_events": n_events, "n_train_decisions": n_decisions}
            if metric == "row_level":
                records.append({**base, **evaluate_row_level(clf, test_df, feature_columns=feature_columns)})
            else:
                scored_test = test_df.copy()
                scored_test["pred_prob"] = score_candidates(clf, scored_test, feature_columns=feature_columns)
                for k in k_values:
                    acc = top_k_accuracy(scored_test, k=k, group_col=eval_group_col, min_candidates=min_candidates)
                    records.append({**base, "k": k, "top_k_acc": acc})
    return pd.DataFrame.from_records(records)
