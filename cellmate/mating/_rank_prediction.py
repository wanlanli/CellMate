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
"""V2 feature/model experiments for the partner-pick classifier, built on
top of `_svm_prediction.py` rather than replacing it (that module's
constants/behavior stay exactly reproducible for the existing
hpm/02_svm_prediction.ipynb notebook and its results).

Three things this adds:
- `add_rank_pct_features`: percentile-rank columns that stay meaningful at
  a 1-candidate decision, unlike `normalize_candidates`'s min-max `_norm`
  (every `_norm` column is forced to 0 there regardless of the candidate's
  real value -- see that function's docstring). Also covers
  `orientation_diff`/`contact_length`, two pair-level geometry columns
  that exist in `get_mating_feature`'s raw output but were never added to
  `FEATURE_COLUMNS`/`CANONICAL_FEATURE_COLUMNS` -- ref_type-agnostic
  already (computed symmetrically between the two cells, like
  `center_dist`/`nearest_dist`/`tip_distance`), so no p/m-side
  canonicalization needed, just previously missing.
- `check_partner_counts` / `drop_decisions_with_bad_partner_count`:
  `_svm_prediction.drop_decisions_without_true_partner` only catches
  decisions with zero True rows; `get_mating_feature`'s contract is
  exactly one, so a decision with two or more is just as much a
  tracking/labeling bug upstream and wasn't being checked for.
- A pairwise/RankSVM-style reformulation (`build_pairwise_dataset`,
  `train_pairwise_classifier`, `score_candidates_pairwise`): trains on
  true-vs-false *feature differences* within a decision instead of an
  absolute 0/1 label per row. This directly matches the task ("is A a
  better pick than B", not "is A a true partner in isolation") and is
  naturally balanced 1:1 regardless of a decision's `N_candidates`, unlike
  `_svm_prediction.train_partner_classifier`'s row-per-candidate training,
  which sees 1 true row for every `N_candidates - 1` false ones.
  `score_candidates_pairwise` returns a `pred_prob`-shaped array, so it
  drops into every evaluation function in `_svm_prediction.py`
  unchanged (`top_k_hits`, `top_k_accuracy_by_value`,
  `pick_one_decision_per_view`, `true_partner_score_trajectory`, ...).
- `add_total_candidates`: how many distinct candidates a reference cell
  considered across its *entire* fusion event, not just the one time
  snapshot `add_partner_columns`'s `N_candidates` counts. Deliberately
  kept separate from `N_candidates` rather than replacing it --
  `N_candidates` is what the top-k evaluation denominator/random
  baseline/`drop_trivial_decisions` need ("how many candidates was this
  decision actually choosing among"); event-wide counting was tried
  there before and reverted (see `add_partner_columns`'s docstring: the
  two disagree for 51% of decisions, always because the event-wide count
  is larger). This is a different signal -- "how crowded was this cell's
  whole search" -- usable as a `value_col` for
  `_svm_prediction.top_k_accuracy_by_value`-style stratification, or as a
  feature in the *absolute* (`_svm_prediction.train_partner_classifier`)
  formulation. It carries no signal for the *pairwise* formulation above,
  though: it's constant across every row of one decision (even one whole
  view), so a true-vs-false difference is always exactly 0 -- don't add
  it to a `feature_columns` list passed to `build_pairwise_dataset`.
"""
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ._svm_prediction import (
    _resolve_group_col, _resolve_fusion_event_group_col, _group_ids,
    CANDIDATE_NORMALIZE_COLUMNS, CANONICAL_FEATURE_COLUMNS,
)


# orientation_diff: folded angle (0..pi/2) between the two cells' major
# axes. contact_length: shared Voronoi border (px) between them. Both
# symmetric pair measurements, same footing as center_dist/nearest_dist/
# tip_distance -- present in get_mating_feature's raw table but never
# wired into FEATURE_COLUMNS/CANONICAL_FEATURE_COLUMNS.
PAIR_GEOMETRY_COLUMNS = ["orientation_diff", "contact_length"]

# Every candidate-relative column worth ranking within a decision, not
# just the three distance columns _svm_prediction.feature_extract already
# ranked: the rest of CANDIDATE_NORMALIZE_COLUMNS, plus the two
# PAIR_GEOMETRY_COLUMNS above.
RANK_PCT_COLUMNS = CANDIDATE_NORMALIZE_COLUMNS + PAIR_GEOMETRY_COLUMNS

# CANONICAL_FEATURE_COLUMNS plus the two previously-unused pair geometry
# columns and every RANK_PCT_COLUMNS column's `_rank_pct` counterpart.
# Requires `canonicalize_candidate_features`, `normalize_candidates`
# (as CANONICAL_FEATURE_COLUMNS already does) *and* `add_rank_pct_features`
# to have been run first.
CANONICAL_FEATURE_COLUMNS_V2 = (
    CANONICAL_FEATURE_COLUMNS
    + PAIR_GEOMETRY_COLUMNS
    + [f"{c}_rank_pct" for c in RANK_PCT_COLUMNS]
)


def add_rank_pct_features(data: pd.DataFrame, columns=RANK_PCT_COLUMNS, group_col=None,
                           ascending: bool = True) -> pd.DataFrame:
    """Add `f"{col}_rank"` (1..N within each `group_col` group, ties->min)
    and `f"{col}_rank_pct"` (0 = top-ranked, 1 = bottom-ranked, scaled to
    use the full 0..1 range regardless of a decision's `N_candidates`) for
    every `columns`. `group_col` defaults per `_resolve_group_col`.
    `ascending` is forwarded to `.rank()` as-is, same direction convention
    as `feature_extract.RANK_COLUMNS` (ascending=True: smallest raw value
    -> rank 1) -- a parameter, not an assumption about which direction is
    biologically "better" (see `normalize_candidates`'s docstring for why
    that distinction matters).

    This is `_rank_pct`'s whole point over `normalize_candidates`'s
    min-max `_norm`: at a 1-candidate decision there's nothing to rank
    against, so that candidate is unambiguously top-ranked in whichever
    direction `ascending` asks for -- `_rank_pct` is pinned to 0, not an
    arbitrary min-max 0 that's indistinguishable from a real
    "smallest/closest of N" 0 at N>1. The `(rank - 1) / (N - 1)` scaling
    would divide by zero there; `N == 1` groups get 0 directly instead.
    """
    group_col = _resolve_group_col(data, group_col)
    out = data.copy()
    columns = list(columns)
    ranks = out.groupby(group_col)[columns].rank(ascending=ascending, method="min")
    sizes = out.groupby(group_col)[columns[0]].transform("size")
    denom = (sizes - 1).where(sizes > 1, 1)
    for col in columns:
        out[f"{col}_rank"] = ranks[col].astype(int)
        pct = (ranks[col] - 1) / denom
        out[f"{col}_rank_pct"] = pct.where(sizes > 1, 0.0)
    return out


def check_partner_counts(data: pd.DataFrame, group_col=None, label_column: str = "flag"):
    """Per-decision count of `label_column` True rows, which
    `get_mating_feature`'s contract says should be exactly 1 everywhere.
    Returns `(n_zero_true, n_multi_true, n_total_decisions)` --
    `_svm_prediction.drop_decisions_without_true_partner` already guards
    against `n_zero_true`'s case at training/eval time; this also surfaces
    `n_multi_true`, which means some decision had the same candidate
    tagged true twice, or two different candidates both tagged true --
    either way a tracking/labeling bug upstream, not a legitimate outcome.
    """
    group_col = _resolve_group_col(data, group_col)
    group_ids = _group_ids(data, group_col)
    counts = data.groupby(group_ids)[label_column].sum()
    n_zero = int((counts == 0).sum())
    n_multi = int((counts > 1).sum())
    return n_zero, n_multi, len(counts)


def drop_decisions_with_bad_partner_count(data: pd.DataFrame, group_col=None, label_column: str = "flag"):
    """Like `_svm_prediction.drop_decisions_without_true_partner`, but also
    drops decisions with *more than one* True row (see
    `check_partner_counts`) -- both are violations of
    `get_mating_feature`'s one-true-partner-per-decision contract, not
    just the zero case. Returns `(clean_data, n_dropped_decisions, n_dropped_rows)`.
    """
    group_col = _resolve_group_col(data, group_col)
    group_ids = _group_ids(data, group_col)
    counts = data.groupby(group_ids)[label_column].transform("sum")
    keep = counts == 1
    n_dropped_rows = int((~keep).sum())
    n_dropped_decisions = int(group_ids[~keep].nunique())
    return data[keep].copy(), n_dropped_decisions, n_dropped_rows


def add_total_candidates(data: pd.DataFrame, group_col=None, partner_col: str = "partner_id",
                          column: str = "N_candidates_total") -> pd.DataFrame:
    """Add `column` (default `N_candidates_total`): how many distinct
    candidates a reference cell considered across its *entire* fusion
    event, i.e. over every time snapshot of its search -- not just the one
    snapshot `add_partner_columns`'s `N_candidates` counts. Requires
    `add_partner_columns` to have been run first (needs `partner_col`).

    `group_col` defaults to the fusion-event-level grouping
    (`_resolve_fusion_event_group_col`: `["image", "fusion_key", "ref_id"]`,
    one reference cell's whole search), not the per-decision one -- same
    level `split_by_fusion_event` groups by, deliberately coarser than
    `N_candidates`'s per-decision grouping (see this module's docstring
    for why the two are kept separate, not merged into one column).
    """
    group_col = _resolve_fusion_event_group_col(data, group_col)
    out = data.copy()
    out[column] = out.groupby(group_col)[partner_col].transform("nunique")
    return out


def build_pairwise_dataset(data: pd.DataFrame, feature_columns, group_col=None,
                            label_column: str = "flag", swap: bool = True) -> pd.DataFrame:
    """Turn each decision's (1 true row, M false rows) into true-vs-false
    feature-*difference* training examples for a pairwise classifier:
    predict "is the first candidate of this pair the more likely true
    partner" instead of fitting `label_column`'s absolute 0/1 directly.
    Naturally balanced 1:1 regardless of `N_candidates`, unlike training
    on `label_column` as-is (1 true row for every `N_candidates - 1` false
    ones).

    For a decision with true row `t` and false rows `f_1..f_M`:
    - `swap=True` (default): emits both `(t - f_i, label=1)` and
      `(f_i - t, label=0)` for every `i` -- `2M` rows per decision,
      explicitly teaching the antisymmetry of the comparison (reversing a
      pair flips the label) rather than relying on the classifier to
      discover that on its own from whatever row order it happens to see.
      `score_candidates_pairwise` assumes the classifier was trained this
      way.
    - `swap=False`: only `(t - f_i, label=1)` -- `M` rows per decision.

    `group_col` defaults per `_resolve_group_col`. Decisions with zero or
    more than one true row are skipped (run
    `drop_decisions_with_bad_partner_count` first to know how many, and
    `drop_trivial_decisions` to drop the 1-candidate decisions that would
    otherwise contribute zero pairs here anyway).

    Returns a DataFrame with `feature_columns` (the raw difference, same
    names, not renamed) plus `_label` and `_group` (the resolved group id,
    for tracing a pair back to its source decision).
    """
    group_col = _resolve_group_col(data, group_col)
    group_ids = _group_ids(data, group_col).to_numpy()
    feature_columns = list(feature_columns)
    X = data[feature_columns].to_numpy(dtype=float)
    y = data[label_column].astype(bool).to_numpy()

    diffs, labels, groups = [], [], []
    for gid in pd.unique(group_ids):
        idx = np.flatnonzero(group_ids == gid)
        true_idx = idx[y[idx]]
        false_idx = idx[~y[idx]]
        if len(true_idx) != 1 or len(false_idx) == 0:
            continue
        t = X[true_idx[0]]
        for fi in false_idx:
            f = X[fi]
            diffs.append(t - f)
            labels.append(1)
            groups.append(gid)
            if swap:
                diffs.append(f - t)
                labels.append(0)
                groups.append(gid)

    out = pd.DataFrame(diffs, columns=feature_columns)
    out["_label"] = labels
    out["_group"] = groups
    return out


def train_pairwise_classifier(pairs_df: pd.DataFrame, feature_columns, estimator,
                                scale: bool = True, shuffle: bool = True, random_state: int = 0):
    """Fit `estimator` (any sklearn-compatible binary classifier exposing
    `predict_proba` -- `SVC(probability=True)`, `LogisticRegression`,
    `RandomForestClassifier`, `HistGradientBoostingClassifier`, ...) on
    `build_pairwise_dataset`'s difference features to predict `_label`
    (1 = the pair's first candidate is the true partner).

    `scale` (default True) wraps `estimator` in a `StandardScaler`
    pipeline first -- matters for distance-/regularization-based
    estimators (SVC, LogisticRegression); harmless but unnecessary for
    tree ensembles, which split on raw thresholds (pass `scale=False` for
    those if you'd rather skip it).
    """
    if shuffle:
        pairs_df = pairs_df.sample(frac=1, random_state=random_state)
    feature_columns = list(feature_columns)
    X = pairs_df[feature_columns]
    y = pairs_df["_label"].astype(int)
    clf = make_pipeline(StandardScaler(), estimator) if scale else estimator
    clf.fit(X, y)
    return clf


def score_candidates_pairwise(clf, data: pd.DataFrame, feature_columns, group_col=None) -> np.ndarray:
    """Per-row score for a classifier trained by `train_pairwise_classifier`,
    a drop-in replacement for `_svm_prediction.score_candidates`'s
    `pred_prob` column everywhere else in that module (`top_k_hits`,
    `top_k_accuracy_by_value`, `pick_one_decision_per_view`,
    `true_partner_score_trajectory`, ...).

    Since the classifier only ever saw *differences* between two
    candidates, scoring one candidate in isolation is meaningless.
    Instead, for each decision, every candidate is compared against every
    other candidate in the same group (all `N * (N - 1)` ordered pairs),
    and each candidate's score is its mean predicted win probability
    across those comparisons -- a Copeland/Borda-style aggregate ("how
    often does this candidate beat a same-decision competitor"). O(N^2)
    pairs per decision: fine at this dataset's group sizes (real data:
    `N_candidates` tops out at 10, median 3) but worth revisiting if that
    grows much further.

    A 1-candidate decision has no competitor to compare against; its row
    gets 0.5 (an uninformative tie, not NaN) -- it's deterministically the
    true partner by construction either way (see `drop_trivial_decisions`),
    and downstream top-k code only needs *some* finite, non-distinguishing
    score for it.
    """
    group_col = _resolve_group_col(data, group_col)
    group_ids = _group_ids(data, group_col).to_numpy()
    feature_columns = list(feature_columns)
    X = data[feature_columns].to_numpy(dtype=float)
    scores = np.full(len(data), 0.5)

    for gid in pd.unique(group_ids):
        idx = np.flatnonzero(group_ids == gid)
        if len(idx) < 2:
            continue
        ii, jj = np.meshgrid(idx, idx, indexing="ij")
        mask = ii != jj
        ii, jj = ii[mask], jj[mask]
        pair_diffs = pd.DataFrame(X[ii] - X[jj], columns=feature_columns)
        win_prob = clf.predict_proba(pair_diffs)[:, 1]  # P(candidate ii beats candidate jj)
        positions = np.searchsorted(idx, ii)
        win_sum = np.bincount(positions, weights=win_prob, minlength=len(idx))
        counts = np.bincount(positions, minlength=len(idx))
        scores[idx] = win_sum / counts

    return scores
