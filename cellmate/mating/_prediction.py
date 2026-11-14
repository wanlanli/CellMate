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
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


RANK_COLUMNS = {
    # feature: (output column, ascending)
    "nearest_dist": ("near_dist_rank", True),
    "center_dist": ("center_dist_rank", True),
    "tip_distance": ("tip_dist_rank", True),
    "contact_length":("contact_length", True),
    "candidate_start": ("candidate_start_rank", True),
    "candidate_area": ("candidate_area_rank", False),
    "candidate_minor": ("candidate_minor_rank", False),
    "candidate_major": ("candidate_major_rank", False),
    "candidate_angle": ("candidate_angle_rank", True),
    "candidate_eccentricity": ("candidate_eccentricity_rank", True),
}

# The single entry point for what the three predictors below train on.
# Override it here, or pass `feature_columns=` to any train_*/score_*
# function below, whenever the upstream raw feature table changes (new/
# renamed columns from `get_mating_feature_v2`, a different
# `add_rank_norm_features` pass, etc.) -- nothing below hardcodes columns
# outside this one list.


def add_rank_norm_features(raw: pd.DataFrame, group_id="decision_id"):

    grouped = raw.groupby(group_id)

    for feature, (rank_column, ascending) in RANK_COLUMNS.items():
        raw[rank_column] = (
            grouped[feature]
            .rank(ascending=ascending, method="min")
            .astype("Int64")
        )
    for feature, (_, ascending) in RANK_COLUMNS.items():
        values = raw[feature]
        gmin = grouped[feature].transform("min")
        gmax = grouped[feature].transform("max")
        span = (gmax - gmin).replace(0, 1)

        offset = values - gmin if ascending else gmax - values
        raw[f"{feature}_norm"] = offset / span
    return raw


def drop_decisions_with_bad_partner_count(data: pd.DataFrame, group_id="decision_id",  label_column: str = "flag"):
    """Like `_svm_prediction.drop_decisions_without_true_partner`, but also
    drops decisions with *more than one* True row (see
    `check_partner_counts`) -- both are violations of
    `get_mating_feature`'s one-true-partner-per-decision contract, not
    just the zero case. `group_col` defaults per `_resolve_group_col` and
    may be a single column name or a list of column names. Returns
    `(clean_data, n_dropped_decisions, n_dropped_rows)`.
    """
    counts = data.groupby(group_id)[label_column].transform("sum")
    keep = counts == 1
    return data[keep].copy(), data[~keep]


def drop_trivial_decisions(data: pd.DataFrame, min_candidates: int = 2, group_col="decision_id",
                            n_candidates_col: str = "n_candidates_frame"):
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
    # group_ids = _group_ids(data, group_col)
    keep = data[n_candidates_col] >= min_candidates
    return data[keep].copy(), data[~keep]


def _group_ids(data: pd.DataFrame, group_col) -> pd.Series:
    """Flat, row-aligned group label, usable with `.unique()`/`.isin()`
    (and `GroupShuffleSplit`, which rejects a multi-column `groups`) even
    when `group_col` is a list of columns.
    """
    if isinstance(group_col, (list, tuple)):
        return data.groupby(list(group_col)).ngroup()
    return data[group_col]


def split_by_fusion_event(data: pd.DataFrame, test_size: float = 0.3, random_state: int = 42, group_col=None,
                           n_candidates_col: str = "n_candidates_frame", decision_col: str = "decision_id",
                           time_col: str = "time_stamp"):
    """Train/test split grouped by fusion event, not by decision -- see
    `_resolve_fusion_event_group_col` for why a decision-level split leaks.
    `group_col` overrides the default (e.g. pass the `_resolve_group_col`
    decision-level grouping explicitly only if you specifically want that
    and accept the leakage it implies).

    Decisions with only one candidate (`n_candidates_col == 1`) are moved
    into `train_df` no matter which side of the split their event landed
    on: with nothing to discriminate between, they're a trivially-"correct",
    wasted row for evaluation (see `drop_trivial_decisions`), but still a
    valid positive example for training.

    Also returns two one-decision-per-event reductions of `test_df`, so
    evaluation isn't dominated by several near-duplicate, time-adjacent
    decisions from the same event: `test_one_random_df` (one randomly
    picked `decision_col` per event) and `test_one_first_df` (the decision
    at that event's smallest `time_col`, i.e. its first snapshot).

    Returns `(train_df, test_df, test_one_random_df, test_one_first_df)`.
    """
    event_ids = _group_ids(data, group_col)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(data, groups=event_ids))
    train_df, test_df = data.iloc[train_idx].copy(), data.iloc[test_idx].copy()

    trivial = test_df[n_candidates_col] == 1
    train_df = pd.concat([train_df, test_df[trivial]])
    test_df = test_df[~trivial].copy()

    test_event_ids = _group_ids(test_df, group_col)

    rng = np.random.default_rng(random_state)
    chosen_random = test_df.groupby(test_event_ids)[decision_col].agg(lambda ids: rng.choice(ids.unique()))
    test_one_random_df = test_df[test_df[decision_col].isin(chosen_random)].copy()

    first_time = test_df.groupby(test_event_ids)[time_col].transform("min")
    chosen_first = test_df.loc[test_df[time_col] == first_time].groupby(test_event_ids)[decision_col].min()
    test_one_first_df = test_df[test_df[decision_col].isin(chosen_first)].copy()

    return train_df, test_df, test_one_random_df, test_one_first_df


# --- Three candidate-selection predictors ---------------------------------
#
# All three share one contract: fit on a decision-level table (one row per
# candidate, grouped by `group_col`, exactly one `label_column`-true row per
# group), and expose a `score_*` counterpart that returns one float per row,
# ranking candidates within their own decision -- so any of them plugs into
# `_svm_prediction.top_k_hits`/`top_k_accuracy` (k=1) the same way. Pick
# `feature_columns` per call, or change `DEFAULT_FEATURE_COLUMNS` above, to
# point all three at a different feature set.

# Auto-filled from the get_mating_feature_v2 + add_rank_norm_features
# output columns. Every feature-shaped column ends up in exactly one of
# the four lists below; ids/labels/metadata -- fusion_id, focal_id,
# focal_type, time_stamp, view_id, decision_id, candidate_id, flag,
# fusion_time, pixel_size, image, image_id, n_candidates_frame,
# n_candidates_view -- are left out of all four. Edit freely.
DISTANCE_FEATURE = [
    "center_dist", "nearest_dist", "contact_length",
    "tip_distance",
    ]
RANK_FEATURE = [
    "nearest_dist_rank", "center_dist_rank", "tip_distance_rank", "candidate_start_rank",
    "near_dist_rank", "tip_dist_rank", "candidate_area_rank", "candidate_minor_rank",
    "candidate_major_rank", "candidate_angle_rank", "candidate_eccentricity_rank",
]
NORM_FEATURE = [
    "focal_angle_norm", "candidate_angle_norm",
    "nearest_dist_norm", "center_dist_norm", "tip_distance_norm", "contact_length_norm",
    "candidate_start_norm", "candidate_area_norm", "candidate_minor_norm",
    "candidate_major_norm", "candidate_eccentricity_norm",
]
# Everything feature-shaped left over once DISTANCE_FEATURE/RANK_FEATURE/
# NORM_FEATURE are taken out: raw focal_*/candidate_* shape, neighbor and
# angle columns, the two tip-meeting flags, and the two pair columns
# (orientation_diff, time_diff) that aren't a plain distance.
FOCAL_FEATURE = ["focal_start", "focal_area", "focal_major", "focal_minor", "focal_eccentricity",
    "focal_neighbor_same", "focal_neighbor_diff", "focal_angle", "focal_angle_index", "focal_in_new_tip",]

CANDI_FEATURE = [
    "candidate_start", "candidate_area", "candidate_major", "candidate_minor", "candidate_eccentricity",
    "candidate_neighbor_same", "candidate_neighbor_diff", "candidate_angle", "candidate_angle_index",
    "orientation_diff",  "candidate_in_new_tip", "time_diff",
]

# Whatever's left once DISTANCE_FEATURE/RANK_FEATURE/NORM_FEATURE/
# FOCAL_FEATURE/CANDI_FEATURE are taken out of the full column list --
# all ids/label/metadata (fusion_id, focal_id, focal_type, time_stamp,
# view_id, decision_id, candidate_id, flag, fusion_time, pixel_size,
# n_candidates_frame, n_candidates_view, image, image_id), none of it a
# real per-candidate feature. `flag` in particular is the label, not a
# feature -- leaving it in DEFAULT_FEATURE_COLUMNS (OTHER_FEATURE is
# summed in below) means training directly on the ground truth. Trim
# this list before training; kept everything here for now since you
# said you'd edit it.
OTHER_FEATURE = [
    "focal_type", "time_stamp", "fusion_time", "pixel_size", "n_candidates_frame", "n_candidates_view",
]

DEFAULT_FEATURE_COLUMNS = DISTANCE_FEATURE + RANK_FEATURE + NORM_FEATURE + CANDI_FEATURE + OTHER_FEATURE


def train_row_classifier(data: pd.DataFrame, feature_columns=None, group_col="decision_id",
                          label_column: str = "flag", estimator=None, scale: bool = True,
                          shuffle: bool = True, random_state: int = 0):
    """Row classifier + top-1: learns P(row is the true partner) per row,
    independent of the other candidates in its decision; at inference,
    the candidate with the highest score within each decision is the
    pick. The simplest of the three -- a good baseline (any sklearn
    `predict_proba` classifier works, e.g. the existing SVM in
    `_svm_prediction.train_partner_classifier` used this way).

    `group_col` isn't used for fitting (each row is independent), only
    accepted for signature symmetry with the other two train_* functions.
    """
    feature_columns = list(feature_columns or DEFAULT_FEATURE_COLUMNS)
    estimator = estimator if estimator is not None else LogisticRegression(max_iter=1000, random_state=random_state)
    df = data.sample(frac=1, random_state=random_state) if shuffle else data
    clf = make_pipeline(StandardScaler(), estimator) if scale else estimator
    clf.fit(df[feature_columns], df[label_column].astype(int))
    clf.feature_columns_ = feature_columns
    return clf


def score_row_classifier(clf, data: pd.DataFrame, feature_columns=None) -> np.ndarray:
    feature_columns = list(feature_columns or getattr(clf, "feature_columns_", DEFAULT_FEATURE_COLUMNS))
    return clf.predict_proba(data[feature_columns])[:, 1]


class ConditionalLogit:
    """Conditional logistic regression / group softmax: a single linear
    score per candidate, fit so that softmax(score) over each decision's
    candidates approximates the one-hot true-partner label -- candidates
    compete directly within their own group during training, unlike
    `train_row_classifier`'s per-row-independent loss. My first choice of
    the three when interpretability of the learned weights matters.
    """

    def __init__(self, feature_columns, l2: float = 1.0):
        self.feature_columns = list(feature_columns)
        self.l2 = l2
        self.coef_ = None
        self.scaler_ = None

    def fit(self, data: pd.DataFrame, group_ids: pd.Series, label_column: str = "flag", scale: bool = True):
        X = data[self.feature_columns].to_numpy(dtype=float)
        if scale:
            self.scaler_ = StandardScaler().fit(X)
            X = self.scaler_.transform(X)
        y = data[label_column].to_numpy(dtype=float)

        order = np.argsort(group_ids.to_numpy(), kind="stable")
        X, y = X[order], y[order]
        _, group_sizes = np.unique(group_ids.to_numpy()[order], return_counts=True)
        boundaries = np.cumsum(group_sizes)[:-1]
        X_groups = np.split(X, boundaries)
        y_groups = np.split(y, boundaries)

        def neg_log_likelihood(w):
            loss = 0.0
            grad = np.zeros_like(w)
            for Xg, yg in zip(X_groups, y_groups):
                s = Xg @ w
                m = s.max()
                exp_s = np.exp(s - m)
                z = exp_s.sum()
                p = exp_s / z
                loss -= float((yg * (s - m - np.log(z))).sum())
                grad += Xg.T @ (p - yg)
            loss += 0.5 * self.l2 * np.sum(w ** 2)
            grad += self.l2 * w
            return loss, grad

        result = minimize(neg_log_likelihood, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B")
        self.coef_ = result.x
        return self

    def predict_score(self, data: pd.DataFrame) -> np.ndarray:
        X = data[self.feature_columns].to_numpy(dtype=float)
        if self.scaler_ is not None:
            X = self.scaler_.transform(X)
        return X @ self.coef_


def train_conditional_logit(data: pd.DataFrame, feature_columns=None, group_col="decision_id",
                             label_column: str = "flag", l2: float = 1.0, scale: bool = True) -> ConditionalLogit:
    feature_columns = list(feature_columns or DEFAULT_FEATURE_COLUMNS)
    model = ConditionalLogit(feature_columns, l2=l2)
    model.fit(data, _group_ids(data, group_col), label_column=label_column, scale=scale)
    return model


def score_conditional_logit(model: ConditionalLogit, data: pd.DataFrame) -> np.ndarray:
    return model.predict_score(data)


def train_learning_to_rank(data: pd.DataFrame, feature_columns=None, group_col="decision_id",
                            label_column: str = "flag", random_state: int = 0, **lgbm_kwargs):
    """Learning to rank: `lightgbm.LGBMRanker` (lambdarank), which learns
    to rank the true partner above its decision's other candidates
    directly, rather than fitting an absolute per-row probability or a
    linear group-softmax score -- worth testing for nonlinear feature
    interactions the other two can't capture.

    Needs the optional `lightgbm` package (not installed in this repo's
    `yeast` env as of writing -- `pip install lightgbm` there first).
    """
    try:
        from lightgbm import LGBMRanker
    except ImportError as exc:
        raise ImportError(
            "train_learning_to_rank needs the optional 'lightgbm' package -- "
            "install it (e.g. `pip install lightgbm` in the yeast env) first."
        ) from exc

    feature_columns = list(feature_columns or DEFAULT_FEATURE_COLUMNS)
    groups = _group_ids(data, group_col)
    order = np.argsort(groups.to_numpy(), kind="stable")
    ordered = data.iloc[order]
    _, group_sizes = np.unique(groups.to_numpy()[order], return_counts=True)

    ranker = LGBMRanker(random_state=random_state, **lgbm_kwargs)
    ranker.fit(ordered[feature_columns].to_numpy(dtype=float),
               ordered[label_column].to_numpy(dtype=int),
               group=group_sizes)
    ranker.feature_columns_ = feature_columns
    return ranker


def score_learning_to_rank(ranker, data: pd.DataFrame, feature_columns=None) -> np.ndarray:
    feature_columns = list(feature_columns or getattr(ranker, "feature_columns_", DEFAULT_FEATURE_COLUMNS))
    return ranker.predict(data[feature_columns])


# def top_k_hits(scored_df: pd.DataFrame, k: int = 1, group_col=None,
#                score_col: str = "pred_prob", label_col: str = "flag",
#                min_candidates: int = None, n_candidates_col: str = "N_candidates") -> pd.Series:
#     """Per-decision 0/1: is the true partner among the top-`k` scored
#     candidates? Indexed by `group_col` (defaults per `_resolve_group_col`);
#     `top_k_accuracy` is this series' mean, and the per-decision values are
#     what `bootstrap_ci` and `cellmate.visualize.plot_top_k_accuracy` want
#     directly.

#     `min_candidates` restricts to decisions with at least that many
#     candidates (a harder, more informative subset -- with 2 candidates,
#     top-1 is nearly guaranteed by construction).

#     Decisions with zero true-partner rows are dropped: `get_mating_feature`
#     guarantees exactly one true partner per decision, so zero signals an
#     upstream tracking/labeling problem, not a legitimate outcome to average
#     classifier performance over. Prefer catching these with
#     `drop_decisions_without_true_partner` before training, not just here.
#     """
#     df = scored_df
#     if min_candidates is not None:
#         df = df[df[n_candidates_col] >= min_candidates]
#     has_true_partner = df.groupby(group_col)[label_col].transform("sum") > 0
#     df = df[has_true_partner]
#     if df.empty:
#         return pd.Series(dtype=float)
#     topk = df.sort_values(score_col, ascending=False).groupby(group_col).head(k)
#     return topk.groupby(group_col)[label_col].max()


# def top_k_accuracy(scored_df: pd.DataFrame, k: int = 1, **kwargs) -> float:
#     """Fraction of decisions where the true partner is among the top-`k`
#     scored candidates -- the mean of `top_k_hits`; see there for `kwargs`.
#     """
#     hits = top_k_hits(scored_df, k=k, **kwargs)
#     return hits.mean() if len(hits) else np.nan
