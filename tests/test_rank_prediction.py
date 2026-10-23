"""Tests for cellmate.mating._rank_prediction -- the v2 feature/model
experiments built on top of _svm_prediction.py: percentile-rank features
that stay meaningful at a 1-candidate decision, the previously-unused
orientation_diff/contact_length pair columns, a stricter decision-group
validator (also catches more-than-one true row, not just zero), and the
pairwise/RankSVM-style reformulation (train on true-vs-false feature
differences, naturally balanced 1:1 regardless of N_candidates).

Synthetic data only, no CellNetwork -- same convention as
test_svm_prediction.py. For the real-data numbers, see
hpm/02_svm_prediction_v2.ipynb (Pipeline_Cell_Behavior_Prediction repo).
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cellmate.mating import (
    add_partner_columns,
    split_by_fusion_event,
    top_k_accuracy,
    random_baseline_top_k,
)
from cellmate.mating._rank_prediction import (
    PAIR_GEOMETRY_COLUMNS,
    add_rank_pct_features,
    check_partner_counts,
    drop_decisions_with_bad_partner_count,
    add_total_candidates,
    build_event_level_dataset,
    build_pairwise_dataset,
    train_pairwise_classifier,
    score_candidates_pairwise,
    learning_curve_pairwise,
    evaluate_feature_groups,
)
from tests.test_svm_prediction import synthetic_fusion_table


def with_pair_geometry(df, seed=0):
    """Add synthetic orientation_diff/contact_length columns -- not part
    of synthetic_fusion_table, which predates them.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["orientation_diff"] = rng.uniform(0, np.pi / 2, size=len(df))
    df["contact_length"] = rng.uniform(0, 100, size=len(df))
    return df


class AddRankPctFeaturesTests(unittest.TestCase):
    def test_rank_pct_spans_0_to_1_within_group_ascending(self):
        df = pd.DataFrame({
            "decision_id": [0, 0, 0, 0],
            "center_dist": [5.0, 1.0, 3.0, 9.0],
        })
        out = add_rank_pct_features(df, columns=["center_dist"], ascending=True)
        # smallest (1.0) -> rank 1 -> pct 0; largest (9.0) -> rank 4 -> pct 1
        self.assertEqual(out.loc[out["center_dist"] == 1.0, "center_dist_rank_pct"].iloc[0], 0.0)
        self.assertEqual(out.loc[out["center_dist"] == 9.0, "center_dist_rank_pct"].iloc[0], 1.0)
        self.assertTrue((out["center_dist_rank_pct"] >= 0).all())
        self.assertTrue((out["center_dist_rank_pct"] <= 1).all())

    def test_single_candidate_group_gets_zero_not_divide_by_zero(self):
        df = pd.DataFrame({"decision_id": [0, 1, 1], "center_dist": [7.0, 2.0, 4.0]})
        out = add_rank_pct_features(df, columns=["center_dist"], ascending=True)
        single = out[out["decision_id"] == 0]
        self.assertEqual(single["center_dist_rank"].iloc[0], 1)
        self.assertEqual(single["center_dist_rank_pct"].iloc[0], 0.0)
        self.assertFalse(out["center_dist_rank_pct"].isna().any())

    def test_ties_use_min_method(self):
        df = pd.DataFrame({"decision_id": [0, 0, 0], "center_dist": [3.0, 3.0, 9.0]})
        out = add_rank_pct_features(df, columns=["center_dist"], ascending=True)
        tied = out[out["center_dist"] == 3.0]["center_dist_rank"]
        self.assertTrue((tied == 1).all())

    def test_descending_direction_flips_which_value_is_top_ranked(self):
        df = pd.DataFrame({"decision_id": [0, 0], "candidate_area": [5.0, 1.0]})
        out = add_rank_pct_features(df, columns=["candidate_area"], ascending=False)
        self.assertEqual(out.loc[out["candidate_area"] == 5.0, "candidate_area_rank_pct"].iloc[0], 0.0)
        self.assertEqual(out.loc[out["candidate_area"] == 1.0, "candidate_area_rank_pct"].iloc[0], 1.0)

    def test_default_columns_cover_pair_geometry(self):
        self.assertIn("orientation_diff", PAIR_GEOMETRY_COLUMNS)
        self.assertIn("contact_length", PAIR_GEOMETRY_COLUMNS)


class AddTotalCandidatesTests(unittest.TestCase):
    def test_counts_distinct_candidates_across_the_whole_view_not_just_one_decision(self):
        # one view (fusion_key=0, ref_id=100) with two decisions (snapshots):
        # decision 0 faces partners {1, 2}, decision 1 faces {2, 3} -- 3
        # distinct candidates over the whole search, not 2.
        df = pd.DataFrame({
            "image": ["img"] * 4, "fusion_key": [0, 0, 0, 0], "ref_id": [100, 100, 100, 100],
            "decision_id": [0, 0, 1, 1],
            "partner_id": [1, 2, 2, 3],
        })
        out = add_total_candidates(df)
        self.assertTrue((out["N_candidates_total"] == 3).all())

    def test_separate_views_counted_independently(self):
        df = pd.DataFrame({
            "image": ["img"] * 4, "fusion_key": [0, 0, 1, 1], "ref_id": [100, 100, 200, 200],
            "decision_id": [0, 0, 1, 1],
            "partner_id": [1, 2, 5, 6],
        })
        out = add_total_candidates(df)
        view_100 = out[out["ref_id"] == 100]
        view_200 = out[out["ref_id"] == 200]
        self.assertTrue((view_100["N_candidates_total"] == 2).all())
        self.assertTrue((view_200["N_candidates_total"] == 2).all())

    def test_differs_from_per_decision_n_candidates_when_candidates_change_over_time(self):
        raw = add_partner_columns(synthetic_multi_snapshot_table_with_turnover())
        out = add_total_candidates(raw)
        # at least one view must show N_candidates_total > any single
        # decision's N_candidates -- otherwise this test fixture isn't
        # exercising the thing add_total_candidates exists for.
        per_view_max_decision_n = out.groupby(["fusion_key", "ref_id"])["N_candidates"].max()
        per_view_total = out.groupby(["fusion_key", "ref_id"])["N_candidates_total"].first()
        self.assertTrue((per_view_total >= per_view_max_decision_n).all())
        self.assertTrue((per_view_total > per_view_max_decision_n).any())


def synthetic_multi_snapshot_table_with_turnover():
    # ref_id=100's search: decision 0 faces {1, 2}, decision 1 faces {2, 3}
    # (1 leaves, 3 arrives) -- N_candidates is 2 at every single decision,
    # but N_candidates_total over the whole view is 3.
    rows = []
    for decision_id, partners in enumerate([[1, 2], [2, 3]]):
        for j, partner in enumerate(partners):
            rows.append(dict(
                image="img", fusion_key=0, ref_id=100, decision_id=decision_id,
                ref_type=1, p_id=100, m_id=partner, flag=(j == 0), time_stamp=decision_id,
            ))
    return pd.DataFrame(rows)


def synthetic_view_with_pre_appearance_frame():
    # ref_id=100's search, true partner is 1:
    # decision 0 (ts=0): faces {2, 3} -- partner 1 isn't trackable yet
    #   (a zero-flag frame, i.e. "before the real partner appears").
    # decision 1 (ts=1): faces {1, 2} -- partner 1 now present, flagged True.
    # decision 2 (ts=2): faces {1, 3} -- partner 1 still True, 2 has left.
    # Distinct candidates over the whole view: {1, 2, 3}.
    rows = []
    frames = {
        0: {2: False, 3: False},
        1: {1: True, 2: False},
        2: {1: True, 3: False},
    }
    for decision_id, partners in frames.items():
        for partner, is_true in partners.items():
            rows.append(dict(
                image="img", fusion_key=0, ref_id=100, decision_id=decision_id,
                partner_id=partner, flag=is_true, time_stamp=decision_id,
            ))
    return pd.DataFrame(rows)


def synthetic_two_views_one_without_any_true_row():
    good = synthetic_view_with_pre_appearance_frame()
    bad = pd.DataFrame([
        dict(image="img", fusion_key=1, ref_id=200, decision_id=3,
             partner_id=p, flag=False, time_stamp=0)
        for p in (7, 8)
    ])
    return pd.concat([good, bad], ignore_index=True)


class BuildEventLevelDatasetTests(unittest.TestCase):
    def test_one_row_per_distinct_candidate_over_the_whole_view(self):
        df = synthetic_view_with_pre_appearance_frame()
        out = build_event_level_dataset(df, frame_selection="first")
        self.assertEqual(sorted(out["partner_id"].tolist()), [1, 2, 3])

    def test_exactly_one_positive_label_matching_the_eventual_partner(self):
        df = synthetic_view_with_pre_appearance_frame()
        out = build_event_level_dataset(df, frame_selection="first")
        self.assertEqual(out["_event_label"].sum(), 1)
        self.assertEqual(out.loc[out["_event_label"] == 1, "partner_id"].iloc[0], 1)

    def test_first_selection_uses_each_candidates_own_earliest_frame(self):
        df = synthetic_view_with_pre_appearance_frame()
        out = build_event_level_dataset(df, frame_selection="first")
        by_candidate = out.set_index("partner_id")
        # candidate 1 only ever appears from decision 1 onward
        self.assertEqual(by_candidate.loc[1, "decision_id"], 1)
        # candidates 2 and 3 are each represented by their own first (pre-appearance) frame
        self.assertEqual(by_candidate.loc[2, "decision_id"], 0)
        self.assertEqual(by_candidate.loc[3, "decision_id"], 0)

    def test_random_selection_still_keeps_exactly_one_row_per_candidate_and_one_positive(self):
        df = synthetic_view_with_pre_appearance_frame()
        out = build_event_level_dataset(df, frame_selection="random", random_state=1)
        self.assertEqual(len(out), 3)
        self.assertEqual(out["_event_label"].sum(), 1)

    def test_view_with_no_true_row_anywhere_is_dropped_entirely(self):
        df = synthetic_two_views_one_without_any_true_row()
        out = build_event_level_dataset(df, frame_selection="first")
        self.assertEqual(sorted(out["ref_id"].unique()), [100])

    def test_invalid_frame_selection_raises(self):
        df = synthetic_view_with_pre_appearance_frame()
        with self.assertRaises(ValueError):
            build_event_level_dataset(df, frame_selection="middle")

    def test_rank_pct_columns_recomputed_within_the_collapsed_candidate_set(self):
        df = synthetic_view_with_pre_appearance_frame()
        # stale per-frame rank_pct: computed within each original decision_id
        # (size 2 each), not within the 3-candidate collapsed event -- each
        # selected row's rank_pct reflects a 2-way, not 3-way, comparison.
        df = add_rank_pct_features(df, columns=["partner_id"], group_col="decision_id")
        out = build_event_level_dataset(df, frame_selection="first")
        self.assertEqual(sorted(out["partner_id_rank_pct"].tolist()), [0.0, 0.0, 1.0])

        recomputed = build_event_level_dataset(df, frame_selection="first", rank_pct_columns=["partner_id"])
        # within the actual collapsed 3-candidate group {1, 2, 3}, partner_id's
        # rank_pct now spans the full 0..1 range across all three, not the
        # stale 2-way value above.
        self.assertEqual(sorted(recomputed["partner_id_rank_pct"].tolist()), [0.0, 0.5, 1.0])

    def test_result_feeds_build_pairwise_dataset_via_event_columns(self):
        df = synthetic_view_with_pre_appearance_frame()
        df["x"] = [10.0, 20.0, 30.0, 1.0, 2.0, 3.0]
        out = build_event_level_dataset(df, frame_selection="first")
        pairs = build_pairwise_dataset(
            out, feature_columns=["x"], group_col="_event_group", label_column="_event_label")
        self.assertEqual(len(pairs), 4)  # 1 true, 2 false -> 2 * 2 with swap


class CheckPartnerCountsTests(unittest.TestCase):
    def test_counts_zero_and_multi_true_decisions_separately(self):
        df = pd.DataFrame({
            "decision_id": [0, 0, 1, 1, 1, 2, 2],
            "flag": [True, False, False, False, False, True, True],
        })
        n_zero, n_multi, n_total = check_partner_counts(df)
        self.assertEqual(n_zero, 1)  # decision 1
        self.assertEqual(n_multi, 1)  # decision 2
        self.assertEqual(n_total, 3)

    def test_all_valid_decisions_report_zero_and_zero(self):
        df = pd.DataFrame({
            "decision_id": [0, 0, 1, 1],
            "flag": [True, False, False, True],
        })
        n_zero, n_multi, n_total = check_partner_counts(df)
        self.assertEqual((n_zero, n_multi, n_total), (0, 0, 2))


class DropDecisionsWithBadPartnerCountTests(unittest.TestCase):
    def test_drops_both_zero_and_multi_true_groups(self):
        df = pd.DataFrame({
            "decision_id": [0, 0, 1, 1, 1, 2, 2],
            "flag": [True, False, False, False, False, True, True],
        })
        clean, n_dropped_decisions, n_dropped_rows = drop_decisions_with_bad_partner_count(df)
        self.assertEqual(n_dropped_decisions, 2)  # decisions 1 and 2
        self.assertEqual(n_dropped_rows, 5)
        self.assertEqual(sorted(clean["decision_id"].unique()), [0])

    def test_keeps_every_row_of_a_valid_decision(self):
        df = pd.DataFrame({
            "decision_id": [0, 0, 0],
            "flag": [True, False, False],
        })
        clean, n_dropped_decisions, n_dropped_rows = drop_decisions_with_bad_partner_count(df)
        self.assertEqual((n_dropped_decisions, n_dropped_rows), (0, 0))
        self.assertEqual(len(clean), 3)


class BuildPairwiseDatasetTests(unittest.TestCase):
    def test_pair_count_per_decision_with_and_without_swap(self):
        df = pd.DataFrame({
            "decision_id": [0, 0, 0, 0],
            "flag": [True, False, False, False],
            "x": [1.0, 2.0, 3.0, 4.0],
        })
        with_swap = build_pairwise_dataset(df, feature_columns=["x"], swap=True)
        no_swap = build_pairwise_dataset(df, feature_columns=["x"], swap=False)
        self.assertEqual(len(with_swap), 6)  # 2 * (4 - 1)
        self.assertEqual(len(no_swap), 3)  # 4 - 1

    def test_true_minus_false_labeled_one_and_reverse_labeled_zero(self):
        df = pd.DataFrame({
            "decision_id": [0, 0],
            "flag": [True, False],
            "x": [10.0, 3.0],
        })
        pairs = build_pairwise_dataset(df, feature_columns=["x"], swap=True)
        pos = pairs[pairs["_label"] == 1]
        neg = pairs[pairs["_label"] == 0]
        self.assertEqual(pos["x"].iloc[0], 10.0 - 3.0)
        self.assertEqual(neg["x"].iloc[0], 3.0 - 10.0)

    def test_skips_decisions_without_exactly_one_true_row(self):
        df = pd.DataFrame({
            "decision_id": [0, 0, 1, 1, 1],
            "flag": [False, False, True, True, False],  # 0: zero true, 1: two true
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        pairs = build_pairwise_dataset(df, feature_columns=["x"])
        self.assertEqual(len(pairs), 0)

    def test_group_column_traces_back_to_source_decision(self):
        df = pd.DataFrame({
            "decision_id": [0, 0, 1, 1, 1],
            "flag": [True, False, False, True, False],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        pairs = build_pairwise_dataset(df, feature_columns=["x"])
        self.assertEqual(set(pairs["_group"].unique()), {0, 1})


class PairwiseClassifierRoundTripTests(unittest.TestCase):
    """Train on separable synthetic data and check the pairwise pipeline
    (build_pairwise_dataset -> train_pairwise_classifier ->
    score_candidates_pairwise -> top_k_accuracy from _svm_prediction)
    beats the random baseline -- a plumbing sanity check, not a benchmark
    of any one estimator.
    """

    def setUp(self):
        raw = synthetic_fusion_table(n_decisions=150, seed=4)
        data = add_partner_columns(raw)
        self.train_df, self.test_df = split_by_fusion_event(data, test_size=0.3, random_state=0)
        self.feature_columns = ["center_dist", "nearest_dist", "tip_distance"]

    def test_beats_random_baseline(self):
        pairs = build_pairwise_dataset(self.train_df, feature_columns=self.feature_columns)
        clf = train_pairwise_classifier(
            pairs, feature_columns=self.feature_columns,
            estimator=LogisticRegression(random_state=0), random_state=0,
        )
        scored = self.test_df.copy()
        scored["pred_prob"] = score_candidates_pairwise(clf, scored, feature_columns=self.feature_columns)

        top1 = top_k_accuracy(scored, k=1)
        random_top1 = random_baseline_top_k(scored, k=1)
        self.assertGreater(top1, random_top1 + 0.15)

    def test_single_candidate_decision_gets_uninformative_score(self):
        pairs = build_pairwise_dataset(self.train_df, feature_columns=self.feature_columns)
        clf = train_pairwise_classifier(
            pairs, feature_columns=self.feature_columns,
            estimator=LogisticRegression(random_state=0), random_state=0,
        )
        single = pd.DataFrame({
            "decision_id": [999],
            "flag": [True],
            "center_dist": [5.0], "nearest_dist": [5.0], "tip_distance": [5.0],
        })
        scores = score_candidates_pairwise(clf, single, feature_columns=self.feature_columns)
        self.assertEqual(scores[0], 0.5)


class LearningCurvePairwiseTests(unittest.TestCase):
    def setUp(self):
        raw = synthetic_fusion_table(n_decisions=200, seed=5)
        data = add_partner_columns(raw)
        self.train_df, self.test_df = split_by_fusion_event(data, test_size=0.3, random_state=0)
        self.feature_columns = ["center_dist", "nearest_dist", "tip_distance"]

    def test_returns_one_row_per_fraction_repeat_k_and_accuracy_improves(self):
        curve = learning_curve_pairwise(
            self.train_df, self.test_df, feature_columns=self.feature_columns,
            estimator=LogisticRegression(random_state=0), fractions=(0.3, 1.0), n_repeats=2,
            random_state=0,
        )
        self.assertEqual(set(curve["fraction"].unique()), {0.3, 1.0})
        self.assertEqual(len(curve), len(curve.drop_duplicates(["fraction", "repeat", "k"])))
        # more training events -> same or better mean accuracy on this separable data
        by_fraction = curve.groupby("fraction")["top_k_acc"].mean()
        self.assertGreaterEqual(by_fraction.loc[1.0], by_fraction.loc[0.3] - 0.05)

    def test_estimator_is_not_mutated_by_repeated_cloning(self):
        est = LogisticRegression(random_state=0)
        learning_curve_pairwise(
            self.train_df, self.test_df, feature_columns=self.feature_columns,
            estimator=est, fractions=(0.5,), n_repeats=3, random_state=0,
        )
        self.assertFalse(hasattr(est, "coef_"))  # the passed-in instance itself was never fit


class EvaluateFeatureGroupsTests(unittest.TestCase):
    def setUp(self):
        raw = synthetic_fusion_table(n_decisions=200, seed=6)
        data = add_partner_columns(raw)
        self.train_df, self.test_df = split_by_fusion_event(data, test_size=0.3, random_state=0)

    def test_informative_group_beats_constant_group(self):
        data_train = self.train_df.copy()
        data_test = self.test_df.copy()
        for df in (data_train, data_test):
            df["constant_col"] = 1.0  # same value for every row -> 0 after differencing
        groups = {
            "distance": ["center_dist", "nearest_dist", "tip_distance"],
            "constant": ["constant_col"],
        }
        out = evaluate_feature_groups(
            data_train, data_test, feature_groups=groups,
            estimator=LogisticRegression(random_state=0), k_values=(1,), random_state=0,
        )
        acc = out.set_index("group")["top_k_acc"]
        self.assertGreater(acc["distance"], acc["constant"] + 0.1)

    def test_constant_only_group_matches_its_own_random_baseline(self):
        data_train = self.train_df.copy()
        data_test = self.test_df.copy()
        for df in (data_train, data_test):
            df["constant_col"] = 1.0
        out = evaluate_feature_groups(
            data_train, data_test, feature_groups={"constant": ["constant_col"]},
            estimator=LogisticRegression(random_state=0), k_values=(1,), random_state=0,
        )
        row = out.iloc[0]
        self.assertAlmostEqual(row["top_k_acc"], row["random_top1"], delta=0.1)

    def test_one_row_per_group_and_k(self):
        groups = {
            "distance": ["center_dist", "nearest_dist", "tip_distance"],
            "center_dist_only": ["center_dist"],
        }
        out = evaluate_feature_groups(
            self.train_df, self.test_df, feature_groups=groups,
            estimator=LogisticRegression(random_state=0), k_values=(1, 2), random_state=0,
        )
        self.assertEqual(len(out), 4)
        self.assertEqual(set(out["group"]), {"distance", "center_dist_only"})
        self.assertEqual(set(out["n_features"]), {3, 1})


if __name__ == "__main__":
    unittest.main()
