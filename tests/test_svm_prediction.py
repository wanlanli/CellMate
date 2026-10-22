"""Tests for cellmate.mating._svm_prediction -- the partner-pick classifier
built on top of feature_extract.get_mating_feature's fusion table: per-decision
min-max normalization, a group-aware train/test split, an SVC that scores
"is this candidate the true partner", and the top-k partner-pick accuracy
(not row-level accuracy -- see module docstring) used to judge it.

Pulled out of the exploratory hpm/02_svm_prediction.ipynb notebook; this is
the unit-level check (synthetic data, no CellNetwork). For the real-data
numbers and the learning-curve datasize check, see that notebook (in the
Pipeline_Cell_Behavior_Prediction repo).
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cellmate.mating import (
    FEATURE_COLUMNS,
    CANONICAL_FEATURE_COLUMNS,
    CANDIDATE_NORMALIZE_COLUMNS,
    canonicalize_candidate_features,
    normalize_candidates,
    add_partner_columns,
    split_by_fusion_event,
    drop_decisions_without_true_partner,
    drop_trivial_decisions,
    train_partner_classifier,
    score_candidates,
    evaluate_row_level,
    top_k_hits,
    top_k_accuracy,
    top_k_accuracy_by_value,
    pick_one_decision_per_view,
    true_partner_score_trajectory,
    random_baseline_top_k,
    learning_curve_by_fusion_event,
)
from cellmate.mating._svm_prediction import NORMALIZE_COLUMNS


def synthetic_fusion_table(n_decisions=60, min_candidates=2, max_candidates=5, seed=0):
    """One mating-event decision per `ref_id`, each with 2-5 candidate
    partners. The true partner (flag=True) is drawn with systematically
    smaller center_dist/nearest_dist/tip_distance -- a learnable signal, with
    enough per-row noise that it isn't a trivial threshold rule -- so a
    trained classifier should clear the random-guess baseline but not hit
    100% top-1 accuracy.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for decision_id in range(n_decisions):
        n_candidates = int(rng.integers(min_candidates, max_candidates + 1))
        true_idx = int(rng.integers(0, n_candidates))
        ref_id = 1000 + decision_id
        time_stamp = int(rng.integers(0, 100))
        for j in range(n_candidates):
            is_true = j == true_idx
            base = 2.0 if is_true else 8.0
            center_dist = max(0.1, base + rng.normal(0, 2.0))
            nearest_dist = max(0.1, center_dist * 0.8 + rng.normal(0, 0.5))
            tip_distance = max(0.1, center_dist * 1.1 + rng.normal(0, 0.5))
            rows.append(dict(
                decision_id=decision_id, group_index=decision_id, ref_type=1, ref_id=ref_id,
                p_id=ref_id, m_id=2000 + decision_id * 10 + j, flag=bool(is_true),
                p_start=0, m_start=int(rng.integers(0, 5)),
                p_area=rng.normal(30, 5), p_major=rng.normal(8, 1), p_minor=rng.normal(4, 0.5),
                p_eccentricity=rng.uniform(0.5, 0.9),
                p_neighbor_same=int(rng.integers(0, 4)), p_neighbor_diff=int(rng.integers(0, 4)),
                m_area=rng.normal(25, 5), m_major=rng.normal(7, 1), m_minor=rng.normal(4, 0.5),
                m_eccentricity=rng.uniform(0.5, 0.9),
                m_neighbor_same=int(rng.integers(0, 4)), m_neighbor_diff=int(rng.integers(0, 4)),
                p_angle=rng.uniform(0, 2 * np.pi), m_angle=rng.uniform(0, 2 * np.pi),
                p_angle_index=int(rng.integers(0, 8)), m_angle_index=int(rng.integers(0, 8)),
                p_angle_norm=rng.uniform(0, 1), m_angle_norm=rng.uniform(0, 1),
                center_dist=center_dist, nearest_dist=nearest_dist, tip_distance=tip_distance,
                time_stamp=time_stamp,
            ))
    df = pd.DataFrame(rows)
    for col, rank_col in [("nearest_dist", "near_dist_rank"), ("center_dist", "center_dist_rank"),
                          ("tip_distance", "tip_dist_rank"), ("p_start", "p_star_time_rank"),
                          ("m_start", "m_star_time_rank")]:
        df[rank_col] = df.groupby("decision_id")[col].rank(method="min")
    df["time_diff"] = df["m_start"] - df["p_start"]
    return df


def synthetic_multi_image_table(n_images=3, n_decisions_per_image=20, seed=0):
    """Concatenate several `synthetic_fusion_table` calls with an `image`
    column, each restarting `decision_id` at 0 -- reproducing exactly how
    `get_mating_feature`'s `decision_id` (an `ngroup()` local to one call)
    collides across images once multiple images' tables are concatenated,
    the way the hpm pipeline's 01_generate_data.ipynb does.
    """
    frames = []
    for image_idx in range(n_images):
        frame = synthetic_fusion_table(n_decisions=n_decisions_per_image, seed=seed + image_idx)
        frame["image"] = f"image_{image_idx}"
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def synthetic_multi_snapshot_table(n_events=20, snapshots_per_event=5,
                                    min_candidates=2, max_candidates=4, seed=0):
    """Several time-snapshot decisions per fusion event (`fusion_key`), like
    real data (~19 decisions/event on average, one every ~10 frames) -- for
    testing that splitting/subsampling must group by fusion event, not by
    decision: decisions of the same event are near-duplicates of each other.
    """
    rng = np.random.default_rng(seed)
    rows = []
    decision_id = 0
    for event in range(n_events):
        ref_id = 1000 + event
        for snap in range(snapshots_per_event):
            n_candidates = int(rng.integers(min_candidates, max_candidates + 1))
            true_idx = int(rng.integers(0, n_candidates))
            for j in range(n_candidates):
                rows.append(dict(
                    decision_id=decision_id, fusion_key=event, group_index=decision_id,
                    ref_type=1, ref_id=ref_id, p_id=ref_id, m_id=2000 + decision_id * 10 + j,
                    flag=bool(j == true_idx), time_stamp=snap * 10,
                    center_dist=1.0, nearest_dist=1.0, tip_distance=1.0,
                ))
            decision_id += 1
    return pd.DataFrame(rows)


class NormalizeCandidatesTests(unittest.TestCase):
    def test_scaled_independently_per_group(self):
        df = pd.DataFrame({
            "decision_id": [0, 0, 0, 1, 1],
            "center_dist": [10.0, 20.0, 30.0, 100.0, 300.0],
        })
        out = normalize_candidates(df, columns=["center_dist"])
        np.testing.assert_allclose(out.loc[out.decision_id == 0, "center_dist_norm"], [0.0, 0.5, 1.0])
        np.testing.assert_allclose(out.loc[out.decision_id == 1, "center_dist_norm"], [0.0, 1.0])

    def test_constant_group_normalizes_to_zero_not_nan(self):
        df = pd.DataFrame({"decision_id": [0, 0, 0], "center_dist": [5.0, 5.0, 5.0]})
        out = normalize_candidates(df, columns=["center_dist"])
        self.assertFalse(out["center_dist_norm"].isna().any())
        np.testing.assert_allclose(out["center_dist_norm"], [0.0, 0.0, 0.0])

    def test_original_columns_untouched(self):
        df = pd.DataFrame({"decision_id": [0, 0], "center_dist": [1.0, 2.0]})
        out = normalize_candidates(df, columns=["center_dist"])
        np.testing.assert_allclose(out["center_dist"], [1.0, 2.0])


class AddPartnerColumnsTests(unittest.TestCase):
    def test_partner_id_follows_ref_type(self):
        df = pd.DataFrame({
            "decision_id": [0, 0, 1],
            "ref_type": [1, 1, 2],
            "ref_id": [10, 10, 20],
            "p_id": [10, 10, 99],
            "m_id": [11, 12, 20],
        })
        out = add_partner_columns(df)
        np.testing.assert_array_equal(out["partner_id"], [11.0, 12.0, 99.0])

    def test_n_candidates_counts_distinct_partners_within_one_decision(self):
        df = pd.DataFrame({
            "decision_id": [0, 0, 0, 1],
            "ref_type": [1, 1, 1, 1],
            "ref_id": [10, 10, 10, 20],
            "p_id": [10, 10, 10, 20],
            "m_id": [11, 12, 12, 21],  # duplicate m_id=12 shouldn't double-count
        })
        out = add_partner_columns(df)
        self.assertEqual(list(out.loc[out.ref_id == 10, "N_candidates"].unique()), [2])
        self.assertEqual(list(out.loc[out.ref_id == 20, "N_candidates"].unique()), [1])

    def test_n_candidates_is_per_decision_not_per_whole_event(self):
        # Same ref_id, same fusion event (group_index), but two different
        # snapshots (decision_id) with different candidate-set sizes --
        # confirmed on real data that this actually happens (candidates
        # come and go over an event's life): N_candidates must track the
        # snapshot a row belongs to, not the event-wide total distinct
        # partners ever seen (the previous, incorrect default).
        df = pd.DataFrame({
            "decision_id": [0, 0, 0, 1, 1],
            "group_index": [0, 0, 0, 0, 0],
            "ref_type": [1, 1, 1, 1, 1],
            "ref_id": [10, 10, 10, 10, 10],
            "p_id": [10, 10, 10, 10, 10],
            "m_id": [11, 12, 13, 11, 12],  # decision 0: 3 candidates; decision 1: 2
        })
        out = add_partner_columns(df)
        self.assertEqual(list(out.loc[out.decision_id == 0, "N_candidates"].unique()), [3])
        self.assertEqual(list(out.loc[out.decision_id == 1, "N_candidates"].unique()), [2])


class CanonicalizeCandidateFeaturesTests(unittest.TestCase):
    """p_*/m_* mean "the reference cell's own feature" or "the candidate's
    own feature" depending on ref_type -- confirmed on real data: p_area is
    constant within 9275/9283 ref_type==1 decisions (it's the one physical
    ref cell, not a per-candidate measurement), while m_area (the actual
    candidate) varies. canonicalize_candidate_features relabels these into
    a ref_type-agnostic candidate_*/ref_* view instead.
    """

    def test_candidate_own_picks_the_varying_side(self):
        df = pd.DataFrame({
            "ref_type": [1, 2], "ref_id": [100, 200], "p_id": [100, 999], "m_id": [999, 200],
            "p_area": [10.0, 20.0], "m_area": [30.0, 40.0],
            "p_major": [1.0, 2.0], "m_major": [3.0, 4.0],
            "p_minor": [1.0, 2.0], "m_minor": [3.0, 4.0],
            "p_eccentricity": [1.0, 2.0], "m_eccentricity": [3.0, 4.0],
            "p_neighbor_same": [1, 2], "m_neighbor_same": [3, 4],
            "p_neighbor_diff": [1, 2], "m_neighbor_diff": [3, 4],
            "p_star_time_rank": [1, 2], "m_star_time_rank": [3, 4],
            "p_angle": [0.1, 0.2], "m_angle": [0.3, 0.4],
            "p_angle_index": [1, 2], "m_angle_index": [3, 4],
            "p_angle_norm": [0.1, 0.2], "m_angle_norm": [0.3, 0.4],
        })
        out = canonicalize_candidate_features(df)
        # ref_type==1 -> ref is p, so candidate is m; ref_type==2 -> reversed
        np.testing.assert_array_equal(out["candidate_area"], [30.0, 20.0])
        np.testing.assert_array_equal(out["candidate_star_time_rank"], [3, 2])

    def test_is_p_ref_preserves_which_side_was_reference(self):
        # which strain is the reference isn't noise -- p/m have a known
        # tip-formation bias asymmetry -- so it must survive canonicalization.
        df = pd.DataFrame({
            "ref_type": [1, 2, 1], "ref_id": [100, 200, 100], "p_id": [100, 999, 100], "m_id": [999, 200, 999],
            "p_area": [10.0, 20.0, 30.0], "m_area": [30.0, 40.0, 50.0],
            "p_major": [0.0] * 3, "m_major": [0.0] * 3, "p_minor": [0.0] * 3, "m_minor": [0.0] * 3,
            "p_eccentricity": [0.0] * 3, "m_eccentricity": [0.0] * 3,
            "p_neighbor_same": [0] * 3, "m_neighbor_same": [0] * 3,
            "p_neighbor_diff": [0] * 3, "m_neighbor_diff": [0] * 3,
            "p_star_time_rank": [0] * 3, "m_star_time_rank": [0] * 3,
            "p_angle": [0.0] * 3, "m_angle": [0.0] * 3,
            "p_angle_index": [0] * 3, "m_angle_index": [0] * 3,
            "p_angle_norm": [0.0] * 3, "m_angle_norm": [0.0] * 3,
        })
        out = canonicalize_candidate_features(df)
        np.testing.assert_array_equal(out["is_p_ref"], [1, 0, 1])

    def test_angle_columns_relabel_both_sides_not_just_candidate(self):
        df = pd.DataFrame({
            "ref_type": [1, 2], "ref_id": [100, 200], "p_id": [100, 999], "m_id": [999, 200],
            "p_angle": [0.1, 0.2], "m_angle": [0.3, 0.4],
            "p_angle_index": [1, 2], "m_angle_index": [3, 4],
            "p_angle_norm": [0.11, 0.22], "m_angle_norm": [0.33, 0.44],
            # unused by the angle pairs but required by _CANDIDATE_OWN_PAIRS
            "p_area": [0.0, 0.0], "m_area": [0.0, 0.0],
            "p_major": [0.0, 0.0], "m_major": [0.0, 0.0],
            "p_minor": [0.0, 0.0], "m_minor": [0.0, 0.0],
            "p_eccentricity": [0.0, 0.0], "m_eccentricity": [0.0, 0.0],
            "p_neighbor_same": [0, 0], "m_neighbor_same": [0, 0],
            "p_neighbor_diff": [0, 0], "m_neighbor_diff": [0, 0],
            "p_star_time_rank": [0, 0], "m_star_time_rank": [0, 0],
        })
        out = canonicalize_candidate_features(df)
        # ref_type==1: ref is p -> ref_angle=p_angle, candidate_angle=m_angle
        self.assertAlmostEqual(out.loc[0, "ref_angle"], 0.1)
        self.assertAlmostEqual(out.loc[0, "candidate_angle"], 0.3)
        # ref_type==2: ref is m -> reversed
        self.assertAlmostEqual(out.loc[1, "ref_angle"], 0.4)
        self.assertAlmostEqual(out.loc[1, "candidate_angle"], 0.2)

    def test_original_p_m_columns_untouched(self):
        df = pd.DataFrame({
            "ref_type": [1], "ref_id": [100], "p_id": [100], "m_id": [999],
            "p_area": [10.0], "m_area": [30.0],
            "p_major": [0.0], "m_major": [0.0], "p_minor": [0.0], "m_minor": [0.0],
            "p_eccentricity": [0.0], "m_eccentricity": [0.0],
            "p_neighbor_same": [0], "m_neighbor_same": [0],
            "p_neighbor_diff": [0], "m_neighbor_diff": [0],
            "p_star_time_rank": [0], "m_star_time_rank": [0],
            "p_angle": [0.0], "m_angle": [0.0],
            "p_angle_index": [0], "m_angle_index": [0],
            "p_angle_norm": [0.0], "m_angle_norm": [0.0],
        })
        out = canonicalize_candidate_features(df)
        self.assertEqual(out.loc[0, "p_area"], 10.0)
        self.assertEqual(out.loc[0, "m_area"], 30.0)


class SplitByFusionEventTests(unittest.TestCase):
    def test_falls_back_to_decision_level_when_no_fusion_key(self):
        # synthetic_fusion_table has no fusion_key column (one decision per
        # event), so split_by_fusion_event falls back to decision-level
        # grouping -- still no leakage, just via _resolve_group_col.
        df = synthetic_fusion_table(n_decisions=40, seed=1)
        train_df, test_df = split_by_fusion_event(df, test_size=0.3, random_state=0)
        self.assertEqual(
            set(train_df["decision_id"]) & set(test_df["decision_id"]), set())
        self.assertEqual(len(train_df) + len(test_df), len(df))

    def test_keeps_every_snapshot_of_one_fusion_event_on_one_side(self):
        # real data: ~19 decisions (time snapshots) per fusion event on
        # average -- splitting at the decision level would scatter one
        # event's highly-correlated snapshots across train and test.
        df = synthetic_multi_snapshot_table(n_events=20, snapshots_per_event=5, seed=0)
        train_df, test_df = split_by_fusion_event(df, test_size=0.3, random_state=0)
        train_events = set(train_df["fusion_key"])
        test_events = set(test_df["fusion_key"])
        self.assertEqual(train_events & test_events, set())
        for fusion_key, group in df.groupby("fusion_key"):
            sides = set()
            if group.index.isin(train_df.index).any():
                sides.add("train")
            if group.index.isin(test_df.index).any():
                sides.add("test")
            self.assertEqual(len(sides), 1, f"fusion_key={fusion_key} split across both sides")

    def test_decision_level_grouping_would_have_leaked(self):
        # contrast case: explicitly forcing decision-level grouping on the
        # same data does let snapshots of one event land on both sides --
        # demonstrating why fusion-event grouping is the right default.
        df = synthetic_multi_snapshot_table(n_events=20, snapshots_per_event=5, seed=0)
        train_df, test_df = split_by_fusion_event(
            df, test_size=0.3, random_state=0, group_col="decision_id")
        train_events = set(train_df["fusion_key"])
        test_events = set(test_df["fusion_key"])
        self.assertTrue(train_events & test_events, "fixture should reproduce the leakage this guards against")

    def test_p_view_and_m_view_of_one_event_are_separate_splittable_groups(self):
        # Every fusion_key has exactly two ref_id views (the p-cell's and
        # the m-cell's candidate search) -- confirmed on real data. They're
        # different cells choosing among different candidates, not
        # duplicates of each other, so -- unlike two decisions of the same
        # view -- they're allowed to land on different sides of the split.
        rows = []
        decision_id = 0
        for event in range(20):
            for ref_id in (f"{event}p", f"{event}m"):
                for snap in range(5):
                    rows.append(dict(
                        decision_id=decision_id, fusion_key=event, ref_id=ref_id,
                        time_stamp=snap * 10, flag=True,
                    ))
                    decision_id += 1
        df = pd.DataFrame(rows)

        train_df, test_df = split_by_fusion_event(df, test_size=0.3, random_state=0)

        # each (fusion_key, ref_id) view's own snapshots stay together
        for (fusion_key, ref_id), group in df.groupby(["fusion_key", "ref_id"]):
            sides = set()
            if group.index.isin(train_df.index).any():
                sides.add("train")
            if group.index.isin(test_df.index).any():
                sides.add("test")
            self.assertEqual(len(sides), 1, f"(fusion_key={fusion_key}, ref_id={ref_id}) split across both sides")

        # but the two views of at least one event land on different sides
        # (not forced together) -- otherwise this fixture wouldn't actually
        # be testing the behavior this test is for.
        train_ref_ids = set(train_df["ref_id"])
        test_ref_ids = set(test_df["ref_id"])
        split_events = {r[:-1] for r in train_ref_ids} & {r[:-1] for r in test_ref_ids}
        self.assertTrue(split_events, "fixture should reproduce at least one event split across both views")


class MultiImageGroupingTests(unittest.TestCase):
    """`decision_id` collides across images (it's an `ngroup()` local to one
    `get_mating_feature` call) -- confirmed on real data: 1503 distinct
    `decision_id` values covered 18730 actual (image, decision_id) decisions.
    Every grouped function must resolve to `["image", "decision_id"]` once an
    `image` column is present, not silently merge unrelated decisions that
    happen to share a `decision_id` number.
    """

    def setUp(self):
        self.df = synthetic_multi_image_table(n_images=4, n_decisions_per_image=15, seed=0)
        # sanity-check the fixture actually reproduces the collision
        self.assertGreater(self.df["decision_id"].duplicated().sum(), 0)
        self.assertEqual(self.df["decision_id"].nunique(), 15)

    def test_normalize_candidates_does_not_mix_images(self):
        normed = normalize_candidates(self.df, columns=["center_dist"])
        # recompute independently per (image, decision_id) and compare
        for (image, decision_id), group in self.df.groupby(["image", "decision_id"]):
            expected_min, expected_max = group["center_dist"].min(), group["center_dist"].max()
            span = expected_max - expected_min or 1
            expected = (group["center_dist"] - expected_min) / span
            got = normed.loc[group.index, "center_dist_norm"]
            np.testing.assert_allclose(got.to_numpy(), expected.to_numpy())

    def test_split_by_fusion_event_no_leakage_across_images(self):
        train_df, test_df = split_by_fusion_event(self.df, test_size=0.3, random_state=0)
        train_keys = set(map(tuple, train_df[["image", "decision_id"]].to_numpy()))
        test_keys = set(map(tuple, test_df[["image", "decision_id"]].to_numpy()))
        self.assertEqual(train_keys & test_keys, set())
        self.assertEqual(len(train_df) + len(test_df), len(self.df))

    def test_top_k_hits_indexed_per_image_decision(self):
        scored = self.df.copy()
        scored["pred_prob"] = np.where(scored["flag"], 0.9, 0.1)
        hits = top_k_hits(scored, k=1)
        self.assertEqual(hits.index.names, ["image", "decision_id"])
        # one hit per real (image, decision_id) decision, not per decision_id number
        self.assertEqual(len(hits), self.df.groupby(["image", "decision_id"]).ngroups)
        self.assertTrue((hits == 1).all())


class TopKAccuracyTests(unittest.TestCase):
    def test_top1_perfect_when_true_partner_scores_highest(self):
        df = pd.DataFrame({
            "decision_id": [0, 0, 1, 1, 1],
            "flag": [True, False, False, True, False],
            "pred_prob": [0.9, 0.1, 0.2, 0.8, 0.3],
            "N_candidates": [2, 2, 3, 3, 3],
        })
        self.assertEqual(top_k_accuracy(df, k=1), 1.0)

    def test_top1_zero_but_top2_one_when_true_partner_scores_second(self):
        df = pd.DataFrame({
            "decision_id": [0, 0, 0],
            "flag": [False, True, False],
            "pred_prob": [0.9, 0.5, 0.1],
            "N_candidates": [3, 3, 3],
        })
        self.assertEqual(top_k_accuracy(df, k=1), 0.0)
        self.assertEqual(top_k_accuracy(df, k=2), 1.0)

    def test_decisions_without_any_true_partner_are_excluded(self):
        df = pd.DataFrame({
            "decision_id": [0, 0, 1, 1],
            "flag": [False, False, True, False],
            "pred_prob": [0.9, 0.1, 0.2, 0.8],
            "N_candidates": [2, 2, 2, 2],
        })
        # decision 0 has no true partner at all (upstream labeling issue) and
        # must not be averaged in -- only decision 1 counts, and it's a miss.
        self.assertEqual(top_k_accuracy(df, k=1), 0.0)

    def test_min_candidates_filters_subset(self):
        df = pd.DataFrame({
            "decision_id": [0, 0, 1, 1, 1],
            "flag": [True, False, False, True, False],
            "pred_prob": [0.9, 0.1, 0.9, 0.1, 0.05],
            "N_candidates": [2, 2, 3, 3, 3],
        })
        # decision 0 (N=2) is a hit, decision 1 (N=3) is a miss.
        self.assertEqual(top_k_accuracy(df, k=1), 0.5)
        self.assertEqual(top_k_accuracy(df, k=1, min_candidates=3), 0.0)


class RandomBaselineTopKTests(unittest.TestCase):
    def test_matches_one_over_n_for_k1(self):
        df = pd.DataFrame({"decision_id": [0, 1, 2], "N_candidates": [2, 4, 5]})
        self.assertAlmostEqual(random_baseline_top_k(df, k=1), np.mean([1 / 2, 1 / 4, 1 / 5]))

    def test_clips_at_one_for_k_greater_than_n(self):
        df = pd.DataFrame({"decision_id": [0, 1], "N_candidates": [2, 4]})
        self.assertAlmostEqual(random_baseline_top_k(df, k=2), np.mean([1.0, 2 / 4]))


class DropDecisionsWithoutTruePartnerTests(unittest.TestCase):
    def test_drops_only_groups_with_zero_true_rows(self):
        df = pd.DataFrame({
            "decision_id": [0, 0, 1, 1, 1],
            "flag": [True, False, False, False, False],
        })
        clean, n_dropped_decisions, n_dropped_rows = drop_decisions_without_true_partner(df)
        self.assertEqual(n_dropped_decisions, 1)
        self.assertEqual(n_dropped_rows, 3)
        self.assertEqual(sorted(clean["decision_id"].unique()), [0])

    def test_does_not_let_a_true_row_in_one_image_rescue_a_colliding_decision_id_in_another(self):
        # image_0's decision_id=0 has a true partner; image_1's decision_id=0
        # (same number, unrelated event) does not. Grouping by decision_id
        # alone would incorrectly "rescue" image_1's group via image_0's True row.
        df = pd.DataFrame({
            "image": ["img0", "img0", "img1", "img1"],
            "decision_id": [0, 0, 0, 0],
            "flag": [True, False, False, False],
        })
        clean, n_dropped_decisions, n_dropped_rows = drop_decisions_without_true_partner(df)
        self.assertEqual(n_dropped_decisions, 1)
        self.assertEqual(n_dropped_rows, 2)
        self.assertEqual(set(clean["image"]), {"img0"})


class DropTrivialDecisionsTests(unittest.TestCase):
    def test_drops_single_candidate_decisions_by_default(self):
        df = pd.DataFrame({
            "decision_id": [0, 1, 1, 2, 2, 2],
            "N_candidates": [1, 2, 2, 3, 3, 3],
        })
        clean, n_dropped_decisions, n_dropped_rows = drop_trivial_decisions(df)
        self.assertEqual(n_dropped_decisions, 1)
        self.assertEqual(n_dropped_rows, 1)
        self.assertEqual(sorted(clean["decision_id"].unique()), [1, 2])

    def test_min_candidates_is_adjustable(self):
        df = pd.DataFrame({
            "decision_id": [0, 1, 1, 2, 2, 2],
            "N_candidates": [1, 2, 2, 3, 3, 3],
        })
        clean, n_dropped_decisions, n_dropped_rows = drop_trivial_decisions(df, min_candidates=3)
        self.assertEqual(n_dropped_decisions, 2)
        self.assertEqual(sorted(clean["decision_id"].unique()), [2])


class TopKAccuracyByValueTests(unittest.TestCase):
    def test_breaks_out_accuracy_and_baseline_per_value(self):
        df = pd.DataFrame({
            "decision_id": [0, 0, 1, 1, 2, 2],
            "time_stamp": [10, 10, 10, 10, 50, 50],
            "flag": [True, False, False, True, True, False],
            # decision 0 (time_stamp 10) is a hit; decision 1 (time_stamp 10) is a miss
            # (true partner scores lower, 0.4 < 0.6); decision 2 (time_stamp 50) is a hit
            "pred_prob": [0.9, 0.1, 0.6, 0.4, 0.8, 0.2],
            "N_candidates": [2, 2, 2, 2, 2, 2],
        })
        out = top_k_accuracy_by_value(df, value_col="time_stamp", k=1, min_count=1)
        row10 = out[out["time_stamp"] == 10].iloc[0]
        row50 = out[out["time_stamp"] == 50].iloc[0]
        self.assertAlmostEqual(row10["top_k_acc"], 0.5)
        self.assertEqual(row10["n_decisions"], 2)
        self.assertAlmostEqual(row50["top_k_acc"], 1.0)
        self.assertEqual(row50["n_decisions"], 1)
        self.assertAlmostEqual(row50["random_baseline"], 0.5)


def _two_view_scored_table(view_times):
    """`view_times`: {view_id: [time_to_fusion, ...]}. Two candidate rows
    per decision (one True, pred_prob 0.9; one False, pred_prob 0.1).
    """
    rows = []
    for view, times in view_times.items():
        for t in times:
            for cand, is_true in [(0, True), (1, False)]:
                rows.append(dict(
                    decision_id=f"{view}_{t}", fusion_key=view, ref_id=view,
                    time_to_fusion=t, flag=is_true, pred_prob=0.9 if is_true else 0.1,
                ))
    return pd.DataFrame(rows)


class PickOneDecisionPerViewTests(unittest.TestCase):
    def test_random_picks_exactly_one_decision_per_view(self):
        df = _two_view_scored_table({"v0": [100, 60, 10], "v1": [80, 40]})
        out = pick_one_decision_per_view(df, strategy="random", random_state=0)
        self.assertEqual(out.groupby("ref_id")["decision_id"].nunique().to_dict(), {"v0": 1, "v1": 1})
        self.assertEqual(len(out), 4)  # 2 views * 2 candidate rows per decision

    def test_closest_to_fusion_picks_smallest_value(self):
        df = _two_view_scored_table({"v0": [100, 60, 10], "v1": [80, 40]})
        out = pick_one_decision_per_view(df, strategy="closest_to_fusion")
        picked = out.drop_duplicates("decision_id").set_index("ref_id")["time_to_fusion"]
        self.assertEqual(picked.to_dict(), {"v0": 10, "v1": 40})

    def test_cutoff_picks_first_crossing_not_closest(self):
        # times [100, 60, 55, 10], cutoff=70 -> eligible {60, 55, 10} ->
        # should pick 60 (largest eligible = first chronological crossing),
        # not 10 (which closest_to_fusion would pick).
        df = _two_view_scored_table({"v": [100, 60, 55, 10]})
        out = pick_one_decision_per_view(df, strategy="cutoff", cutoff=70)
        picked_time = out.drop_duplicates("decision_id")["time_to_fusion"].iloc[0]
        self.assertEqual(picked_time, 60)

    def test_cutoff_drops_views_that_never_qualify(self):
        df = _two_view_scored_table({"v0": [100, 60, 10], "v1": [80, 40]})
        out = pick_one_decision_per_view(df, strategy="cutoff", cutoff=50)
        picked = out.drop_duplicates("decision_id").set_index("ref_id")["time_to_fusion"]
        self.assertEqual(picked.to_dict(), {"v0": 10, "v1": 40})

        # neither view ever gets within 5 frames of fusion -> both dropped
        out_empty = pick_one_decision_per_view(df, strategy="cutoff", cutoff=5)
        self.assertTrue(out_empty.empty)

    def test_cutoff_requires_cutoff_argument(self):
        df = _two_view_scored_table({"v0": [10]})
        with self.assertRaises(ValueError):
            pick_one_decision_per_view(df, strategy="cutoff")

    def test_unknown_strategy_raises(self):
        df = _two_view_scored_table({"v0": [10]})
        with self.assertRaises(ValueError):
            pick_one_decision_per_view(df, strategy="bogus")

    def test_works_with_a_non_unique_row_index(self):
        # Confirmed on real data: fusion_data_table.csv's row index is ~92%
        # duplicated (an unreset index surviving an upstream pd.concat).
        # idxmin()/idxmax()-based selection must not silently break or
        # raise on this.
        df = _two_view_scored_table({"v0": [100, 60, 10], "v1": [80, 40]})
        df.index = [0] * len(df)
        out = pick_one_decision_per_view(df, strategy="closest_to_fusion")
        picked = out.drop_duplicates("decision_id").set_index("ref_id")["time_to_fusion"]
        self.assertEqual(picked.to_dict(), {"v0": 10, "v1": 40})


class TruePartnerScoreTrajectoryTests(unittest.TestCase):
    def test_one_row_per_decision_tagged_by_view(self):
        df = _two_view_scored_table({"v0": [100, 60, 10], "v1": [80, 40]})
        traj = true_partner_score_trajectory(df)
        self.assertEqual(len(traj), 5)  # 3 + 2 decisions, one row each
        self.assertEqual(set(traj.columns), {"view_id", "time_to_fusion", "pred_prob"})
        # always the True row's own score
        self.assertTrue((traj["pred_prob"] == 0.9).all())
        # 2 distinct, opaque view ids (one per real view) -- not necessarily
        # the original ref_id labels, just a unique id per view for grouping
        self.assertEqual(traj["view_id"].nunique(), 2)
        self.assertEqual(traj.groupby("view_id").size().sort_values().tolist(), [2, 3])

    def test_works_with_a_non_unique_row_index(self):
        df = _two_view_scored_table({"v0": [100, 60, 10], "v1": [80, 40]})
        df.index = [0] * len(df)
        traj = true_partner_score_trajectory(df)
        self.assertEqual(len(traj), 5)
        self.assertTrue((traj["pred_prob"] == 0.9).all())


class ClassifierRoundTripTests(unittest.TestCase):
    """Train on separable synthetic data and check the full pipeline beats
    the random baseline by a wide margin -- a sanity check on the plumbing
    (normalize -> split -> train -> score -> top_k_accuracy), not a
    benchmark of SVC itself.
    """

    def setUp(self):
        raw = synthetic_fusion_table(n_decisions=150, seed=2)
        normed = normalize_candidates(raw, columns=NORMALIZE_COLUMNS)
        self.data = add_partner_columns(normed)
        self.train_df, self.test_df = split_by_fusion_event(self.data, test_size=0.3, random_state=0)

    def test_beats_random_baseline(self):
        clf = train_partner_classifier(self.train_df, feature_columns=FEATURE_COLUMNS, random_state=0)
        scored = self.test_df.copy()
        scored["pred_prob"] = score_candidates(clf, scored, feature_columns=FEATURE_COLUMNS)

        top1 = top_k_accuracy(scored, k=1)
        random_top1 = random_baseline_top_k(scored, k=1)
        self.assertGreater(top1, random_top1 + 0.15)

        row_metrics = evaluate_row_level(clf, self.test_df, feature_columns=FEATURE_COLUMNS)
        for value in row_metrics.values():
            self.assertTrue(0.0 <= value <= 1.0)

    def test_canonical_feature_set_also_beats_random_baseline(self):
        raw = synthetic_fusion_table(n_decisions=150, seed=2)
        canon = canonicalize_candidate_features(raw)
        normed = normalize_candidates(canon, columns=CANDIDATE_NORMALIZE_COLUMNS)
        data = add_partner_columns(normed)
        train_df, test_df = split_by_fusion_event(data, test_size=0.3, random_state=0)

        clf = train_partner_classifier(train_df, feature_columns=CANONICAL_FEATURE_COLUMNS, random_state=0)
        scored = test_df.copy()
        scored["pred_prob"] = score_candidates(clf, scored, feature_columns=CANONICAL_FEATURE_COLUMNS)

        top1 = top_k_accuracy(scored, k=1)
        random_top1 = random_baseline_top_k(scored, k=1)
        self.assertGreater(top1, random_top1 + 0.15)


class LearningCurveTests(unittest.TestCase):
    def test_output_shape_and_more_data_does_not_collapse_accuracy(self):
        raw = synthetic_fusion_table(n_decisions=150, seed=3)
        normed = normalize_candidates(raw, columns=NORMALIZE_COLUMNS)
        data = add_partner_columns(normed)
        train_df, test_df = split_by_fusion_event(data, test_size=0.3, random_state=0)

        fractions = (0.3, 1.0)
        curve = learning_curve_by_fusion_event(
            train_df, test_df, fractions=fractions, n_repeats=3,
            feature_columns=FEATURE_COLUMNS, k_values=(1,), random_state=0)

        self.assertEqual(set(curve.columns),
                         {"fraction", "repeat", "n_train_events", "n_train_decisions", "k", "top_k_acc"})
        self.assertEqual(len(curve), len(fractions) * 3)
        self.assertTrue((curve.groupby("fraction")["n_train_events"].nunique() == 1).all())

        mean_by_fraction = curve.groupby("fraction")["top_k_acc"].mean()
        # Loose bound (not strict monotonicity) -- 5x the training decisions
        # shouldn't make partner-pick accuracy meaningfully worse.
        self.assertGreater(mean_by_fraction.loc[1.0], mean_by_fraction.loc[0.3] - 0.1)

    def test_subsamples_whole_events_not_individual_decisions(self):
        # with 5 snapshots/event, n_train_decisions should track as a
        # multiple of n_train_events (every sampled event contributes all
        # of its decisions), not vary independently of it.
        df = synthetic_multi_snapshot_table(n_events=20, snapshots_per_event=5, seed=0)
        train_df, test_df = split_by_fusion_event(df, test_size=0.3, random_state=0)
        curve = learning_curve_by_fusion_event(
            train_df, test_df, fractions=(0.5, 1.0), n_repeats=1,
            feature_columns=["center_dist"], k_values=(1,), random_state=0)
        ratios = curve["n_train_decisions"] / curve["n_train_events"]
        self.assertTrue((ratios == 5).all())

    def test_row_level_metric_shape(self):
        raw = synthetic_fusion_table(n_decisions=150, seed=3)
        normed = normalize_candidates(raw, columns=NORMALIZE_COLUMNS)
        data = add_partner_columns(normed)
        train_df, test_df = split_by_fusion_event(data, test_size=0.3, random_state=0)

        fractions = (0.3, 1.0)
        curve = learning_curve_by_fusion_event(
            train_df, test_df, fractions=fractions, n_repeats=3,
            feature_columns=FEATURE_COLUMNS, metric="row_level", random_state=0)

        self.assertEqual(set(curve.columns),
                         {"fraction", "repeat", "n_train_events", "n_train_decisions",
                          "accuracy", "recall", "f1"})
        self.assertEqual(len(curve), len(fractions) * 3)
        for col in ("accuracy", "recall", "f1"):
            self.assertTrue(((curve[col] >= 0) & (curve[col] <= 1)).all())

    def test_unknown_metric_raises(self):
        raw = synthetic_fusion_table(n_decisions=20, seed=0)
        normed = normalize_candidates(raw, columns=NORMALIZE_COLUMNS)
        data = add_partner_columns(normed)
        train_df, test_df = split_by_fusion_event(data, test_size=0.3, random_state=0)
        with self.assertRaises(ValueError):
            learning_curve_by_fusion_event(train_df, test_df, metric="bogus")


if __name__ == "__main__":
    unittest.main()
