import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional
from scipy.optimize import linear_sum_assignment


# =========================================================
# Configuration
# =========================================================

@dataclass
class MatchConfig:
    n_agents_per_side: int = 50     # number of A(p) and B(m) agents
    n_trials: int = 100
    sensing_radius: Optional[float] = None   # None = all visible
    topk_list: Tuple[int, ...] = (1, 2, 3, 5)
    seed: int = 42

# =========================================================
# Utilities
# =========================================================
def generate_agents(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate 2D coordinates."""
    return rng.uniform(0, size=(n, 2))


def pairwise_dist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distance matrix, shape (len(A), len(B))."""
    diff = A[:, None, :] - B[None, :, :]
    return np.linalg.norm(diff, axis=2)


def mask_by_radius(D: np.ndarray, radius: Optional[float]) -> np.ndarray:
    """
    Visible edges mask. True means visible/allowed.
    If radius is None, everything is visible.
    """
    if radius is None:
        return np.ones_like(D, dtype=bool)
    return D <= radius


# =========================================================
# Global optimal baseline
# =========================================================
def optimal_matching(D: np.ndarray, visible_mask: Optional[np.ndarray] = None) -> Dict[int, int]:
    """
    Solve global minimum-cost one-to-one matching from A to B.
    Uses Hungarian algorithm (linear_sum_assignment), so this is bipartite.

    Returns:
        match_A_to_B: dict {a_idx: b_idx}
    """
    cost = D.copy()

    if visible_mask is not None:
        # Large penalty for invisible edges
        large = 1e9
        cost = np.where(visible_mask, cost, large)

    row_ind, col_ind = linear_sum_assignment(cost)

    # sanity check: reject if some selected edge is actually forbidden
    if visible_mask is not None:
        if not np.all(visible_mask[row_ind, col_ind]):
            raise ValueError("No feasible perfect matching under current sensing radius.")

    return {int(i): int(j) for i, j in zip(row_ind, col_ind)}


def optimal_partial_matching(D: np.ndarray, visible_mask: Optional[np.ndarray] = None) -> Dict[int, int]:
    """
    Solve minimum-cost one-to-one matching from A to B, allowing unmatched nodes.

    If a perfect matching is impossible, keep the matchable subset and leave the
    remaining A or B unmatched.

    Parameters
    ----------
    D : (nA, nB) array
        Cost matrix.
    visible_mask : (nA, nB) bool array, optional
        True where edge A_i -> B_j is allowed.
    Returns
    -------
    match_A_to_B : dict
        {a_idx: b_idx} for real matched pairs only
    unmatched_A : list
        Indices of A that were left unmatched
    unmatched_B : list
        Indices of B that were left unmatched
    """
    D = np.asarray(D, dtype=float)
    nA, nB = D.shape

    if visible_mask is None:
        visible_mask = np.ones_like(D, dtype=bool)
    else:
        visible_mask = np.asarray(visible_mask, dtype=bool)

    # Large forbidden cost
    INF = 1e12

    # Real edge costs
    real_cost = np.where(visible_mask, D, INF)

    # Choose unmatched penalty automatically:
    # prefer any feasible real match over leaving both sides unmatched
    feasible_vals = real_cost[real_cost < INF]

    if feasible_vals.size == 0:
        unmatched_penalty = 1.0
    else:
        unmatched_penalty = feasible_vals.max() + 1.0

    # Build augmented square matrix of size (nA + nB) x (nA + nB)
    #
    # block structure:
    # [ real A -> real B      | real A -> dummy B ]
    # [ dummy A -> real B     | dummy A -> dummy B ]
    #
    # Each A_i can match its own dummy B_i => unmatched A_i
    # Each B_j can match from its own dummy A_j => unmatched B_j
    N = nA + nB
    C = np.full((N, N), INF, dtype=float)

    # Top-left: real A -> real B
    C[:nA, :nB] = real_cost

    # Top-right: real A -> dummy B  (leave A unmatched)
    for i in range(nA):
        C[i, nB + i] = unmatched_penalty

    # Bottom-left: dummy A -> real B  (leave B unmatched)
    for j in range(nB):
        C[nA + j, j] = unmatched_penalty

    # Bottom-right: dummy A -> dummy B
    # zero cost, so unused dummy nodes can pair freely
    C[nA:, nB:] = 0.0

    row_ind, col_ind = linear_sum_assignment(C)

    match_A_to_B: Dict[int, int] = {}
    unmatched_A: List[int] = []
    unmatched_B: List[int] = []

    for r, c in zip(row_ind, col_ind):
        # real A matched to real B
        if r < nA and c < nB:
            if C[r, c] >= INF:
                continue
            match_A_to_B[int(r)] = int(c)

        # real A matched to dummy => unmatched A
        elif r < nA and c >= nB:
            unmatched_A.append(int(r))

        # dummy matched to real B => unmatched B
        elif r >= nA and c < nB:
            unmatched_B.append(int(c))

    return match_A_to_B


# =========================================================
# Baseline local policies
# =========================================================
def random_matching(D: np.ndarray, visible_mask: np.ndarray, rng: np.random.Generator) -> Dict[int, int]:
    """
    Sequential random one-to-one matching.
    """
    nA, nB = D.shape
    available_B = set(range(nB))
    order = rng.permutation(nA)

    match = {}
    for i in order:
        candidates = [j for j in available_B if visible_mask[i, j]]
        if not candidates:
            continue
        j = rng.choice(candidates)
        match[int(i)] = int(j)
        available_B.remove(j)

    return match


def greedy_nearest_matching(D: np.ndarray, visible_mask: np.ndarray) -> Dict[int, int]:
    """
    Sequential nearest-neighbor one-to-one matching.
    """
    nA, nB = D.shape
    available_B = set(range(nB))
    unmatched_A = set(range(nA))
    match = {}

    while unmatched_A and available_B:
        best_pair = None
        best_cost = np.inf

        for i in unmatched_A:
            valid_js = [j for j in available_B if visible_mask[i, j]]
            if not valid_js:
                continue
            j_best = min(valid_js, key=lambda j: D[i, j])
            c = D[i, j_best]
            if c < best_cost:
                best_cost = c
                best_pair = (i, j_best)

        if best_pair is None:
            break

        i, j = best_pair
        match[int(i)] = int(j)
        unmatched_A.remove(i)
        available_B.remove(j)

    return match


# =========================================================
# Yeast local policy template
# =========================================================
def reaction(l, kd=1):
    return l/(l+kd+1e-6)


def diffusion_1d(x, t, D=1.0, M=1.0):
    """
    1D diffusion from point source.

    Parameters
    ----------
    x : float or np.ndarray
        position(s)
    t : float or np.ndarray
        time(s), must be > 0
    D : float
        diffusion coefficient
    M : float
        total released amount

    Returns
    -------
    c : float or np.ndarray
        concentration
    """
    x = np.asarray(x)
    t = np.asarray(t)

    # avoid division by zero
    t = np.maximum(t, 1e-12)

    return M / np.sqrt(4 * np.pi * D * t) * np.exp(-x**2 / (4 * D * t))


class YeastMatching():
    def __init__(self, A, B, D):
        self.A = A
        self.B = B
        self.D = D

        self.commit_threshold = 1
        self.response_threshold = 0.1
        self.decay = -0.02

        self.diffusion_a = 10
        self.diffusion_b = 1
        self.m_a = 10
        self.m_b = 1

        self.max_iter = 1000

    def select_non_conflicting_pairs(self, commit_map, score_map, active_A, active_B):
        """
        From all committed candidate pairs, pick a set of pairs such that
        each A and each B is used at most once in this round.

        Greedy: sort candidate pairs by score descending.
        """
        candidate_rc = np.argwhere(commit_map)

        if len(candidate_rc) == 0:
            return []

        # keep only active rows/cols
        filtered = []
        for i, j in candidate_rc:
            if active_A[i] and active_B[j]:
                filtered.append((i, j, score_map[i, j]))

        if not filtered:
            return []

        filtered.sort(key=lambda x: x[2], reverse=True)

        chosen = []
        used_A = set()
        used_B = set()

        for i, j, s in filtered:
            if i not in used_A and j not in used_B:
                chosen.append((i, j))
                used_A.add(i)
                used_B.add(j)

        return chosen

    def agent_update_matrix(self, A):
        """
        A: 2D matrix, choose one column for each row
        returns:
            final_idx: chosen column index for each row
            final_vals: chosen value for each row
        """
        n_rows, n_cols = A.shape

        max_vals = np.max(A, axis=1)
        argmax_idx = np.argmax(A, axis=1)

        rand_idx = np.random.randint(0, n_cols, size=n_rows)
        use_max = max_vals > self.response_threshold

        final_idx = np.where(use_max, argmax_idx, rand_idx)
        final_vals = A[np.arange(n_rows), final_idx]

        return final_idx, final_vals

    def simulate_pairing(self, map_A_pheno, map_B_pheno,):
        """
        Iteratively update A/B pheromone maps and commit pairs.
        Committed A and B are removed from future consideration.
        """

        map_A_pheno = map_A_pheno.copy()
        map_B_pheno = map_B_pheno.copy()

        nA, nB = map_A_pheno.shape

        active_A = np.ones(nA, dtype=bool)
        active_B = np.ones(nB, dtype=bool)

        committed_pairs = []

        for step in range(self.max_iter):
            if not active_A.any() or not active_B.any():
                break

            # Only active rows/cols participate
            valid_mask = np.outer(active_A, active_B)

            # -------- A update --------
            # inactive positions are ignored
            A_view_for_choice = np.where(valid_mask, map_B_pheno, -np.inf)

            # only active A rows choose
            active_A_idx = np.where(active_A)[0]
            if len(active_A_idx) > 0 and active_B.any():
                chosen_B_local, sensed_B = self.agent_update_matrix(A_view_for_choice[active_A_idx, :])

                updated_A_map = np.zeros_like(map_A_pheno) + self.decay

                chosen_B = chosen_B_local
                row_idx = active_A_idx

                # add released signal only to chosen targets
                new_pheno_A = diffusion_1d(
                    self.D[row_idx, chosen_B],
                    t=1,
                    D=self.diffusion_b,
                    M=reaction(sensed_B)
                )*self.m_a

                updated_A_map[row_idx, chosen_B] += new_pheno_A

                # optionally stop decay on invalid positions:
                updated_A_map[~valid_mask] = 0

                map_A_pheno += updated_A_map
                map_A_pheno = np.clip(map_A_pheno, 0, None)

            # -------- B update --------
            # each active B chooses one active A
            B_view_for_choice = np.where(valid_mask, map_A_pheno, -np.inf)

            active_B_idx = np.where(active_B)[0]
            if len(active_B_idx) > 0 and active_A.any():
                # transpose so each B is now a row
                chosen_A_local, sensed_A = self.agent_update_matrix(
                    B_view_for_choice[:, active_B_idx].T)

                updated_B_map = np.zeros_like(map_B_pheno) + self.decay

                col_idx = active_B_idx
                chosen_A = chosen_A_local

                new_pheno_B = diffusion_1d(
                    self.D[chosen_A, col_idx],
                    t=1,
                    D=self.diffusion_a,
                    M=reaction(sensed_A),
                )*self.m_b

                updated_B_map[chosen_A, col_idx] += new_pheno_B

                # optionally stop decay on invalid positions:
                updated_B_map[~valid_mask] = 0

                map_B_pheno += updated_B_map
                map_B_pheno = np.clip(map_B_pheno, 0, None)

            # -------- commit detection --------
            commit_map = (
                (map_A_pheno >= self.commit_threshold) &
                (map_B_pheno >= self.commit_threshold) &
                valid_mask
            )

            # score used to rank simultaneous candidate commitments
            score_map = map_A_pheno + map_B_pheno

            new_pairs = self.select_non_conflicting_pairs(
                commit_map, score_map, active_A, active_B
            )

            if new_pairs:
                committed_pairs.extend(new_pairs)

                for i, j in new_pairs:
                    active_A[i] = False
                    active_B[j] = False

                    # optional: zero out committed row/col so they never influence again
                    map_A_pheno[i, :] = 0
                    map_A_pheno[:, j] = 0
                    map_B_pheno[i, :] = 0
                    map_B_pheno[:, j] = 0

            # # optional early stop: nothing can happen anymore
            # if not new_pairs and step > self.max_iter:
            #     # keep or remove this depending on whether you want long accumulation
            #     pass
        # return {
        #     "committed_pairs": committed_pairs,
        #     "map_A_pheno": map_A_pheno,
        #     "map_B_pheno": map_B_pheno,
        #     "active_A": active_A,
        #     "active_B": active_B,
        # }
        return dict(committed_pairs)

    def matching(self):
        map_A_pheno = diffusion_1d(self.D, t=1)
        map_B_pheno = diffusion_1d(self.D, t=1)
        committed_pairs = self.simulate_pairing(map_A_pheno, map_B_pheno)
        return committed_pairs


class Evaluator():
    # =========================================================
    # Evaluation
    # =========================================================
    def matching_cost(self, match: Dict[int, int], D: np.ndarray) -> float:
        return float(sum(D[i, j] for i, j in match.items()))

    def exact_match_accuracy(self, match: Dict[int, int], optimal_match: Dict[int, int], nA: int) -> float:
        correct = sum(1 for i, j in match.items() if optimal_match.get(i, None) == j)
        return correct / nA

    def unmatched_fraction(self, match: Dict[int, int], nA: int) -> float:
        return 1.0 - len(match) / nA


# =========================================================
# One trial
# =========================================================

# def run_one_trial(cfg: MatchConfig, rng: np.random.Generator) -> Dict[str, Dict]:
#     A = generate_agents(cfg.n_agents_per_side, rng)
#     B = generate_agents(cfg.n_agents_per_side, rng)
#     D = pairwise_dist(A, B)
#     visible_mask = mask_by_radius(D, cfg.sensing_radius)

#     # Global optimal
#     optimal = optimal_matching(D, visible_mask)
#     optimal_cost = matching_cost(optimal, D)

#     # Baselines + your policy
#     methods = {
#         "random": random_matching(D, visible_mask, rng),
#         "greedy": greedy_nearest_matching(D, visible_mask),
#         "yeast_policy": yeast_policy_pairing(
#             A=A,
#             B=B,
#             distance=D,
#             rng=rng,
#             # Theta=Theta,
#             # visible_mask=visible_mask,
#             # score_fn=local_policy_matching,
#             # conflict_mode="mutual_best",
#         ),
#     }

#     out = {}
#     for name, match in methods.items():
#         cost = matching_cost(match, D)
#         out[name] = {
#             "cost": cost,
#             "cost_ratio": cost / optimal_cost if optimal_cost > 0 else np.nan,
#             "accuracy": exact_match_accuracy(match, optimal, cfg.n_agents_per_side),
#             "unmatched_fraction": unmatched_fraction(match, cfg.n_agents_per_side),
#             "topk": topk_hit_rates(match, D, visible_mask, cfg.topk_list),
#             "n_matches": len(match),
#         }

#     out["optimal"] = {
#         "cost": optimal_cost,
#         "cost_ratio": 1.0,
#         "accuracy": 1.0,
#         "unmatched_fraction": 0.0,
#         "topk": {k: 1.0 for k in cfg.topk_list},
#         "n_matches": len(optimal),
#     }

#     return out


# # =========================================================
# # Multi-trial benchmark
# # =========================================================

# def summarize_trials(trial_results: List[Dict[str, Dict]], topk_list=(1, 2, 3, 5)) -> Dict[str, Dict]:
#     methods = trial_results[0].keys()
#     summary = {}

#     for method in methods:
#         cost = np.array([r[method]["cost"] for r in trial_results], dtype=float)
#         cost_ratio = np.array([r[method]["cost_ratio"] for r in trial_results], dtype=float)
#         accuracy = np.array([r[method]["accuracy"] for r in trial_results], dtype=float)
#         unmatched = np.array([r[method]["unmatched_fraction"] for r in trial_results], dtype=float)
#         n_matches = np.array([r[method]["n_matches"] for r in trial_results], dtype=float)

#         topk_summary = {}
#         for k in topk_list:
#             arr = np.array([r[method]["topk"][k] for r in trial_results], dtype=float)
#             topk_summary[k] = {
#                 "mean": float(np.mean(arr)),
#                 "std": float(np.std(arr)),
#             }

#         summary[method] = {
#             "cost_mean": float(np.mean(cost)),
#             "cost_std": float(np.std(cost)),
#             "cost_ratio_mean": float(np.mean(cost_ratio)),
#             "cost_ratio_std": float(np.std(cost_ratio)),
#             "accuracy_mean": float(np.mean(accuracy)),
#             "accuracy_std": float(np.std(accuracy)),
#             "unmatched_mean": float(np.mean(unmatched)),
#             "unmatched_std": float(np.std(unmatched)),
#             "n_matches_mean": float(np.mean(n_matches)),
#             "n_matches_std": float(np.std(n_matches)),
#             "topk": topk_summary,
#         }

#     return summary


# def print_summary(summary: Dict[str, Dict], topk_list=(1, 2, 3, 5)) -> None:
#     for method, stats in summary.items():
#         print(f"\n=== {method} ===")
#         print(f"cost              : {stats['cost_mean']:.3f} ± {stats['cost_std']:.3f}")
#         print(f"cost ratio        : {stats['cost_ratio_mean']:.3f} ± {stats['cost_ratio_std']:.3f}")
#         print(f"exact match acc   : {stats['accuracy_mean']:.3f} ± {stats['accuracy_std']:.3f}")
#         print(f"unmatched frac    : {stats['unmatched_mean']:.3f} ± {stats['unmatched_std']:.3f}")
#         print(f"n matches         : {stats['n_matches_mean']:.2f} ± {stats['n_matches_std']:.2f}")
#         for k in topk_list:
#             print(f"top-{k} hit rate      : {stats['topk'][k]['mean']:.3f} ± {stats['topk'][k]['std']:.3f}")


# def run_benchmark(cfg: SimConfig) -> Dict[str, Dict]:
#     rng = np.random.default_rng(cfg.seed)
#     trial_results = []

#     for _ in range(cfg.n_trials):
#         res = run_one_trial(cfg, rng)
#         trial_results.append(res)

#     summary = summarize_trials(trial_results, cfg.topk_list)
#     print_summary(summary, cfg.topk_list)
#     return summary


# =========================================================
# Main
# =========================================================

# if __name__ == "__main__":
#     cfg = SimConfig(
#         n_agents_per_side=50,
#         box_size=100.0,
#         n_trials=100,
#         sensing_radius=None,   # try e.g. 30.0 for local visibility
#         topk_list=(1, 2, 3, 5),
#         seed=42,
#     )

#     summary = run_benchmark(cfg)
