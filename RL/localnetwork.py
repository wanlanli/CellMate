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
    # box_size: float = 100.0         # 2D square [0, box_size]^2
    n_trials: int = 100
    sensing_radius: Optional[float] = None   # None = all visible
    topk_list: Tuple[int, ...] = (1, 2, 3, 5)
    commit_threshold = 1
    
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


def pairwise_angle(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Angle from each A_i to each B_j, in radians, shape (len(A), len(B)).
    Can be useful if your strategy uses direction.
    """
    diff = B[None, :, :] - A[:, None, :]
    return np.arctan2(diff[..., 1], diff[..., 0])


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


def my_local_policy_score(
    d_ij: float,
    theta_ij: float,
    local_rank_by_distance: int,
) -> float:
    """
    ---------------------------------------------------------
    Replace this with YOUR strategy.
    Higher score = more preferred.
    ---------------------------------------------------------

    Current default:
        prefer shorter distance
    """
    score = -d_ij
    return score


def local_policy_matching(
    D: np.ndarray,
    Theta: np.ndarray,
    visible_mask: np.ndarray,
    score_fn: Callable[[float, float, int], float],
    conflict_mode: str = "mutual_best",
) -> Dict[int, int]:
    """
    Local policy:
      1. each A selects one preferred B using only local info
      2. conflicts are resolved

    conflict_mode:
      - "mutual_best": keep only mutual best pairs, then iterate
      - "priority_distance": resolve B conflicts by smallest distance
    """
    nA, nB = D.shape
    remaining_A = set(range(nA))
    remaining_B = set(range(nB))
    match: Dict[int, int] = {}

    while remaining_A and remaining_B:
        proposals_A_to_B = {}
        proposals_B_to_A = {}

        # A -> preferred B
        for i in remaining_A:
            candidates = [j for j in remaining_B if visible_mask[i, j]]
            if not candidates:
                continue

            sorted_candidates = sorted(candidates, key=lambda j: D[i, j])
            best_j = None
            best_score = -np.inf

            for rank, j in enumerate(sorted_candidates, start=1):
                s = score_fn(
                    d_ij=float(D[i, j]),
                    theta_ij=float(Theta[i, j]),
                    local_rank_by_distance=rank,
                )
                if s > best_score:
                    best_score = s
                    best_j = j

            proposals_A_to_B[i] = best_j

        if not proposals_A_to_B:
            break

        if conflict_mode == "priority_distance":
            # Many A may propose same B; B keeps the closest one
            grouped: Dict[int, List[int]] = {}
            for i, j in proposals_A_to_B.items():
                grouped.setdefault(j, []).append(i)

            accepted_pairs = []
            for j, proposers in grouped.items():
                i_best = min(proposers, key=lambda i: D[i, j])
                accepted_pairs.append((i_best, j))

        elif conflict_mode == "mutual_best":
            # B also chooses best A among remaining_A using same local rule
            for j in remaining_B:
                candidates = [i for i in remaining_A if visible_mask[i, j]]
                if not candidates:
                    continue

                sorted_candidates = sorted(candidates, key=lambda i: D[i, j])
                best_i = None
                best_score = -np.inf

                for rank, i in enumerate(sorted_candidates, start=1):
                    s = score_fn(
                        d_ij=float(D[i, j]),
                        theta_ij=float(Theta[i, j] + np.pi),  # rough reverse angle
                        local_rank_by_distance=rank,
                    )
                    if s > best_score:
                        best_score = s
                        best_i = i

                proposals_B_to_A[j] = best_i

            accepted_pairs = []
            for i, j in proposals_A_to_B.items():
                if proposals_B_to_A.get(j, None) == i:
                    accepted_pairs.append((i, j))

        else:
            raise ValueError(f"Unknown conflict_mode: {conflict_mode}")

        if not accepted_pairs:
            break

        for i, j in accepted_pairs:
            match[int(i)] = int(j)

        matched_A = {i for i, _ in accepted_pairs}
        matched_B = {j for _, j in accepted_pairs}
        remaining_A -= matched_A
        remaining_B -= matched_B

    return match


# =========================================================
# Yeast local policy template
# =========================================================
def select_non_conflicting_pairs(commit_map, score_map, active_A, active_B):
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


def agent_update_matrix(A, response_threshold):
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
    use_max = max_vals > response_threshold

    final_idx = np.where(use_max, argmax_idx, rand_idx)
    final_vals = A[np.arange(n_rows), final_idx]

    return final_idx, final_vals


def simulate_pairing(
    map_A_pheno,
    map_B_pheno,
    distance,
    response_threshold,
    commit_threshold,
    decay,
    reaction,
    diffusion_1d,
    max_iter=800,
):
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

    for step in range(max_iter):
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
            chosen_B_local, sensed_B = agent_update_matrix(
                A_view_for_choice[active_A_idx, :],
                response_threshold
            )

            updated_A_map = np.zeros_like(map_A_pheno) + decay

            chosen_B = chosen_B_local
            row_idx = active_A_idx

            # add released signal only to chosen targets
            new_pheno_A = diffusion_1d(
                distance[row_idx, chosen_B],
                t=1,
                M=reaction(sensed_B)
            )

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
            chosen_A_local, sensed_A = agent_update_matrix(
                B_view_for_choice[:, active_B_idx].T,
                response_threshold
            )

            updated_B_map = np.zeros_like(map_B_pheno) + decay

            col_idx = active_B_idx
            chosen_A = chosen_A_local

            new_pheno_B = diffusion_1d(
                distance[chosen_A, col_idx],
                t=1,
                M=reaction(sensed_A)
            )

            updated_B_map[chosen_A, col_idx] += new_pheno_B

            # optionally stop decay on invalid positions:
            updated_B_map[~valid_mask] = 0

            map_B_pheno += updated_B_map
            map_B_pheno = np.clip(map_B_pheno, 0, None)

        # -------- commit detection --------
        commit_map = (
            (map_A_pheno >= commit_threshold) &
            (map_B_pheno >= commit_threshold) &
            valid_mask
        )

        # score used to rank simultaneous candidate commitments
        score_map = map_A_pheno + map_B_pheno

        new_pairs = select_non_conflicting_pairs(
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

        # optional early stop: nothing can happen anymore
        if not new_pairs and step > max_iter:
            # keep or remove this depending on whether you want long accumulation
            pass
    # return {
    #     "committed_pairs": committed_pairs,
    #     "map_A_pheno": map_A_pheno,
    #     "map_B_pheno": map_B_pheno,
    #     "active_A": active_A,
    #     "active_B": active_B,
    # }
    return dict(committed_pairs)


def reaction(l, kd=1):
    return l/(l+kd)


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


def yeast_policy_pairing(A, B, distance, rng, decay=-0.02, response_threshold=0.1, commit_threshold=1,):
    map_A_pheno = diffusion_1d(distance, t=1)
    map_B_pheno = diffusion_1d(distance, t=1)
    committed_pairs = simulate_pairing(map_A_pheno, 
                                       map_B_pheno, 
                                       distance, 
                                       response_threshold, 
                                       commit_threshold, 
                                       decay, 
                                       reaction,
                                       diffusion_1d)
    return committed_pairs

# =========================================================
# Evaluation
# =========================================================


def matching_cost(match: Dict[int, int], D: np.ndarray) -> float:
    return float(sum(D[i, j] for i, j in match.items()))


def exact_match_accuracy(match: Dict[int, int], optimal_match: Dict[int, int], nA: int) -> float:
    correct = sum(1 for i, j in match.items() if optimal_match.get(i, None) == j)
    return correct / nA


def topk_hit_rates(match: Dict[int, int], D: np.ndarray, visible_mask: np.ndarray, topk_list=(1, 2, 3, 5)) -> Dict[int, float]:
    results = {k: 0 for k in topk_list}
    nA = D.shape[0]

    for i, chosen_j in match.items():
        valid_js = np.where(visible_mask[i])[0]
        ranked_js = valid_js[np.argsort(D[i, valid_js])]

        for k in topk_list:
            if chosen_j in ranked_js[:k]:
                results[k] += 1

    return {k: v / nA for k, v in results.items()}


def unmatched_fraction(match: Dict[int, int], nA: int) -> float:
    return 1.0 - len(match) / nA


# =========================================================
# One trial
# =========================================================

def run_one_trial(cfg: SimConfig, rng: np.random.Generator) -> Dict[str, Dict]:
    A = generate_agents(cfg.n_agents_per_side, rng)
    B = generate_agents(cfg.n_agents_per_side, rng)

    D = pairwise_dist(A, B)
    Theta = pairwise_angle(A, B)
    visible_mask = mask_by_radius(D, cfg.sensing_radius)

    # Global optimal
    optimal = optimal_matching(D, visible_mask)
    optimal_cost = matching_cost(optimal, D)

    # Baselines + your policy
    methods = {
        "random": random_matching(D, visible_mask, rng),
        "greedy": greedy_nearest_matching(D, visible_mask),
        "local_policy": local_policy_matching(
            D=D,
            Theta=Theta,
            visible_mask=visible_mask,
            score_fn=my_local_policy_score,
            conflict_mode="mutual_best",
        ),
        "yeast_policy": yeast_policy_pairing(
            A=A,
            B=B,
            distance=D,
            rng=rng,
            # Theta=Theta,
            # visible_mask=visible_mask,
            # score_fn=local_policy_matching,
            # conflict_mode="mutual_best",
        ),
    }

    out = {}
    for name, match in methods.items():
        cost = matching_cost(match, D)
        out[name] = {
            "cost": cost,
            "cost_ratio": cost / optimal_cost if optimal_cost > 0 else np.nan,
            "accuracy": exact_match_accuracy(match, optimal, cfg.n_agents_per_side),
            "unmatched_fraction": unmatched_fraction(match, cfg.n_agents_per_side),
            "topk": topk_hit_rates(match, D, visible_mask, cfg.topk_list),
            "n_matches": len(match),
        }

    out["optimal"] = {
        "cost": optimal_cost,
        "cost_ratio": 1.0,
        "accuracy": 1.0,
        "unmatched_fraction": 0.0,
        "topk": {k: 1.0 for k in cfg.topk_list},
        "n_matches": len(optimal),
    }

    return out


# =========================================================
# Multi-trial benchmark
# =========================================================

def summarize_trials(trial_results: List[Dict[str, Dict]], topk_list=(1, 2, 3, 5)) -> Dict[str, Dict]:
    methods = trial_results[0].keys()
    summary = {}

    for method in methods:
        cost = np.array([r[method]["cost"] for r in trial_results], dtype=float)
        cost_ratio = np.array([r[method]["cost_ratio"] for r in trial_results], dtype=float)
        accuracy = np.array([r[method]["accuracy"] for r in trial_results], dtype=float)
        unmatched = np.array([r[method]["unmatched_fraction"] for r in trial_results], dtype=float)
        n_matches = np.array([r[method]["n_matches"] for r in trial_results], dtype=float)

        topk_summary = {}
        for k in topk_list:
            arr = np.array([r[method]["topk"][k] for r in trial_results], dtype=float)
            topk_summary[k] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
            }

        summary[method] = {
            "cost_mean": float(np.mean(cost)),
            "cost_std": float(np.std(cost)),
            "cost_ratio_mean": float(np.mean(cost_ratio)),
            "cost_ratio_std": float(np.std(cost_ratio)),
            "accuracy_mean": float(np.mean(accuracy)),
            "accuracy_std": float(np.std(accuracy)),
            "unmatched_mean": float(np.mean(unmatched)),
            "unmatched_std": float(np.std(unmatched)),
            "n_matches_mean": float(np.mean(n_matches)),
            "n_matches_std": float(np.std(n_matches)),
            "topk": topk_summary,
        }

    return summary


def print_summary(summary: Dict[str, Dict], topk_list=(1, 2, 3, 5)) -> None:
    for method, stats in summary.items():
        print(f"\n=== {method} ===")
        print(f"cost              : {stats['cost_mean']:.3f} ± {stats['cost_std']:.3f}")
        print(f"cost ratio        : {stats['cost_ratio_mean']:.3f} ± {stats['cost_ratio_std']:.3f}")
        print(f"exact match acc   : {stats['accuracy_mean']:.3f} ± {stats['accuracy_std']:.3f}")
        print(f"unmatched frac    : {stats['unmatched_mean']:.3f} ± {stats['unmatched_std']:.3f}")
        print(f"n matches         : {stats['n_matches_mean']:.2f} ± {stats['n_matches_std']:.2f}")
        for k in topk_list:
            print(f"top-{k} hit rate      : {stats['topk'][k]['mean']:.3f} ± {stats['topk'][k]['std']:.3f}")


def run_benchmark(cfg: SimConfig) -> Dict[str, Dict]:
    rng = np.random.default_rng(cfg.seed)
    trial_results = []

    for _ in range(cfg.n_trials):
        res = run_one_trial(cfg, rng)
        trial_results.append(res)

    summary = summarize_trials(trial_results, cfg.topk_list)
    print_summary(summary, cfg.topk_list)
    return summary


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":
    cfg = SimConfig(
        n_agents_per_side=50,
        box_size=100.0,
        n_trials=100,
        sensing_radius=None,   # try e.g. 30.0 for local visibility
        topk_list=(1, 2, 3, 5),
        seed=42,
    )

    summary = run_benchmark(cfg)
