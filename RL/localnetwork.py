import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
from scipy.optimize import linear_sum_assignment
from .diffustion import distance_concentration_kernel

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


def greedy_global_matching(D: np.ndarray, visible: np.ndarray):
    """
    Sort all visible edges by distance ascending, then greedily pick non-conflicting edges.
    """
    nA, nB = D.shape
    edges = [(D[i, j], i, j) for i in range(nA) for j in range(nB) if visible[i, j]]
    edges.sort(key=lambda x: x[0])

    used_A = set()
    used_B = set()
    match_A_to_B = {}

    for dist, i, j in edges:
        if i not in used_A and j not in used_B:
            match_A_to_B[i] = j
            used_A.add(i)
            used_B.add(j)
    return match_A_to_B


def greedy_rowwise_matching(D: np.ndarray, visible: np.ndarray, row_order: Optional[List[int]] = None,):
    """
    Traverse A in order. Each A_i chooses the nearest currently available visible B_j.
    """
    nA, nB = D.shape
    if row_order is None:
        row_order = list(range(nA))

    available_B = set(range(nB))
    match_A_to_B = {}

    for i in row_order:
        candidates = [j for j in available_B if visible[i, j]]
        if len(candidates) == 0:
            continue
        j_best = min(candidates, key=lambda j: D[i, j])
        match_A_to_B[i] = j_best
        available_B.remove(j_best)
    return match_A_to_B

import numpy as np


def _prepare_visible_mask(D, visible_mask=None):
    D = np.asarray(D, dtype=float)
    if visible_mask is None:
        visible_mask = np.ones_like(D, dtype=bool)
    else:
        visible_mask = np.asarray(visible_mask, dtype=bool)
        if visible_mask.shape != D.shape:
            raise ValueError("visible_mask must have the same shape as D")
    return D, visible_mask


def _argmin_random_tie(cost_row):
    """
    cost_row: 1D array, smaller is better, inf means invalid
    returns chosen index, or -1 if no finite entry exists
    """
    finite = np.isfinite(cost_row)
    if not finite.any():
        return -1
    min_val = np.min(cost_row[finite])
    candidates = np.where(cost_row == min_val)[0]
    return int(np.random.choice(candidates))


def _active_cost_matrix(D, visible_mask, active_A, active_B):
    """
    invalid / inactive edges -> inf
    """
    valid = np.outer(active_A, active_B) & visible_mask
    return np.where(valid, D, np.inf)


def local_nearest_neighbor(D, visible_mask=None):
    """
    Each active A proposes to its nearest visible active B.
    Each B accepts the nearest proposer.
    Repeat until no more matches can be made.

    Returns
    -------
    dict {a_idx: b_idx}
    """
    D, visible_mask = _prepare_visible_mask(D, visible_mask)
    nA, nB = D.shape

    active_A = np.ones(nA, dtype=bool)
    active_B = np.ones(nB, dtype=bool)
    matches = {}

    while active_A.any() and active_B.any():
        cost = _active_cost_matrix(D, visible_mask, active_A, active_B)

        # Each active A chooses nearest active visible B
        proposals = {}  # b -> list of a
        any_proposal = False

        for i in np.where(active_A)[0]:
            j = _argmin_random_tie(cost[i])
            if j >= 0:
                proposals.setdefault(j, []).append(i)
                any_proposal = True

        if not any_proposal:
            break

        new_pairs = []
        for j, proposers in proposals.items():
            # B accepts the proposer with smallest D[i, j]
            dvals = np.array([D[i, j] for i in proposers], dtype=float)
            min_d = np.min(dvals)
            best_candidates = [i for i in proposers if D[i, j] == min_d]
            i_star = int(np.random.choice(best_candidates))
            new_pairs.append((i_star, j))

        if not new_pairs:
            break

        # Commit all non-conflicting pairs (they should already be unique in j;
        # i is also unique because each A proposes once)
        for i, j in new_pairs:
            if active_A[i] and active_B[j]:
                matches[i] = j
                active_A[i] = False
                active_B[j] = False

    return matches

def mutual_nearest_neighbor(D, visible_mask=None):
    """
    Match only mutual nearest neighbors among currently active agents.

    Returns
    -------
    dict {a_idx: b_idx}
    """
    D, visible_mask = _prepare_visible_mask(D, visible_mask)
    nA, nB = D.shape

    active_A = np.ones(nA, dtype=bool)
    active_B = np.ones(nB, dtype=bool)
    matches = {}

    while active_A.any() and active_B.any():
        cost = _active_cost_matrix(D, visible_mask, active_A, active_B)

        chosen_B = np.full(nA, -1, dtype=int)
        for i in np.where(active_A)[0]:
            chosen_B[i] = _argmin_random_tie(cost[i])

        chosen_A = np.full(nB, -1, dtype=int)
        for j in np.where(active_B)[0]:
            chosen_A[j] = _argmin_random_tie(cost[:, j])

        new_pairs = []
        for i in np.where(active_A)[0]:
            j = chosen_B[i]
            if j >= 0 and active_B[j] and chosen_A[j] == i:
                new_pairs.append((i, j))

        if not new_pairs:
            break

        # They should already be conflict-free by construction,
        # but keep safety checks.
        used_A = set()
        used_B = set()
        committed = []
        for i, j in new_pairs:
            if i not in used_A and j not in used_B and active_A[i] and active_B[j]:
                committed.append((i, j))
                used_A.add(i)
                used_B.add(j)

        if not committed:
            break

        for i, j in committed:
            matches[i] = j
            active_A[i] = False
            active_B[j] = False

    return matches


def thresholded_local_commitment(D, distance_threshold, visible_mask=None):
    """
    Instant local commitment:
    An agent only commits if its nearest visible partner is within threshold,
    and commitment happens only for mutual choices.

    Parameters
    ----------
    distance_threshold : float
        Smaller than this threshold means acceptable local commitment.

    Returns
    -------
    dict {a_idx: b_idx}
    """
    D, visible_mask = _prepare_visible_mask(D, visible_mask)
    nA, nB = D.shape

    active_A = np.ones(nA, dtype=bool)
    active_B = np.ones(nB, dtype=bool)
    matches = {}

    while active_A.any() and active_B.any():
        cost = _active_cost_matrix(D, visible_mask, active_A, active_B)

        chosen_B = np.full(nA, -1, dtype=int)
        A_accept = np.zeros(nA, dtype=bool)

        for i in np.where(active_A)[0]:
            j = _argmin_random_tie(cost[i])
            if j >= 0:
                chosen_B[i] = j
                A_accept[i] = (D[i, j] <= distance_threshold)

        chosen_A = np.full(nB, -1, dtype=int)
        B_accept = np.zeros(nB, dtype=bool)

        for j in np.where(active_B)[0]:
            i = _argmin_random_tie(cost[:, j])
            if i >= 0:
                chosen_A[j] = i
                B_accept[j] = (D[i, j] <= distance_threshold)

        new_pairs = []
        for i in np.where(active_A)[0]:
            j = chosen_B[i]
            if (
                j >= 0
                and active_B[j]
                and A_accept[i]
                and chosen_A[j] == i
                and B_accept[j]
            ):
                new_pairs.append((i, j))

        if not new_pairs:
            break

        for i, j in new_pairs:
            if active_A[i] and active_B[j]:
                matches[i] = j
                active_A[i] = False
                active_B[j] = False

    return matches


def asynchronous_nearest_neighbor(
    D,
    visible_mask=None,
    max_steps=10000,
    patience=None,
):
    """
    Asynchronous decentralized nearest-neighbor matching.
    At each step, randomly pick one active agent (from A or B).
    It proposes to its nearest visible active partner.
    If the partner also sees it as nearest, they commit immediately.

    Parameters
    ----------
    max_steps : int
        Hard stop for simulation.
    patience : int or None
        Stop if no new match has occurred for this many steps.
        If None, defaults to 5 * (nA + nB).

    Returns
    -------
    dict {a_idx: b_idx}
    """
    D, visible_mask = _prepare_visible_mask(D, visible_mask)
    nA, nB = D.shape

    active_A = np.ones(nA, dtype=bool)
    active_B = np.ones(nB, dtype=bool)
    matches = {}

    if patience is None:
        patience = 5 * (nA + nB)

    no_progress = 0

    for _ in range(max_steps):
        if not active_A.any() or not active_B.any():
            break
        if no_progress >= patience:
            break

        side_choices = []
        if active_A.any():
            side_choices.append("A")
        if active_B.any():
            side_choices.append("B")
        side = np.random.choice(side_choices)

        cost = _active_cost_matrix(D, visible_mask, active_A, active_B)
        made_match = False

        if side == "A":
            i = int(np.random.choice(np.where(active_A)[0]))
            j = _argmin_random_tie(cost[i])
            if j >= 0:
                # check whether i is also B_j's nearest
                i_back = _argmin_random_tie(cost[:, j])
                if i_back == i and active_B[j]:
                    matches[i] = j
                    active_A[i] = False
                    active_B[j] = False
                    made_match = True

        else:  # side == "B"
            j = int(np.random.choice(np.where(active_B)[0]))
            i = _argmin_random_tie(cost[:, j])
            if i >= 0:
                # check whether j is also A_i's nearest
                j_back = _argmin_random_tie(cost[i])
                if j_back == j and active_A[i]:
                    matches[i] = j
                    active_A[i] = False
                    active_B[j] = False
                    made_match = True

        if made_match:
            no_progress = 0
        else:
            no_progress += 1

    return matches


# =========================================================
# Yeast local policy template
# =========================================================
def reaction(l, kd=0.1, alpha=1, beta=0.1):
    l += 1e-3
    den = l + kd + 1e-6
    out = np.divide(l, den, out=np.zeros_like(den, dtype=float), where=np.isfinite(l) & np.isfinite(den))
    out = alpha*out+beta*l
    return out


class YeastMatching():
    def __init__(self, A, B, D):
        self.A = A
        self.B = B
        self.D = D

        self.commit_threshold = 3
        # self.response_threshold = 0.1
        self.decay = -0.2

        self.diffusion_a = 0.05
        self.diffusion_b = 0.1
        # self.m_a = 20
        # self.m_b = 20

        self.max_iter = 100
        self.map_A_pheno = np.zeros(self.D.shape)
        self.map_B_pheno = np.zeros(self.D.shape)

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

    def random_argmax_fast(self, arr):
        arr = np.asarray(arr)
        row_max = arr.max(axis=1, keepdims=True)
        mask = arr == row_max

        row_index = np.array([
            np.random.choice(np.flatnonzero(row_mask))
            for row_mask in mask])

        return row_index, row_max.flatten()

    def simulate_pairing(self, map_A_pheno, map_B_pheno,):
        """
        Iteratively update A/B pheromone maps and commit pairs.
        Committed A and B are removed from future consideration.
        """
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
                chosen_B_local, sensed_B = self.random_argmax_fast(A_view_for_choice[active_A_idx, :])
                # chosen_B_local, sensed_B = self.agent_update_matrix(A_view_for_choice[active_A_idx, :])

                updated_A_map = np.zeros_like(map_A_pheno) + self.decay

                chosen_B = chosen_B_local
                row_idx = active_A_idx

                # # add released signal only to chosen targets
                # sensed_B_diff = distance_concentration_kernel(
                #     self.D[row_idx, chosen_B],
                #     spread=self.diffusion_b,
                #     amplitude=sensed_B,
                # )
                sensed_B_diff = sensed_B.copy()
                sensed_B_diff[sensed_B_diff <= 0] = 1e-3
                new_pheno_A = reaction(sensed_B_diff, kd=0.01)

                updated_A_map[row_idx, chosen_B] += new_pheno_A
                # print("step: ", step, "    sensed_B: ",new_pheno_A, "new_pheno_A")
                # optionally stop decay on invalid positions:
                updated_A_map[~valid_mask] = 0

                # map_A_pheno += updated_A_map
                # map_A_pheno = np.clip(map_A_pheno, 0, None)

            # -------- B update --------
            # each active B chooses one active A
            B_view_for_choice = np.where(valid_mask, map_A_pheno, -np.inf)

            active_B_idx = np.where(active_B)[0]
            if len(active_B_idx) > 0 and active_A.any():
                # transpose so each B is now a row
                chosen_A_local, sensed_A = self.random_argmax_fast(
                    B_view_for_choice[:, active_B_idx].T)

                updated_B_map = np.zeros_like(map_B_pheno) + self.decay

                col_idx = active_B_idx
                chosen_A = chosen_A_local

                # sensed_A_diff = distance_concentration_kernel(
                #     self.D[chosen_A, col_idx],
                #     spread=self.diffusion_a,
                #     amplitude=sensed_A,
                # )
                sensed_A_diff = sensed_A.copy()
                sensed_A_diff[sensed_A_diff <= 0] = 1e-3
                new_pheno_B = reaction(sensed_A_diff, kd=0.01)

                updated_B_map[chosen_A, col_idx] += new_pheno_B
                # print("step: ", step, "    sensed_A: ",sensed_A, "new_pheno_B: ", new_pheno_B)
                # optionally stop decay on invalid positions:
                updated_B_map[~valid_mask] = 0


                map_A_pheno += updated_A_map
                map_A_pheno = np.clip(map_A_pheno, 0, None)

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
        return dict(committed_pairs)

    def matching(self):
        self.map_A_pheno = distance_concentration_kernel(self.D, self.diffusion_a, 0.1) # np.zeros(self.D.shape) # distance_concentration_kernel(self.D, self.diffusion_a, 1)  # np.zeros(self.D.shape)
        self.map_B_pheno = distance_concentration_kernel(self.D, self.diffusion_b, 0.1)
        committed_pairs = self.simulate_pairing(self.map_A_pheno, self.map_B_pheno)
        return committed_pairs


class Evaluator():
    def __init__(self, match_optimal, D):
        self.match_optimal = match_optimal
        self.D = D
        self.K_opt = len(self.match_optimal)
        self.C_opt = self.matching_cost(self.match_optimal, D)
        self.avg_C_opt = self.C_opt / self.K_opt if self.K_opt > 0 else 0.0,
        self.E_opt = self.matching_edges(match_optimal)

    # =========================================================
    # Evaluation
    # =========================================================
    def extract_matched_distances(self, match_A_to_B: Dict[int, int], D: np.ndarray) -> np.ndarray:
        if len(match_A_to_B) == 0:
            return np.array([], dtype=float)
        return np.array([D[i, j] for i, j in match_A_to_B.items()], dtype=float)

    def matching_cost(self, match: Dict[int, int], D: np.ndarray) -> float:
        return self.extract_matched_distances(match, D).sum()

    def exact_match_accuracy(self, match: Dict[int, int], optimal_match: Dict[int, int], nA: int) -> float:
        correct = sum(1 for i, j in match.items() if optimal_match.get(i, None) == j)
        return correct / nA

    def unmatched_fraction(self, match: Dict[int, int], nA: int) -> float:
        return 1.0 - len(match) / nA

    def matching_edges(self, match):
        return set((int(i), int(j)) for i, j in match.items())

    def mean_lower_percent(self, x, percent=0.9):
        x = np.asarray(x)
        n = len(x)
        k = int(np.ceil(n * percent))  # number to keep (lowest 80%)

        x_sorted = np.sort(x)
        low = x_sorted[:k]

        return np.mean(low)

    def metric(self, match):
        D = self.D
        dists = self.extract_matched_distances(match, D)
        nA = D.shape[0]
        nB = D.shape[1]
        K_curr = len(match)
        C_curr = float(dists.sum()) if K_curr else 0.0
        overlap = len(self.matching_edges(match) & self.E_opt)
        precision = overlap / K_curr if K_curr > 0 else np.nan
        recall = overlap / self.K_opt if self.K_opt > 0 else np.nan
        f1 = (2 * precision * recall / (precision + recall) if K_curr > 0 and self.K_opt > 0 and (precision + recall) > 0 else np.nan)
        mec = {
            "matching_rate": K_curr / min(nA, nB) if min(nA, nB) > 0 else 0.0,
            "A_coverage": K_curr / nA if nA > 0 else 0.0,
            "B_coverage": K_curr / nB if nB > 0 else 0.0,
            "total_distance": C_curr,
            "avg_distance": float(dists.mean()) if K_curr else np.nan,
            "avg_distance_90": self.mean_lower_percent(dists),
            "median_distance": float(np.median(dists)) if K_curr else np.nan,
            "std_distance": float(dists.std()) if K_curr else np.nan,
            "cardinality_gap":  self.K_opt - K_curr,
            "cardinality_ratio": K_curr / self.K_opt if self.K_opt > 0 else np.nan,
            "cost_gap": (C_curr - self.C_opt) if K_curr == self.K_opt else np.nan,
            "relative_cost_gap": ((C_curr - self.C_opt) / self.C_opt) if (K_curr == self.K_opt and self.C_opt > 0) else np.nan,
            "edge_overlap": overlap / self.K_opt if self.K_opt > 0 else np.nan,
            "precision_vs_opt": precision,
            "recall_vs_opt": recall,
            "f1_vs_opt": f1,
        }
        return mec
