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
"""Plots for inspecting mating-feature extraction on a `CellNetwork`:
cropped-mask neighbor highlights, candidate-feature time trajectories, and
single-frame pairwise geometry. Pulled out of exploratory notebook cells so
they're reusable (and testable) outside any one notebook.
"""
import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba


def plot_cell_neighbors(cellnet, center_id=None, frame=None, strain="diff",
                         competent_only=False, rng=None, ax=None):
    """Crop around `center_id` at `frame` and highlight its neighbors on the
    label mask: neighbors in bold color by `strain_type` (green=1, red=2,
    yellow=3, gray=other), everything else faded, `center_id` outlined in
    black. `center_id`/`frame` default to a random cell / random frame of
    its own life if omitted.

    Returns a dict with the `center_id`, `frame`, and `neighbors` used (so
    the random choices made are visible to the caller), plus the axes.
    """
    rng = rng or random
    if center_id is None:
        center_id = rng.choice(list(cellnet.cells.keys()))
    if frame is None:
        frame = rng.choice(cellnet.cells[center_id].frames)

    crop_half_size = cellnet.neighbor_threshold
    neighbors = set(cellnet.neighbor(center_id, frame, strain=strain,
                                      competent_only=competent_only))
    neighbors.discard(center_id)

    label_image = cellnet.image[frame]
    label_map = cellnet.label_map[frame]
    center_label = label_map[center_id]

    centers = cellnet.center_overtime(center_id)
    cy, cx = centers[frame - cellnet.cells[center_id].start]
    cy, cx = int(round(cy)), int(round(cx))

    height, width = label_image.shape
    y0 = max(0, cy - crop_half_size)
    y1 = min(height, cy + crop_half_size + 1)
    x0 = max(0, cx - crop_half_size)
    x1 = min(width, cx + crop_half_size + 1)
    cropped_labels = label_image[y0:y1, x0:x1]

    background_colors = {
        0: to_rgba("lightgray", alpha=0.45),
        1: to_rgba("lightgreen", alpha=0.45),
        2: to_rgba("lightcoral", alpha=0.45),
        3: to_rgba("lightyellow", alpha=0.45),
    }
    neighbor_colors = {
        0: to_rgba("gray", alpha=0.45),
        1: to_rgba("green", alpha=0.85),
        2: to_rgba("red", alpha=0.85),
        3: to_rgba("yellow", alpha=0.45),
    }

    overlay = np.zeros((*cropped_labels.shape, 4), dtype=np.float32)
    for cell_id, label in label_map.items():
        mask = cropped_labels == label
        if not mask.any():
            continue
        cell_type = cellnet.cells[cell_id].strain_type
        is_neighbor = cell_id in neighbors
        palette = neighbor_colors if is_neighbor else background_colors
        default_color = to_rgba(
            "gray" if is_neighbor else "lightgray",
            alpha=0.85 if is_neighbor else 0.45,
        )
        overlay[mask] = palette.get(cell_type, default_color)

    created_fig = ax is None
    if created_fig:
        _, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(overlay)

    center_mask = cropped_labels == center_label
    if center_mask.any():
        ax.contour(center_mask.astype(float), levels=[0.5], colors="black", linewidths=1.5)
        rows, cols = np.nonzero(center_mask)
        ax.text(cols.mean(), rows.mean(), str(center_id), color="black",
                 ha="center", va="center", fontsize=9)

    ax.set_title(f"frame {frame} -- center cell {center_id}")
    ax.axis("off")
    if created_fig:
        plt.tight_layout()
        plt.show()

    return dict(center_id=center_id, frame=frame, neighbors=neighbors, ax=ax)


def mating_candidate_features(cellnet, parents_ids, time_step=10):
    """`potential_mating_feature` output for `parents_ids`, plus a
    `partner_id` column: whichever cell in each row isn't the reference
    (p_id when ref_id == m_id, m_id when ref_id == p_id)."""
    data = cellnet.potential_mating_feature(list(parents_ids), time_step=time_step)
    if data is None:
        return None
    data = data.copy()
    data["partner_id"] = np.where(data["ref_id"] == data["p_id"], data["m_id"], data["p_id"])
    return data


def find_fusion_time(cellnet, parents_ids):
    """Frame the two parents actually fused, or None if `parents_ids` isn't
    a real fusion pair in this network."""
    parents_set = set(parents_ids)
    for fusion_id in cellnet.fusion_cells():
        if set(cellnet.cells[fusion_id].parents) == parents_set:
            return cellnet.cells[fusion_id].start
    return None


def plot_mating_candidate_features(
    cellnet, parents_ids, time_step=10,
    distance_features=("nearest_dist", "center_dist", "tip_distance"),
    angle_features=("p_angle", "m_angle"),
):
    """For each parent in turn (as reference), plot every mating-competent
    candidate's distance/angle features over time -- the real eventual
    partner highlighted in red, every other candidate in light gray.
    """
    data = mating_candidate_features(cellnet, parents_ids, time_step=time_step)
    if data is None:
        print(f"{list(parents_ids)}: same strain_type (or unclassified) -- nothing to plot")
        return None

    fusion_time = find_fusion_time(cellnet, parents_ids)
    features = list(distance_features) + list(angle_features)
    ref_ids = sorted(data["ref_id"].unique())

    fig, axes = plt.subplots(
        len(features), len(ref_ids), figsize=(5.5 * len(ref_ids), 3 * len(features)),
        sharex="col", squeeze=False,
    )

    for col, ref_id in enumerate(ref_ids):
        ref_data = data[data["ref_id"] == ref_id]
        for row, feature in enumerate(features):
            ax = axes[row][col]
            for partner_id, group in ref_data.groupby("partner_id"):
                group = group.sort_values("time_stamp")
                is_true = bool(group["flag"].any())
                ax.plot(
                    group["time_stamp"], group[feature],
                    color="crimson" if is_true else "lightgray",
                    linewidth=2.5 if is_true else 1,
                    marker="o" if is_true else None,
                    zorder=3 if is_true else 1,
                    label=f"true partner ({partner_id})" if is_true else None,
                )
            if fusion_time is not None:
                ax.axvline(fusion_time, color="black", linestyle="--", linewidth=1, label="fusion")

            if row == 0:
                ax.set_title(f"reference cell {ref_id}")
            if col == 0:
                ax.set_ylabel(feature)
            if row == len(features) - 1:
                ax.set_xlabel("time_stamp (frame)")
            ax.legend(frameon=False, fontsize=8)
            ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"Mating candidates for parents {list(parents_ids)}", fontsize=14)
    plt.tight_layout()
    plt.show()
    return data


# Strain-type color palette for plot_pair_snapshot (1 = P/GFP, 2 = M/mCherry).
_PAIR_SNAPSHOT_MAIN = {1: "#4F8098", 2: "#BD6975"}
_PAIR_SNAPSHOT_FILL = {1: "#8FAFBE", 2: "#D49AA3"}
_PAIR_SNAPSHOT_LIGHT = {1: "#DCE7EC", 2: "#F1DFE3"}
_PAIR_SNAPSHOT_TEXT = "#283238"
_PAIR_SNAPSHOT_SECONDARY = "#5B666C"
_PAIR_SNAPSHOT_UNKNOWN = "#A3ADB3"
_PAIR_SNAPSHOT_ACCENT = "#B58E2C"


def plot_pair_snapshot(cellnet, center_id, target_id=None, time=None, margin=15, ax=None, rng=None):
    """Draw cell `center_id` and a target cell at `time` on the cropped
    label mask, with:
      - center-to-center line (center_dist)
      - nearest-point line, the two closest contour points (nearest_dist)
      - tip-to-tip line, the two closest skeleton endpoints (tip_distance)
      - at each cell's own center: its ray to its own nearest-approach
        point and its ray to its own nearest tip, annotated with the angle
        between them (p_angle at center_id, m_angle at target_id -- same
        convention as pair_feature/potential_mating_feature).

    Colors encode strain_type (1 = P/GFP, 2 = M/mCherry, otherwise gray):
    `center_id`/`target_id` get the strongest fill and their own outline
    (from `measure.coordinate`), `center_id`'s other mating-competent,
    opposite-type neighbors at `time` get a medium fill, everything else is
    pale background.

    `target_id` and/or `time` are optional:
      - `target_id` omitted: picked at random from `center_id`'s
        mating-competent, opposite-strain-type neighbors at `time`.
      - `time` omitted (and `target_id` given): picked at random from the
        frames `center_id` and `target_id` coexist.
      - both omitted: `time` is first picked at random from `center_id`'s
        own frames, then `target_id` at random among its neighbors there.

    Returns the plotted values as a dict (plus `target_id` and `time`, so
    the random choices made are visible to the caller).
    """
    rng = rng or random

    if target_id is None:
        if time is None:
            time = rng.choice(cellnet.cells[center_id].frames)
        candidates = sorted(set(cellnet.neighbor(center_id, time, strain="diff", competent_only=True)) - {center_id})
        if not candidates:
            raise ValueError(f"cell {center_id} has no mating-competent, opposite-type neighbor at frame {time}")
        target_id = rng.choice(candidates)
    else:
        if time is None:
            shared_frames = sorted(set(cellnet.cells[center_id].frames) & set(cellnet.cells[target_id].frames))
            if not shared_frames:
                raise ValueError(f"cells {center_id} and {target_id} never coexist")
            time = rng.choice(shared_frames)
        candidates = sorted(set(cellnet.neighbor(center_id, time, strain="diff", competent_only=True)) - {center_id})

    id1, id2 = center_id, target_id
    measure = cellnet.measure[time]
    label_map = cellnet.label_map[time]
    label1, label2 = label_map[id1], label_map[id2]

    center1 = np.asarray(measure.center(label=label1))
    center2 = np.asarray(measure.center(label=label2))
    target_point, source_point = measure.nearest_point(source=label1, target=label2, ptype="label")
    point_in_1, point_in_2 = np.asarray(source_point), np.asarray(target_point)
    tip_dist, tip1, tip2 = measure.tips_distance(label1, label2, ptype="label")
    tip1, tip2 = np.asarray(tip1), np.asarray(tip2)
    p_angle, m_angle = measure.between_angle(label1, label2, ptype="label")
    center_dist, nearest_dist = measure.distance(label1, label2, ptype="label")[0, 0, :2]
    coord1 = measure.coordinate(label=label1)
    coord2 = measure.coordinate(label=label2)

    # the own tip each angle is actually measured against: whichever tip is
    # closer to that cell's own nearest-approach point (point2tips_angle's rule)
    tips1 = np.asarray(measure.tip(label=label1))
    tips2 = np.asarray(measure.tip(label=label2))
    active_tip1 = tips1[np.argmin(np.linalg.norm(tips1 - point_in_1, axis=1))]
    active_tip2 = tips2[np.argmin(np.linalg.norm(tips2 - point_in_2, axis=1))]

    # Crop around center_id, radius = neighbor_threshold (no bbox dependency
    # on target_id, so candidate neighbors outside the pair's own bbox still show).
    label_image = cellnet.image[time]
    cy, cx = np.rint(center1).astype(int)
    crop_half_size = int(np.ceil(cellnet.neighbor_threshold)) + margin
    height, width = label_image.shape
    y0 = max(0, cy - crop_half_size)
    y1 = min(height, cy + crop_half_size + 1)
    x0 = max(0, cx - crop_half_size)
    x1 = min(width, cx + crop_half_size + 1)
    crop = label_image[y0:y1, x0:x1]

    def to_xy(point):
        # measure coordinates are (row, col); imshow's x is col, y is row
        return point[1] - x0, point[0] - y0

    selected_ids = {id1, id2}
    candidate_ids = set(candidates)
    overlay = np.zeros((*crop.shape, 4), dtype=np.float32)
    for cell_id, label in label_map.items():
        mask = crop == label
        if not mask.any():
            continue
        cell_type = cellnet.cells[cell_id].strain_type
        if cell_id in selected_ids:
            color, alpha = _PAIR_SNAPSHOT_MAIN.get(cell_type, _PAIR_SNAPSHOT_UNKNOWN), 0.38
        elif cell_id in candidate_ids:
            color, alpha = _PAIR_SNAPSHOT_FILL.get(cell_type, _PAIR_SNAPSHOT_UNKNOWN), 0.65
        else:
            color, alpha = _PAIR_SNAPSHOT_LIGHT.get(cell_type, "#E2E7EA"), 0.75
        overlay[mask] = to_rgba(color, alpha=alpha)

    created_fig = ax is None
    if created_fig:
        _, ax = plt.subplots(figsize=(85 / 25.4, 85 / 25.4), layout="constrained")
    ax.set_facecolor("white")
    ax.imshow(overlay, interpolation="nearest")

    color1 = _PAIR_SNAPSHOT_MAIN.get(cellnet.cells[id1].strain_type, _PAIR_SNAPSHOT_SECONDARY)
    color2 = _PAIR_SNAPSHOT_MAIN.get(cellnet.cells[id2].strain_type, _PAIR_SNAPSHOT_SECONDARY)
    for coord, color in ((coord1, color1), (coord2, color2)):
        ax.plot(coord[:, 1] - x0, coord[:, 0] - y0, color=color)

    def draw_line(p, q, color, label=None, linestyle="-", marker="o", linewidth=1.2, alpha=1.0):
        (xp, yp), (xq, yq) = to_xy(p), to_xy(q)
        ax.plot([xp, xq], [yp, yq], color=color, linestyle=linestyle, linewidth=linewidth,
                 marker=marker, markersize=3, alpha=alpha, label=label, zorder=4)

    draw_line(center1, center2, _PAIR_SNAPSHOT_TEXT, f"center_dist = {center_dist:.1f}", linestyle="--")
    draw_line(point_in_1, point_in_2, _PAIR_SNAPSHOT_ACCENT, f"nearest_dist = {nearest_dist:.1f}", marker="s")
    draw_line(tip1, tip2, _PAIR_SNAPSHOT_SECONDARY, f"tip_distance = {tip_dist:.1f}", linestyle=":", marker="^")

    for center, point, tip, color in (
        (center1, point_in_1, active_tip1, color1),
        (center2, point_in_2, active_tip2, color2),
    ):
        draw_line(center, point, color, marker=None, linewidth=0.9, alpha=0.85)
        draw_line(center, tip, color, marker=None, linewidth=0.9, linestyle="--", alpha=0.85)

    for center, angle, cell_id in ((center1, p_angle, id1), (center2, m_angle, id2)):
        ax.annotate(
            f"{cell_id}: {np.degrees(angle):.0f}°", to_xy(center), xytext=(6, 6),
            textcoords="offset points", color=_PAIR_SNAPSHOT_TEXT, fontsize=8,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5), zorder=5,
        )

    ax.set_title(f"cell {id1} vs {id2} -- frame {time}", fontsize=9, color=_PAIR_SNAPSHOT_TEXT)
    ax.legend(frameon=False, fontsize=7, labelcolor=_PAIR_SNAPSHOT_TEXT, loc="upper right")
    ax.axis("off")
    if created_fig:
        plt.show()

    return dict(center_dist=center_dist, nearest_dist=nearest_dist, tip_distance=tip_dist,
                p_angle=p_angle, m_angle=m_angle, target_id=target_id, time=time)
