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
"""Plots for `cellmate.mating._svm_prediction`'s partner-pick classifier:
model-vs-random top-k accuracy bars (`plot_top_k_accuracy`), the learning-
curve datasize check (`plot_learning_curve`), accuracy broken out by a
value like time-to-fusion (`plot_top_k_accuracy_by_value`), and individual
per-view score trajectories (`plot_score_trajectories`). Pulled out of
exploratory notebook cells so they're reusable (and testable) outside any
one notebook.
"""
import numpy as np
import matplotlib.pyplot as plt

from cellmate.mating._svm_prediction import bootstrap_ci


def plot_top_k_accuracy(groups, ax=None, bar_width=0.28):
    """Grouped bar chart: model top-k accuracy vs its random-guess baseline,
    one group of bars per entry in `groups`.

    `groups`: list of dicts, each `{"label": str, "model": array-like of
    per-decision hits (0/1), "random": array-like}` -- e.g. `"model"` from
    `topk.groupby("decision_id")["flag"].max()` and `"random"` from
    `k / scored_df.drop_duplicates("decision_id")["N_candidates"]`. Bars show
    the mean of each array with a bootstrap 95% CI.
    """
    created_fig = ax is None
    if created_fig:
        _, ax = plt.subplots(figsize=(1.2 + 1.4 * len(groups), 2.8))

    centers = np.arange(len(groups), dtype=float)
    labels, model_means, model_cis, random_means, random_cis = [], [], [], [], []
    for g in groups:
        labels.append(g["label"])
        model = np.asarray(g["model"], dtype=float)
        random = np.asarray(g["random"], dtype=float)
        model_means.append(model.mean())
        random_means.append(random.mean())
        model_cis.append(bootstrap_ci(model))
        random_cis.append(bootstrap_ci(random))

    model_means, random_means = np.array(model_means), np.array(random_means)
    model_cis, random_cis = np.array(model_cis), np.array(random_cis)

    def _yerr(means, cis):
        return np.array([means - cis[:, 0], cis[:, 1] - means])

    x_random, x_model = centers - bar_width / 2, centers + bar_width / 2
    ax.bar(x_random, random_means, yerr=_yerr(random_means, random_cis), capsize=2,
           width=bar_width, color="#C0C0C0", edgecolor="black", label="Random")
    ax.bar(x_model, model_means, yerr=_yerr(model_means, model_cis), capsize=2,
           width=bar_width, color="#1f77b4", edgecolor="black", label="Model")

    for x, y in zip(x_random, random_means):
        ax.text(x, y + 0.04, f"{y:.2f}", ha="center", va="bottom")
    for x, y in zip(x_model, model_means):
        ax.text(x, y + 0.04, f"{y:.2f}", ha="center", va="bottom")

    ax.set_xticks(centers)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Top-k accuracy")
    ax.set_ylim(0, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)

    if created_fig:
        plt.tight_layout()
        plt.show()
    return ax


def plot_learning_curve(curve_df, k=1, ax=None, color="#1f77b4"):
    """Mean top-k accuracy (+/- bootstrap 95% CI across repeats) vs training
    set size, from `learning_curve_by_fusion_event`'s output -- is accuracy still
    climbing with more labeled decisions, or has it plateaued?

    Repeats are aggregated by `fraction` (the deterministic key -- every
    repeat at a given fraction samples the same *number* of events, just a
    different random subset), not by `n_train_decisions`: that varies
    slightly per repeat even at a fixed fraction (events differ in length),
    so grouping by it directly would fragment repeats across many nearby
    x-positions instead of aggregating them into one point per fraction.
    The x-axis itself still shows `n_train_decisions` (mean per fraction) --
    the more physically meaningful "how much data" quantity.
    """
    created_fig = ax is None
    if created_fig:
        _, ax = plt.subplots(figsize=(4.2, 2.8))

    sub = curve_df[curve_df["k"] == k]
    grouped = sub.groupby("fraction")
    sizes = grouped["n_train_decisions"].mean().to_numpy()
    means = grouped["top_k_acc"].mean().to_numpy()
    cis = np.array([bootstrap_ci(g.to_numpy()) for _, g in grouped["top_k_acc"]])
    order = np.argsort(sizes)
    sizes, means, cis = sizes[order], means[order], cis[order]

    ax.plot(sizes, means, marker="o", color=color, label=f"top-{k}")
    ax.fill_between(sizes, cis[:, 0], cis[:, 1], color=color, alpha=0.2)

    ax.set_xlabel("Training decisions (n)")
    ax.set_ylabel("Top-k accuracy")
    ax.set_ylim(0, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)

    if created_fig:
        plt.tight_layout()
        plt.show()
    return ax


def plot_row_level_learning_curve(curve_df, metrics=("accuracy", "recall", "f1"), ax=None,
                                   colors=("#1f77b4", "#d62728", "#2ca02c")):
    """Mean row-level accuracy/recall/f1 (+/- bootstrap 95% CI across
    repeats) vs training set size, from `learning_curve_by_fusion_event`'s
    `metric="row_level"` output -- cheaper, standard-ML-style counterpart
    to `plot_learning_curve`'s partner-pick curve, one line per metric.

    Repeats are aggregated by `fraction`, same reasoning as
    `plot_learning_curve`; the x-axis shows `n_train_decisions` (mean per
    fraction).
    """
    created_fig = ax is None
    if created_fig:
        _, ax = plt.subplots(figsize=(4.2, 2.8))

    grouped = curve_df.groupby("fraction")
    sizes_by_fraction = grouped["n_train_decisions"].mean()

    for metric, color in zip(metrics, colors):
        sizes = sizes_by_fraction.to_numpy()
        means = grouped[metric].mean().to_numpy()
        cis = np.array([bootstrap_ci(g.to_numpy()) for _, g in grouped[metric]])
        order = np.argsort(sizes)
        sizes, means, cis = sizes[order], means[order], cis[order]

        ax.plot(sizes, means, marker="o", color=color, label=metric)
        ax.fill_between(sizes, cis[:, 0], cis[:, 1], color=color, alpha=0.15)

    ax.set_xlabel("Training decisions (n)")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)

    if created_fig:
        plt.tight_layout()
        plt.show()
    return ax


def plot_score_trajectories(trajectory_df, value_col="time_to_fusion", score_col="pred_prob",
                             view_col="view_id", ax=None, max_lines=200, random_state=0,
                             color="#1f77b4", alpha=0.25):
    """Spaghetti plot: one line per view, the true partner's predicted
    score vs `value_col` (e.g. `time_to_fusion`) -- from
    `cellmate.mating.true_partner_score_trajectory`'s output. Shows the
    actual shape of how confidence in the correct candidate builds (or
    doesn't) across a cell's search -- climbing steadily, jumping late, or
    flat -- which `plot_top_k_accuracy_by_value`'s binned population
    average can't show. `max_lines` randomly subsamples views for
    legibility on large test sets (`None` to plot every view).
    """
    created_fig = ax is None
    if created_fig:
        _, ax = plt.subplots(figsize=(4.6, 3.2))

    views = trajectory_df[view_col].unique()
    if max_lines is not None and len(views) > max_lines:
        rng = np.random.default_rng(random_state)
        views = rng.choice(views, size=max_lines, replace=False)

    for view in views:
        sub = trajectory_df[trajectory_df[view_col] == view].sort_values(value_col)
        ax.plot(sub[value_col], sub[score_col], color=color, alpha=alpha, linewidth=0.8)

    ax.set_xlabel(value_col)
    ax.set_ylabel("Predicted probability of true partner")
    ax.set_ylim(0, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if created_fig:
        plt.tight_layout()
        plt.show()
    return ax


def plot_top_k_accuracy_by_value(by_value_df, value_col, ax=None, color="#1f77b4"):
    """Model top-k accuracy and its random baseline vs a breakout value (e.g.
    `time_stamp`), from `top_k_accuracy_by_value`'s output -- are decisions
    closer to fusion easier to call correctly?
    """
    created_fig = ax is None
    if created_fig:
        _, ax = plt.subplots(figsize=(4.2, 2.8))

    df = by_value_df.sort_values(value_col)
    ax.plot(df[value_col], df["top_k_acc"], marker="o", color=color, label="Model")
    ax.plot(df[value_col], df["random_baseline"], marker="o", color="#C0C0C0", label="Random")

    ax.set_xlabel(value_col)
    ax.set_ylabel("Top-k accuracy")
    ax.set_ylim(0, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)

    if created_fig:
        plt.tight_layout()
        plt.show()
    return ax
