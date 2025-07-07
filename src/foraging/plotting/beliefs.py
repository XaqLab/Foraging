from functools import reduce
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from numpy.typing import ArrayLike
from scipy.optimize import minimize_scalar
from scipy.stats import gamma

from foraging.config.constants import BOX_COLORS, BOX_LABELS
from foraging.plotting import (
    HEATMAP_PALETTE,
    PALETTE,
    bp,
    fig_init,
    legend_handler,
    titler,
)
from foraging.plotting._base import subject_plotter
from foraging.utils import INDEX, MIN_INDEX
from foraging.utils._base import kwargs_handler
from foraging.utils.beliefs import get_mean_beliefs, get_std_beliefs
from foraging.utils.data import (
    bin_data,
    filter_df,
    process_block_safely,
    process_blocks,
)
from foraging.utils.models import BeliefModule, Observation


def likelihood_single_obs(
    obs_model: Observation,
    latents: ArrayLike,
    obs: Any,
    ax: Optional[plt.Axes] = None,
    **kwargs: Any,
) -> plt.Axes:
    """
    Computes and plots the likelihood for a single observation across multiple latents.

    Args:
        obs_model: The observation model used to calculate the likelihood.
        latents: A list or array of latents over which the likelihood is calculated.
        obs: The observation for which the likelihood is calculated.
        ax: Optional, an existing matplotlib Axes object. If None, a new one is created.
        kwargs: Additional keyword arguments passed to matplotlib's `subplots` function for figure creation.

    Returns:
        The matplotlib Axes object with the plot.
    """

    # Create axes if none provided
    fig, ax = fig_init(ax, **kwargs)

    # Compute likelihood for each latent
    likelihoods = [obs_model.probability(obs, latent) for latent in latents]

    # Plot the likelihoods
    ax.plot(latents, likelihoods)
    ax.set_xlabel("Latent")
    ax.set_ylabel("Likelihood")
    ax.set_title(f"Likelihood given observation {obs}")
    return ax


@legend_handler
def plot_fisher_info(
    t: Any,
    schedules: list[float],
    palette: dict = PALETTE,
    alpha: float = 1,
    figsize: tuple[float, float] = (10, 5),
    title: str = "Fisher Information",
):

    # Calculate fisher information
    def _fisher_info(t, schedule, alpha):
        scale = schedule / alpha
        cdf = gamma.cdf(t, a=alpha, scale=scale)
        return (
            (t / schedule) ** 2
            * gamma.pdf(t, a=alpha, scale=scale) ** 2
            / (cdf * (1 - cdf))
        )

    # Find optimal push time for each schedule
    optimal_fisher_t = {}
    optimal_fisher_info = {}
    fishers = []
    _, ax = plt.subplots()
    colors = list(palette.values())
    for i, schedule in enumerate(schedules):
        res = minimize_scalar(lambda x: -_fisher_info(x, schedule, alpha))

        # Result
        x_max = res.x
        f_max = _fisher_info(x_max, schedule, alpha)

        print(
            f"Optimal Fisher info for schedule {schedule} with shape = {alpha} occurs at t = {x_max:.4f}, with value {f_max:.4f}"
        )
        optimal_fisher_info[schedule] = f_max
        optimal_fisher_t[schedule] = x_max
        fishers.append(_fisher_info(t, schedule, alpha))
        ax.plot(t, fishers[i], label=schedule, color=colors[i])
    ax.set_ylabel("Fisher information")
    ax.set_xlabel("push time (s)")
    ymax = np.array(fishers).max()
    ax.vlines(
        x=optimal_fisher_t.values(),
        ymin=0,
        ymax=1.1 * ymax,
        color=colors,
        linestyle="--",
        label=r"$t^*=\underset{t}{\mathrm{argmax}}J_{\lambda}(t)$",
    )
    ax.set_title(title)

    # Create a black dashed line for the legend
    optimal_line = Line2D(
        [0], [1], color="black", linestyle="--", label="optimal fisher info"
    )

    # Add legend with the optimal line
    ax.legend(handles=[optimal_line])
    return ax


@legend_handler
def plot_optimal_fisher_uncertainty(
    n: int,
    schedules: list[float],
    alpha: float = 1,
    title: str = "Fisher-optimal uncertainty as a function of number of pushes",
):
    n = np.arange(n) + 1

    # Calculate fisher information
    def _fisher_info(t, schedule, alpha):
        scale = schedule / alpha
        cdf = gamma.cdf(t, a=alpha, scale=scale)
        return (
            (t / schedule) ** 2
            * gamma.pdf(t, a=alpha, scale=scale) ** 2
            / (cdf * (1 - cdf))
        )

    # Find optimal push time for each schedule
    optimal_fisher_t = {}
    optimal_fisher_info = {}
    for schedule in schedules:
        res = minimize_scalar(lambda x: -_fisher_info(x, schedule, alpha))

        # Result
        x_max = res.x
        f_max = _fisher_info(x_max, schedule, alpha)
        optimal_fisher_info[schedule] = f_max
        optimal_fisher_t[schedule] = x_max

    _, ax = plt.subplots()
    colors = list(PALETTE.values())
    for i, (k, v) in enumerate(optimal_fisher_info.items()):
        ax.plot(n, np.sqrt(1 / (n * v)), label=k, color=colors[i])
    ax.set_title(title)
    ax.set_xlabel("# pushes")
    ax.set_ylabel("standard deviation")
    ax.legend()
    return ax


def reward_beliefs3d(
    df: pd.DataFrame,
    reward_beliefs: np.ndarray,
    box_labels: list = BOX_LABELS,
    box_colors: list = BOX_COLORS,
    fontsize: int = 10,
    title: str = "Belief about reward availability at time of push",
    ax: Optional[plt.Axes] = None,
    **kwargs: Any,
) -> plt.Axes:
    """
    Plots a 3D scatter plot of reward beliefs for each box at the time of a push.

    Args:
        df: DataFrame containing session data, including 'box rank'.
        reward_beliefs: 2D numpy array where each row corresponds to beliefs about reward availability.
        box_labels: List of labels corresponding to different box ranks.
        box_colors: List of colors for each box, used in the scatter plot.
        fontsize: Size of the font for axis labels.
        title: Title of the plot.
        ax: Optional, existing matplotlib Axes object. If None, a new one will be created.
        kwargs: Additional keyword arguments passed to matplotlib's `scatter` function.

    Returns:
        ax: The matplotlib Axes object with the plot.
    """

    # Create axes if none provided
    if ax is None:
        fig = plt.figure(**kwargs.pop("fig_kwargs", {}))
        ax = fig.add_subplot(projection="3d")

    # Plot beliefs
    plt_kwargs = {"cmap": ListedColormap(box_colors)} | kwargs.pop("plt_kwargs", {})
    p = ax.scatter(
        reward_beliefs[:, 0],
        reward_beliefs[:, 1],
        reward_beliefs[:, 2],
        c=df["box rank"],
        **plt_kwargs,
    )

    # Set axis limits and labels
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])
    ax.set_xlabel("Belief that reward is available at fast box", fontsize=fontsize)
    ax.set_ylabel("Belief that reward is available at medium box", fontsize=fontsize)
    ax.set_zlabel("Belief that reward is available at slow box", fontsize=fontsize)
    ax.set_title(title)
    ax.view_init(elev=20, azim=-135)

    # Add colorbar with discrete labels
    cbar = plt.colorbar(p, ax=ax, ticks=[0, 1, 2])
    cbar.set_label("Schedule")
    cbar.set_ticklabels(box_labels)
    return ax


@legend_handler(loc="upper left", bbox=(1.1, 1))
def plot_schedule_beliefs_in_block(
    df: pd.DataFrame,
    beliefs: dict[tuple, BeliefModule],
    conds: dict[str, Any],
    palette: dict = PALETTE,
    heatmap_palette: dict = HEATMAP_PALETTE,
    show_stats: bool = False,
    ax: plt.Axes = None,
    **kwargs,
) -> plt.Axes:
    """
    Plots the beliefs about the schedule for a specific block in the experiment, with uncertainty bands
    and reward outcomes.

    Args:
        df: DataFrame containing session data.
        index: Index of the block to analyze in the DataFrame.
        x: The value from `df` used for the x-axis, typically 'push #' or 'push times'.
        ax: Optional, existing matplotlib Axes object. If None, a new one will be created.

        kwargs:
            - 'fig_kwargs': Dictionary to specify figure properties when creating a new figure (passed to `plt.subplots`).
            - 'plt_kwargs': Dictionary to specify additional plotting properties for the line plot (passed to `bp wrapper`).
            - 'lgd_kwargs': Dictionary of keyword arguments for customizing the legend (passed to `plt.legend`).

    Returns:
        ax: The matplotlib Axes object with the plot.
    """

    # Get block data
    df_block = filter_df(df, conds)
    n_boxes = df_block["n boxes"].values[0]
    push_times = df_block["push times"].values
    posteriors = np.array(beliefs[tuple(conds.values())].features)
    schedule_candidates = beliefs[tuple(conds.values())].support[0]
    schedules = np.sort(df_block["schedule"].unique())
    mean_across_pushes = get_mean_beliefs(posteriors, schedule_candidates)[1:]

    # Create a new array to hold the interpolated data
    # The new number of columns will be the difference between the max and min time points, creating 1 second time bins
    new_num_cols = int(push_times[-1] + 1)
    interpolated_beliefs = np.zeros(
        (new_num_cols, posteriors.shape[1], posteriors.shape[2])
    )
    start = 0
    for i, t in enumerate(push_times):
        end = int(t)
        interpolated_beliefs[start:end, :, :] = posteriors[i, :, :]
        start = end

    if show_stats:
        mean_across_time = get_mean_beliefs(
            interpolated_beliefs, schedule_candidates
        )  # E[lambda] at each timepoint
        std_across_time = np.nan_to_num(
            get_std_beliefs(interpolated_beliefs, schedule_candidates)
        )
    fig, ax = fig_init(
        ax,
        **kwargs.pop(
            "fig_kwargs",
            {
                "nrows": n_boxes,
                "ncols": 1,
                "sharex": True,
                "sharey": True,
                "figsize": (15, 10),
            },
        ),
    )
    box_labels = list(heatmap_palette.keys())
    for i in range(n_boxes):
        # Plot belief probabilities
        sns.heatmap(
            interpolated_beliefs[:, i, :].T,
            cmap=heatmap_palette[box_labels[i]],
            ax=ax[i],
        )
        if show_stats:
            ax[i].plot(
                mean_across_time[:, i], color=palette[box_labels[i]], label="mean"
            )
            ax[i].plot(
                mean_across_time[:, i] - std_across_time[:, i],
                color=palette[box_labels[i]],
                linestyle=":",
                label="s.d.",
            )
            ax[i].plot(
                mean_across_time[:, i] + std_across_time[:, i],
                color=palette[box_labels[i]],
                linestyle=":",
            )

        # Plot true schedule
        ax[i].axhline(schedules[i], color="black", linestyle="--")
        ax[i].set_title(f"Belief about schedule for {box_labels[i]} box")
        ax[i].set_xlabel("Time (s)")
        ax[i].set_ylabel("Possible schedules")
        current_yticks = ax[i].get_yticks()
        ax[i].set_yticklabels(
            [
                schedule_candidates[int(j)]
                for j in current_yticks
                if int(j) < len(schedule_candidates)
            ]
        )

        # Plot reward outcomes
        reward_mask = df_block["reward outcomes"].values
        box_mask = df_block["box rank"].values == i
        ax[i].scatter(
            push_times[reward_mask & box_mask],
            mean_across_pushes[reward_mask & box_mask, i],
            c="black",
            marker="^",
            s=50,
        )
        ax[i].scatter(
            push_times[~reward_mask & box_mask],
            mean_across_pushes[~reward_mask & box_mask, i],
            edgecolors="black",
            marker="v",
            s=50,
            facecolors="none",
        )

        # Add legend
        if i == 0:
            handles = [
                Line2D([0], [0], color="black", linestyle="--", label="true schedule"),
                Line2D(
                    [0], [0], color="black", linestyle="", marker="^", label="rewarded"
                ),
                Line2D(
                    [0],
                    [0],
                    color="black",
                    linestyle="",
                    marker="v",
                    markerfacecolor="none",
                    label="no reward",
                ),
            ]
            if show_stats:
                handles.append(
                    Line2D([0], [0], color="black", linestyle="-", label="mean")
                )
                handles.append(
                    Line2D([0], [0], color="black", linestyle=":", label="s.d.")
                )
            ax[i].legend(handles=handles)

    fig.suptitle(titler("Beliefs about schedule", conds=conds))
    fig.tight_layout()
    return ax


def plot_schedule_beliefs_mean_and_std_across_blocks(
    df: pd.DataFrame,
    beliefs: dict[tuple, BeliefModule],
    conds: dict[str, Any] = None,
    x: str = "push times",
    palette: dict = PALETTE,
    **kwargs,
) -> plt.Axes:
    """
    Plots the beliefs about the schedule for a specific block in the experiment, with uncertainty bands
    and reward outcomes.

    Args:
        df: DataFrame containing session data.
        index: Index of the block to analyze in the DataFrame.
        x: The value from `df` used for the x-axis, typically 'push #' or 'push times'.
        ax: Optional, existing matplotlib Axes object. If None, a new one will be created.

        kwargs:
            - 'fig_kwargs': Dictionary to specify figure properties when creating a new figure (passed to `plt.subplots`).
            - 'plt_kwargs': Dictionary to specify additional plotting properties for the line plot (passed to `bp wrapper`).
            - 'lgd_kwargs': Dictionary of keyword arguments for customizing the legend (passed to `plt.legend`).

    Returns:
        ax: The matplotlib Axes object with the plot.
    """

    df = filter_df(df, conds)

    # Get beliefs of each block
    def _inner(df: pd.DataFrame, index: tuple):
        df_block = df.loc[index]
        posteriors = np.array(beliefs[index].features)
        schedule_candidates = beliefs[index].support[0]

        # Get mean and std of beliefs
        mean = get_mean_beliefs(posteriors, schedule_candidates)
        std = get_std_beliefs(posteriors, schedule_candidates)
        push_times = df_block["push times"].values
        box = df_block["box"].values
        box_rank = df_block["box rank"].values
        res = {
            (index + (i,)): [
                mean[i + 1, box_rank[i]],
                std[i + 1, box_rank[i]],
                push_times[i],
                box[i],
                i,
            ]
            for i in range(len(mean) - 1)
        }

        # Add prior
        for i in df_block["box rank"].unique():
            res[(index + (-i,))] = [mean[0, i], std[0, i], 0, box[0], 0]
        return res

    res, _ = process_blocks(df, _inner)
    res = reduce(lambda x, y: {**x, **y}, list(res.values()), {})
    df_beliefs = pd.DataFrame.from_dict(
        res,
        orient="index",
        columns=["mean of belief", "s.d. of belief", "push times", "box", "n pushes"],
    )
    df_beliefs.index = pd.MultiIndex.from_tuples(
        df_beliefs.index, names=INDEX[:MIN_INDEX]
    )

    # Bin pushes by time
    if x == "push times":
        bin_kwargs = kwargs_handler(kwargs, "bin_kwargs", dict(bin_width=60))
        df_beliefs[x] = bin_data(df_beliefs, x, **bin_kwargs).values

    fig_kwargs = kwargs_handler(
        kwargs, "fig_kwargs", {"nrows": 2, "ncols": 1, "figsize": (10, 10)}
    )

    @legend_handler
    def _plot(i, subj, **kwargs):
        fig, ax = fig_init(**fig_kwargs)
        conds = {"subject": subj}
        df_subj = filter_df(df_beliefs, conds=conds)
        bp(sns.lineplot)(
            df_subj,
            x=x,
            y="mean of belief",
            hue="box",
            palette=palette,
            ax=ax[0],
            **kwargs,
        )
        bp(sns.lineplot)(
            df_subj,
            x=x,
            y="s.d. of belief",
            hue="box",
            palette=palette,
            ax=ax[1],
            legend=False,
            **kwargs,
        )
        fig.suptitle(titler(title_prefix="Beliefs about schedule", conds=conds))
        fig.tight_layout()
        return ax

    subject_plotter(df.index.unique("subject"), _plot, **kwargs)
