from functools import reduce
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from numpy.typing import ArrayLike
from scipy.optimize import minimize_scalar

from foraging.config.constants import BOX_COLORS, BOX_LABELS, HEATMAP_PALETTE, PALETTE
from foraging.plotting import (
    bp,
    fig_init,
    legend_handler,
    multiplot,
    titler,
)
from foraging.plotting._base import (
    across_conditions_plotter,
    plot_block_average_or_traces,
)
from foraging.utils import INDEX, MIN_INDEX
from foraging.utils._base import kwargs_handler
from foraging.utils.beliefs import (
    fisher_info_reward_observations,
    get_entropy_beliefs_over_time,
    get_mean_beliefs_over_time,
    get_std_beliefs_over_time,
)
from foraging.utils.data import (
    bin_data,
    filter_df,
    map_box_positions_to_ranks,
    process_block_safely,
    process_blocks,
)
from foraging.utils.models import AbstractBelief, BeliefModule
from foraging.utils.stats import moving_average


def likelihood_single_obs(
    obs_model: AbstractBelief,
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
    likelihoods = [obs_model.likelihood(obs, latent) for latent in latents]

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
    title: str = "Fisher Information Reward Observations",
):

    # Find optimal push time for each schedule
    optimal_fisher_t = {}
    optimal_fisher_info = {}
    fishers = []
    _, ax = plt.subplots()
    colors = list(palette.values())
    for i, schedule in enumerate(schedules):
        res = minimize_scalar(
            lambda x: -fisher_info_reward_observations(x, schedule, alpha)
        )

        # Result
        x_max = res.x
        f_max = fisher_info_reward_observations(x_max, schedule, alpha)

        print(
            f"Optimal Fisher info for schedule {schedule} with shape = {alpha} occurs at t = {x_max:.4f}, with value {f_max:.4f}"
        )
        optimal_fisher_info[schedule] = f_max
        optimal_fisher_t[schedule] = x_max
        fishers.append(fisher_info_reward_observations(t, schedule, alpha))
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
def plot_cramer_rao_lb(
    n: int,
    schedules: list[float],
    palette: dict = PALETTE,
    alpha: float = 1,
    title: str = "Cramer-Rao Lower Bound as a Function of # Observations",
):

    # Plot Cramer-Rao lower bound for perfect observations vs reward observations
    n = np.arange(n) + 1
    colors = list(PALETTE.values())
    for i, schedule in enumerate(schedules):

        # Find optimal push time for each schedule
        res = minimize_scalar(
            lambda x: -fisher_info_reward_observations(x, schedule, alpha)
        )
        x_max = res.x
        fisher_info_reward = fisher_info_reward_observations(x_max, schedule, alpha)
        fisher_info_perfect = alpha / schedule**2

        std_crlb_perfect = np.sqrt(1 / (n * fisher_info_perfect))
        std_crlb_reward = np.sqrt(1 / (n * fisher_info_reward))

        plt.plot(n, std_crlb_perfect, color=colors[i])
        plt.plot(n, std_crlb_reward, color=colors[i], linestyle="--")

    # Custom legend
    legend_elements = []

    # Add schedule patches
    labels = list(PALETTE.keys())
    for i, schedule in enumerate(schedules):
        legend_elements.append(Patch(color=colors[i], label=labels[i]))

    # Add line style indicators
    legend_elements.append(
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="-",
            linewidth=2,
            label="Perfect observations",
        )
    )
    legend_elements.append(
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            linewidth=2,
            label="Reward observations",
        )
    )

    plt.xlabel("# observations")
    plt.ylabel("standard deviation")
    plt.title(title)
    plt.legend(handles=legend_elements)
    return plt.gca()


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


@legend_handler(bbox=(1.15, 1))
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
    mean_across_pushes = get_mean_beliefs_over_time(posteriors, schedule_candidates)[1:]

    # Create a new array to hold the interpolated data
    # The new number of columns will be the difference between the max and min time points, creating 1 second time bins
    new_num_cols = int(push_times[-1] + 1)
    interpolated_beliefs = np.zeros(
        (new_num_cols, posteriors.shape[1], posteriors.shape[2])
    )
    start = 0
    for box_pos, t in enumerate(push_times):
        end = int(t)
        interpolated_beliefs[start:end, :, :] = posteriors[box_pos, :, :]
        start = end

    if show_stats:
        mean_across_time = get_mean_beliefs_over_time(
            interpolated_beliefs, schedule_candidates
        )  # E[lambda] at each timepoint
        std_across_time = np.nan_to_num(
            get_std_beliefs_over_time(interpolated_beliefs, schedule_candidates)
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
    pos_to_rank = map_box_positions_to_ranks(df_block)
    for box_pos in range(n_boxes):
        box_rank = pos_to_rank.loc[box_pos].values[0]

        # Plot belief probabilities
        sns.heatmap(
            interpolated_beliefs[:, box_pos, :].T,
            cmap=heatmap_palette[box_labels[box_rank]],
            cbar_kws={"label": "probability"},
            ax=ax[box_pos],
        )
        if show_stats:
            ax[box_pos].plot(
                mean_across_time[:, box_pos],
                color=palette[box_labels[box_rank]],
                label="mean",
            )
            ax[box_pos].plot(
                mean_across_time[:, box_pos] - std_across_time[:, box_pos],
                color=palette[box_labels[box_rank]],
                linestyle=":",
                label="s.d.",
            )
            ax[box_pos].plot(
                mean_across_time[:, box_pos] + std_across_time[:, box_pos],
                color=palette[box_labels[box_rank]],
                linestyle=":",
            )

        # Plot true schedule
        ax[box_pos].axhline(schedules[box_rank], color="black", linestyle="--")
        ax[box_pos].set_title(f"Belief about schedule for {box_labels[box_rank]} box")
        ax[box_pos].set_xlabel("Time (s)")
        ax[box_pos].set_ylabel("Possible schedules")
        current_yticks = ax[box_pos].get_yticks()
        ax[box_pos].set_yticklabels(
            [
                schedule_candidates[int(j)]
                for j in current_yticks
                if int(j) < len(schedule_candidates)
            ]
        )

        # Plot reward outcomes
        reward_mask = df_block["reward outcomes"].values
        box_mask = df_block["box rank"].values == box_rank
        ax[box_pos].scatter(
            push_times[reward_mask & box_mask],
            mean_across_pushes[reward_mask & box_mask, box_pos],
            c="black",
            marker="^",
            s=50,
        )
        ax[box_pos].scatter(
            push_times[~reward_mask & box_mask],
            mean_across_pushes[~reward_mask & box_mask, box_pos],
            edgecolors="black",
            marker="v",
            s=50,
            facecolors="none",
        )

        # Add legend
        if box_pos == 0:
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
            ax[box_pos].legend(handles=handles)

    fig.suptitle(titler("Beliefs about schedule", conds=conds))
    fig.tight_layout()
    return ax


def plot_schedule_beliefs_mean_and_std_across_block(
    df: pd.DataFrame,
    beliefs: dict[tuple, BeliefModule],
    conds: dict[str, Any] = None,
    x: str = "push times",
    palette: dict = PALETTE,
    show_traces: bool = False,
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
        df_block = df.loc[index].reset_index()
        posteriors = np.array(beliefs[index].features)
        schedule_candidates = beliefs[index].support[0]

        # Get mean and std of beliefs
        mean = get_mean_beliefs_over_time(posteriors, schedule_candidates)
        std = get_std_beliefs_over_time(posteriors, schedule_candidates)
        x_vals = df_block[x].values
        box = df_block["box"].values
        box_positions = df_block["box position"].values
        block_id = df_block["block_id"].values[0]
        res = {
            (index + (i + 1,)): [
                mean[i + 1, box_positions[i]],
                std[i + 1, box_positions[i]],
                x_vals[i],
                box[i],
                block_id,
            ]
            for i in range(len(mean) - 1)
        }

        # Add prior
        box_labels = list(PALETTE.keys())
        for i in df_block["box position"].unique():
            res[(index + (-i,))] = [mean[0, i], std[0, i], 0, box_labels[i]]
        return res

    res, _ = process_blocks(df, _inner)
    res = reduce(lambda x, y: {**x, **y}, list(res.values()), {})
    df_beliefs = pd.DataFrame.from_dict(
        res,
        orient="index",
        columns=["mean", "standard deviation", x, "box", "block_id"],
    )

    df_beliefs.index = pd.MultiIndex.from_tuples(
        df_beliefs.index, names=INDEX[:MIN_INDEX]
    )

    # Average data over time
    groupers = ["box", "block_id"]
    if x == "push times":
        smooth_kwargs = kwargs_handler(kwargs, "smooth_kwargs")
        ma_mean = moving_average(
            df_beliefs,
            x=x,
            y="mean",
            y_name="mean",
            bin_func=lambda x: x["mean"].mean(),
            groupers=groupers,
            **smooth_kwargs,
        )
        ma_std = moving_average(
            df_beliefs,
            x=x,
            y="standard deviation",
            y_name="standard deviation",
            bin_func=lambda x: x["standard deviation"].mean(),
            groupers=groupers,
            **smooth_kwargs,
        )
        df_beliefs = ma_mean
        df_beliefs["standard deviation"] = ma_std["standard deviation"]
        x = "time"

    fig_kwargs = kwargs_handler(
        kwargs, "fig_kwargs", {"nrows": 2, "ncols": 1, "figsize": (10, 10)}
    )

    @legend_handler
    def _plot(i, subj, **kwargs):
        fig, ax = fig_init(**fig_kwargs)
        conds = {"subject": subj}
        df_subj = filter_df(df_beliefs, conds=conds)

        params = {
            "x": x,
            "y": "mean",
            "hue": "box",
            "palette": palette,
            "ax": ax[0],
            **kwargs,
        }
        plot_block_average_or_traces(df_subj, show_traces=show_traces, **params)

        # Plot schedules on top of means
        schedules = sorted(filter_df(df, conds=conds)["schedule"].unique())
        colors = list(palette.values())
        for i, schedule in enumerate(schedules):
            ax[0].axhline(schedule, color=colors[i], linestyle="--")

        actual_schedule_line = Line2D(
            [0], [0], color="black", linestyle="--", label="actual schedule"
        )

        # If you have an existing legend, you can add this to the existing handles
        existing_handles, existing_labels = ax[0].get_legend_handles_labels()
        all_handles = existing_handles + [actual_schedule_line]
        all_labels = existing_labels + ["actual schedule"]

        # Remove the old legend and create a new one
        if ax[0].get_legend():
            ax[0].get_legend().remove()
            ax[0].legend(handles=all_handles, labels=all_labels)

        params.update(
            {
                "y": "standard deviation",
                "legend": False,
                "ax": ax[1],
            }
        )
        plot_block_average_or_traces(df_subj, show_traces=show_traces, **params)

        fig.suptitle(titler(title="Beliefs about schedule", conds=conds))
        fig.tight_layout()
        return ax

    across_conditions_plotter(df.index.unique("subject"), _plot, **kwargs)


def plot_accuracy_across_block(
    df: pd.DataFrame,
    accuracies: dict[tuple, ArrayLike],
    conds: dict[str, Any] = None,
    x: str = "push times",
    palette: dict = PALETTE,
    show_traces: bool = False,
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
        df_block = df.loc[index].reset_index()
        accuracy = accuracies[index]

        x_vals = df_block[x].values
        block_id = df_block["block_id"].values[0]
        res = {
            (index + (i + 1,)): [
                accuracy[i + 1],
                x_vals[i],
                block_id,
            ]
            for i in range(len(accuracy) - 1)
        }

        res[(index + (0,))] = [accuracy[0], 0, block_id]
        return res

    res, _ = process_blocks(df, _inner)
    res = reduce(lambda x, y: {**x, **y}, list(res.values()), {})
    df_beliefs = pd.DataFrame.from_dict(
        res,
        orient="index",
        columns=["accuracy", x, "block_id"],
    )

    df_beliefs.index = pd.MultiIndex.from_tuples(
        df_beliefs.index, names=INDEX[:MIN_INDEX]
    )

    # Average data over time
    groupers = ["block_id"]
    if x == "push times":
        smooth_kwargs = kwargs_handler(kwargs, "smooth_kwargs")
        df_beliefs = moving_average(
            df_beliefs,
            x=x,
            y="accuracy",
            y_name="accuracy",
            bin_func=lambda x: x["accuracy"].mean(),
            groupers=groupers,
            **smooth_kwargs,
        )
        x = "time"

    fig_kwargs = kwargs_handler(kwargs, "fig_kwargs")

    @legend_handler
    def _plot(i, subj, **kwargs):
        fig, ax = fig_init(**fig_kwargs)
        conds = {"subject": subj}
        df_subj = filter_df(df_beliefs, conds=conds)

        params = {
            "x": x,
            "y": "accuracy",
            "palette": palette,
            "ax": ax,
            **kwargs,
        }
        plot_block_average_or_traces(df_subj, show_traces=show_traces, **params)

        fig.suptitle(titler(title="Theoretical accuracy", conds=conds))
        fig.tight_layout()
        return ax

    across_conditions_plotter(df.index.unique("subject"), _plot, **kwargs)


def plot_schedule_beliefs_entropy_across_blocks(
    df: pd.DataFrame,
    beliefs: dict[tuple, BeliefModule],
    conds: dict[str, Any] = None,
    x: str = "push times",
    palette: dict = PALETTE,
    average_blocks: bool = False,
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
        df_block = df.loc[index].reset_index()
        posteriors = beliefs[index].features

        # Get entropy of beliefs over time in block
        entropies = get_entropy_beliefs_over_time(posteriors)
        x_vals = df_block[x].values
        box = df_block["box"].values
        box_positions = df_block["box position"].values
        block_id = df_block["block_id"].values[0]
        res = {
            (index + (i + 1,)): [entropies[i + 1], x_vals[i], block_id]
            for i in range(len(entropies) - 1)
        }

        # Add prior
        res[(index + (0,))] = [entropies[0], 0, block_id]
        return res

    res, _ = process_blocks(df, _inner)
    res = reduce(lambda x, y: {**x, **y}, list(res.values()), {})
    columns = ["entropy", x, "block_id"]
    df_beliefs = pd.DataFrame.from_dict(
        res,
        orient="index",
        columns=columns,
    )

    df_beliefs.index = pd.MultiIndex.from_tuples(
        df_beliefs.index, names=INDEX[:MIN_INDEX]
    )
    if x in df_beliefs.index.names:
        df_beliefs = df_beliefs.drop(x, axis=1)

    # Bin pushes by time
    if x == "push times":
        bin_kwargs = kwargs_handler(kwargs, "bin_kwargs", dict(bin_width=60))
        df_beliefs[x] = bin_data(df_beliefs, x, **bin_kwargs).values

    fig_kwargs = kwargs_handler(kwargs, "fig_kwargs", {"nrows": 1, "ncols": 1})

    @legend_handler
    def _plot(i, subj, **kwargs):
        fig, ax = fig_init(**fig_kwargs)
        conds = {"subject": subj}
        df_subj = filter_df(df_beliefs, conds=conds)

        base_params = {
            "x": x,
            "y": "entropy",
            "color": "black",
            "ax": ax,
            **kwargs,
        }
        plot_block_average_or_traces(
            df_subj, show_traces=average_blocks, params=base_params
        )

        fig.suptitle(titler(title="Entropy of beliefs about schedule", conds=conds))
        fig.tight_layout()
        return ax

    across_conditions_plotter(df.index.unique("subject"), _plot, **kwargs)
