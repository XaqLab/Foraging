"""
This module contains functions for plotting beliefs.
"""

from functools import reduce
from typing import Any, Callable, Iterable, Optional

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from numpy.typing import ArrayLike
from scipy.optimize import minimize_scalar

from foraging.models._base import HashableDict
from foraging.models.distribution import FactorizedPosterior, Posterior
from foraging.models.experiment import Experiment
from foraging.plotting import (
    bp,
    fig_init,
    legend_corrector,
    multiplot,
    plot_average_or_traces,
    titler,
)
from foraging.plotting._base import (
    BasePlotter,
    Embeddable,
    across_conditions_plotter,
)
from foraging.utils._base import kwargs_handler
from foraging.utils.belief import (
    fisher_info_reward_observations,
    get_entropy_beliefs_over_time,
    get_map_indices_over_time,
    get_map_over_time,
    get_mean_beliefs_over_time,
    get_std_beliefs_over_time,
)
from foraging.utils.data import (
    bin_data,
    filter_df,
    map_box_positions_to_ranks,
)
from foraging.utils.stats import moving_average


def likelihood_single_obs(
    obs_model: Any,
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


@legend_corrector
def plot_fisher_info(
    t: Any,
    schedules: list[float],
    palette: dict,
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


@legend_corrector
def plot_cramer_rao_lb(
    n: int,
    schedules: list[float],
    palette: dict,
    alpha: float = 1,
    title: str = "Cramer-Rao Lower Bound as a Function of # Observations",
):

    # Plot Cramer-Rao lower bound for perfect observations vs reward observations
    n = np.arange(n) + 1
    colors = list(palette.values())
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
    labels = list(palette.keys())
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


class BeliefPlotter(BasePlotter):
    def __init__(
        self,
        beliefs: dict[HashableDict, FactorizedPosterior],
        dataset: Experiment,
        config: dict | Iterable,
    ):
        super().__init__(dataset, config)
        self.beliefs = beliefs

    def _init_vars(self, **kwargs):
        if "beliefs" in kwargs and kwargs["beliefs"] is None:
            kwargs["beliefs"] = self.beliefs
        return super()._init_vars(**kwargs)

    def reward_beliefs3d(
        self,
        reward_beliefs: np.ndarray,
        dataset: Experiment = None,
        conds: dict[str, Any] = None,
        palette: dict[str, Any] = None,
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
        dataset, palette = self._init_vars(dataset=dataset, palette=palette)
        df = dataset.filter(conds).df

        # Create axes if none provided
        if ax is None:
            fig = plt.figure(**kwargs.pop("fig_kwargs", {}))
            ax = fig.add_subplot(projection="3d")

        # Plot beliefs
        plt_kwargs = {"cmap": ListedColormap(list(palette.values()))} | kwargs.pop(
            "plt_kwargs", {}
        )
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
        ax.set_ylabel(
            "Belief that reward is available at medium box", fontsize=fontsize
        )
        ax.set_zlabel("Belief that reward is available at slow box", fontsize=fontsize)
        ax.set_title(title)
        ax.view_init(elev=20, azim=-135)

        # Add colorbar with discrete labels
        cbar = plt.colorbar(p, ax=ax, ticks=[0, 1, 2])
        cbar.set_label("Schedule")
        cbar.set_ticklabels(list(palette.keys()))
        return ax

    @legend_corrector(bbox=(1.15, 1))
    def plot_schedule_beliefs_in_block(
        self,
        beliefs: dict[HashableDict, FactorizedPosterior] = None,
        dataset: Experiment = None,
        conds: dict[str, Any] = None,
        palette: dict[str, Any] = None,
        heatmap_palette: dict = None,
        show_stats: bool = False,
        dt: float = 1,
        ax: plt.Axes = None,
        **kwargs,
    ) -> plt.Axes:
        """
        Plots the beliefs about the schedule for a specific block in the experiment, with heatmaps showing
        belief probabilities over time for each box, optional statistics (mean and standard deviation),
        true schedule lines, and reward outcomes.

        Args:
            beliefs: Dictionary mapping condition tuples to FactorizedPosterior objects containing
                the belief distributions over time.
            dataset: Experiment object containing the dataset. If None, uses the default dataset.
            conds: Dictionary of conditions to filter the dataset. If None, uses default conditions.
            palette: Dictionary mapping box labels to colors for plotting. If None, uses default palette.
            heatmap_palette: Dictionary mapping box labels to colormaps for heatmaps. If None, uses default.
            show_stats: If True, overlays mean and standard deviation lines on the heatmaps.
            dt: timescale to display belief data
            ax: Optional, existing matplotlib Axes object or array of Axes. If None, a new figure will be created.

            kwargs:
                - 'fig_kwargs': Dictionary to specify figure properties when creating a new figure (passed to `plt.subplots`).

        Returns:
            ax: The matplotlib Axes object or array of Axes with the plot.
        """
        beliefs, dataset, palette, heatmap_palette = self._init_vars(
            beliefs=beliefs,
            dataset=dataset,
            palette=palette,
            heatmap_palette=heatmap_palette,
        )
        conds = HashableDict(conds)

        # Get block data
        df_block = dataset.filter(conds).df.reset_index()
        n_boxes = df_block["n boxes"].values[0]
        push_times = df_block["push times"].values
        posteriors = np.asarray(beliefs[conds].representation)
        schedule_candidates = beliefs[conds].prior[0].support
        schedules = np.sort(df_block["assigned schedules"].iloc[0])
        mean_across_pushes = get_mean_beliefs_over_time(
            posteriors, schedule_candidates
        )[1:]

        # Create a new array to hold the interpolated data
        # The new number of columns will be the difference between the max and min time points, creating 1 second time bins
        new_num_cols = int((push_times[-1] + 1) / dt)
        interpolated_beliefs = np.zeros(
            (new_num_cols, n_boxes, len(schedule_candidates))
        )
        start = 0
        for push_idx, t in enumerate(push_times):
            end = int(t / dt)
            interpolated_beliefs[start:end, :, :] = posteriors[push_idx, :, :]
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
            
            # Set x-tick labels to reflect actual time values
            current_xticks = ax[box_pos].get_xticks()
            ax[box_pos].set_xticklabels(
                [
                    f"{tick * dt:.1f}" if 0 <= tick < new_num_cols else ""
                    for tick in current_xticks
                ]
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
            ax[box_pos].set_title(
                f"Belief about schedule for {box_labels[box_rank]} box"
            )
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
                push_times[reward_mask & box_mask] / dt,
                mean_across_pushes[reward_mask & box_mask, box_pos],
                c="black",
                marker="^",
                s=50,
            )
            ax[box_pos].scatter(
                push_times[~reward_mask & box_mask] / dt,
                mean_across_pushes[~reward_mask & box_mask, box_pos],
                edgecolors="black",
                marker="v",
                s=50,
                facecolors="none",
            )

            # Add legend
            if box_pos == 0:
                handles = [
                    Line2D(
                        [0], [0], color="black", linestyle="--", label="true schedule"
                    ),
                    Line2D(
                        [0],
                        [0],
                        color="black",
                        linestyle="",
                        marker="^",
                        label="rewarded",
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

    @legend_corrector(bbox=(1.15, 1))
    def plot_permutation_beliefs_in_block(
        self,
        beliefs: dict[HashableDict, Posterior] = None,
        dataset: Experiment = None,
        conds: dict[str, Any] = None,
        palette: dict[str, Any] = None,
        heatmap_palette: str = "Greys",
        dt: float = 1,
        ax: plt.Axes = None,
        **kwargs,
    ) -> plt.Axes:
        """
        Plots the beliefs about the permutation for a specific block in the experiment, with heatmaps showing
        belief probabilities over time for each box, optional statistics (mean and standard deviation),
        true permutation lines, and reward outcomes.

        Args:
            beliefs: Dictionary mapping condition tuples to Posterior objects containing
                the belief distributions over time.
            dataset: Experiment object containing the dataset. If None, uses the default dataset.
            conds: Dictionary of conditions to filter the dataset. If None, uses default conditions.
            palette: Dictionary mapping box labels to colors for plotting. If None, uses default palette.
            heatmap_palette: Colormap for the heatmap. Defaults to "Greys".
            show_stats: If True, overlays mean and standard deviation lines on the heatmaps.
            dt: timescale to display belief data
            ax: Optional, existing matplotlib Axes object or array of Axes. If None, a new figure will be created.

            kwargs:
                - 'fig_kwargs': Dictionary to specify figure properties when creating a new figure (passed to `plt.subplots`).

        Returns:
            ax: The matplotlib Axes object or array of Axes with the plot.
        """
        #TODO: change 
        beliefs, dataset, palette, heatmap_palette = self._init_vars(
            beliefs=beliefs,
            dataset=dataset,
            palette=palette,
            heatmap_palette=heatmap_palette,
        )
        conds = HashableDict(conds)

        # Get block data
        df_block = dataset.filter(conds).df.reset_index()
        n_boxes = df_block["n boxes"].values[0]
        push_times = df_block["push times"].values
        posteriors = np.asarray(beliefs[conds].representation)
        permutation_candidates = beliefs[conds].prior.support

        # Get the true permutation from assigned schedules
        assigned_schedules = df_block["assigned schedules"].iloc[0]
        # Find which permutation matches the assigned schedules
        true_permutation_idx = None
        for idx, perm in enumerate(permutation_candidates):
            if hasattr(perm, 'permutation') and list(perm.permutation) == list(assigned_schedules):
                true_permutation_idx = idx + 0.5
                break

        map_across_pushes = get_map_indices_over_time(
            posteriors, permutation_candidates
        )[1:] + 0.5

        # Create a new array to hold the interpolated data
        # The new number of columns will be the difference between the max and min time points, creating 1 second time bins
        new_num_cols = int((push_times[-1] + 1) / dt)
        interpolated_beliefs = np.zeros(
            (new_num_cols, posteriors.shape[1])
        )
        start = 0
        for push_idx, t in enumerate(push_times):
            end = int(t / dt)
            interpolated_beliefs[start:end, :] = posteriors[push_idx, :]
            start = end

        fig, ax = fig_init(
            ax,
            **kwargs.pop(
                "fig_kwargs",
                {
                    "nrows": 1,
                    "ncols": 1,
                    "sharex": True,
                    "sharey": True,
                    "figsize": (15, 10),
                },
            ),
        )

        # Plot belief probabilities
        sns.heatmap(
            interpolated_beliefs.T,
            cmap=heatmap_palette,
            cbar_kws={"label": "probability"},
            ax=ax,
        )
        
        # Set x-tick labels to reflect actual time values
        current_xticks = ax.get_xticks()
        ax.set_xticklabels(
            [
                f"{tick * dt:.2f}" if 0 <= tick < new_num_cols else ""
                for tick in current_xticks
            ]
        )
        
        # Plot true permutation
        if true_permutation_idx is not None:
            ax.axhline(true_permutation_idx, color="black", linestyle="--")
        ax.set_title(
            f"Belief about permutation"
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Possible permutations")
        current_yticks = ax.get_yticks()
        ax.set_yticklabels(
            [
                str(tuple(permutation_candidates[int(j)].permutation))
                if hasattr(permutation_candidates[int(j)], 'permutation')
                else str(permutation_candidates[int(j)])
                for j in current_yticks
                if int(j) < len(permutation_candidates)
            ],
            rotation=45
        )

        # Plot reward outcomes
        pos_to_rank = map_box_positions_to_ranks(df_block)
        box_labels = list(palette.keys())
        for box_pos in range(n_boxes):
            box_rank = pos_to_rank.loc[box_pos].values[0]
            box_mask = df_block["box rank"].values == box_rank
            reward_mask = df_block["reward outcomes"].values
            ax.scatter(
                push_times[reward_mask & box_mask] / dt,
                map_across_pushes[reward_mask & box_mask],
                color=palette[box_labels[box_rank]],
                marker="^",
                s=50,
            )
            ax.scatter(
                push_times[~reward_mask & box_mask] / dt,
                map_across_pushes[~reward_mask & box_mask],
                edgecolors=palette[box_labels[box_rank]],
                marker="v",
                s=50,
                facecolors="none",
            )

        handles = [
            Line2D(
                [0], [0], color="black", linestyle="--", label="true permutation"
            ),
            Line2D(
                [0],
                [0],
                color="black",
                linestyle="",
                marker="^",
                label="rewarded",
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
        ax.legend(handles=handles)
        fig.suptitle(titler(f"Beliefs about permutation (true schedules) {assigned_schedules}", conds=conds))
        fig.tight_layout()
        return ax


    def plot_schedule_beliefs_mean_and_std_across_block(
        self,
        beliefs: dict[HashableDict, FactorizedPosterior] = None,
        dataset: Experiment = None,
        conds: dict[str, Any] = None,
        palette: dict[str, Any] = None,
        x: str = "push times",
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
            **kwargs: Unlisted keyword arguments get passed to `plot_block_average_or_traces`.
                - 'fig_kwargs': Dictionary to specify figure properties when creating a new figure (passed to `plt.subplots`).

        Returns:
            ax: The matplotlib Axes object with the plot.
        """
        beliefs, dataset, palette = self._init_vars(
            beliefs=beliefs, dataset=dataset, palette=palette
        )
        dataset = dataset.filter(conds)
        df = dataset.df

        # Get beliefs of each block
        def _inner(
            dataset: "Experiment",
            block_key: dict[str, Any],
            block: pd.DataFrame,
            *args,
            **kwargs,
        ):
            block = block.reset_index()
            posteriors = np.asarray(beliefs[block_key].representation)
            schedule_candidates = beliefs[block_key].prior[0].support

            # Get mean and std of beliefs
            mean = get_mean_beliefs_over_time(posteriors, schedule_candidates)
            std = get_std_beliefs_over_time(posteriors, schedule_candidates)
            x_vals = block[x].values
            box = block["box"].values
            box_positions = block["box position"].values
            block_id = block["block_id"].values[0]
            index = tuple(block_key.values())

            # Extract beliefs only when they change ie. each x value is assigned the relevant belief that changed as a result of that x value
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
            box_labels = list(palette.keys())
            for i in block["box position"].unique():
                res[(index + (-i,))] = [
                    mean[0, i],
                    std[0, i],
                    0,
                    box_labels[i],
                    block_id,
                ]
            return res

        res, _ = dataset.process_blocks(_inner)
        res = reduce(lambda x, y: {**x, **y}, list(res.values()), {})
        df_beliefs = pd.DataFrame.from_dict(
            res,
            orient="index",
            columns=["mean", "standard deviation", x, "box", "block_id"],
        )

        df_beliefs.index = pd.MultiIndex.from_tuples(
            df_beliefs.index,
            names=dataset.block_identifiers + dataset.within_block_identifiers,
        )
        fig_kwargs = kwargs_handler(
            kwargs, "fig_kwargs", {"nrows": 2, "ncols": 1, "figsize": (10, 10)}
        )

        @legend_corrector
        def _plot(embeddables: Iterable[Embeddable], **kwargs):
            subj = embeddables[0].value
            fig, ax = plt.subplots(**fig_kwargs)
            conds = {"subject": subj}
            df_subj = filter_df(df_beliefs, conds=conds)

            # Plot mean
            params = {
                "x": x,
                "y": "mean",
                "hue": "box",
                "palette": palette,
                "ax": ax[0],
                **kwargs,
            }
            plot_average_or_traces(df_subj, **params)

            # Plot schedules on top of means
            schedules = sorted(dataset.filter(conds=conds).get("assigned schedules")[0])
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

            # Now plot std
            params.update(
                {
                    "y": "standard deviation",
                    "legend": False,
                    "ax": ax[1],
                }
            )
            plot_average_or_traces(df_subj, **params)

            fig.suptitle(titler(title="Beliefs about schedule", conds=conds))
            fig.tight_layout()
            return ax

        across_conditions_plotter(
            "subject", df.index.unique("subject"), _plot, **kwargs
        )

    def plot_accuracy_across_block(
        self,
        accuracies: dict[tuple, ArrayLike],
        dataset: Experiment = None,
        conds: dict[str, Any] = None,
        x: str = "push times",
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
        dataset = self._init_vars(dataset=dataset)
        dataset = dataset.filter(conds)

        # Get beliefs of each block
        def _inner(
            dataset: Experiment,
            block_key: dict[str, Any],
            block: pd.DataFrame,
            *args,
            **kwargs,
        ):
            block = block.reset_index()
            accuracy = accuracies[block_key]

            x_vals = block[x].values
            block_id = block["block_id"].values[0]
            index = tuple(block_key.values())
            res = {
                (index + (i + 1,)): [
                    accuracy[i + 1],
                    x_vals[i],
                    block_id,
                ]
                for i in range(len(accuracy) - 1)
            }

            # Add prior
            res[(index + (0,))] = [accuracy[0], 0, block_id]
            return res

        res, _ = dataset.process_blocks(_inner)
        res = reduce(lambda x, y: {**x, **y}, list(res.values()), {})
        df_beliefs = pd.DataFrame.from_dict(
            res,
            orient="index",
            columns=["accuracy", x, "block_id"],
        )

        df_beliefs.index = pd.MultiIndex.from_tuples(
            df_beliefs.index,
            names=dataset.block_identifiers + dataset.within_block_identifiers,
        )

        params = {
            "x": x,
            "y": "accuracy",
            "ax": ax,
            "conds": conds,
            **kwargs,
        }
        return plot_average_or_traces(df_beliefs, **params)

        # # Average data over time
        # fig_kwargs = kwargs_handler(kwargs, "fig_kwargs")

        # @legend_corrector
        # def _plot(i, subj, **kwargs):
        #     fig, ax = fig_init(**fig_kwargs)
        #     conds = {"subject": subj}
        #     df_subj = filter_df(df_beliefs, conds=conds)
        #     params = {
        #         "x": x,
        #         "y": "accuracy",
        #         "ax": ax,
        #         **kwargs,
        #     }
        #     plot_average_or_traces(df_subj, **params)

        #     fig.suptitle(titler(title="Theoretical accuracy", conds=conds))
        #     fig.tight_layout()
        #     return ax

        # across_conditions_plotter(df.index.unique("subject"), _plot, **kwargs)


# def plot_schedule_beliefs_efficiency_across_block(
#     df: pd.DataFrame,
#     beliefs: dict[tuple, FactorizedPosterior],
#     conds: dict[str, Any] = None,
#     x: str = "push times",
#     palette: dict = PALETTE,
#     show_traces: bool = False,
#     **kwargs,
# ) -> plt.Axes:
#     """
#     Plots the beliefs about the schedule for a specific block in the experiment, with uncertainty bands
#     and reward outcomes.

#     Args:
#         df: DataFrame containing session data.
#         index: Index of the block to analyze in the DataFrame.
#         x: The value from `df` used for the x-axis, typically 'push #' or 'push times'.
#         ax: Optional, existing matplotlib Axes object. If None, a new one will be created.

#         kwargs:
#             - 'fig_kwargs': Dictionary to specify figure properties when creating a new figure (passed to `plt.subplots`).
#             - 'plt_kwargs': Dictionary to specify additional plotting properties for the line plot (passed to `bp wrapper`).
#             - 'lgd_kwargs': Dictionary of keyword arguments for customizing the legend (passed to `plt.legend`).

#     Returns:
#         ax: The matplotlib Axes object with the plot.
#     """

#     df = filter_df(df, conds)

#     # Find optimal push time for each schedule
#     res = minimize_scalar(
#         lambda x: -fisher_info_reward_observations(x, schedule, alpha)
#     )
#     x_max = res.x
#     fisher_info_reward = fisher_info_reward_observations(x_max, schedule, alpha)

#     std_crlb_reward = np.sqrt(1 / (n * fisher_info_reward))


#     # Get beliefs of each block
#     def _inner(df: pd.DataFrame, index: tuple):
#         df_block = df.loc[index].reset_index()
#         posteriors = np.asarray(beliefs[index].representation)
#         schedule_candidates = beliefs[index].prior[0].support
#         schedules = sorted(df_block["schedule"].unique())

#         #todo: get efficiency by dividing std by crlb
#         # Get std of beliefs
#         std = get_std_beliefs_over_time(posteriors, schedule_candidates)
#         x_vals = df_block[x].values
#         box = df_block["box"].values
#         box_positions = df_block["box position"].values
#         block_id = df_block["block_id"].values[0]
#         res = {
#             (index + (i + 1,)): [
#                 std[i + 1, box_positions[i]],
#                 x_vals[i],
#                 box[i],
#                 block_id,
#             ]
#             for i in range(len(std) - 1)
#         }


#     res, _ = process_blocks(df, _inner)
#     res = reduce(lambda x, y: {**x, **y}, list(res.values()), {})
#     df_beliefs = pd.DataFrame.from_dict(
#         res,
#         orient="index",
#         columns=["mean", "standard deviation", x, "box", "block_id"],
#     )

#     df_beliefs.index = pd.MultiIndex.from_tuples(
#         df_beliefs.index, names=INDEX[:MIN_INDEX]
#     )

#     # Average data over time
#     groupers = ["box", "block_id"]
#     if x == "push times":
#         smooth_kwargs = kwargs_handler(kwargs, "smooth_kwargs", {"fill_value": np.nan, "min_periods": 1})
#         ma_mean = moving_average(
#             df_beliefs,
#             x=x,
#             y="mean",
#             y_name="mean",
#             groupers=groupers,
#             **smooth_kwargs,
#         )
#         ma_std = moving_average(
#             df_beliefs,
#             x=x,
#             y="standard deviation",
#             y_name="standard deviation",
#             groupers=groupers,
#             **smooth_kwargs,
#         )
#         df_beliefs = ma_mean
#         df_beliefs["standard deviation"] = ma_std["standard deviation"]
#         x = "time"

#     fig_kwargs = kwargs_handler(
#         kwargs, "fig_kwargs", {"nrows": 2, "ncols": 1, "figsize": (10, 10)}
#     )

#     @legend_corrector
#     def _plot(i, subj, **kwargs):
#         fig, ax = fig_init(**fig_kwargs)
#         conds = {"subject": subj}
#         df_subj = filter_df(df_beliefs, conds=conds)

#         params = {
#             "x": x,
#             "y": "mean",
#             "hue": "box",
#             "palette": palette,
#             "ax": ax[0],
#             **kwargs,
#         }
#         plot_average_or_traces(df_subj, show_traces=show_traces, **params)

#         # Plot schedules on top of means
#         schedules = sorted(filter_df(df, conds=conds)["assigned schedules"].iloc[0])
#         colors = list(palette.values())
#         for i, schedule in enumerate(schedules):
#             ax[0].axhline(schedule, color=colors[i], linestyle="--")

#         actual_schedule_line = Line2D(
#             [0], [0], color="black", linestyle="--", label="actual schedule"
#         )

#         # If you have an existing legend, you can add this to the existing handles
#         existing_handles, existing_labels = ax[0].get_legend_handles_labels()
#         all_handles = existing_handles + [actual_schedule_line]
#         all_labels = existing_labels + ["actual schedule"]

#         # Remove the old legend and create a new one
#         if ax[0].get_legend():
#             ax[0].get_legend().remove()
#             ax[0].legend(handles=all_handles, labels=all_labels)

#         params.update(
#             {
#                 "y": "standard deviation",
#                 "legend": False,
#                 "ax": ax[1],
#             }
#         )
#         plot_average_or_traces(df_subj, show_traces=show_traces, **params)

#         fig.suptitle(titler(title="Beliefs about schedule", conds=conds))
#         fig.tight_layout()
#         return ax

#     across_conditions_plotter(df.index.unique("subject"), _plot, **kwargs)


# def plot_schedule_beliefs_entropy_across_blocks(
#     df: pd.DataFrame,
#     beliefs: dict[tuple, Posterior],
#     conds: dict[str, Any] = None,
#     x: str = "push times",
#     palette: dict = PALETTE,
#     average_blocks: bool = False,
#     **kwargs,
# ) -> plt.Axes:
#     """
#     Plots the beliefs about the schedule for a specific block in the experiment, with uncertainty bands
#     and reward outcomes.

#     Args:
#         df: DataFrame containing session data.
#         index: Index of the block to analyze in the DataFrame.
#         x: The value from `df` used for the x-axis, typically 'push #' or 'push times'.
#         ax: Optional, existing matplotlib Axes object. If None, a new one will be created.

#         kwargs:
#             - 'fig_kwargs': Dictionary to specify figure properties when creating a new figure (passed to `plt.subplots`).
#             - 'plt_kwargs': Dictionary to specify additional plotting properties for the line plot (passed to `bp wrapper`).
#             - 'lgd_kwargs': Dictionary of keyword arguments for customizing the legend (passed to `plt.legend`).

#     Returns:
#         ax: The matplotlib Axes object with the plot.
#     """

#     df = filter_df(df, conds)

#     # Get beliefs of each block
#     def _inner(df: pd.DataFrame, index: tuple):
#         df_block = df.loc[index].reset_index()
#         posteriors = beliefs[index].features

#         # Get entropy of beliefs over time in block
#         entropies = get_entropy_beliefs_over_time(posteriors)
#         x_vals = df_block[x].values
#         box = df_block["box"].values
#         box_positions = df_block["box position"].values
#         block_id = df_block["block_id"].values[0]
#         res = {
#             (index + (i + 1,)): [entropies[i + 1], x_vals[i], block_id]
#             for i in range(len(entropies) - 1)
#         }

#         # Add prior
#         res[(index + (0,))] = [entropies[0], 0, block_id]
#         return res

#     res, _ = process_blocks(df, _inner)
#     res = reduce(lambda x, y: {**x, **y}, list(res.values()), {})
#     columns = ["entropy", x, "block_id"]
#     df_beliefs = pd.DataFrame.from_dict(
#         res,
#         orient="index",
#         columns=columns,
#     )

#     df_beliefs.index = pd.MultiIndex.from_tuples(
#         df_beliefs.index, names=INDEX[:MIN_INDEX]
#     )
#     if x in df_beliefs.index.names:
#         df_beliefs = df_beliefs.drop(x, axis=1)

#     # Bin pushes by time
#     if x == "push times":
#         bin_kwargs = kwargs_handler(kwargs, "bin_kwargs", dict(bin_width=60))
#         df_beliefs[x] = bin_data(df_beliefs, x, **bin_kwargs).values

#     fig_kwargs = kwargs_handler(kwargs, "fig_kwargs", {"nrows": 1, "ncols": 1})

#     @legend_corrector
#     def _plot(i, subj, **kwargs):
#         fig, ax = fig_init(**fig_kwargs)
#         conds = {"subject": subj}
#         df_subj = filter_df(df_beliefs, conds=conds)

#         base_params = {
#             "x": x,
#             "y": "entropy",
#             "color": "black",
#             "ax": ax,
#             **kwargs,
#         }
#         plot_average_or_traces(
#             df_subj, show_traces=average_blocks, params=base_params
#         )

#         fig.suptitle(titler(title="Entropy of beliefs about schedule", conds=conds))
#         fig.tight_layout()
#         return ax

#     across_conditions_plotter(df.index.unique("subject"), _plot, **kwargs)
