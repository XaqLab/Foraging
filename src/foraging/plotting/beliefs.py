from typing import Optional, Any

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from numpy.typing import ArrayLike

from foraging.utils import models
from foraging.plotting import BOX_COLORS, BOX_LABELS, bp


def likelihood_single_obs(obs_model: models.Observation, latents: ArrayLike, obs: Any, ax: Optional[plt.Axes] = None,
                          **kwargs: Any) -> plt.Axes:
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
    if ax is None:
        fig, ax = plt.subplots(**kwargs.pop('fig_kwargs', {}))

    # Compute likelihood for each latent
    likelihoods = [obs_model.probability(obs, latent) for latent in latents]

    # Plot the likelihoods
    ax.plot(latents, likelihoods)
    ax.set_xlabel("Latent")
    ax.set_ylabel("Likelihood")
    ax.set_title(f"Likelihood given observation {obs}")
    return ax


def reward_beliefs3d(df: pd.DataFrame, reward_beliefs: np.ndarray, box_labels: list = BOX_LABELS,
                     box_colors: list = BOX_COLORS, fontsize: int = 10,
                     title: str = "Belief about reward availability at time of push",
                     ax: Optional[plt.Axes] = None, **kwargs: Any) -> plt.Axes:
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
        fig = plt.figure(**kwargs.pop('fig_kwargs', {}))
        ax = fig.add_subplot(projection='3d')

    # Plot beliefs
    plt_kwargs = {'cmap': ListedColormap(box_colors)} | kwargs.pop('plt_kwargs', {})
    p = ax.scatter(reward_beliefs[:, 0], reward_beliefs[:, 1], reward_beliefs[:, 2], c=df['box rank'], **plt_kwargs)

    # Set axis limits and labels
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])
    ax.set_xlabel('Belief that reward is available at slow box', fontsize=fontsize)
    ax.set_ylabel('Belief that reward is available at medium box', fontsize=fontsize)
    ax.set_zlabel('Belief that reward is available at fast box', fontsize=fontsize)
    ax.set_title(title)
    ax.view_init(elev=20, azim=-135)

    # Add colorbar with discrete labels
    cbar = plt.colorbar(p, ax=ax, ticks=[0, 1, 2])
    cbar.set_label('Schedule')
    cbar.set_ticklabels(box_labels)
    return ax


def schedule_beliefs_block(df: pd.DataFrame, index: tuple, x_col: str = 'push #', box_labels: list = BOX_LABELS, box_colors: list = BOX_COLORS, ax: Optional[plt.Axes] = None,
                           **kwargs) -> plt.Axes:
    """
    Plots the beliefs about the schedule for a specific block in the experiment, with uncertainty bands
    and reward outcomes.

    Args:
        df: DataFrame containing session data.
        index: Index of the block to analyze in the DataFrame.
        x_col: The value from `df` used for the x-axis, typically 'push #' or 'push times'.
        ax: Optional, existing matplotlib Axes object. If None, a new one will be created.

        kwargs:
            - 'fig_kwargs': Dictionary to specify figure properties when creating a new figure (passed to `plt.subplots`).
            - 'plt_kwargs': Dictionary to specify additional plotting properties for the line plot (passed to `bp wrapper`).
            - 'lgd_kwargs': Dictionary of keyword arguments for customizing the legend (passed to `plt.legend`).

    Returns:
        ax: The matplotlib Axes object with the plot.
    """
    # Create axes if none provided
    if ax is None:
        fig, ax = plt.subplots(**kwargs.pop('fig_kwargs', {}))

    sub_df = df.loc[index]

    # Plot the mean schedule as a line plot
    bp(sns.lineplot)(sub_df, x=x_col, y='mean schedule', title_prefix=f'Beliefs about schedule for {index}',
                     **kwargs.pop('plt_kwargs', {}), ax = ax)

    # Loop over each box and add error bands using fill_between
    for i, (cat, sub_df2) in enumerate(sub_df.groupby("box rank")):
        if x_col in sub_df2.columns:
            ax.fill_between(sub_df2[x_col], sub_df2["mean schedule"] - sub_df2["uncertainty schedule"],
                             sub_df2["mean schedule"] + sub_df2["uncertainty schedule"],
                             color=box_colors[cat], alpha=0.1)
        else:
            ax.fill_between(sub_df2.index.get_level_values(x_col),
                             sub_df2["mean schedule"] - sub_df2["uncertainty schedule"],
                             sub_df2["mean schedule"] + sub_df2["uncertainty schedule"],
                             color=box_colors[cat], alpha=0.1)

    # Draw horizontal lines for reward rates
    schedules = -np.sort(-sub_df['schedule'].unique())
    [ax.axhline(schedules[i], color=box_colors[i], linestyle='--') for i in range(len(schedules))]

    # Plot reward outcomes as scatter plot with different colors
    if x_col in sub_df.columns:
        x = sub_df[x_col]
    else:
        x = sub_df.index.get_level_values(x_col)
    y = sub_df['mean schedule']
    mask = sub_df['reward outcomes'] == True
    ax.scatter(x[mask], y[mask], c='g', marker='^')  # Rewarded
    ax.scatter(x[~mask], y[~mask], c='r', marker='v')  # Not rewarded

    # Add legend with proxy artists
    legend_kwargs = {'loc': 'upper right'} | kwargs.pop('lgd_kwargs', {})
    legend_elements = ([Line2D([0], [0], color=box_colors[i], linestyle='-', label=box_labels[i]) for i in
                        range(len(box_labels))]
                       + [Line2D([0], [0], color='green', linestyle='', marker='^', label='rewarded'),
                          Line2D([0], [0], color='red', linestyle='', marker='v', label='no reward')])
    ax.legend(handles=legend_elements, **legend_kwargs)
    return ax