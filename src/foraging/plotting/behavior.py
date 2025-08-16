import logging
import pickle
from copy import deepcopy
from typing import Callable

import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from cycler import cycler
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from numpy.typing import ArrayLike
from scipy.spatial.distance import jensenshannon
from scipy.stats import expon, fit, kstest

from foraging.config.constants import (
    BIN_WIDTH,
    BOX_COLORS,
    BOX_POSITIONS,
    KAPPA_LEVELS,
    PALETTE,
    PALETTE_DARK,
    SEED,
    STEP,
    WINDOW_SIZE,
)
from foraging.plotting import (
    across_conditions_plotter,
    bp,
    enhanced_violinplot,
    fig_init,
    legend_handler,
    multiplot,
    titler,
    unitler,
)
from foraging.plotting._base import (
    get_bar_heights,
    palette_handler,
    plot_block_average_or_traces,
    plot_quantity_across_block,
    regplot,
)
from foraging.utils import INDEX, MIN_INDEX, kwargs_handler
from foraging.utils.data import (
    bin_data,
    filter_df,
    get_blocks,
    get_continuous_from_df_to_dict,
    process_block_safely,
    process_blocks,
)
from foraging.utils.stats import moving_average

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@legend_handler
def plot_experiment_overview(
    df: pd.DataFrame,
    conds: dict = None,
    title: str = "Overview of pushes over entire experiment",
    palette: dict = PALETTE,
    label_rotation: float = 35,
    annotate_block: bool = False,
    ax: plt.Axes = None,
    **kwargs,
) -> plt.Axes:
    """
    Plot the pushes over all blocks in the experiment, organized by sessions. This assumes one subject is specified in the `conds` dictionary.

    Args:
        df: DataFrame.
        conds: Dictionary to filter df.
        title: Title of figure.
        palette: Dictionary mapping box schedules to colors. Can also be a list of just colors.
        label_rotation: Angle to rotate y-tick labels by.
        annotate_block: if True, also display the block parameters above each block.
        ax: Axes to plot on. If None, a new figure and axes are created using plt.subplots. Specify keyword arguments in `fig_kwargs`.
        **kwargs: Keyword arguments passed to seaborn. May also contain nested kwargs.
            - 'fig_kwargs': Dictionary to specify figure properties when creating a new figure (passed to `plt.subplots`).

    Returns:
        The axes.
    """
    df = filter_df(df, conds)

    # Offset x-coord
    x_offset = get_blocks(df)["duration"].last()
    x_offset.iloc[1:] = x_offset.groupby(["subject", "session"]).cumsum().iloc[:-1]
    session_start = (
        x_offset.reset_index(level="block")
        .groupby(["subject", "session"])["block"]
        .first()
    )
    for (
        idx,
        x,
    ) in (
        session_start.items()
    ):  # Make sure each row (session) starts from 0 on the x-axis
        x_offset.loc[idx + (x,)] = 0
    df_temp = df.join(x_offset, rsuffix="_offset", on=INDEX[: MIN_INDEX - 1])
    df_temp["x"] = df_temp["push times"] + df_temp["duration_offset"]

    # Offset y-coord
    session_order = sorted(df_temp.index.unique("session"))
    session_offsets = {session: i for i, session in enumerate(session_order)}
    box_order = sorted(df_temp["box position"].unique())
    box_offsets = {box: box - 1 for box in box_order}

    df_temp["y_offset_1"] = df_temp["box position"].map(box_offsets)
    df_temp["y_offset_2"] = df_temp.index.map(
        lambda x: session_offsets[x[INDEX.index("session")]]
    )

    # Change multiplier to control spacing between sessions and rows
    y_offset_1_factor = 1
    y_offset_2_factor = 6
    df_temp["y"] = (
        1
        + y_offset_1_factor * df_temp["y_offset_1"]
        + y_offset_2_factor * df_temp["y_offset_2"]
    )

    # Create ax if none provided
    fig_kwargs = kwargs_handler(kwargs, "fig_kwargs", dict(figsize=(40, 50)))
    fig, ax = fig_init(ax, **fig_kwargs)

    legend = True
    for session in session_order:
        bp(sns.scatterplot)(
            filter_df(df_temp, {"session": session}),
            x="x",
            y="y",
            marker="|",
            s=100,
            hue="box",
            palette=palette,
            title=None,
            legend=legend,
            ax=ax,
            **kwargs,
        )
        legend = False

    # Annotate block parameters
    y_text_offset = 0.5
    if annotate_block:
        for session in session_order:
            df_session = filter_df(df_temp, {"session": session})
            y_text = df_session["y"].max() + y_text_offset
            blocks = df_session.index.get_level_values("block")
            kappas = df_session.index.get_level_values("kappa")
            kappas = kappas[np.insert(blocks[1:] != blocks[:-1], 0, True)]
            shapes = df_session.index.get_level_values("shape")
            shapes = shapes[np.insert(blocks[1:] != blocks[:-1], 0, True)]
            x_text = df_session["duration_offset"].unique()
            for i in range(len(kappas)):
                ax.text(
                    x_text[i], y_text, rf"$\kappa$={kappas[i]},$\alpha$={shapes[i]}"
                )

    # Demarcate blocks
    for session in session_order:
        df_session = filter_df(df_temp, {"session": session})
        x_text = df_session["duration_offset"].unique()[1:]
        ax.vlines(
            x_text,
            y_offset_2_factor * df_session["y_offset_2"].unique()[0] - 0.5,
            y_offset_2_factor * df_session["y_offset_2"].unique()[0] + 2.5,
            linestyles="dotted",
            colors="black",
        )

    # Tidy up axes
    ax.set_yticks(
        [
            y_offset_2_factor * offset + 0.5
            for offset in sorted(df_temp["y_offset_2"].unique())
        ],
        [str(s) for s in session_order],
    )
    ax.tick_params(axis="y", labelrotation=label_rotation)
    ax.set_xlabel("time in block (s)")
    ax.set_ylabel("session")
    ax.set_title(titler(title=title, conds=conds))
    fig.tight_layout()
    return ax


def plot_push_percentiles(df: pd.DataFrame, percentiles: dict = None, **kwargs):
    """
    Plot the percentiles of consecutive push intervals for each subject.

    Args:
        df: A DataFrame containing the data to be plotted.
        percentiles: A dictionary mapping subjects to their specific percentile thresholds.
        **kwargs: Additional keyword arguments passed to the plotting function.

    Returns:
        None
    """

    def _plot(i, subj, **kwargs):
        # Plot each push's percentiles
        ax = bp(sns.scatterplot)(
            df,
            x="consecutive push intervals",
            y="push percentiles",
            conds={"subject": subj},
            title="Percentiles of consecutive push intervals",
            x_unit="s",
            legend=False,
            **kwargs,
        )
        y = df.loc[(subj,), "push percentiles"].sort_values()
        x = df.loc[(subj,), "consecutive push intervals"].sort_values()

        # Mark specific percentile
        if percentiles:
            percentile = percentiles[subj]
            perc_idx = np.argmin((y - percentile) ** 2)
            push_perc = x.iloc[perc_idx]
            print(f"{subj}'s {round(percentile * 100, 2)}% push {push_perc}")
            ax.axvline(
                push_perc,
                color="black",
                linestyle="dashed",
                label=f"{round(percentile * 100, 2)}% push",
            )
            ax.legend(loc="upper right")

    return across_conditions_plotter(df.index.unique("subject"), _plot, **kwargs)


def plot_long_push_blocks(
    df: pd.DataFrame, top_n: int, figsize: tuple[float, float] = (20, 2.5), **kwargs
):
    """
    Plot the blocks containing the top N longest push intervals for each subject.
    This function sorts the data by the magnitude of push intervals in descending order and plots the blocks containing the top N longest push intervals for each subject.

    Args:
        df: A DataFrame containing the data to be plotted.
        top_n: The number of top longest push intervals to plot.
        figsize: A tuple specifying the size of the figure.
        **kwargs: Additional keyword arguments passed to the plotting function.

    Returns:
        None
    """
    # Sort data by magnitude of push interval in descending order
    df_sorted = df.sort_values(by="consecutive push intervals", ascending=False)

    def _plot(i, subj, **kwargs):
        fig, axes = plt.subplots(1, top_n, figsize=figsize)
        df_subject = filter_df(df_sorted, {"subject": subj})

        # Plot each block containing the `top_n` pushes
        for i, (idx, g) in enumerate(get_blocks(df_subject, sort=False)):
            if i >= top_n:
                break
            conds = dict(zip(INDEX[: MIN_INDEX - 1], idx))
            plot_pushes(
                g.sort_index(),
                conds=conds,
                title="",
                legend=False,
                ax=axes[i],
                **kwargs,
            )
            axes[i].set_title(f"session {conds['session']} block {conds['block']}")
        fig.suptitle(subj)
        fig.tight_layout()

    return across_conditions_plotter(df.index.unique("subject"), _plot, **kwargs)


def plot_pushes(
    df: pd.DataFrame,
    conds: dict = None,
    title: str = "Pushes",
    palette: dict = PALETTE,
    box_labels: list = BOX_POSITIONS,
    legend: bool = True,
    ax: plt.Axes = None,
    **kwargs,
) -> plt.Axes:
    """
    Plot the pushes in the block by the box they occur at.

    Args:
        df: DataFrame.
        conds: Dictionary to filter df.
        title: Title of figure.
        palette: Dictionary mapping box schedules to colors. Can also be a list of just colors.
        box_labels: Labels on y-axis for each box.
        legend: If True, display legend. Specify keyword arguments in `legend_kwargs`.
        ax: Axes to plot on. If None, a new figure and axes are created using plt.subplots. Specify keyword arguments in `fig_kwargs`.
        **kwargs: Additional keyword arguments passed to `plot_block_events`.

    Returns:
        The axes.
    """

    ax = plot_block_events(
        df,
        conds=conds,
        title=title,
        palette=palette,
        legend=legend,
        ax=ax,
        **kwargs,
    )

    # Custom plotting logic
    df_block = filter_df(df, conds).reset_index()
    ax.set_xlim([0, df_block["push times"].max() + 1])
    box_labels = [box_labels[i] for i in sorted(df_block["box position"].unique())]
    ax.set_yticks(range(len(box_labels)), box_labels, rotation=90, va="center")
    ax.set_ylabel("")
    return ax


@legend_handler
def plot_block_events(
    df: pd.DataFrame,
    conds: dict = None,
    x: str = "push times",
    y: str = "box position",
    x_unit: str = "s",
    y_unit: str = None,
    title: str = "Block activity",
    palette: dict = PALETTE,
    legend: bool = True,
    ax: plt.Axes = None,
    **kwargs,
) -> plt.Axes:
    """
    Plot the push-related variable in the block.

    Args:
        df: DataFrame.
        conds: Dictionary to filter df.
        x: Name of x variable in DataFrame. Defaults to `push times`.
        y: Name of y variable in DataFrame. Defaults to `box rank`.
        x_unit: Unit to assign to x. Defaults to `s` for seconds. Ignored if None.
        y_unit: Unit to assign to y. Defaults to None. Ignored if None.
        title: Title of figure.
        palette: Dictionary mapping box schedules to colors. Can also be a list of just colors.
        legend: If True, display legend. Specify keyword arguments in `legend_kwargs`.
        ax: Axes to plot on. If None, a new figure and axes are created using plt.subplots. Specify keyword arguments in `fig_kwargs`.
        **kwargs: Additional keyword arguments.
            - 'fig_kwargs': Dictionary to specify figure properties when creating a new figure (passed to `plt.subplots`).
            - 'line_kwargs': Dictionary to specify line properties (passed to 'LineCollection').
            - 'legend_kwargs': Dictionary of keyword arguments for customizing the legend (passed to `ax.legend`).

    Returns:
        The axes.
    """

    # Create ax if none provided
    fig_kwargs = kwargs_handler(kwargs, "fig_kwargs")
    fig, ax = fig_init(ax, **fig_kwargs)

    # Get block data and metadata
    df_block = filter_df(df, conds)
    schedules = sorted(df_block["schedule"].unique())
    kappa = df_block.index.unique("kappa")
    stim_type = df_block.index.unique("stimulus type")
    shape = df_block.index.unique("shape")

    if conds is None:
        conds = {}
    else:
        conds = deepcopy(conds)
    conds["kappa"] = kappa[0]
    conds["stim type"] = stim_type[0]
    conds["shape"] = shape[0]
    # Create switch segments (x, y) pairs for LineCollection
    x_vals = df_block[x].values
    y_vals = df_block[y].values
    colors = np.array(["black"] * (len(y_vals) - 1))
    # styles = ['dashed' if x else 'solid' for x in df_block['stay/switch'].values[1:]]
    segments = [
        [(x_vals[i], y_vals[i]), (x_vals[i + 1], y_vals[i + 1])]
        for i in range(len(x_vals) - 1)
    ]

    # Create the LineCollection
    line_kwargs = kwargs_handler(
        kwargs, "line_kwargs", dict(linestyles="--", linewidth=1, zorder=0)
    )
    lc = LineCollection(segments, colors=colors, **line_kwargs)

    # Set labels
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_title(titler(title=title, conds=conds))
    ax.set_ylabel(unitler(y, y_unit))
    ax.set_xlabel(unitler(x, x_unit))

    # Add reward outcomes with shaded (rewarded) and empty (not rewarded) markers
    colors = np.array([palette[i] for i in df_block["box"].values])
    mask = df_block["reward outcomes"].values
    ax.scatter(x_vals[mask], y_vals[mask], c=colors[mask], marker="^", s=80, zorder=2)
    ax.scatter(
        x_vals[~mask],
        y_vals[~mask],
        edgecolors=colors[~mask],
        marker="v",
        s=80,
        zorder=2,
        facecolors="none",
    )
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # Create legend manually with proxy artists
    if legend:
        legend_kwargs = kwargs_handler(
            kwargs, "legend_kwargs", {"loc": "upper left", "bbox_to_anchor": (1.05, 1)}
        )
        palette = palette_handler(palette, df_block["box"].unique())
        legend_elements = [
            Line2D([0], [0], color=palette[j], linestyle="-", label=schedules[i])
            for i, j in enumerate(palette.keys())
        ] + [
            Line2D([0], [0], color="black", linestyle="", marker="^", label="rewarded"),
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
        ax.legend(handles=legend_elements, **legend_kwargs)
    return ax


@legend_handler
def plot_recent_rewards_vs_push_percentiles(
    df: pd.DataFrame,
    n_samples: int = 5000,
    window: float = 30,
    seed: int = SEED,
    invert_reward: bool = False,
    **kwargs,
):
    """
    Plot reward outcomes in a time window preceding each push as a function of push interval.

    Args:
        df: A DataFrame containing push data.
        n_samples: The number of samples to draw for analysis.
        window: The time window in seconds to consider for reward calculation.
        seed: The seed for random number generation.
        invert_reward: If True, invert the reward outcomes.
        **kwargs: Additional keyword arguments passed to the plotting function.
          - fig_kwargs: Dictionary to specify figure properties when creating a new figure (passed to `plt.subplots`).

    Returns:
        The axes containing the plots.
    """

    reward_label = "# rewards" if not invert_reward else "# failures"
    df_rate_content = {"subject": [], reward_label: [], "push percentiles": []}
    rng = np.random.default_rng(seed)

    def _helper(subject: str):
        df_subject = filter_df(df, {"subject": subject})
        for row in df_subject.sample(
            n_samples, replace=True, random_state=rng
        ).itertuples():
            idx = row.Index

            # Get push immediately before current push
            push_num = idx[INDEX.index("push #")]
            new_idx = df.index.get_loc(idx) - 1
            if push_num == 1:
                continue

            # Identify time window
            new_row = df.iloc[new_idx]
            df_block = df.loc[idx[: MIN_INDEX - 1]]
            df_window = df_block.loc[
                (df_block["push times"] <= new_row["push times"])
                & (df_block["push times"] >= new_row["push times"] - window)
            ]
            row = df.loc[idx]

            # Populate data arrays if there are pushes in the window
            if len(df_window) > 0:
                df_rate_content["subject"].append(subject)
                if invert_reward:
                    df_rate_content[reward_label].append(
                        len(df_window) - df_window["reward outcomes"].sum()
                    )
                else:
                    df_rate_content[reward_label].append(
                        df_window["reward outcomes"].sum()
                    )
                df_rate_content["push percentiles"].append(new_row["push percentiles"])

    # For each subject, sample pushes calculate the reward fraction of the past `window` seconds prior to each push
    for subject in df.index.unique("subject"):
        _helper(subject)

    # Plot the reward fraction vs. push interval
    fig_kwargs = kwargs_handler(
        kwargs, "fig_kwargs", dict(nrows=2, ncols=1, sharex=True)
    )
    fig, axes = plt.subplots(**fig_kwargs)
    df_rate = pd.DataFrame(df_rate_content).set_index("subject").dropna()
    title = (
        f"# of rewards " if not invert_reward else "# of failures "
    ) + f"{window} s before push interval"
    sns.scatterplot(
        df_rate,
        x="push percentiles",
        y=reward_label,
        hue="subject",
        ax=axes[0],
        **kwargs,
    )
    axes[0].set_title(title)

    df_rate["push percentiles"] = bin_data(df_rate["push percentiles"], 10)
    # stacked_barplot(df=df_rate, x='percentiles', y=reward_label, hue='subject', ax=axes[1], **kwargs)
    sns.lineplot(
        df_rate,
        x="push percentiles",
        y=reward_label,
        hue="subject",
        ax=axes[1],
        legend=False,
        **kwargs,
    )
    axes[1].set_title(title)
    fig.tight_layout()
    return axes


def plot_previous_push_interval_vs_push_interval(
    df: pd.DataFrame, n_samples: int = 5000, seed: int = SEED, **kwargs
):
    """
    Plot previous push interval vs current push interval.

    Args:
        df: A DataFrame containing push data.
        n_samples: The number of samples to draw for analysis.
        seed: The seed for random number generation.
        **kwargs: Additional keyword arguments passed to the plotting function.
          - fig_kwargs: Dictionary to specify figure properties when creating a new figure (passed to `plt.subplots`).

    Returns:
        The axes containing the plots.
    """

    rng = np.random.default_rng(seed)
    fig_kwargs = kwargs_handler(kwargs, "fig_kwargs")

    @legend_handler
    def _plot(i, subj, **kwargs):

        df_rate_content = {
            "previous push intervals (s)": [],
            "push intervals (s)": [],
            "stay/switch": [],
        }

        # For each subject, sample pushes calculate the reward fraction of the past `window` seconds prior to each push
        df_subject = filter_df(df, {"subject": subj})
        for row in df_subject.sample(
            n_samples, replace=True, random_state=rng
        ).itertuples():
            idx = row.Index

            # Get push immediately before current push
            push_num = idx[INDEX.index("push #")]
            new_idx = df.index.get_loc(idx) - 1
            if push_num == 1:
                continue

            # Identify time window
            new_row = df.iloc[new_idx]
            row = df.loc[idx]

            # Populate data arrays
            df_rate_content["previous push intervals (s)"].append(
                new_row["consecutive push intervals"]
            )
            df_rate_content["push intervals (s)"].append(
                row["consecutive push intervals"]
            )
            df_rate_content["stay/switch"].append(row["stay/switch"])

        # Plot the previous push interval vs. push interval
        fig, ax = plt.subplots(**fig_kwargs)
        df_rate = pd.DataFrame(df_rate_content).dropna()
        sns.scatterplot(
            df_rate,
            x="push intervals (s)",
            y="previous push intervals (s)",
            hue="stay/switch",
            ax=ax,
            s=5,
            **kwargs,
        )
        ax.set_title(f"Previous push interval vs current push interval for {subj}")
        ax.set(yscale="log")
        ax.set(xscale="log")

        # Add a line of unity
        min_val, max_val = (
            df_rate[["push intervals (s)", "previous push intervals (s)"]].min().min(),
            df_rate[["push intervals (s)", "previous push intervals (s)"]].max().max(),
        )
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            linestyle="--",
            color="black",
            label="unity",
        )
        ax.legend()
        ax.set_aspect("equal")
        fig.tight_layout()
        return ax

    return across_conditions_plotter(df.index.unique("subject"), _plot, **kwargs)


@legend_handler
def plot_push_interval_autocorrelation(df: pd.DataFrame, lags: int = 10, **kwargs):
    """
    Plot the autocorrelation of consecutive push intervals for each subject over a range of lags, aggregated over blocks.

    Args:
        df: A DataFrame containing push data.
        lags: The number of lags to calculate the autocorrelation for.
        **kwargs: Additional keyword arguments passed to the plotting function.
          - fig_kwargs: Dictionary to specify figure properties when creating a new figure (passed to `plt.subplots`).

    Returns:
        The axes containing the plots.
    """

    # Prepare data for plotting
    autocorr_data = {
        "subject": [],
        "lag": [],
        "autocorrelation": [],
    }

    for subject in df.index.unique("subject"):
        df_subject = filter_df(df, {"subject": subject})
        for _, df_block in df_subject.groupby(level=["session", "block"]):
            push_intervals = df_block["consecutive push intervals"].dropna()
            if len(push_intervals) > 1:
                for lag in range(1, lags + 1):
                    autocorr_data["subject"].append(subject)
                    autocorr_data["lag"].append(lag)
                    autocorr_data["autocorrelation"].append(
                        push_intervals.autocorr(lag)
                    )

    # Convert to DataFrame
    df_autocorr = pd.DataFrame(autocorr_data)

    # Plot the autocorrelation
    fig_kwargs = kwargs_handler(kwargs, "fig_kwargs")
    fig, ax = plt.subplots(**fig_kwargs)
    sns.lineplot(
        data=df_autocorr,
        x="lag",
        y="autocorrelation",
        hue="subject",
        ax=ax,
        **kwargs,
    )
    ax.set_title("Autocorrelation of Push Intervals")
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.axhline(0, linestyle="--", color="black", linewidth=1)

    fig.tight_layout()
    return ax


@legend_handler
def plot_session_onsets_vs_push_percentiles(df: pd.DataFrame, **kwargs):
    """
    Plot the session onset times of pushes as a function of push percentile.

    Args:
        df: A DataFrame containing push data.
        n_samples: The number of samples to draw for analysis.
        seed: The seed for random number generation.
        **kwargs: Additional keyword arguments passed to the plotting function.
          - fig_kwargs: Dictionary to specify figure properties when creating a new figure (passed to `plt.subplots`).

    Returns:
        The axes containing the plot.
    """

    # Get time of each push in the session, not just block
    x_offset = get_blocks(df)["duration"].last()
    x_offset.iloc[1:] = x_offset.groupby(["subject", "session"]).cumsum().iloc[:-1]
    session_start = (
        x_offset.reset_index(level="block")
        .groupby(["subject", "session"])["block"]
        .first()
    )
    for (
        idx,
        x,
    ) in (
        session_start.items()
    ):  # Make sure each row (session) starts from 0 on the x-axis
        x_offset.loc[idx + (x,)] = 0
    df_temp = df.join(x_offset, rsuffix="_offset", on=INDEX[: MIN_INDEX - 1])
    df_temp["push time in session"] = df_temp["push times"] + df_temp["duration_offset"]
    df_temp["onset (s)"] = get_blocks(df_temp["push time in session"]).shift().fillna(0)

    # Plot the onset of push in session
    fig_kwargs = kwargs_handler(kwargs, "fig_kwargs")
    fig, ax = plt.subplots(**fig_kwargs)
    df_temp["push percentiles"] = bin_data(df_temp["push percentiles"])
    sns.lineplot(df_temp, x="push percentiles", y="onset (s)", hue="subject", ax=ax)
    ax.set_title("Onset of push in session")

    fig.tight_layout()
    return ax


@legend_handler
def plot_block_onsets_vs_push_percentiles(df: pd.DataFrame, **kwargs):
    """
    Plot the block onset times of pushes as a function of push percentile.

    Args:
        df: A DataFrame containing push data.
        **kwargs: Additional keyword arguments passed to the plotting function.
          - fig_kwargs: Dictionary to specify figure properties when creating a new figure (passed to `plt.subplots`).

    Returns:
        The axes containing the plot.
    """

    # Get time of each push in the block
    df = df.copy()
    df["onset (s)"] = get_blocks(df["push times"]).shift().fillna(0)

    # Plot the onset of push in block
    fig_kwargs = kwargs_handler(kwargs, "fig_kwargs")
    fig, ax = plt.subplots(**fig_kwargs)
    df["push percentiles"] = bin_data(df["push percentiles"])
    sns.lineplot(df, x="push percentiles", y="onset (s)", hue="subject", ax=ax)
    ax.set_title("Onset of push in block")

    fig.tight_layout()
    return ax


def plot_vertical_position_in_block(df: pd.DataFrame, conds: dict, data_dir: str):
    """
    Plot the vertical position of subjects within a block over time.
    This function retrieves vertical position data for a specified block and plots the vertical position over time, highlighting the position at the time of each push.

    Args:
        df: A DataFrame containing block data.
        conds: A dictionary specifying conditions to filter the DataFrame.
        data_dir: The directory containing continuous data files.

    Returns:
        None
    """

    # Get vertical data for block specified by `conds`
    df_block = filter_df(df, conds).copy()
    continuous_data, errors = get_continuous_from_df_to_dict(df_block, data_dir)
    time = continuous_data[tuple(conds.values())]["time"]
    vertical = continuous_data[tuple(conds.values())]["position"][:, 2]

    # Plot all vertical positions
    fig, ax = plt.subplots()
    ax.plot(time, vertical, c="grey", linewidth=1)

    # Plot vertical position at time of push
    idx = np.abs(time[None, :] - df_block["push times"].values[:, None]).argmin(axis=1)
    df_block["z-coordinate"] = vertical[idx]
    ax = plot_block_events(
        df_block,
        conds=conds,
        y="z-coordinate",
        y_unit="mm",
        title="Vertical position in block",
        ax=ax,
    )
    ax.axhline(y=500, label="screen", linestyle=":", c="black")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("vertical position (mm)")
    ax.set_title("Vertical position in block")
    return ax


@multiplot
def plot_vertical_position_vs_push_percentiles(
    df: pd.DataFrame, data_dir: str, **kwargs
):
    """
    Plot the average vertical position of subjects during push intervals against push percentiles.
    This function calculates the average vertical position during each push interval and plots it against the push percentiles for each subject.

    Args:
        df: A DataFrame containing push data.
        data_dir: The directory containing continuous data files.
        **kwargs: Additional keyword arguments passed to the plotting function.
          - fig_kwargs: Dictionary to specify figure properties when creating a new figure (passed to `plt.subplots`).

    Returns:
        None
    """
    continuous_data, _ = get_continuous_from_df_to_dict(df, data_dir)
    dfs_cont = []

    # Only keep blocks that have real position data
    for block, data in continuous_data.items():
        if np.any(data["position"] != np.nan):
            dfs_cont.append(
                df.xs(block, level=("subject", "session", "block"), drop_level=False)
            )
    df = pd.concat(dfs_cont)

    fig_kwargs = kwargs_handler(
        kwargs,
        "fig_kwargs",
        dict(figsize=(15, 5), nrows=1, ncols=3, sharex=True, sharey=True),
    )
    fig, axes = plt.subplots(**fig_kwargs)

    def _plot(i, subject, **kwargs):
        df_subject = filter_df(df, {"subject": subject})
        df_vertical = {
            "push intervals": [],
            "average vertical position": [],
            "push percentiles": [],
        }
        # Find average vertical position over each push interval
        for idx, row in df_subject.iterrows():
            try:
                conds = dict(
                    subject=subject,
                    session=idx[INDEX.index("session")],
                    block=idx[INDEX.index("block")],
                )
                time = continuous_data[tuple(conds.values())]["time"]
                vertical = continuous_data[tuple(conds.values())]["position"][:, 2]
                push_interval_end = row["push times"]
                push_interval_start = (
                    push_interval_end - row["consecutive push intervals"]
                )
                x = np.abs(
                    time[None, :]
                    - np.array([[push_interval_start], [push_interval_end]])
                ).argmin(axis=1)
                v = vertical[x[0] : x[1]]
                mean_vertical = v[~np.isnan(v)].mean()
                if not np.isnan(mean_vertical):
                    df_vertical["average vertical position"].append(mean_vertical)
                    df_vertical["push intervals"].append(
                        row["consecutive push intervals"]
                    )
                    df_vertical["push percentiles"].append(row["push percentiles"])
            except:
                continue
        sns.scatterplot(
            df_vertical, x="push percentiles", y="average vertical position", ax=axes[i]
        )
        axes[i].set_title(
            f"{subject}'s average vertical position\n occupied during push interval"
        )
        return axes[i]

    # Plot each subject's vertical position distribution
    return across_conditions_plotter(df.index.unique("subject"), _plot, **kwargs)


# TODO: visualize xy-position in block in 2d and super-impose block activity
# TODO: rework aggregating xy statistics-- is it velocity or position?
def plot_xy_velocity_long_vs_medium_pushes(
    df: pd.DataFrame,
    data_dir: str,
    percentiles: dict = None,
    n_samples: int = 5000,
    long_label: str = "long push intervals",
    medium_label: str = "medium push intervals",
    **kwargs,
):
    continuous_data, _ = get_continuous_from_df_to_dict(df, data_dir)
    dfs_cont = []
    rng = np.random.default_rng(SEED)

    # Only keep blocks that have real position data
    for block, data in continuous_data.items():
        if np.any(data["position"] != np.nan):
            dfs_cont.append(
                df.xs(block, level=("subject", "session", "block"), drop_level=False)
            )
    df = pd.concat(dfs_cont)

    def _plot(i, subject):
        df_subject = filter_df(df, {"subject": subject})
        xy_pos = {"x": [], "y": [], "type": []}

        # Find x-y positions for medium pushes
        for idx, row in df_subject.sample(frac=1, random_state=rng).iterrows():
            if row["push percentiles"] >= 0.25 and row["push percentiles"] <= 0.75:
                conds = dict(
                    subject=subject,
                    session=idx[INDEX.index("session")],
                    block=idx[INDEX.index("block")],
                )
                time = continuous_data[tuple(conds.values())]["time"]
                xy = continuous_data[tuple(conds.values())]["position"][:, :2]
                mask = ~np.isnan(xy).any(axis=1)
                xy_pos["x"] += list(xy[mask][:, 0])
                xy_pos["y"] += list(xy[mask][:, 1])
                xy_pos["type"] += [medium_label] * len(xy[mask])
                if len(xy_pos["x"]) >= n_samples:
                    break

        # Find x-y positions for medium pushes
        for idx, row in df_subject.sample(frac=1, random_state=rng).iterrows():
            if row["push percentiles"] >= percentiles[subject]:
                conds = dict(
                    subject=subject,
                    session=idx[INDEX.index("session")],
                    block=idx[INDEX.index("block")],
                )
                time = continuous_data[tuple(conds.values())]["time"]
                xy = continuous_data[tuple(conds.values())]["position"][:, :2]
                mask = ~np.isnan(xy).any(axis=1)
                xy_pos["x"] += list(xy[mask][:, 0])
                xy_pos["y"] += list(xy[mask][:, 1])
                xy_pos["type"] += [long_label] * len(xy[mask])
                if len(xy_pos["x"]) >= 2 * n_samples:
                    break
        fig, axes = plt.subplots(1, 2, **kwargs)
        xy_pos = pd.DataFrame(xy_pos)

        sns.histplot(xy_pos[xy_pos["type"] == medium_label], x="x", y="y", ax=axes[0])
        sns.histplot(xy_pos[xy_pos["type"] == long_label], x="x", y="y", ax=axes[1])
        axes[0].set_title(
            f"{subject}'s xy positions occupied\n during medium push interval"
        )
        axes[1].set_title(
            f"{subject}'s xy positions occupied\n during long push interval"
        )
        fig.tight_layout()

    # Plot each subject's vertical position distribution
    return across_conditions_plotter(df.index.unique("subject"), _plot)


@legend_handler(bbox=(1.1, 1))
def plot_hmm_probabilities_in_block(
    df: pd.DataFrame, filepath: str, block_idx: int = 3
):
    """
    Plot HMM probabilities over a specific block.
    This function loads Hidden Markov Model (HMM) probabilities from a file and overlays them on top of a specific block's data, visualizing the probabilities of different HMM policies.

    Args:
        df: A DataFrame containing block data.
        filepath: The path to the file containing saved HMM probabilities.
        block_idx: The index of the block to overlay the HMM probabilities on.

    Returns:
        The axes with the plotted HMM probabilities.
    """

    # Load HMM probabilities
    with open(filepath, "rb") as f:
        saved = pickle.load(f)

    subject, kappa, block_ids = saved["subject"], saved["kappa"], saved["block_ids"]
    pos, gaze = saved["pos"], saved["gaze"]
    push, move, look = saved["push"], saved["move"], saved["look"]
    p_policy = saved["p_policy"]

    # Overlay probabilities on top of a specific block
    session_id, block = block_ids[block_idx]
    hmm_probs = p_policy[block_idx]
    ax = plot_pushes(
        df, conds=dict(subject=subject, session=int(session_id), block=block + 1)
    )
    ax2 = ax.twinx()

    # Set up custom plotting
    cycle = cycler(color=["darkgreen", "lime"], linestyle=["-", "-"])
    ax2.set_prop_cycle(cycle)
    p = ax2.plot(hmm_probs, label=["HMM policy 1", "HMM policy 2"])
    ax2.set_ylabel("HMM Policy Probability")

    # Get existing legend handles and labels
    legend = ax.get_legend()
    existing_handles = legend.legend_handles
    existing_labels = [text.get_text() for text in legend.get_texts()]

    # Combine existing handles with custom lines
    existing_handles.extend(p)
    existing_labels.extend(["HMM policy 1", "HMM policy 2"])
    legend.remove()
    ax2.legend(existing_handles, existing_labels)
    return ax2


def plot_experiment_parameters(
    df: pd.DataFrame,
    conds: dict,
    title: str = "Experiment parameters by session",
    label_rotation: float = 35,
    ax: plt.Axes = None,
    **kwargs,
) -> plt.Axes:
    """
    Plots the distribution of experiment parameters (kappa, stimulus type, shape) across different sessions.
    Displays the number of blocks associated with each parameter and session.

    Args:
        df: DataFrame containing experiment session data with hierarchical index ('session', 'stimulus type', 'shape', 'kappa').
        conds: Dictionary of conditions used to filter the DataFrame before plotting.
        title: Title of the plot.
        label_rotation: degrees to rotate xtick labels.
        ax: Axes to plot on. If None, a new figure and axes are created using plt.subplots. Specify keyword arguments in `fig_kwargs`.
        kwargs: Additional keyword arguments.
            - 'fig_kwargs': Dictionary for customizing the figure properties when creating a new figure (passed to `plt.subplots`).
            - 'x_ticks': Ticks for the x-axis (optional).
            - 'y_ticks': Custom y-axis ticks (optional).
            - 'fontsize': Font size for axis labels and annotations (default is 10).
            - 'label_color': Color for parameter value labels (default is 'black').

    Returns:
        ax: The Axes object with the plot.
    """
    # Create axes if none provided
    if ax is None:
        fig_kwargs = kwargs.pop("fig_kwargs", {})
        _, ax = plt.subplots(**fig_kwargs)

    # Get all unique experiment parameters
    kappas = df.index.unique("kappa").sort_values()
    stim_types = df.index.unique("stimulus type").sort_values()
    shapes = df.index.unique("shape").sort_values()
    n_params = len(kappas) + len(stim_types) + len(shapes)

    # Filter df according to conditions
    df = filter_df(df, conds)
    sessions = df.index.unique("session").sort_values()
    y_labels = (
        [str(s) for s in shapes]
        + [str(s) for s in stim_types]
        + [str(k) for k in kappas]
    )
    a, b, c = 2, 2, 1  # Constants to control spacing
    v_offset = 0.25  # Vertical offset for annotation alignment
    h_offset = 0.05  # Horizontal offset for annotation alignment
    shape_ticks = [0, 1 * a]
    stim_type_ticks = [2 * b, 3 * b]
    kappa_ticks = [i * c + max(stim_type_ticks) + 1 for i in range(1, len(kappas) + 1)]

    # Generate n_param + 1 y-ticks
    ax.scatter(np.ones(n_params + 1), np.arange(n_params + 1), alpha=0)
    for i, sess in enumerate(sessions):
        # Count parameter values for the current session
        kappa_counts = (
            df.xs(sess, level="session")
            .reset_index()
            .groupby("kappa")["block"]
            .nunique()
        )
        stim_types_counts = (
            df.xs(sess, level="session")
            .reset_index()
            .groupby("stimulus type")["block"]
            .nunique()
        )
        shapes_counts = (
            df.xs(sess, level="session")
            .reset_index()
            .groupby("shape")["block"]
            .nunique()
        )

        # Determine y-coordinates for parameter value annotations
        y_kappas = np.searchsorted(kappas, kappa_counts.index.values)
        y_stim_types = np.searchsorted(
            stim_types, stim_types_counts.index.values
        ) + len(shapes)
        y_shapes = np.searchsorted(shapes, shapes_counts.index.values)

        # Annotate the count of blocks associated with each parameter value
        [
            ax.annotate(
                shapes_counts.values[j],
                (i - h_offset, y * a - v_offset),
                c="c",
                fontsize=10,
            )
            for j, y in enumerate(y_shapes)
        ]
        [
            ax.annotate(
                stim_types_counts.values[j],
                (i - h_offset, y * b - v_offset),
                c="m",
                fontsize=10,
            )
            for j, y in enumerate(y_stim_types)
        ]
        [
            ax.annotate(
                kappa_counts.values[j],
                (i - h_offset, (y + 1) + max(stim_type_ticks) + 1 - v_offset),
                c="g",
                fontsize=10,
            )
            for j, y in enumerate(y_kappas)
        ]

    ax.set_xticks(range(len(sessions)), sessions)
    ax.tick_params(axis="x", labelrotation=label_rotation)
    ax.set_yticks(shape_ticks + stim_type_ticks + kappa_ticks, y_labels, fontsize=10)
    ax.set_ylabel(
        "kappa\nstim type\nshape",
        rotation="horizontal",
        labelpad=55,
        multialignment="left",
        va="center",
        linespacing=7,
        fontsize=15,
    )
    ax.set_ylim(-1, max(kappa_ticks) + 1)
    ax.set_title(title)
    return ax


def plot_push_intervals_by_sessions(
    df: pd.DataFrame, label_rotation: float = 35, **kwargs
):
    """
    Plot push intervals across sessions for monkey subjects.
    This function visualizes the distribution of push intervals across different sessions for specified monkey subjects, using swarm plots to show push intervals and adding weekday labels for context.

    Args:
        df: A DataFrame containing session data.
        label_rotation: The angle to rotate x-tick labels by.
        **kwargs: Additional keyword arguments passed to the plotting function.
          - fig_kwargs: Dictionary to specify figure properties when creating a new figure (passed to `plt.subplots`).

    Returns:
        None
    """

    # Examine monkeys separately from humans
    monkey_subjects = ["dylan", "marco", "viktor"]
    conds = {"subject": monkey_subjects}
    df_monkey = filter_df(df, conds).copy()

    # Reformat time data
    df_monkey["session id"] = pd.to_datetime(
        df_monkey.index.get_level_values("session"), format="%Y%m%d"
    )
    df_ref = df_monkey.groupby("subject")["session id"].min()
    df_monkey = df_monkey.join(df_ref, how="left", rsuffix="_ref", on=["subject"])
    df_monkey["day"] = (df_monkey["session id"] - df_monkey["session id_ref"]).dt.days
    fig_kwargs = kwargs_handler(
        kwargs, "fig_kwargs", dict(figsize=(25, 10), height_ratios=[1, 2], sharex=True)
    )

    def _plot(i, subj, **kwargs):
        # Plot experiment overview and push interval distribution
        fig, axes = plt.subplots(2, 1, **fig_kwargs)
        plot_experiment_parameters(df_monkey, conds={"subject": subj}, ax=axes[0])
        bp(sns.swarmplot)(
            df_monkey,
            x="day",
            y="consecutive push intervals",
            conds={"subject": subj},
            hue="box",
            palette=PALETTE,
            title_override="Push intervals across sessions",
            size=1,
            log_scale=True,
            legend=False,
            ax=axes[1],
            **kwargs,
        )  # legend_kwargs={'markerscale': 5}

        # Add weekday labels
        labels = axes[1].get_xticklabels()
        days = (
            df_monkey.xs(subj, level="subject")
            .reset_index()
            .groupby(["session"], as_index=False)["week day"]
            .max()["week day"]
        )
        for j, l in enumerate(labels):
            tmp = l.get_text()
            labels[j] = tmp + "\n" + days[j]
        axes[1].set_xticklabels(labels)
        axes[1].tick_params(axis="x", labelrotation=label_rotation)
        fig.suptitle(f"Push intervals for {subj}")
        fig.tight_layout()

    return across_conditions_plotter(monkey_subjects, _plot, **kwargs)


@multiplot
def plot_push_intervals(
    df: pd.DataFrame,
    stim_reliabilities: dict = KAPPA_LEVELS,
    palette: dict = PALETTE,
    palette_dark: dict = PALETTE_DARK,
    **kwargs,
):
    """
    This function visualizes the distribution of push intervals across different stimulus reliabilities for each subject.

    Args:
        df: A DataFrame containing push interval data.
        stim_reliabilities: A dictionary mapping subjects to their stimulus reliability levels.
        palette: A dictionary mapping box schedules to colors.
        palette_dark: A dictionary mapping box schedules to darkened colors.
        **kwargs: Additional keyword arguments passed to the plotting function.
          - fig_kwargs: Dictionary to specify figure properties when creating a new figure (passed to `plt.subplots`).
          - swarm_kwargs: Dictionary to specify properties for the swarm plot.

    Returns:
        None
    """

    fig_kwargs = kwargs_handler(kwargs, "fig_kwargs")

    def _plot(i, subj, **kwargs):
        fig, ax = fig_init(**fig_kwargs)
        df_subj = filter_df(df, {"subject": subj})
        swarm_kwargs = kwargs_handler(
            kwargs, "swarm_kwargs", dict(size=0.5, log_scale=True, dodge=True)
        )
        bp(sns.swarmplot)(
            df_subj,
            x="stimulus reliability",
            order=list(stim_reliabilities[subj].keys()),
            y="push intervals",
            hue="box",
            palette=palette_dark,
            legend=False,
            ax=ax,
            **swarm_kwargs,
        )
        bp(enhanced_violinplot)(
            df_subj,
            x="stimulus reliability",
            order=list(stim_reliabilities[subj].keys()),
            y="push intervals",
            hue="box",
            palette=palette,
            y_unit="s",
            cut=0,
            inner=None,
            log_scale=True,
            common_norm=True,
            ax=ax,
            **kwargs,
        )
        fig.suptitle(f"Push intervals for {subj}")
        fig.tight_layout()

    return across_conditions_plotter(df.index.unique("subject"), _plot, **kwargs)


def plot_stay_switch_pushes(
    df: pd.DataFrame,
    palette: dict = PALETTE,
    palette_dark: dict = PALETTE_DARK,
    null_model: bool = False,
    **kwargs,
):
    """
    Plot the stay and switch push intervals.

    Args:
        df: DataFrame.
        palette: Dictionary mapping box schedules to colors.
        palette_dark: Dictionary mapping box schedules to darkened colors.
        null_model: If True, perform Kolmogorov-Smirnov Test to see if push intervals can be well described by an exponential distribution
        **kwargs: Additional keyword arguments.
            - 'fig_kwargs': Dictionary to specify figure properties when creating a new figure (passed to `plt.subplots`).

    Returns:
        None
    """

    n_boxes = len(palette)
    box_labels = palette.keys()
    fig_kwargs = kwargs_handler(
        kwargs,
        "fig_kwargs",
        dict(nrows=n_boxes, ncols=n_boxes, figsize=(12, 10), sharex=True, sharey=True),
    )

    def _plot(i, subj, **kwargs):
        fig, axes = fig_init(**fig_kwargs)
        df_subj = filter_df(df, {"subject": subj})
        for i, source in enumerate(box_labels):
            for j, dest in enumerate(box_labels):
                ax = axes[i, j]
                subset = df_subj[
                    (df_subj["prev box"] == source) & (df_subj["box"] == dest)
                ]
                color_fill = palette[dest]
                color_fill_dark = palette_dark[dest]
                color_outline = palette[source]
                if not subset.empty:
                    sns.kdeplot(
                        data=subset,
                        x="consecutive push intervals",
                        fill=True,
                        ax=ax,
                        color=color_fill,
                        alpha=0.6,
                    )
                    if source != dest:
                        sns.kdeplot(
                            data=subset,
                            x="consecutive push intervals",
                            fill=True,
                            ax=ax,
                            color=color_fill_dark,
                            alpha=0.6,
                        )
                        sns.kdeplot(
                            data=subset,
                            x="consecutive push intervals",
                            fill=False,
                            ax=ax,
                            color=color_outline,
                            linewidth=2,
                        )
                    if null_model:
                        delta = subset["consecutive push intervals"].min()
                        fit_res = fit(
                            expon, subset["consecutive push intervals"] - delta
                        )
                        print(
                            f"Fitting geometric distribution to ({source} -> {dest}) push intervals with offset {delta}",
                            fit_res.params,
                        )
                        res = kstest(
                            subset["consecutive push intervals"] - delta,
                            expon.cdf,
                            fit_res.params,
                        )
                        print(
                            f"KS-test of ({source} -> {dest}) push intervals",
                            res.pvalue,
                        )
                        x = np.sort(subset["consecutive push intervals"].unique())
                        ax.plot(
                            x,
                            expon.pdf(x - delta, fit_res.params[0]),
                            color="black",
                            linestyle="-",
                        )

                ax.set_xlabel("")
                if i == 0:
                    ax.set_title(dest, fontsize=15)
                else:
                    ax.set_title("")

                if j == 0:
                    ax.set_ylabel(source, fontsize=15)
                    if i == n_boxes - 1:
                        ax.set_xlabel("push interval (s)")
                else:
                    ax.set_ylabel("")
        fig.suptitle(f"Stay and switch times for {subj}", y=1)
        fig.text(0.5, 0.95, "TO", ha="center")
        fig.text(0.0, 0.5, "FROM", va="center", rotation="vertical")
        fig.tight_layout()

    return across_conditions_plotter(df.index.unique("subject"), _plot, **kwargs)


@multiplot
def plot_runlengths(
    df: pd.DataFrame,
    palette: dict = PALETTE,
    stim_reliabilities: dict = KAPPA_LEVELS,
    null_model: bool = False,
    disp_js: bool = False,
    **kwargs,
):
    """
    Plot the runlengths of consecutive pushes and their distribution across different boxes.
    This function calculates the runlengths of consecutive pushes and visualizes their distribution across different boxes, optionally comparing them to a null model based on visitation frequencies.

    Args:
        df: A DataFrame containing push data.
        palette: A dictionary mapping box schedules to colors.
        stim_reliabilities: A dictionary mapping subjects to their stimulus reliability levels.
        null_model: If True, overlay random dice probabilities from a geometric distribution.
        disp_js: If True, display Jensen-Shannon distance between empirical and null distributions.
        **kwargs: Additional keyword arguments.
          - fig_kwargs: Dictionary to specify figure properties when creating a new figure (passed to `plt.subplots`).

    Returns:
        None
    """

    @legend_handler
    def _plot(i, subj, **kwargs):
        # Identify consecutive pushes and when they switch
        df_subj = filter_df(df, {"subject": subj})
        x = df_subj.index.get_level_values("push #")
        consecutive_mask = x[1:] - x[:-1] == 1
        change_mask = (df_subj["stay/switch"] == "switch") & np.insert(
            consecutive_mask, 0, True
        )
        push_nums = (
            get_blocks(df_subj)["push times"].rank().astype(int)
        )  # Calculate from scratch in case pushes got dropped
        change_mask[push_nums == 1] = True

        # Count the runlengths at different boxes
        group_labels = change_mask.cumsum()
        labeled_lengths = pd.DataFrame(
            {
                "group": group_labels,
                "box": df_subj["box"],
                "next box": get_blocks(df_subj["box"]).shift(-1).fillna("missing"),
            }
        ).set_index(df_subj.index)
        labeled_lengths_all = (
            labeled_lengths.groupby(["stimulus reliability", "box", "group"])
            .size()
            .to_frame()
            .rename(columns={0: "length"})
        )
        labeled_lengths_all = labeled_lengths_all[
            (labeled_lengths_all["length"] > 1) & (labeled_lengths_all["length"] <= 10)
        ]

        # Calculate the distribution of runlengths under a dice that is rolled by visitation frequencies
        visit_freqs = (
            df_subj.groupby(["stimulus reliability"])["box"]
            .value_counts(normalize=True)
            .to_frame()
        )
        kappas = stim_reliabilities[subj].keys()
        fig_kwargs = kwargs_handler(
            kwargs, "fig_kwargs", {"ncols": len(kappas), "sharey": True, "sharex": True}
        )
        fig, ax = fig_init(**fig_kwargs)
        for i, kappa in enumerate(kappas):
            try:
                bp(sns.histplot)(
                    labeled_lengths_all.reset_index(),
                    x="length",
                    conds={"stimulus reliability": kappa},
                    hue="box",
                    palette=PALETTE,
                    discrete=True,
                    common_norm=True,
                    multiple="stack",
                    legend=i == len(kappas) - 1,
                    ax=ax[i],
                    **kwargs,
                )
            except:
                continue

            # Overlay random dice probabilities from geometric distribution
            if null_model:
                bars = ax[i].patches
                bar_width = bars[0].get_width()  # Width of one bar
                probs = visit_freqs.loc[kappa]  # Visit probabilities
                boxes = sorted(probs.index.unique("box"))
                handles = ax[i].get_legend().legend_handles
                labels = [t.get_text() for t in ax[i].get_legend().get_texts()]
                for b, box in enumerate(boxes):
                    try:
                        p = probs.loc[box].iloc[0]
                        run_lengths = (
                            labeled_lengths_all.loc[(kappa, box), "length"]
                            .sort_values()
                            .unique()
                        )
                        geom = p**run_lengths * (1 - p)
                        offset = (b - (len(boxes) - 1) / 2) * bar_width
                        x = run_lengths + offset
                        ax[i].plot(x, geom, c=palette[box], label="random dice")
                        if disp_js:
                            bar_heights = get_bar_heights(ax[i], x_centers=run_lengths)
                            # for k, bar in enumerate(bar_heights[box]):
                            #     axes[i, j].text(x[k], bar, f'{jensenshannon(geom, bar):.1f}', ha='center', va='bottom', fontsize = 7)
                            ax[i].set_title(
                                ax[i].get_title()
                                + f"\nJS-distance = {jensenshannon(geom, bar_heights[box])}"
                            )
                            # print(f"Jensen-shannon distance between empirical distribution and null distribution of ({subj}, {kappa}, {box}): {jensenshannon(geom, bar_heights[box])}")
                    except:
                        continue
                ax[i].legend(
                    handles=handles
                    + [
                        Line2D(
                            [0], [0], color="black", linestyle="-", label="random dice"
                        )
                    ],
                    labels=labels + ["random dice"],
                )
        fig.suptitle(f"Runlengths for {subj}")
        fig.tight_layout()
        return ax

    return across_conditions_plotter(df.index.unique("subject"), _plot, **kwargs)


@multiplot
def plot_push_intervals_vs_reward_intervals(
    df: pd.DataFrame,
    conds: dict = None,
    title: str = "Push intervals vs reward intervals",
    title_override: str = None,
    palette: dict = PALETTE,
    stim_reliabilities: dict = KAPPA_LEVELS,
    annotate_reg: bool = False,
    **kwargs,
):
    """
    Plot linear regression of push intervals against reward intervals in a block.

    Args:
        df: A DataFrame of experiment data for a given block.
        palette: A dictionary mapping box schedules to colors.
        stim_reliabilities: A dictionary mapping subjects to their stimulus reliability levels.
        unity: If True, display data in square and color region by reward outcome.
        annotate_reg: If True, annotate the regression slope on the plot.
        **kwargs: Additional keyword arguments for seaborn.
          - fig_kwargs: Dictionary to specify figure properties when creating a new figure (passed to `plt.subplots`).

    Returns:
        None
    """
    df = filter_df(df, conds)
    # Remove first push from each box, since reward time is messed up for first pushes
    # df = df.drop(df[df['push # by box'] == 1].index)
    fig_kwargs = kwargs_handler(kwargs, "fig_kwargs", {"sharey": True, "sharex": True})

    @legend_handler
    def _plot(i, subj, **kwargs):
        kappas = stim_reliabilities[subj].keys()
        fig_kwargs["ncols"] = len(kappas)
        fig, axes = fig_init(**fig_kwargs)
        fit_results = []
        max_x = 0
        df_subj = filter_df(df, {"subject": subj})
        for i, kappa in enumerate(kappas):
            bp(sns.scatterplot)(
                df_subj,
                x="reward intervals",
                y="push intervals",
                conds={"stimulus reliability": kappa},
                hue="box",
                palette=palette,
                ax=axes[i],
                legend=i == len(kappas) - 1,
                **kwargs,
            )
            df_subset = filter_df(df_subj, conds={"stimulus reliability": kappa})
            fit_results.append(
                regplot(
                    df_subset["reward intervals"].to_numpy(),
                    df_subset["push intervals"].to_numpy(),
                    line_kws={"color": "black"},
                    ax=axes[i],
                    **kwargs,
                )
            )
            max_x = max(max_x, axes[i].get_xlim()[1], axes[i].get_ylim()[1])

        # Add some aesthetics
        for i in range(len(kappas)):
            # max_x = max(axes[i].get_xlim()[1], axes[i].get_ylim()[1])
            x = np.arange(max_x)
            axes[i].plot([0, max_x], [0, max_x], linestyle="dashed", color="black")
            axes[i].fill_between(x, x, max_x, color="green", alpha=0.1)
            axes[i].fill_between(x, x, color="red", alpha=0.1)

            if annotate_reg:
                axes[i].text(
                    0.75,
                    0.1,
                    f"slope={fit_results[i].params[1]:.2f}",
                    transform=axes[i].transAxes,
                    fontsize=10,
                )
        fig.suptitle(
            titler(
                title=title + " for " + subj,
                conds=conds,
                title_override=(
                    title_override + " for " + subj if title_override else None
                ),
            )
        )
        fig.tight_layout()
        return axes

    return across_conditions_plotter(df.index.unique("subject"), _plot, **kwargs)


def plot_next_push_surprise(
    df: pd.DataFrame,
    stim_reliabilities: dict = KAPPA_LEVELS,
    palette: dict = PALETTE,
    palette_dark: dict = PALETTE_DARK,
    **kwargs,
):
    """Plot the change in push interval after each push.

    Args:
        df: DataFrame containing experiment data
        stim_reliabilities: Dictionary mapping subjects to their stimulus reliability levels
        palette: Dictionary mapping box schedules to colors
        palette_dark: Dictionary mapping box schedules to darkened colors
        **kwargs: Additional keyword arguments
    """

    df = df.copy()

    # Calculate the change in push interval
    push_deltas = df.groupby(["subject", "session", "block", "box"])
    df["consecutive wait"] = push_deltas["push # by box"].diff().fillna(1)
    df = df.loc[df["consecutive wait"] == 1]
    df["change in next push interval"] = -push_deltas["push intervals"].diff(-1)
    df["rewarded"] = df["reward outcomes"].map({True: "yes", False: "no"})

    # Track whether the subject stayed or switched after each push
    df["stay/switch"] = df["stay/switch"].shift(-1)

    fig_kwargs = kwargs_handler(
        kwargs,
        "fig_kwargs",
        {"nrows": 2, "sharey": True, "sharex": True, "figsize": (20, 10)},
    )

    @legend_handler
    def _plot(i, subj, **kwargs):
        df_subj = filter_df(df, {"subject": subj})
        kappas = stim_reliabilities[subj].keys()
        fig_kwargs["ncols"] = len(kappas)
        fig, axes = fig_init(**fig_kwargs)
        cnt = 0
        for i, ro in enumerate(df_subj["rewarded"].unique()):
            for j, kappa in enumerate(kappas):
                cnt += 1
                bp(sns.scatterplot)(
                    df_subj,
                    x="push intervals",
                    y="change in next push interval",
                    conds={"stimulus reliability": kappa, "rewarded": ro},
                    hue="box",
                    palette=palette if ro == "yes" else palette_dark,
                    style="stay/switch",
                    alpha=0.5,
                    ax=axes[i][j],
                    legend=cnt == len(kappas),
                    **kwargs,
                )
                axes[i][j].hlines(
                    0, 0, axes[i][j].get_xlim()[1], linestyles="dashed", colors="black"
                )
                axes[i][j].set_xlim([0, 40])
                axes[i][j].set_ylim([-40, 40])
        fig.suptitle(
            f"Change in push interval as a function of reward outcome for {subj}"
        )
        fig.tight_layout()
        return axes

    return across_conditions_plotter(df.index.unique("subject"), _plot, **kwargs)


@multiplot
def plot_stay_probabilities(
    df: pd.DataFrame,
    stim_reliabilities: dict = KAPPA_LEVELS,
    bin_width: float = 10,
    **kwargs,
):
    """
    Plot the probability of staying at the same box after a push, based on push intervals.
    This function calculates and visualizes the probability of a subject staying at the same box after a push, using push intervals binned by a specified width.

    Args:
        df: A DataFrame containing push data.
        stim_reliabilities: A dictionary mapping subjects to their stimulus reliability levels.
        bin_width: The width of the bins for push intervals.
        **kwargs: Additional keyword arguments.
          - fig_kwargs: Dictionary to specify figure properties when creating a new figure (passed to `plt.subplots`).

    Returns:
        None
    """

    df = df.copy()

    # Track whether the push was rewarded and whether the subject stayed or switched after each push
    df["rewarded"] = df["reward outcomes"].map({True: "yes", False: "no"})
    df["time"] = bin_data(df["push intervals"], bin_width=bin_width)
    df["P(stay)"] = df["stay/switch"].shift(-1).map({"stay": 1, "switch": 0})

    fig_kwargs = kwargs_handler(kwargs, "fig_kwargs", {"sharey": True, "sharex": True})

    @legend_handler
    def _plot(i, subj, **kwargs):
        df_subj = filter_df(df, {"subject": subj})
        kappas = stim_reliabilities[subj].keys()
        fig_kwargs["ncols"] = len(kappas)
        fig, ax = fig_init(**fig_kwargs)
        for i, kappa in enumerate(kappas):
            bp(sns.lineplot)(
                df_subj,
                conds={"stimulus reliability": kappa},
                x="time",
                y="P(stay)",
                x_unit="s",
                hue="rewarded",
                hue_order=["no", "yes"],
                errorbar="se",
                ax=ax[i],
                **kwargs,
            )
        fig.suptitle(f"P(stay) as a function of push interval for {subj}")
        fig.tight_layout()
        return ax

    return across_conditions_plotter(df.index.unique("subject"), _plot, **kwargs)


@multiplot
def plot_push_rates_across_block(
    df: pd.DataFrame,
    stim_reliabilities: dict = KAPPA_LEVELS,
    palette: dict = PALETTE,
    by_box: bool = False,
    show_traces: bool = False,
    **kwargs,
) -> list[plt.Axes]:
    """
    This function calculates and visualizes the reward rates across different blocks, smoothing the data over a specified window and optionally separating the data by box.

    Args:
        df: A DataFrame containing push data.
        stim_reliabilities: A dictionary mapping subjects to their stimulus reliability levels.
        palette: A dictionary mapping box schedules to colors.
        by_box: If True, separate the reward rates by box.
        show_traces: If True, show each block's trace instead of averaging over blocks.
        **kwargs: Additional keyword arguments.
            - smooth_kwargs: Dictionary to specify window properties for smoothing the reward rate (passed to `moving_average`).
    Returns:
        None
    """
    kwargs = kwargs.copy()
    smooth_kwargs = kwargs_handler(
        kwargs,
        "smooth_kwargs",
        {"rate": True, "bin_func": lambda x: x.size()},
    )
    return plot_quantity_across_block(
        df,
        stim_reliabilities=stim_reliabilities,
        palette=palette,
        by_box=by_box,
        show_traces=show_traces,
        x="push times",
        y="reward outcomes",
        y_name="push rate",
        fig_title="Push rate",
        smooth_kwargs=smooth_kwargs,
        **kwargs,
    )


@multiplot
def plot_reward_rates_across_block(
    df: pd.DataFrame,
    stim_reliabilities: dict = KAPPA_LEVELS,
    palette: dict = PALETTE,
    by_box: bool = False,
    show_traces: bool = False,
    **kwargs,
) -> list[plt.Axes]:
    """
    This function calculates and visualizes the reward rates across different blocks, smoothing the data over a specified window and optionally separating the data by box.

    Args:
        df: A DataFrame containing push data.
        stim_reliabilities: A dictionary mapping subjects to their stimulus reliability levels.
        palette: A dictionary mapping box schedules to colors.
        by_box: If True, separate the reward rates by box.
        show_traces: If True, show each block's trace instead of averaging over blocks.
        **kwargs: Additional keyword arguments.
            - smooth_kwargs: Dictionary to specify window properties for smoothing the reward rate (passed to `moving_average`).
    Returns:
        None
    """
    kwargs = kwargs.copy()
    smooth_kwargs = kwargs_handler(
        kwargs,
        "smooth_kwargs",
        {"rate": True, "bin_func": lambda x: x["reward outcomes"].sum()},
    )
    return plot_quantity_across_block(
        df,
        stim_reliabilities=stim_reliabilities,
        palette=palette,
        by_box=by_box,
        show_traces=show_traces,
        x="push times",
        y="reward outcomes",
        y_name="reward rate",
        fig_title="Reward rate",
        smooth_kwargs=smooth_kwargs,
        **kwargs,
    )


@multiplot
def plot_reward_per_push_across_block(
    df: pd.DataFrame,
    stim_reliabilities: dict = KAPPA_LEVELS,
    palette: dict = PALETTE,
    by_box: bool = False,
    show_traces: bool = False,
    fig_title: str = "Reward-per-push",
    **kwargs,
):
    """
    This function calculates and visualizes the reward rates across different blocks, smoothing the data over a specified window and optionally separating the data by box.

    Args:
        df: A DataFrame containing push data.
        stim_reliabilities: A dictionary mapping subjects to their stimulus reliability levels.
        palette: A dictionary mapping box schedules to colors.
        by_box: If True, separate the reward rates by box.
        show_traces: If True, show each block's trace instead of averaging over blocks.
        **kwargs: Additional keyword arguments.
            - bin_kwargs: Dictionary to specify binning properties for time (passed to `bin_data`).
            - fig_kwargs: Dictionary to specify figure properties when creating a new figure (passed to `plt.subplots`).
            - smooth_kwargs: Dictionary to specify window properties for smoothing the reward rate (passed to `moving_average`).
    Returns:
        None
    """
    kwargs = kwargs.copy()
    smooth_kwargs = kwargs_handler(
        kwargs,
        "smooth_kwargs",
        {"rate": True},
    )

    # Average data over time
    groupers = ["stimulus reliability", "block_id"]
    if by_box:
        groupers.append("box")

    fig_kwargs = kwargs_handler(kwargs, "fig_kwargs", {"sharey": True, "sharex": True})
    kwargs.update({"x": "time", "y": "reward-per-push", "x_unit": "s"})

    if by_box:
        kwargs.update({"hue": "box", "palette": palette})
    else:
        kwargs.update({"color": "black"})

    def _by_kappa(i: int, kappa: str, df2: pd.DataFrame, axes: ArrayLike, **kwargs):
        kwargs.update(
            {
                "conds": {"stimulus reliability": kappa},
                "ax": axes[i],
            }
        )
        plot_block_average_or_traces(df2, show_traces=show_traces, **kwargs)
        return axes[i]

    @legend_handler
    def _plot(i, subj, **kwargs):
        df_subj = filter_df(df, {"subject": subj})
        ma_push = moving_average(
            df_subj,
            x="push times",
            y="reward outcomes",
            y_name="push rate",
            bin_func=lambda x: x.size(),
            groupers=groupers,
            **smooth_kwargs,
        )

        ma_rew = moving_average(
            df_subj,
            x="push times",
            y="reward outcomes",
            y_name="reward rate",
            bin_func=lambda x: x["reward outcomes"].sum(),
            groupers=groupers,
            **smooth_kwargs,
        )

        ma_rew["reward-per-push"] = ma_rew["reward rate"] / ma_push["push rate"]
        kappas = stim_reliabilities[subj].keys()
        fig_kwargs["ncols"] = len(kappas)
        fig, axes = fig_init(**fig_kwargs)
        legend = {
            kappa: {"legend": i == len(kappas) - 1} for i, kappa in enumerate(kappas)
        }
        across_conditions_plotter(
            kappas, _by_kappa, df2=ma_rew, axes=axes, cond_kwargs=legend, **kwargs
        )
        if fig_title:
            fig.suptitle(f"{fig_title} for {subj}")
        fig.tight_layout()
        return axes

    across_conditions_plotter(df.index.unique("subject"), _plot, **kwargs)


@multiplot
def plot_matching_law(
    df: pd.DataFrame,
    stim_reliabilities: list = KAPPA_LEVELS,
    palette: dict = PALETTE,
    time_bin: tuple[float, float] = None,
    **kwargs,
):
    """
    Calculate and visualize the slopes and intercepts of the matching law for each subject over time.

    Args:
        df: A DataFrame containing push data.
        stim_reliabilities: A list of stimulus reliability levels for each subject.
        palette: A dictionary mapping box schedules to colors.
        time_bin: A tuple specifying the start and end times of the time bin to plot. If None, the entire block is plotted.
        **kwargs: Additional keyword arguments.
            - bin_kwargs: Dictionary to specify binning properties for time (passed to `bin_data`).
            - fig_kwargs: Dictionary to specify figure properties when creating a new figure (passed to `plt.subplots`).

    Returns:
        None
    """

    if time_bin:
        df = df[df["push times"].between(time_bin[0], time_bin[1])]

    # Bin pushes by time, group by box, count rewards and pushes
    grouped = get_blocks(df, ["stimulus reliability", "box"])
    rr = grouped["reward outcomes"].sum().to_frame()
    rr["pushes"] = grouped.size()  # .reset_index()[0]

    # Calculate totals across boxes for each block (exclude box dimension)
    total_pushes = get_blocks(rr, ["stimulus reliability"])["pushes"].sum()
    total_rewards = get_blocks(rr, ["stimulus reliability"])["reward outcomes"].sum()

    # Reset index to get box column back
    rr = rr.reset_index()

    # Merge totals back to original dataframe (exclude box from merge keys)
    rr = pd.merge(
        rr,
        total_pushes,
        on=["subject", "session", "block", "stimulus reliability"],
        suffixes=["", "_total"],
    )
    rr = pd.merge(
        rr,
        total_rewards,
        on=["subject", "session", "block", "stimulus reliability"],
        suffixes=["", "_total"],
    )
    rr["relative pushes"] = rr["pushes"] / rr["pushes_total"]
    rr["relative rewards"] = rr["reward outcomes"] / rr["reward outcomes_total"]

    # Drop rows with NaN or infinite values
    rr = rr.replace([np.inf, -np.inf], np.nan).dropna()

    fig_kwargs = kwargs_handler(kwargs, "fig_kwargs")

    @legend_handler
    def _plot(i, subj, **kwargs):
        kappas = stim_reliabilities[subj].keys()
        fig_kwargs["ncols"] = len(kappas)
        fig, ax = fig_init(**fig_kwargs)
        rr_subj = filter_df(rr, {"subject": subj})
        max_pt = max(
            rr_subj["relative rewards"].max(), rr_subj["relative pushes"].max()
        )
        min_pt = min(
            rr_subj["relative rewards"].min(), rr_subj["relative pushes"].min()
        )
        for i, kappa in enumerate(kappas):
            ax[i].set_xlim(min_pt, max_pt)
            ax[i].set_ylim(min_pt, max_pt)
            ax[i].set_aspect("equal")
            bp(sns.scatterplot)(
                rr_subj,
                conds={"stimulus reliability": kappa},
                x="relative rewards",
                y="relative pushes",
                hue="box",
                palette=palette,
                ax=ax[i],
                legend=i == len(kappas) - 1,
                **kwargs,
            )

            # Best fit line
            rr_subj_kappa = filter_df(rr_subj, conds={"stimulus reliability": kappa})
            fit_results = regplot(
                rr_subj_kappa["relative rewards"].to_numpy(),
                rr_subj_kappa["relative pushes"].to_numpy(),
                line_kws={"color": "black"},
                ax=ax[i],
                **kwargs,
            )
            ax[i].text(
                0.75,
                0.1,
                f"slope={fit_results.params[1]:.2f}\nintercept={fit_results.params[0]:.2f}",
                transform=ax[i].transAxes,
                fontsize=10,
            )

            # if i == 0:
            #     ax[i].set_ylabel(r"$\frac{\text{# rewards}}{\text{# pushes}}$")
        fig.suptitle(f"Matching Law for {subj}")
        fig.tight_layout()
        return ax

    across_conditions_plotter(rr["subject"].unique(), _plot, **kwargs)


@multiplot
def plot_matching_law_coefficients(
    df: pd.DataFrame,
    stim_reliabilities: list = KAPPA_LEVELS,
    min_obs: int = 10,
    **kwargs,
):
    """
    Calculate and visualize the slopes and intercepts of the matching law for each subject over time.

    Args:
        df: A DataFrame containing push data.
        stim_reliabilities: A list of stimulus reliability levels for each subject.
        min_obs: The minimum number of observations required in a time bin to include it in the analysis.
        **kwargs: Additional keyword arguments.
            - bin_kwargs: Dictionary to specify binning properties for time (passed to `bin_data`).
            - fig_kwargs: Dictionary to specify figure properties when creating a new figure (passed to `plt.subplots`).

    Returns:
        None
    """

    # Bin pushes by time, group by box, count rewards and pushes
    grouped = get_blocks(df, ["stimulus reliability", "box"])
    rr = grouped["reward outcomes"].sum().to_frame()  # .reset_index()
    rr["pushes"] = grouped.size()  # .reset_index()[0]

    # Convert to relative rates in each time bin
    total_pushes = get_blocks(rr, ["stimulus reliability"])["pushes"].sum()
    total_rewards = get_blocks(rr, ["stimulus reliability"])["reward outcomes"].sum()
    rr = pd.merge(
        rr,
        total_pushes,
        on=["subject", "session", "block", "stimulus reliability"],
        suffixes=["", "_total"],
    )
    rr = pd.merge(
        rr,
        total_rewards,
        on=["subject", "session", "block", "stimulus reliability"],
        suffixes=["", "_total"],
    )
    rr["relative_pushes"] = rr["pushes"] / rr["pushes_total"]
    rr["relative_rewards"] = rr["reward outcomes"] / rr["reward outcomes_total"]

    # Drop rows with NaN or infinite values
    rr = rr.replace([np.inf, -np.inf], np.nan).dropna()

    # For each block, fit matching law
    @process_block_safely
    def _inner(df: pd.DataFrame, index: tuple):
        df_block = df.loc[index]
        slopes, intercepts, observed_time_bins = [], [], []
        if len(df_block) < 2:
            return None
        X = sm.add_constant(df_block["relative_rewards"])
        y = df_block["relative_pushes"]
        model = sm.OLS(y, X).fit()
        try:
            slopes.append(model.params[1])
            intercepts.append(model.params[0])
        except:
            return None

        return pd.DataFrame({"slope": slopes, "intercept": intercepts})

    result = process_blocks(rr, _inner)

    # Merge all matching law fits into one DataFrame
    merged_df_list = []
    for (subject, session, block), _df in result[0].items():
        # Add columns for subject, session, and block
        _df["subject"] = subject
        _df["session"] = session
        _df["block"] = block
        _df["stimulus reliability"] = df.loc[(subject, session, block)].index.unique(
            "stimulus reliability"
        )[0]

        # Append the DataFrame to the list
        merged_df_list.append(_df)

    # Concatenate all DataFrames in the list
    merged_df = pd.concat(merged_df_list, ignore_index=True)
    fig_kwargs = kwargs_handler(
        kwargs,
        "fig_kwargs",
        {"sharey": True, "sharex": True, "nrows": 2, "ncols": 1, "figsize": (5, 5)},
    )

    @legend_handler
    def _plot(i, subj, **kwargs):
        df_subj = filter_df(merged_df, {"subject": subj})
        fig, ax = fig_init(**fig_kwargs)
        if min_obs:
            df_subj = df_subj.groupby(
                ["stimulus reliability"], observed=True, as_index=False
            ).filter(lambda g: len(g) >= min_obs)
        sns.barplot(
            df_subj,
            x="stimulus reliability",
            order=list(stim_reliabilities[subj].keys()),
            y="slope",
            hue="stimulus reliability",
            hue_order=list(stim_reliabilities[subj].keys()),
            ax=ax[0],
            **kwargs,
        )
        sns.barplot(
            df_subj,
            x="stimulus reliability",
            order=list(stim_reliabilities[subj].keys()),
            y="intercept",
            hue="stimulus reliability",
            hue_order=list(stim_reliabilities[subj].keys()),
            ax=ax[1],
            **kwargs,
        )

        fig.suptitle(f"Matching law for {subj}")
        fig.tight_layout()
        return ax

    across_conditions_plotter(merged_df["subject"].unique(), _plot, **kwargs)


@multiplot
def plot_matching_law_coefficients_across_block(
    df: pd.DataFrame,
    stim_reliabilities: list = KAPPA_LEVELS,
    min_obs: int = 10,
    **kwargs,
):
    """
    Calculate and visualize the slopes and intercepts of the matching law for each subject over time.

    Args:
        df: A DataFrame containing push data.
        stim_reliabilities: A list of stimulus reliability levels for each subject.
        min_obs: The minimum number of observations required in a time bin to include it in the analysis.
        **kwargs: Additional keyword arguments.
            - bin_kwargs: Dictionary to specify binning properties for time (passed to `bin_data`).
            - fig_kwargs: Dictionary to specify figure properties when creating a new figure (passed to `plt.subplots`).

    Returns:
        None
    """

    # Bin pushes by time, group by box, count rewards and pushes
    x_bins = "time"
    df = df.copy()
    bin_kwargs = kwargs_handler(kwargs, "bin_kwargs", dict(bin_width=30))
    df[x_bins] = bin_data(df, "push times", **bin_kwargs)
    grouped = get_blocks(df, ["stimulus reliability", "time", "box rank"])
    rr = grouped["reward outcomes"].sum().to_frame()
    rr["pushes"] = grouped.size()
    rr["theoretical_rewards"] = grouped["schedule"].apply(lambda x: 1 / x.unique()[0])

    # Convert to relative rates in each time bin
    grouped = get_blocks(rr, ["stimulus reliability", "time"])
    total_pushes = grouped["pushes"].sum()
    total_rewards = grouped["reward outcomes"].sum()
    total_theoretical_rewards = grouped["theoretical_rewards"].sum()
    rr = pd.merge(
        rr,
        total_pushes,
        on=["subject", "session", "block", "stimulus reliability", "time"],
        suffixes=["", "_total"],
    )
    rr = pd.merge(
        rr,
        total_rewards,
        on=["subject", "session", "block", "stimulus reliability", "time"],
        suffixes=["", "_total"],
    )
    rr = pd.merge(
        rr,
        total_theoretical_rewards,
        on=["subject", "session", "block", "stimulus reliability", "time"],
        suffixes=["", "_total"],
    )
    rr["relative_pushes"] = rr["pushes"] / rr["pushes_total"]
    rr["relative_rewards"] = rr["reward outcomes"] / rr["reward outcomes_total"]
    rr["relative_theoretical_rewards"] = (
        rr["theoretical_rewards"] / rr["theoretical_rewards_total"]
    )

    # Drop rows with NaN or infinite values
    rr = rr.replace([np.inf, -np.inf], np.nan).dropna()

    # For each block, fit matching law
    @process_block_safely
    def _inner(df: pd.DataFrame, index: tuple):
        df_block = df.loc[index]
        time_bins = sorted(df_block.index.unique("time"))
        slopes, intercepts, observed_time_bins = [], [], []
        for time_bin in time_bins:
            df_time = filter_df(df_block, {"time": time_bin})
            if len(df_time) < 2:
                continue
            X = sm.add_constant(df_time["relative_rewards"])
            # X = sm.add_constant(df_time["relative_theoretical_rewards"])
            y = df_time["relative_pushes"]
            model = sm.OLS(y, X).fit()
            try:
                slopes.append(model.params[1])
                intercepts.append(model.params[0])
                observed_time_bins.append(time_bin)
            except:
                continue
        return pd.DataFrame(
            {"time": observed_time_bins, "slope": slopes, "intercept": intercepts}
        )

    result = process_blocks(rr, _inner)

    # Merge all matching law fits into one DataFrame
    merged_df_list = []
    for (subject, session, block), _df in result[0].items():
        # Add columns for subject, session, and block
        _df["subject"] = subject
        _df["session"] = session
        _df["block"] = block
        _df["stimulus reliability"] = df.loc[(subject, session, block)].index.unique(
            "stimulus reliability"
        )[0]

        # Append the DataFrame to the list
        merged_df_list.append(_df)

    # Concatenate all DataFrames in the list
    merged_df = pd.concat(merged_df_list, ignore_index=True)
    fig_kwargs = kwargs_handler(
        kwargs, "fig_kwargs", {"sharey": True, "sharex": True, "nrows": 2, "ncols": 1}
    )

    @legend_handler
    def _plot(i, subj, **kwargs):
        df_subj = filter_df(merged_df, {"subject": subj})
        fig, ax = fig_init(**fig_kwargs)
        if min_obs:
            df_subj = df_subj.groupby(
                ["time", "stimulus reliability"], observed=True, as_index=False
            ).filter(lambda g: len(g) >= min_obs)
        sns.lineplot(
            df_subj,
            x="time",
            y="slope",
            hue="stimulus reliability",
            hue_order=list(stim_reliabilities[subj].keys()),
            ax=ax[0],
            **kwargs,
        )
        sns.lineplot(
            df_subj,
            x="time",
            y="intercept",
            hue="stimulus reliability",
            hue_order=list(stim_reliabilities[subj].keys()),
            ax=ax[1],
            **kwargs,
        )

        fig.suptitle(f"Matching law for {subj}")
        fig.tight_layout()
        return ax

    across_conditions_plotter(merged_df["subject"].unique(), _plot, **kwargs)


def plot_frequencies_over_experiment(
    df: pd.DataFrame,
    category: str,
    conds: dict = None,
    title: str = None,
    title_override: str = None,
    palette: list = BOX_COLORS,
    label_rotation: float = 35,
    ax: plt.Axes = None,
    **kwargs,
):

    # Get frequencies for specified category
    visit_freqs = filter_df(
        get_blocks(df)[category].value_counts(normalize=True).to_frame(), conds=conds
    ).reset_index()

    # Define horizontal offset for each session
    session_order = sorted(visit_freqs["session"].unique())
    session_offsets = {session: i for i, session in enumerate(session_order)}
    visit_freqs["y_offset"] = visit_freqs["session"].map(session_offsets)
    visit_freqs["y"] = 1 - visit_freqs["proportion"] + visit_freqs["y_offset"]

    # Create ax if none provided
    fig_kwargs = kwargs_handler(kwargs, "fig_kwargs")
    fig, ax = fig_init(ax, **fig_kwargs)
    legend = True
    category_order = sorted(df[category].unique())
    for session in session_order:
        sns.lineplot(
            data=visit_freqs[visit_freqs["session"] == session].sort_values(by="block"),
            x="block",
            y="y",
            hue=category,
            hue_order=category_order,
            palette=list(palette),
            ax=ax,
            legend=legend,
            **kwargs,
        )
        legend = False

    # Tidy up axes
    ax.set_yticks(
        [offset + 0.5 for offset in session_offsets.values()],
        [str(s) for s in session_order],
    )
    ax.tick_params(axis="y", labelrotation=label_rotation)
    ax.set_xlabel("blocks")
    ax.set_ylabel("session")
    ax.set_title(titler(title=title, conds=conds, title_override=title_override))

    # Draw the scale bar
    scale_length = 1
    x_start = visit_freqs.loc[
        visit_freqs["session"] == session_order[0], "block"
    ].max()  # x-position of the scale
    y_start = 0  # y-position of the scale bar
    x_offset = 0.05
    ax.plot(
        [x_start + x_offset, x_start + x_offset],
        [y_start, y_start + scale_length],
        color="black",
        lw=2,
    )
    ax.text(x_start, y_start, "1", ha="center", va="bottom")
    ax.text(x_start, y_start + scale_length + 0.5, "0", ha="center", va="bottom")
    ax.invert_yaxis()
    ax.legend(loc="upper right", title=category)
    fig.tight_layout()
    return ax


def plot_fisher(
    df: pd.DataFrame,
    conds: dict = None,
    title: str = "Fisher information",
    title_override: str = None,
    box_colors: list = BOX_COLORS,
    box_labels: list = None,
    legend: bool = True,
    specific: bool = False,
    ax: plt.Axes = None,
    **kwargs,
) -> plt.Axes:
    """
    Plots the push intervals for each box, with different colors and line styles based on the stay/switch behavior,
    and displays reward outcomes with markers. Optionally adds a custom legend.

    Args:
        df: Dataframe
        conds: Dictionary to filter df. Should specify a block.
        x: x-axis
        title: Title of figure.
        title_override: Overrides title. See `titler` for more details.
        box_colors: List of colors for each box
        box_labels: Labels of each box
        legend: If True, a custom legend is added to the plot.
        specific: if True, then plot the specific information
        ax: Axes to plot on. If none, a new figure and axes are created using plt.subplots. Specify keyword arguments in `fig_kwargs`.
        **kwargs: Additional keyword arguments.
            - 'fig_kwargs': Dictionary to specify figure properties when creating a new figure (passed to `plt.subplots`).
            - 'legend_kwargs': Dictionary of keyword arguments for customizing the legend (passed to `ax.legend`).

    Returns:
        the axes
    """

    # Create ax if none provided
    fig, ax = fig_init(ax, **kwargs.pop("fig_kwargs", {}))

    # Get data from block
    x = "push times"
    df_block = filter_df(df, conds)
    x_vals = df_block[x].values
    y = "specific fisher" if specific else "fisher"
    y_vals = df_block[y].values
    colors = np.array(["black"] * len(y_vals))

    # Create segments (x, y) pairs for LineCollection
    segments = [[(0, 0), (x_vals[0], y_vals[0])]] + [
        [(x_vals[i], y_vals[i]), (x_vals[i + 1], y_vals[i + 1])]
        for i in range(len(x_vals) - 1)
    ]

    # Create the LineCollection
    lc = LineCollection(segments, colors=colors, linewidth=2)

    # Create ax if none provided
    if ax is None:
        fig_kwargs = kwargs.pop("fig_kwargs", {})
        _, ax = plt.subplots(**fig_kwargs)
    ax.add_collection(lc)
    ax.autoscale()

    # Get block metadata
    schedules = np.sort(df_block["schedule"].unique())
    kappa = df_block.index.unique("kappa")
    stim_type = df_block.index.unique("stimulus type")
    shape = df_block.index.unique("shape")

    if conds is None:
        conds = {}
    else:
        conds = deepcopy(conds)
    conds["kappa"] = kappa[0]
    conds["stim type"] = stim_type[0]
    conds["shape"] = shape[0]
    box_labels = box_labels if box_labels else schedules

    ax.set_title(titler(title=title, conds=conds, title_override=title_override))
    ax.set_ylabel("Fisher information")
    ax.set_xlabel(unitler(x, "s"))

    # Add reward outcomes with shaded (rewarded) and empty (not rewarded) markers
    colors = np.array([box_colors[i] for i in df_block["box rank"].values])
    mask = df_block["reward outcomes"] == True
    ax.scatter(x_vals[mask], y_vals[mask], c=colors[mask], marker="^", s=80, zorder=2)
    ax.scatter(
        x_vals[~mask],
        y_vals[~mask],
        edgecolors=colors[~mask],
        marker="v",
        s=80,
        zorder=2,
        facecolors="none",
    )

    # Create legend manually with proxy artists
    if legend:
        legend_kwargs = kwargs.pop("legend_kwargs", {"loc": "upper right"})
        legend_elements = [
            Line2D([0], [0], color=box_colors[j], linestyle="-", label=box_labels[i])
            for i, j in enumerate(sorted(df_block["box rank"].unique()))
        ] + [
            Line2D([0], [0], color="black", linestyle="", marker="^", label="rewarded"),
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
        ax.legend(handles=legend_elements, **legend_kwargs)
    return ax


def plot_continuous3d_dict(
    continuous_data: dict,
    list_blocks: list,
    x: str,
    title: str = None,
    color_key: str = "time",
    ax: plt.Axes = None,
    **kwargs,
) -> tuple:
    """
    Plots 3D scatter data for specified blocks from a dictionary of continuous data. The plot is color-coded
    based on a specified key (e.g., 'time').

    Args:
        continuous_data: Dictionary where keys are block names and values are DataFrames containing continuous data.
        list_blocks: List of block names to include in the plot.
        x: The key for the column in each block's DataFrame to plot on the x, y, and z axes.
        title: Optional title for the plot.
        color_key: The key in each block's DataFrame that will be used for coloring the scatter points (default is 'time').
        ax: Optional, existing matplotlib Axes object. If None, a new one will be created.
        kwargs: Additional keyword arguments.
            - 'fig_kwargs': Dictionary of parameters for creating the figure (passed to `plt.figure`).
            - 'plt_kwargs': Dictionary of parameters for `scatter` (e.g., marker size, color map, etc.).
            - 'cbar_kwargs': Dictionary of parameters for customizing the color bar (e.g., label, ticks).
            - 'view_kwargs': Dictionary for setting the elevation and azimuth of the 3D view (passed to `view_init`).

    Returns:
        tuple: The function returns a tuple (ax, p), where `ax` is the matplotlib Axes object and `p` is the
        collection of points from the scatter plot.
    """
    # Accumulate points across specified blocks
    points = []
    c = []
    for block in list_blocks:
        if block in continuous_data:
            points.append(continuous_data[block][x])
            c.append(continuous_data[block][color_key])

    points = np.vstack(points)
    c = np.hstack(c)

    # Plot 3D data
    if ax is None:
        fig_kwargs = kwargs.pop("fig_kwargs", {})
        fig = plt.figure(**fig_kwargs)
        ax = fig.add_subplot(projection="3d")

    plt_kwargs = kwargs.pop("plt_kwargs", {})
    p = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=c, **plt_kwargs)
    if title:
        ax.set_title(title)

    # Configure plot
    cbar_kwargs = {"label": color_key} | kwargs.pop("cbar_kwargs", {})
    plt.colorbar(p, ax=ax, **cbar_kwargs)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.set_ticks([])
    view_kwargs = {"elev": 0, "azim": -45} | kwargs.pop("view_kwargs", {})
    ax.view_init(**view_kwargs)

    return ax, p


def plot_continuous3d_df(
    df: pd.DataFrame,
    continuous_data: dict,
    x: str,
    title: str = None,
    color_key: str = "time since last push (s)",
    color_filter=None,
    ax: plt.Axes = None,
    **kwargs,
) -> tuple:
    """
    Plots 3D scatter data for specified push intervals from a DataFrame containing block-level data.
    The plot is color-coded based on a specified key (e.g., 'time since last push').

    Args:
        df: DataFrame containing block-level data with 'push times' and 'consecutive push intervals'.
        continuous_data: Dictionary where keys are tuples identifying blocks (e.g., (session, block, stim_type)),
                          and values are DataFrames with continuous data (including 'time' and other variables).
        x: The key for the column in each block's DataFrame to plot on the x, y, and z axes.
        title: Optional title for the plot.
        color_key: The key in each block's DataFrame that will be used for coloring the scatter points (default is 'time since last push (s)').
        color_filter: Optional filter function to apply to the color array (e.g., for custom filtering of color data).
        ax: Optional, existing matplotlib Axes object. If None, a new one will be created.
        kwargs: Additional keyword arguments.
            - 'fig_kwargs': Dictionary of parameters for creating the figure (passed to `plt.figure`).
            - 'plt_kwargs': Dictionary of parameters for `scatter` (e.g., marker size, color map, etc.).
            - 'cbar_kwargs': Dictionary of parameters for customizing the color bar (e.g., label, ticks).
            - 'view_kwargs': Dictionary for setting the elevation and azimuth of the 3D view (passed to `view_init`).

    Returns:
        tuple: The function returns a tuple (ax, p), where `ax` is the matplotlib Axes object and `p` is the
        collection of points from the scatter plot.
    """
    points = []
    c = []
    for block in df.iterrows():
        # Go through each push interval and get continuous data in that interval
        end = block[1]["push times"]
        start = end - block[1]["consecutive push intervals"]
        key = (block[0][0], block[0][1], block[0][2])
        if key in continuous_data:
            continuous_data_block = continuous_data[key]
            start_idx, end_idx = np.searchsorted(
                continuous_data_block["time"], [start, end]
            )
            segment = continuous_data_block[x][start_idx:end_idx]
            not_nans = ~np.isnan(segment).any(axis=1)
            segment = segment[not_nans]  # drop rows with nans
            if len(segment) == 0:
                continue

            # Add segment to points
            points.append(segment)

            # Add color
            if color_key in block[1]:
                c.append(block[1][color_key] * np.ones(len(segment)))
            elif color_key in continuous_data_block:
                c.append(continuous_data_block[color_key][start_idx:end_idx][not_nans])
            elif color_key == "time since last push (s)":
                t_vec = continuous_data_block["time"][start_idx:end_idx][not_nans]
                t_vec -= continuous_data_block["time"][start_idx]
                c.append(t_vec)
            else:
                raise Exception("color key not in dataframe nor in data dictionary")

    points = np.vstack(points)
    c = np.hstack(c)

    if color_filter:
        points = points[color_filter(c)]
        c = c[color_filter(c)]

    # Plot 3D data
    if ax is None:
        fig_kwargs = kwargs.pop("fig_kwargs", {})
        fig = plt.figure(**fig_kwargs)
        ax = fig.add_subplot(projection="3d")

    plt_kwargs = kwargs.pop("plt_kwargs", {})
    p = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=c, **plt_kwargs)
    if title:
        ax.set_title(title)

    # Configure color bar
    cbar_kwargs = {"label": color_key} | kwargs.pop("cbar_kwargs", {})
    cbar = plt.colorbar(p, ax=ax, **cbar_kwargs)
    cbar.solids.set_alpha(1.0)

    # Remove axis ticks
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.set_ticks([])

    # Set 3D view parameters
    view_kwargs = {"elev": 0, "azim": -45} | kwargs.pop("view_kwargs", {})
    ax.view_init(**view_kwargs)
    return ax, p


def plot_continuous2d_df(
    df: pd.DataFrame,
    continuous_data: dict,
    x: str,
    dims: tuple = (0, 1),
    title: str = None,
    color_key: str = "time since last push (s)",
    color_filter=None,
    ax: plt.Axes = None,
    **kwargs,
) -> tuple:
    """
    Plots 2D scatter data for specified push intervals from a DataFrame containing block-level data.
    The plot is color-coded based on a specified key (e.g., 'time since last push').

    Args:
        df: DataFrame containing block-level data with 'push times' and 'consecutive push intervals'.
        continuous_data: Dictionary where keys are tuples identifying blocks (e.g., (session, block, stim_type)),
                          and values are DataFrames with continuous data (including 'time' and other variables).
        x: The key for the column in each block's DataFrame to plot on the x, y axes.
        dims: Tuple of integers (i, j) representing the dimensions (columns) to plot as x and y axes. Default is (0, 1).
        title: Optional title for the plot.
        color_key: The key in each block's DataFrame that will be used for coloring the scatter points (default is 'time since last push (s)').
        color_filter: Optional filter function to apply to the color array (e.g., for custom filtering of color data).
        ax: Optional, existing matplotlib Axes object. If None, a new one will be created.
        kwargs: Additional keyword arguments.
            - 'fig_kwargs': Dictionary of parameters for creating the figure (passed to `plt.subplots`).
            - 'plt_kwargs': Dictionary of parameters for `scatter` (e.g., marker size, color map, etc.).
            - 'cbar_kwargs': Dictionary of parameters for customizing the color bar (e.g., label, ticks).

    Returns:
        tuple: The function returns a tuple (ax, p), where `ax` is the matplotlib Axes object and `p` is the
        collection of points from the scatter plot.
    """
    points = []
    c = []
    for block in df.iterrows():
        # Go through each push interval and get continuous data in that interval
        end = block[1]["push times"]
        start = end - block[1]["consecutive push intervals"]
        key = (block[0][0], block[0][1], block[0][2])
        if key in continuous_data:
            continuous_data_block = continuous_data[key]
            start_idx, end_idx = np.searchsorted(
                continuous_data_block["time"], [start, end]
            )
            segment = continuous_data_block[x][start_idx:end_idx]
            not_nans = ~np.isnan(segment).any(axis=1)
            segment = segment[not_nans]
            if len(segment) == 0:
                continue

            segment = segment[:, [dims[0], dims[1]]]  # drop rows with nans

            # Add segment to points
            points.append(segment)

            # Add color
            if color_key in block[1]:
                c.append(block[1][color_key] * np.ones(len(segment)))
            elif color_key in continuous_data_block:
                c.append(continuous_data_block[color_key][start_idx:end_idx][not_nans])
            elif color_key == "time since last push (s)":
                t_vec = continuous_data_block["time"][start_idx:end_idx][not_nans]
                t_vec -= continuous_data_block["time"][start_idx]
                c.append(t_vec)
            else:
                raise Exception("color key not in dataframe nor in data dictionary")

    points = np.vstack(points)
    c = np.hstack(c)

    if color_filter:
        points = points[color_filter(c)]
        c = c[color_filter(c)]

    # Plot 2D data
    if ax is None:
        fig_kwargs = kwargs.pop("fig_kwargs", {})
        fig, ax = plt.subplots(**fig_kwargs)

    plt_kwargs = kwargs.pop("plt_kwargs", {})
    p = ax.scatter(points[:, 0], points[:, 1], c=c, **plt_kwargs)
    if title:
        ax.set_title(title)

    cbar_kwargs = {"label": color_key} | kwargs.pop("cbar_kwargs", {})
    cbar = plt.colorbar(p, ax=ax, **cbar_kwargs)
    cbar.solids.set_alpha(1.0)

    # Remove axis ticks
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_ticks([])

    return ax, p


def inset(
    plot_func,
    inset_func,
    add_inset: bool = True,
    inset_bounds: tuple = (0.6, 0.6, 0.35, 0.35),
    inset_xlim: tuple = (0, 60),
    inset_title: str = "Zoom",
    **inset_kwargs,
):
    """
    Adds an inset to any plotting function.

    Args:
        plot_func: The main plotting function
        inset_func: The function to call for the inset
        add_inset: whether to add inset (default True)
        inset_bounds: [left, bottom, width, height] for inset position
        inset_xlim: x-axis limits for inset
        inset_ylim: y-axis limits for inset
        inset_title: title for inset

    Returns:
        A new function that creates the original plot with an inset
    """

    def plot_with_inset(*args, **kwargs):

        # Call the main plotting function
        result = plot_func(*args, **kwargs)

        # Only add inset if requested
        if not add_inset:
            return result

        # Handle different return types from plotting functions
        axes_to_modify = []

        if hasattr(result, "__iter__") and not isinstance(result, str):
            if hasattr(result, "shape"):  # numpy array of axes
                axes_to_modify = list(result.flat)
            else:  # list or tuple of axes
                axes_to_modify = list(result)
        else:
            # Single axis or other return type
            if hasattr(result, "plot"):  # matplotlib axes
                axes_to_modify = [result]
            else:
                # If we can't determine axes, try to get current axes
                axes_to_modify = [plt.gca()]

        # Add inset to each axis
        for ax in axes_to_modify:
            if hasattr(ax, "inset_axes"):
                # Create inset axes
                inset_ax = ax.inset_axes(inset_params["inset_bounds"])

                # Call the inset plotting function with the inset axes
                inset_result = inset_func(*args, ax=inset_ax, **kwargs)

                # Set inset limits if provided
                if "inset_xlim" in inset_params:
                    inset_ax.set_xlim(inset_params["inset_xlim"])
                if "inset_ylim" in inset_params:
                    inset_ax.set_ylim(inset_params["inset_ylim"])
                if "inset_title" in inset_params:
                    inset_ax.set_title(inset_params["inset_title"])

        return result

    return plot_with_inset


# Example usage:
#
# # Create a version of plot_reward_rates_across_block with first-minute inset
# plot_reward_rates_with_inset = with_first_minute_inset(plot_reward_rates_across_block)
#
# # Or use the more flexible add_inset_to_plot function
# plot_reward_rates_with_custom_inset = add_inset_to_plot(
#     plot_reward_rates_across_block,
#     inset_xlim=(0, 30),  # First 30 seconds
#     inset_title='First 30s',
#     inset_bounds=[0.7, 0.7, 0.25, 0.25]
# )
#
# # Usage:
# plot_reward_rates_with_inset(df, stim_reliabilities, palette)
#
# # To disable inset for a specific call:
# plot_reward_rates_with_inset(df, stim_reliabilities, palette, add_inset=False)
#
# # You can also pass inset parameters directly:
# plot_reward_rates_with_inset(df, stim_reliabilities, palette,
#                             inset_xlim=(0, 45), inset_title='First 45s')
