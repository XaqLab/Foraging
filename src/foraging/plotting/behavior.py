"""
This module contains functions for plotting behavior.
"""

import logging
import pickle
from copy import deepcopy
from typing import Any, Callable, Iterable

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

from foraging import (
    BIN_WIDTH,
    MULTIPLOT_FIGSIZE,
    SEED,
    STEP,
    WINDOW_SIZE,
)
from foraging.models import HashableDict
from foraging.models.experiment import Experiment
from foraging.plotting import (
    BasePlotter,
    Embeddable,
    across_conditions_plotter,
    bp,
    embeddable_to_conds,
    enhanced_violinplot,
    fig_init,
    legend_corrector,
    multiplot,
    titler,
    unitler,
)
from foraging.plotting._base import (
    get_bar_heights,
    get_figure_from_axes,
    palette_corrector,
    plot_average_or_traces,
    regplot,
)
from foraging.utils import kwargs_handler
from foraging.utils.data import (
    bin_data,
    filter_df,
    get_continuous_from_df_to_dict,
)
from foraging.utils.stats import moving_average

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class BehaviorPlotter(BasePlotter):
    def __init__(self, dataset: Experiment, config: dict | Iterable):
        super().__init__(dataset, config)

    @legend_corrector
    def plot_experiment_overview(
        self,
        dataset: Experiment = None,
        conds: dict[str, Any] = None,
        palette: dict[str, Any] = None,
        title: str = "Overview of pushes over entire experiment",
        row: str = "session",
        col: str = "block",
        label_rotation: float = 35,
        annotate_block: list[str] = None,
        ax: plt.Axes = None,
        **kwargs,
    ) -> plt.Axes:
        """
        Plot the pushes over all blocks in the experiment, organized by sessions. This assumes one subject is specified in the `conds` dictionary.

        Args:
            dataset: Experiment dataset.
            conds: Dictionary to filter dataset.
            palette: Dictionary mapping box schedules to colors. Can also be a list of just colors.
            title: Title of figure.
            row: Condition to plot on the y-axis.
            col: Condition to plot on the x-axis.
            label_rotation: Angle to rotate y-tick labels by.
            annotate_block: if True, also display the block parameters above each block.
            ax: Axes to plot on. If None, a new figure and axes are created using plt.subplots. Specify keyword arguments in `fig_kwargs`.
            **kwargs: Keyword arguments passed to seaborn. May also contain nested kwargs.
                - 'fig_kwargs': Dictionary to specify figure properties when creating a new figure (passed to `plt.subplots`).

        Returns:
            The axes.
        """
        dataset, palette = self._init_vars(dataset=dataset, palette=palette)
        df = dataset.filter(conds).df

        # Offset x-coord
        x_offset = df.groupby(dataset.block_identifiers)["duration"].last()
        x_offset.iloc[1:] = x_offset.groupby(["subject", row]).cumsum().iloc[:-1]
        y_start = x_offset.reset_index(level=col).groupby(["subject", row])[col].first()
        for (
            idx,
            x,
        ) in (
            y_start.items()
        ):  # Make sure each row (session) starts from 0 on the x-axis
            x_offset.loc[idx + (x,)] = 0
        df_temp = df.join(x_offset, rsuffix="_offset", on=dataset.block_identifiers)
        df_temp["x"] = df_temp["push times"] + df_temp["duration_offset"]

        # Offset y-coord
        y_order = sorted(df_temp.index.unique(row))
        y_offsets = {y: i for i, y in enumerate(y_order)}
        box_order = sorted(df_temp["box position"].unique())
        box_offsets = {box: box - 1 for box in box_order}

        df_temp["y_offset_1"] = df_temp["box position"].map(box_offsets)
        df_temp["y_offset_2"] = df_temp.index.map(
            lambda x: y_offsets[x[dataset.index.index(row)]]
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
        for y in y_order:
            bp(sns.scatterplot)(
                filter_df(df_temp, {row: y}),
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
            for y in y_order:
                df_row = filter_df(df_temp, {row: y})
                y_text = df_row["y"].max() + y_text_offset
                blocks = df_row.index.get_level_values("block")
                flag = np.insert(blocks[1:] != blocks[:-1], 0, True)
                annotation_labels = []
                for anot in annotate_block:
                    labels = df_row[anot][flag].values
                    annotation_labels.append(labels)
                x_text = df_row["duration_offset"].unique()
                for i in range(len(annotation_labels[0])):
                    label = []
                    for j, anot in enumerate(annotate_block):
                        label.append(rf"{anot}={annotation_labels[j][i]}")
                    ax.text(x_text[i], y_text, ", ".join(label))

        # Demarcate blocks
        for y in y_order:
            df_row = filter_df(df_temp, {row: y})
            x_text = df_row["duration_offset"].unique()[1:]
            ax.vlines(
                x_text,
                y_offset_2_factor * df_row["y_offset_2"].unique()[0] - 0.5,
                y_offset_2_factor * df_row["y_offset_2"].unique()[0] + 2.5,
                linestyles="dotted",
                colors="black",
            )

        # Tidy up axes
        ax.set_yticks(
            [
                y_offset_2_factor * offset + 0.5
                for offset in sorted(df_temp["y_offset_2"].unique())
            ],
            [str(s) for s in y_order],
        )
        ax.tick_params(axis="y", labelrotation=label_rotation)
        ax.set_xlabel(col)
        ax.set_ylabel(row)
        ax.set_title(titler(title=title, conds=conds))
        fig.tight_layout()
        return ax

    def plot_pushes(
        self,
        dataset: Experiment = None,
        conds: dict[str, Any] = None,
        title: str = "Pushes",
        palette: dict[str, Any] = None,
        legend: bool = True,
        ax: plt.Axes = None,
        **kwargs,
    ) -> plt.Axes:
        """
        Plot the pushes in the block by the box they occur at.

        Args:
            dataset: Experiment dataset.
            conds: Dictionary to filter dataset.
            title: Title of figure.
            palette: Dictionary mapping box schedules to colors. Can also be a list of just colors.
            box_labels: Labels on y-axis for each box.
            legend: If True, display legend. Specify keyword arguments in `legend_kwargs`.
            ax: Axes to plot on. If None, a new figure and axes are created using plt.subplots. Specify keyword arguments in `fig_kwargs`.
            **kwargs: Additional keyword arguments passed to `plot_block_events`.

        Returns:
            The axes.
        """

        ax = super().plot_block_events(
            dataset=dataset,
            conds=conds,
            title=title,
            palette=palette,
            legend=legend,
            ax=ax,
            **kwargs,
        )

        # Custom plotting logic
        dataset = self._init_vars(dataset=dataset)
        block = dataset.filter(conds)
        box_labels = dataset.constants.BOX_POSITIONS
        ax.set_xlim([0, block.get("push times").max() + 1])
        box_labels = [box_labels[i] for i in sorted(block.get_unique("box position"))]
        ax.set_yticks(range(len(box_labels)), box_labels, rotation=90, va="center")
        ax.set_ylabel("")
        return ax

    @legend_corrector
    def plot_push_percentiles(
        self, dataset: Experiment = None, conds: dict = None, **kwargs
    ) -> plt.Axes:
        """
        Plot the percentiles of consecutive push intervals for each subject.

        Args:
            dataset: An Experiment dataset.
            conds: A dictionary of conditions to filter the data.
            **kwargs: Additional keyword arguments passed to the plotting function.

        Returns:
            The axes.
        """
        dataset = self._init_vars(dataset=dataset)
        df = dataset.filter(conds).df
        ax = bp(sns.scatterplot)(
            df,
            x="consecutive push intervals",
            y="push percentiles",
            conds=conds,
            title="Percentiles of consecutive push intervals",
            x_unit="s",
            legend=False,
            **kwargs,
        )
        return ax

    @legend_corrector
    def plot_long_push_blocks(
        self,
        top_n: int,
        dataset: Experiment = None,
        palette: dict[str, Any] = None,
        figsize: tuple[float, float] = (20, 2.5),
        **kwargs,
    ) -> plt.Axes:
        """
        Plot the blocks containing the top N longest push intervals for each subject.
        This function sorts the data by the magnitude of push intervals in descending order and plots the blocks containing the top N longest push intervals for each subject.

        Args:
            top_n: The number of top longest push intervals to plot.
            dataset: An Experiment dataset.
            palette: Dictionary mapping box schedules to colors. Can also be a list of just colors.
            figsize: A tuple specifying the size of the figure.
            **kwargs: Additional keyword arguments passed to the plotting function.

        Returns:
            The axes.
        """
        dataset, palette = self._init_vars(dataset=dataset, palette=palette)

        # Sort data by magnitude of push interval in descending order
        dataset = dataset.wrap(
            dataset.df.sort_values(by="consecutive push intervals", ascending=False)
        )

        @embeddable_to_conds
        def _plot(conds: dict = None, **kwargs):
            fig, axes = plt.subplots(1, top_n, figsize=figsize)
            dataset_subject = dataset.filter(conds)

            # Plot each block containing the `top_n` pushes
            subj = ""
            for i, (idx, g) in enumerate(dataset_subject.get_blocks(sort=False)):
                if i >= top_n:
                    break
                conds = dict(zip(dataset.block_identifiers, idx))
                self.plot_pushes(
                    dataset.wrap(g.sort_index()),
                    conds=conds,
                    title="",
                    legend=False,
                    palette=palette,
                    box_labels=dataset.constants.BOX_POSITIONS,
                    ax=axes[i],
                    **kwargs,
                )
                subj = conds.pop("subject", None)
                axes[i].set_title(titler(title="", conds=conds))
            fig.suptitle(subj)
            fig.tight_layout()
            return axes

        return across_conditions_plotter(
            "subject", dataset.get_unique("subject"), _plot, **kwargs
        )

    @legend_corrector
    def plot_recent_rewards_vs_push_percentiles(
        self,
        dataset: Experiment = None,
        conds: dict = None,
        n_samples: int = 5000,
        window: float = 30,
        seed: int = SEED,
        invert_reward: bool = False,
        **kwargs,
    ) -> plt.Axes:
        """
        Plot reward outcomes in a time window preceding each push as a function of push interval.

        Args:
            dataset: An Experiment dataset.
            conds: A dictionary of conditions to filter the data.
            n_samples: The number of samples to draw for analysis.
            window: The time window in seconds to consider for reward calculation.
            seed: The seed for random number generation.
            invert_reward: If True, invert the reward outcomes.
            **kwargs: Additional keyword arguments passed to the plotting function.
            - fig_kwargs: Dictionary to specify figure properties when creating a new figure (passed to `plt.subplots`).

        Returns:
            The axes.
        """
        dataset = self._init_vars(dataset=dataset)
        df = dataset.filter(conds).df
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
                push_num = idx[dataset.index.index("push #")]
                new_idx = df.index.get_loc(idx) - 1
                if push_num == 1:
                    continue

                # Identify time window
                new_row = df.iloc[new_idx]
                block_idx = tuple(
                    [idx[dataset.index.index(i)] for i in dataset.block_identifiers]
                )
                df_block = df.loc[block_idx]
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
                    df_rate_content["push percentiles"].append(
                        new_row["push percentiles"]
                    )

        # For each subject, sample pushes calculate the reward fraction of the past `window` seconds prior to each push
        for subject in dataset.get_unique("subject"):
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

    @legend_corrector
    def plot_previous_push_interval_vs_push_interval(
        self,
        dataset: Experiment = None,
        conds: dict = None,
        n_samples: int = 5000,
        seed: int = SEED,
        ax: plt.Axes = None,
        **kwargs,
    ) -> plt.Axes:
        """
        Plot previous push interval vs current push interval.

        Args:
            dataset: An Experiment dataset.
            conds: A dictionary of conditions to filter the data.
            n_samples: The number of samples to draw for analysis.
            seed: The seed for random number generation.
            **kwargs: Additional keyword arguments passed to the plotting function.
            - fig_kwargs: Dictionary to specify figure properties when creating a new figure (passed to `plt.subplots`).

        Returns:
            The axes.
        """
        dataset = self._init_vars(dataset=dataset)
        df = dataset.df
        rng = np.random.default_rng(seed)

        df_rate_content = {
            "previous push intervals (s)": [],
            "push intervals (s)": [],
            "stay/switch": [],
        }

        # For each subject, sample pushes calculate the reward fraction of the past `window` seconds prior to each push
        df_subject = dataset.filter(conds).df
        for row in df_subject.sample(
            n_samples, replace=True, random_state=rng
        ).itertuples():
            idx = row.Index

            # Get push immediately before current push
            push_num = idx[dataset.index.index("push #")]
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
        return ax

    @legend_corrector
    def plot_session_onsets_vs_push_percentiles(
        self, dataset: Experiment = None, **kwargs
    ) -> plt.Axes:
        """
        Plot the session onset times of pushes as a function of push percentile.

        Args:
            dataset: An Experiment dataset.
            **kwargs: Additional keyword arguments passed to the plotting function.
            - fig_kwargs: Dictionary to specify figure properties when creating a new figure (passed to `plt.subplots`).

        Returns:
            The axes.
        """
        dataset = self._init_vars(dataset=dataset)
        df = dataset.df

        # Get time of each push in the session, not just block
        x_offset = dataset.get_blocks()["duration"].last()
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

        df_temp = df.join(x_offset, rsuffix="_offset", on=dataset.block_identifiers)
        df_temp["push time in session"] = (
            df_temp["push times"] + df_temp["duration_offset"]
        )
        df_temp["onset (s)"] = (
            dataset.wrap(df_temp).blocks["push time in session"].shift().fillna(0)
        )

        # Plot the onset of push in session
        fig_kwargs = kwargs_handler(kwargs, "fig_kwargs")
        fig, ax = plt.subplots(**fig_kwargs)
        df_temp["push percentiles"] = bin_data(df_temp["push percentiles"])
        sns.lineplot(df_temp, x="push percentiles", y="onset (s)", hue="subject", ax=ax)
        ax.set_title("Onset of push in session")
        fig.tight_layout()
        return ax

    @legend_corrector
    def plot_block_onsets_vs_push_percentiles(
        self, dataset: Experiment = None, **kwargs
    ) -> plt.Axes:
        """
        Plot the block onset times of pushes as a function of push percentile.

        Args:
            dataset: An Experiment dataset.
            **kwargs: Additional keyword arguments passed to the plotting function.
            - fig_kwargs: Dictionary to specify figure properties when creating a new figure (passed to `plt.subplots`).

        Returns:
            The axes.
        """

        # Get time of each push in the block
        dataset = self._init_vars(dataset=dataset)
        df = deepcopy(dataset.df)
        df["onset (s)"] = dataset.blocks["push times"].shift().fillna(0)

        # Plot the onset of push in block
        fig_kwargs = kwargs_handler(kwargs, "fig_kwargs")
        fig, ax = plt.subplots(**fig_kwargs)
        df["push percentiles"] = bin_data(df["push percentiles"])
        sns.lineplot(df, x="push percentiles", y="onset (s)", hue="subject", ax=ax)
        ax.set_title("Onset of push in block")

        fig.tight_layout()
        return ax

    def plot_vertical_position_in_block(
        self, dataset: Experiment = None, conds: dict = None, data_dir: str = None
    ) -> plt.Axes:
        """
        Plot the vertical position of subjects within a block over time.
        This function retrieves vertical position data for a specified block and plots the vertical position over time, highlighting the position at the time of each push.

        Args:
            dataset: An Experiment dataset.
            conds: A dictionary specifying conditions to filter the DataFrame.
            data_dir: The directory containing continuous data files.

        Returns:
            The axes.
        """

        # Get vertical data for block specified by `conds`
        dataset = self._init_vars(dataset=dataset)
        df_block = dataset.filter(conds).df
        continuous_data, errors = get_continuous_from_df_to_dict(
            dataset.wrap(df_block), data_dir
        )
        block_key = HashableDict(conds)
        time = continuous_data[block_key]["time"]
        vertical = continuous_data[block_key]["position"][:, 2]

        # Plot all vertical positions
        fig, ax = plt.subplots()
        ax.plot(time, vertical, c="grey", linewidth=1)

        # Plot vertical position at time of push
        idx = np.abs(time[None, :] - df_block["push times"].values[:, None]).argmin(
            axis=1
        )
        df_block["z-coordinate"] = vertical[idx]
        ax = super().plot_block_events(
            dataset.wrap(df_block),
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
        self, dataset: Experiment = None, data_dir: str = None, **kwargs
    ) -> plt.Axes:
        """
        Plot the average vertical position of subjects during push intervals against push percentiles.
        This function calculates the average vertical position during each push interval and plots it against the push percentiles for each subject.

        Args:
            dataset: An Experiment dataset.
            data_dir: The directory containing continuous data files.
            **kwargs: Additional keyword arguments passed to the plotting function.
            - fig_kwargs: Dictionary to specify figure properties when creating a new figure (passed to `plt.subplots`).

        Returns:
            The axes.
        """
        dataset = self._init_vars(dataset=dataset)
        continuous_data, _ = get_continuous_from_df_to_dict(dataset, data_dir)
        dfs_cont = []

        # Only keep blocks that have real position data
        for block, data in continuous_data.items():
            if np.any(data["position"] != np.nan):
                dfs_cont.append(
                    dataset.df.xs(
                        tuple(block.values()),
                        level=("subject", "session", "block"),
                        drop_level=False,
                    )
                )
        dataset_cont = dataset.wrap(pd.concat(dfs_cont))
        fig_kwargs = kwargs_handler(
            kwargs,
            "fig_kwargs",
            dict(
                figsize=MULTIPLOT_FIGSIZE,
                nrows=1,
                ncols=len(dataset_cont.get_unique("subject")),
                sharex=True,
                sharey=True,
            ),
        )
        fig, axes = plt.subplots(**fig_kwargs)

        # For each subject, sample pushes calculate the reward fraction of the past `window` seconds prior to each push
        @legend_corrector
        def _plot(embeddables: Iterable[Embeddable], **kwargs):
            embeddable = embeddables[0]
            i = embeddable.index
            subject = embeddable.value
            conds = {"subject": subject}
            df_subject = dataset_cont.filter(conds).df
            df_vertical = {
                "push intervals": [],
                "average vertical position": [],
                "push percentiles": [],
            }

            # Find average vertical position over each push interval
            for idx, row in df_subject.iterrows():
                try:
                    conds2 = HashableDict(
                        subject=subject,
                        session=idx[dataset_cont.index.index("session")],
                        block=idx[dataset_cont.index.index("block")],
                    )
                    time = continuous_data[conds2]["time"]
                    vertical = continuous_data[conds2]["position"][:, 2]
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
                df_vertical,
                x="push percentiles",
                y="average vertical position",
                ax=axes[i],
            )
            axes[i].set_title(
                titler(
                    "Average vertical position\n occupied during push interval",
                    conds=conds,
                )
            )
            return axes[i]

        # Plot each subject's vertical position distribution
        return across_conditions_plotter(
            "subject", dataset_cont.get_unique("subject"), _plot, **kwargs
        )

    @legend_corrector(bbox=(1.1, 1))
    def plot_hmm_probabilities_in_block(
        self, filepath: str = None, block_idx: int = 3
    ) -> plt.Axes:
        """
        Plot HMM probabilities over a specific block.
        This function loads Hidden Markov Model (HMM) probabilities from a file and overlays them on top of a specific block's data, visualizing the probabilities of different HMM policies.

        Args:
            filepath: The path to the file containing saved HMM probabilities.
            block_idx: The index of the block to overlay the HMM probabilities on.

        Returns:
            The axes.
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
        ax = self.plot_pushes(
            conds=dict(subject=subject, session=int(session_id), block=block + 1)
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

    # TODO: this only works on angelaki data
    def plot_experiment_parameters(
        self,
        dataset: Experiment = None,
        conds: dict = None,
        title: str = "Experiment parameters by session",
        label_rotation: float = 35,
        ax: plt.Axes = None,
        **kwargs,
    ) -> plt.Axes:
        """
        Plots the distribution of experiment parameters (kappa, stimulus type, shape) across different sessions.
        Displays the number of blocks associated with each parameter and session.

        Args:
            dataset: Experiment dataset.
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
        dataset = self._init_vars(dataset=dataset)
        df = dataset.df

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
        df = dataset.filter(conds).df
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
        kappa_ticks = [
            i * c + max(stim_type_ticks) + 1 for i in range(1, len(kappas) + 1)
        ]

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
        ax.set_yticks(
            shape_ticks + stim_type_ticks + kappa_ticks, y_labels, fontsize=10
        )
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

    # TODO: this only works on angelaki data
    def plot_push_intervals_by_sessions(
        self,
        dataset: Experiment = None,
        palette: dict[str, Any] = None,
        label_rotation: float = 35,
        **kwargs,
    ) -> plt.Axes:
        """
        Plot push intervals across sessions for monkey subjects.
        This function visualizes the distribution of push intervals across different sessions for specified monkey subjects, using swarm plots to show push intervals and adding weekday labels for context.

        Args:
            dataset: An Experiment dataset.
            palette: Dictionary mapping box schedules to colors. Can also be a list of just colors.
            label_rotation: The angle to rotate x-tick labels by.
            **kwargs: Additional keyword arguments passed to the plotting function.
            - fig_kwargs: Dictionary to specify figure properties when creating a new figure (passed to `plt.subplots`).

        Returns:
            The axes.
        """
        dataset, palette = self._init_vars(dataset=dataset, palette=palette)

        # Examine monkeys separately from humans
        monkey_subjects = [x for x in dataset.get_unique("subject") if x != "humans"]
        conds = {"subject": monkey_subjects}
        df_monkey = deepcopy(dataset.filter(conds).df)

        # Reformat time data
        df_monkey["session id"] = pd.to_datetime(
            df_monkey.index.get_level_values("session"), format="%Y%m%d"
        )
        df_ref = df_monkey.groupby("subject")["session id"].min()
        df_monkey = df_monkey.join(df_ref, how="left", rsuffix="_ref", on=["subject"])
        df_monkey["day"] = (
            df_monkey["session id"] - df_monkey["session id_ref"]
        ).dt.days
        dataset = dataset.wrap(df_monkey)
        fig_kwargs = kwargs_handler(
            kwargs,
            "fig_kwargs",
            dict(figsize=(25, 10), height_ratios=[1, 2], sharex=True),
        )

        def _plot(embeddables: Iterable[Embeddable], **kwargs):
            embeddable = embeddables[0]
            i = embeddable.index
            subject = embeddable.value
            conds = {"subject": subject}

            # Plot experiment overview and push interval distribution
            fig, axes = plt.subplots(2, 1, **fig_kwargs)
            self.plot_experiment_parameters(dataset, conds=conds, ax=axes[0])
            bp(sns.swarmplot)(
                df_monkey,
                x="day",
                y="consecutive push intervals",
                conds=conds,
                hue="box",
                palette=palette,
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
                df_monkey.xs(subject, level="subject")
                .reset_index()
                .groupby(["session"], as_index=False)["weekday"]
                .max()["weekday"]
            )
            for j, l in enumerate(labels):
                tmp = l.get_text()
                labels[j] = tmp + "\n" + days[j]

            axes[1].set_xticklabels(labels)
            axes[1].tick_params(axis="x", labelrotation=label_rotation)
            fig.suptitle(f"Push intervals for {subject}")
            fig.tight_layout()

        across_conditions_plotter("subject", monkey_subjects, _plot, **kwargs)

    def plot_push_intervals(
        self,
        dataset: Experiment = None,
        conds: dict = None,
        palette: dict[str, Any] = None,
        palette_dark: dict[str, Any] = None,
        x: str = "stimulus reliability",
        ax: plt.Axes = None,
        **kwargs,
    ) -> plt.Axes:
        """
        This function visualizes the distribution of push intervals conditioned on different `x` values.

        Args:
            dataset: An Experiment dataset.
            conds: A dictionary of conditions to filter the data.
            palette: Dictionary mapping box schedules to colors. Can also be a list of just colors.
            palette_dark: Dictionary mapping box schedules to darkened colors. Can also be a list of just colors.
            x: The variable to condition on.
            ax: A matplotlib Axes object to plot on.
        Returns:
            The axes.
        """

        dataset, palette, palette_dark = self._init_vars(
            dataset=dataset, palette=palette, palette_dark=palette_dark
        )
        df = dataset.filter(conds).df
        swarm_kwargs = kwargs_handler(
            kwargs, "swarm_kwargs", dict(size=0.5, log_scale=True, dodge=True)
        )
        x_order = dataset.label_order[x]
        x_order = (
            df.reset_index()[[x, x_order]]
            .drop_duplicates()
            .sort_values(x_order)[x]
            .tolist()
        )
        bp(sns.swarmplot)(
            df,
            x=x,
            order=x_order,
            y="push intervals",
            hue="box",
            palette=palette_dark,
            legend=False,
            ax=ax,
            **swarm_kwargs,
        )
        bp(enhanced_violinplot)(
            df,
            x=x,
            order=x_order,
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

        return ax

    def plot_stay_switch_pushes(
        self,
        dataset: Experiment = None,
        conds: dict = None,
        palette: dict[str, Any] = None,
        palette_dark: dict[str, Any] = None,
        null_model: bool = False,
        **kwargs,
    ) -> plt.Axes:
        """
        Plot the stay and switch push intervals.

        Args:
            dataset: An Experiment dataset.
            conds: A dictionary of conditions to filter the data.
            null_model: If True, perform Kolmogorov-Smirnov Test to see if push intervals can be well described by an exponential distribution
            **kwargs: Additional keyword arguments.
                - 'fig_kwargs': Dictionary to specify figure properties when creating a new figure (passed to `plt.subplots`).

        Returns:
            The axes.
        """

        dataset, palette, palette_dark = self._init_vars(
            dataset=dataset, palette=palette, palette_dark=palette_dark
        )
        dataset = dataset.filter(conds)
        box_labels = self.get_config_value("box_labels")
        n_boxes = len(box_labels)
        fig_kwargs = kwargs_handler(
            kwargs,
            "fig_kwargs",
            dict(
                nrows=n_boxes, ncols=n_boxes, figsize=(12, 10), sharex=True, sharey=True
            ),
        )

        def _plot(embeddables: Iterable[Embeddable], **kwargs):
            embeddable = embeddables[0]
            i = embeddable.index
            subject = embeddable.value
            conds = {"subject": subject}
            fig, axes = fig_init(**fig_kwargs)
            df_subj = dataset.filter(conds).df
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
            fig.suptitle(f"Stay and switch times for {subject}", y=1)
            fig.text(0.5, 0.95, "TO", ha="center")
            fig.text(0.0, 0.5, "FROM", va="center", rotation="vertical")
            fig.tight_layout()

        return across_conditions_plotter(
            "subject", dataset.get_unique("subject"), _plot, **kwargs
        )

    def plot_runlengths(
        self,
        dataset: Experiment = None,
        conds: dict = None,
        palette: dict[str, Any] = None,
        null_model: bool = False,
        disp_js: bool = False,
        ax: plt.Axes = None,
        **kwargs,
    ) -> plt.Axes:
        """
        Plot the runlengths of consecutive pushes and their distribution across different boxes.
        This function calculates the runlengths of consecutive pushes and visualizes their distribution across different boxes, optionally comparing them to a null model based on visitation frequencies.

        Args:
            dataset: An Experiment dataset.
            conds: A dictionary of conditions to filter the data.
            palette: Dictionary mapping box schedules to colors. Can also be a list of just colors.
            null_model: If True, overlay random dice probabilities from a geometric distribution.
            disp_js: If True, display Jensen-Shannon distance between empirical and null distributions.
            ax: A matplotlib Axes object to plot on.
            **kwargs: Additional keyword arguments.
            - fig_kwargs: Dictionary to specify figure properties when creating a new figure (passed to `plt.subplots`).

        Returns:
            The axes.
        """
        dataset, palette = self._init_vars(dataset=dataset, palette=palette)
        dataset = dataset.filter(conds)
        x = dataset.get("push #")
        consecutive_mask = x[1:] - x[:-1] == 1
        change_mask = (dataset.get("stay/switch") == "switch") & np.insert(
            consecutive_mask, 0, True
        )
        push_nums = (
            dataset.blocks["push times"].rank().astype(int)
        )  # Calculate from scratch in case pushes got dropped
        change_mask[push_nums == 1] = True

        # Count the runlengths at different boxes
        group_labels = change_mask.cumsum()
        labeled_lengths = pd.DataFrame(
            {
                "group": group_labels,
                "box": dataset.get("box"),
                "next box": dataset.blocks["box"].shift(-1).fillna("missing"),
            }
        ).set_index(dataset.df.index)
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
        visit_freqs = dataset.df["box"].value_counts(normalize=True).to_frame()
        try:
            bp(sns.histplot)(
                labeled_lengths_all.reset_index(),
                conds=conds,
                x="length",
                hue="box",
                palette=palette,
                discrete=True,
                multiple="stack",
                ax=ax,
                **kwargs,
            )
        except:
            pass

        # Overlay random dice probabilities from geometric distribution
        if null_model:
            bars = ax.patches
            bar_width = bars[0].get_width()  # Width of one bar
            probs = visit_freqs  # Visit probabilities
            boxes = sorted(probs.index.unique("box"))
            handles = ax.get_legend().legend_handles
            labels = [t.get_text() for t in ax.get_legend().get_texts()]
            for b, box in enumerate(boxes):
                try:
                    p = probs.loc[box].iloc[0]
                    run_lengths = (
                        labeled_lengths_all.loc[box, "length"].sort_values().unique()
                    )
                    geom = p**run_lengths * (1 - p)
                    offset = (b - (len(boxes) - 1) / 2) * bar_width
                    x = run_lengths + offset
                    ax.plot(x, geom, c=palette[box], label="random dice")
                    if disp_js:
                        bar_heights = get_bar_heights(ax, x_centers=run_lengths)
                        ax.set_title(
                            ax.get_title()
                            + f"\nJS-distance = {jensenshannon(geom, bar_heights[box])}"
                        )
                        # print(f"Jensen-shannon distance between empirical distribution and null distribution of ({subj}, {kappa}, {box}): {jensenshannon(geom, bar_heights[box])}")
                except:
                    continue
            ax.legend(
                handles=handles
                + [Line2D([0], [0], color="black", linestyle="-", label="random dice")],
                labels=labels + ["random dice"],
            )
        return ax

    def plot_push_intervals_vs_reward_intervals(
        self,
        dataset: Experiment = None,
        conds: dict = None,
        palette: dict[str, Any] = None,
        annotate_reg: bool = False,
        max_x=60,
        ax: plt.Axes = None,
        **kwargs,
    ) -> plt.Axes:
        """
        Plot linear regression of push intervals against reward intervals in a block.

        Args:
            dataset: An Experiment dataset.
            conds: A dictionary of conditions to filter the data.
            palette: Dictionary mapping box schedules to colors. Can also be a list of just colors.
            annotate_reg: If True, annotate the regression slope on the plot.
            max_x: The maximum value of the x-axis.
            ax: Axes to plot on. If none, a new figure and axes are created using plt.subplots. Specify keyword arguments in `fig_kwargs`.
            **kwargs: Additional keyword arguments for seaborn.
            - fig_kwargs: Dictionary to specify figure properties when creating a new figure (passed to `plt.subplots`).

        Returns:
            None
        """
        dataset, palette = self._init_vars(dataset=dataset, palette=palette)
        df = dataset.filter(conds).df
        bp(sns.scatterplot)(
            df,
            x="reward intervals",
            y="push intervals",
            conds=conds,
            hue="box",
            palette=palette,
            ax=ax,
            **kwargs,
        )

        # Add best fit line
        fit_results = regplot(
            df["reward intervals"].to_numpy(),
            df["push intervals"].to_numpy(),
            line_kws={"color": "black"},
            ax=ax,
            **kwargs,
        )

        # Add nice aesthetics
        x = np.arange(max_x)
        ax.plot([0, max_x], [0, max_x], linestyle="dashed", color="black")
        ax.fill_between(x, x, max_x, color="green", alpha=0.1)
        ax.fill_between(x, x, color="red", alpha=0.1)
        ax.set_xlim([0, max_x])
        ax.set_ylim([0, max_x])
        ax.set_aspect("equal")

        if annotate_reg:
            ax.text(
                0.75,
                0.1,
                f"slope={fit_results.params[1]:.2f}",
                transform=ax.transAxes,
                fontsize=10,
            )
        return ax

    def plot_next_push_surprise(
        self,
        dataset: Experiment = None,
        conds: dict = None,
        palette: dict[str, Any] = None,
        palette_dark: dict[str, Any] = None,
        ax: plt.Axes = None,
        **kwargs,
    ) -> plt.Axes:
        """Plot the change in push interval after each push.

        Args:
            dataset: An Experiment dataset.
            conds: A dictionary of conditions to filter the data.
            palette: Dictionary mapping box schedules to colors. Can also be a list of just colors.
            palette_dark: Dictionary mapping box schedules to darkened colors. Can also be a list of just colors.
            ax: Axes to plot on. If none, a new figure and axes are created using plt.subplots. Specify keyword arguments in `fig_kwargs`.
            **kwargs: Additional keyword arguments.

        Returns:
            The axes.
        """
        dataset, palette, palette_dark = self._init_vars(
            dataset=dataset, palette=palette, palette_dark=palette_dark
        )
        dataset = dataset.filter(conds)

        # First, identify consecutive pushes for each box
        push_deltas = dataset.get_blocks(groupers=["box"])
        df = dataset.df.copy()
        df.loc[:, "consecutive wait"] = push_deltas["push # by box"].diff().fillna(1)
        df = df.loc[df["consecutive wait"] == 1].copy()

        # Calculate the change in push interval
        push_deltas = dataset.get_blocks(groupers=["box"])
        df.loc[:, "change in next push interval"] = -push_deltas["push intervals"].diff(
            -1
        )

        # Track whether the subject stayed or switched after each push
        df.loc[:, "stay/switch"] = df["stay/switch"].shift(-1)
        ro = conds["rewarded"]
        bp(sns.scatterplot)(
            df,
            x="push intervals",
            y="change in next push interval",
            conds=conds,
            hue="box",
            palette=palette if ro == "yes" else palette_dark,
            style="stay/switch",
            alpha=0.5,
            ax=ax,
            **kwargs,
        )
        ax.hlines(0, 0, ax.get_xlim()[1], colors="black")
        ax.plot([0, 40], [0, -40], linestyle="dashed", color="black")
        ax.set_xlim([0, 40])
        ax.set_ylim([-40, 40])
        return ax

    def plot_stay_probabilities(
        self,
        dataset: Experiment = None,
        conds: dict = None,
        bin_width: float = 10,
        ax: plt.Axes = None,
        **kwargs,
    ) -> plt.Axes:
        """
        Plot the probability of staying at the same box after a push, based on push intervals.
        This function calculates and visualizes the probability of a subject staying at the same box after a push, using push intervals binned by a specified width.

        Args:
            dataset: An Experiment dataset.
            conds: A dictionary of conditions to filter the data.
            bin_width: The width of the bins for push intervals.
            ax: Axes to plot on. If none, a new figure and axes are created using plt.subplots. Specify keyword arguments in `fig_kwargs`.
            **kwargs: Additional keyword arguments.

        Returns:
            The axes.
        """

        dataset = self._init_vars(dataset=dataset)
        df = dataset.filter(conds).df.copy()

        # Track whether the push was rewarded and whether the subject stayed or switched after each push
        df["rewarded"] = df["reward outcomes"].map({True: "yes", False: "no"})
        df["time"] = bin_data(df["push intervals"], bin_width=bin_width)
        df["P(stay)"] = df["stay/switch"].shift(-1).map({"stay": 1, "switch": 0})
        bp(sns.lineplot)(
            df,
            conds=conds,
            x="time",
            y="P(stay)",
            x_unit="s",
            hue="rewarded",
            hue_order=["no", "yes"],
            errorbar="se",
            ax=ax,
            **kwargs,
        )
        return ax

    def plot_push_rates_across_block(
        self,
        **kwargs,
    ) -> list[plt.Axes]:
        """
        This function calculates and visualizes the push rates across different blocks, smoothing the data over a specified window and optionally separating the data by box.

        Args:
            **kwargs: Additional keyword arguments.
                - smooth_kwargs: Dictionary to specify window properties for smoothing the push rate (passed to `moving_average`).
        Returns:
            None
        """
        smooth_kwargs = kwargs_handler(
            kwargs,
            "smooth_kwargs",
            {"rate": True, "bin_func": lambda x: x.size()},
            no_pop=True,
        )

        return self.plot_quantity_across_block(
            x="push times",
            y="reward outcomes",
            y_name="push rate",
            smooth_kwargs=smooth_kwargs,
            **kwargs,
        )

    def plot_reward_rates_across_block(
        self,
        **kwargs,
    ) -> list[plt.Axes]:
        """
        This function calculates and visualizes the reward rates across different blocks, smoothing the data over a specified window and optionally separating the data by box.

        Args:
            **kwargs: Additional keyword arguments.
                - smooth_kwargs: Dictionary to specify window properties for smoothing the reward rate (passed to `moving_average`).
        Returns:
            None
        """
        smooth_kwargs = kwargs_handler(
            kwargs,
            "smooth_kwargs",
            {"rate": True, "bin_func": lambda x: x["reward outcomes"].sum()},
            no_pop=True,
        )

        return self.plot_quantity_across_block(
            x="push times",
            y="reward outcomes",
            y_name="reward rate",
            smooth_kwargs=smooth_kwargs,
            **kwargs,
        )

    def plot_reward_per_push_across_block(
        self,
        dataset: Experiment = None,
        conds: dict = None,
        palette: dict = None,
        by_box: bool = False,
        ax: plt.Axes = None,
        **kwargs,
    ):
        """
        This function calculates and visualizes the reward rates across different blocks, smoothing the data over a specified window and optionally separating the data by box.

        Args:
            dataset: An Experiment dataset.
            conds: A dictionary of conditions to filter the data.
            palette: A dictionary mapping box schedules to colors.
            by_box: If True, separate the reward rates by box.
            ax: Axes to plot on. If none, a new figure and axes are created using plt.subplots. Specify keyword arguments in `fig_kwargs`.
            **kwargs: Additional keyword arguments.
                - bin_kwargs: Dictionary to specify binning properties for time (passed to `bin_data`).
                - fig_kwargs: Dictionary to specify figure properties when creating a new figure (passed to `plt.subplots`).
                - smooth_kwargs: Dictionary to specify window properties for smoothing the reward rate (passed to `moving_average`).
        Returns:
            None
        """
        kwargs = deepcopy(kwargs)
        smooth_kwargs = kwargs_handler(
            kwargs,
            "smooth_kwargs",
            {"rate": True},
        )

        # Average data over time
        dataset, palette = self._init_vars(dataset=dataset, palette=palette)
        dataset = dataset.filter(conds)
        groupers = ["stimulus reliability", "block_id"]
        if by_box:
            groupers.append("box")

        fig_kwargs = kwargs_handler(
            kwargs, "fig_kwargs", {"sharey": True, "sharex": True}
        )
        kwargs.update({"x": "time", "y": "reward-per-push", "x_unit": "s"})

        if by_box:
            kwargs.update({"hue": "box", "palette": palette})
        else:
            kwargs.update({"color": "black"})

        ma_push = moving_average(
            dataset,
            x="push times",
            y="reward outcomes",
            y_name="push rate",
            bin_func=lambda x: x.size(),
            groupers=groupers,
            **smooth_kwargs,
        )

        ma_rew = moving_average(
            dataset,
            x="push times",
            y="reward outcomes",
            y_name="reward rate",
            bin_func=lambda x: x["reward outcomes"].sum(),
            groupers=groupers,
            **smooth_kwargs,
        )

        ma_rew["reward-per-push"] = ma_rew["reward rate"] / ma_push["push rate"]
        return plot_average_or_traces(ma_rew, ax=ax, **kwargs)

    def plot_matching_law(
        self,
        dataset: Experiment = None,
        conds: dict = None,
        palette: dict = None,
        time_bin: tuple[float, float] = None,
        ax: plt.Axes = None,
        **kwargs,
    ) -> plt.Axes:
        """
        Fit the empirical matching law for each subject over time.

        Args:
            dataset: An Experiment dataset.
            conds: A dictionary of conditions to filter the data.
            palette: A dictionary mapping box schedules to colors.
            time_bin: A tuple specifying the start and end times of the time bin to plot. If None, the entire block is plotted.
            ax: Axes to plot on. If none, a new figure and axes are created using plt.subplots. Specify keyword arguments in `fig_kwargs`.
            **kwargs: Additional keyword arguments.
                - bin_kwargs: Dictionary to specify binning properties for time (passed to `bin_data`).

        Returns:
            The axes.
        """
        dataset, palette = self._init_vars(dataset=dataset, palette=palette)
        dataset = dataset.filter(conds)
        df = dataset.df
        if time_bin:
            df = df[df["push times"].between(time_bin[0], time_bin[1])]
            dataset = dataset.wrap(df)

        # Bin pushes by time, group by box, count rewards and pushes
        grouped = dataset.get_blocks(groupers=["stimulus reliability", "box"])
        rr = grouped["reward outcomes"].sum().to_frame()
        rr["pushes"] = grouped.size()  # .reset_index()[0]

        # Calculate totals across boxes for each block (exclude box dimension)
        totals = rr.groupby(dataset.block_identifiers + ["stimulus reliability"]).agg(
            {"pushes": "sum", "reward outcomes": "sum"}
        )

        # Reset index to get box column back
        rr = rr.reset_index()

        # Merge totals back to original dataframe (exclude box from merge keys)
        rr = pd.merge(
            rr,
            totals,
            on=["subject", "session", "block", "stimulus reliability"],
            suffixes=["", "_total"],
        )
        rr["relative pushes"] = rr["pushes"] / rr["pushes_total"]
        rr["relative rewards"] = rr["reward outcomes"] / rr["reward outcomes_total"]

        # Drop rows with NaN or infinite values
        rr = rr.replace([np.inf, -np.inf], np.nan).dropna()
        max_pt = max(rr["relative rewards"].max(), rr["relative pushes"].max())
        min_pt = min(rr["relative rewards"].min(), rr["relative pushes"].min())
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        bp(sns.scatterplot)(
            rr,
            conds=conds,
            x="relative rewards",
            y="relative pushes",
            hue="box",
            palette=palette,
            ax=ax,
            **kwargs,
        )

        # Best fit line
        fit_results = regplot(
            rr["relative rewards"].to_numpy(),
            rr["relative pushes"].to_numpy(),
            line_kws={"color": "black"},
            ax=ax,
            **kwargs,
        )
        ax.text(
            0.75,
            0.1,
            f"slope={fit_results.params[1]:.2f}\nintercept={fit_results.params[0]:.2f}",
            transform=ax.transAxes,
            fontsize=10,
        )
        return ax


## OLD
def plot_fisher(
    df: pd.DataFrame,
    conds: dict = None,
    title: str = "Fisher information",
    title_override: str = None,
    box_colors: list = None,
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
