import gc
import logging
import math
import os
from copy import deepcopy
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from kneed import KneeLocator
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from numpy.typing import ArrayLike
from scipy.optimize import curve_fit
from scipy.spatial.distance import euclidean
from tqdm import tqdm

from foraging.config.constants import BOX_LABELS, MULTIPLOT_FIGSIZE
from foraging.utils import flatten, kwargs_handler
from foraging.utils.data import filter_df

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def fig_init(ax: plt.Axes = None, **kwargs):
    if ax is None:
        return plt.subplots(**kwargs)
    return ax.get_figure(), ax


def titler(title: str = None, conds: dict = None, title_override: str = None):
    if title_override is not None:
        return title_override
    if title is None:
        return None
    if conds is None or len(conds) == 0:
        return title
    conds_str = ", ".join([k + " = " + str(v) for k, v in conds.items()])
    if len(title) > 0:
        return title + "\n" + conds_str
    return conds_str


def unitler(label: str, unit: str):
    if unit is None:
        return label
    return label + " (" + unit + ")"


def format_yticks(axes, func):
    for ax in flatten(axes):
        ax.yaxis.set_major_formatter(FuncFormatter(func))


def get_bar_positions(
    ax: plt.Axes, hue_order: list = BOX_LABELS, x_centers: ArrayLike = None
):
    bars = ax.patches

    # Group bar patches by hue group
    bar_width = bars[0].get_width()  # Width of one bar
    n_groups = len(hue_order)
    if x_centers is None:
        x_centers = np.arange(len(bars) // n_groups)

    # Organize bar positions by hue group
    positions_by_group = {group: [] for group in hue_order}
    for x_center in x_centers:
        for i, group in enumerate(hue_order):
            # Leftmost group's left edge is offset furthest left
            offset = (i - (n_groups - 1) / 2) * bar_width
            x_left = x_center + offset
            positions_by_group[group].append(x_left)
    return {k: np.array(v) for k, v in positions_by_group.items()}


def get_bar_heights(
    ax: plt.Axes, hue_order: list = BOX_LABELS, x_centers: ArrayLike = None
):
    # First map bar positions to heights
    bars = ax.patches
    bar_width = bars[0].get_width()  # Width of one bar
    n_groups = len(hue_order)
    proto_pos = []
    proto_heights = []
    for i, bar in enumerate(bars):
        x = bar.get_x() + bar_width / 2  # Center of the bar
        height = bar.get_height()
        proto_pos.append(x)
        proto_heights.append(height)

    # Now get the heights in order
    if x_centers is None:
        x_centers = np.arange(len(bars) // n_groups)
    heights_by_group = {group: [] for group in hue_order}
    for x_center in x_centers:
        for i, group in enumerate(hue_order):
            # Leftmost group's left edge is offset furthest left
            offset = (i - (n_groups - 1) / 2) * bar_width
            x_left = x_center + offset
            idx = np.argmin(np.abs(np.array(proto_pos) - x_left))
            if math.isclose(proto_pos[idx], x_left, abs_tol=1e-4):
                heights_by_group[group].append(proto_heights[idx])
    return {k: np.array(v) for k, v in heights_by_group.items()}


def palette_handler(palette: dict | list, categories: list):
    """Corrects for any mismatch between `palette` and observed `categories`. If `palette` is list, return first `len(categories)` entries."""
    return (
        {k: v for k, v in palette.items() if k in categories}
        if type(palette) == dict
        else palette[: len(categories)]
    )


def bp(func: Callable):
    """
    Wraps seaborn-style function with custom figure settings

    Args:
        func: function to wrap, typically seaborn-style function that takes the following as input:
            - df: DataFrame.
            - x: Name of x variable to be plotted.
            - hue (Optional): Name of variable to color-code data by.
            - hue_order (Optional): List specifying the order to assign colors to the `hue` variable.
            - palette (Optional): List of colors to map onto `hue` ordered by `hue_order`.
            - ax (Optional): axes to plot on.

    Returns:
        wrapped function
    """

    @wraps(func)
    @legend_handler
    def wrapper(
        df: pd.DataFrame = None,
        x: str = None,
        hue: str = None,
        palette: list = None,
        conds: dict = None,
        single_block: bool = False,
        title: str = "",
        title_override: str = None,
        legend: Any = "auto",
        x_unit: str = None,
        y_unit: str = None,
        min_obs: int = None,
        attempt_index: bool = True,
        ax: plt.Axes = None,
        **kwargs,
    ) -> Any:
        """
        Convenience decorator that customizes figure in formulaic fashion

        Args:
            df: DataFrame of block(s) data.
            x: Name of x variable to be plotted.
            hue: Name of hue variable to be plotted.
            palette: List or dictionary mapping hue levels to colors.
            conds: Dictionary mapping level keys to values to be used to filter `df`.
            single_block: True indicates `df` should be treated as a single block.
            title: Title for figure.
            title_override: Override title for figure.
            legend: If True, display figure.
            x_unit: Unit of the x-axis. If None, then ignored.
            y_unit: Unit of the y-axis. If None, then ignored.
            min_obs: Threshold for min number of observations a bin must have to be displayed. Only used if not None.
            attempt_index: Refer to `filter_df` for more details.
            ax: Axis to plot on (not None if reusing premade figure and axis object).
            **kwargs: Additional keyword arguments.
                - fig_kwargs: keyword arguments to be passed to `plt.subplots`.
                - legend_kwargs: keyword arguments to be passed to `Axes.legend`.
                - title_kwargs: keyword arguments to be passed to `Axes.set_title`.
                - xlabel_kwargs: keyword arguments to be passed to `Axes.set_xlabel`.
                - ylabel_kwargs: keyword arguments to be passed to `Axes.set_ylabel`.
                - additional keyword arguments get passed to wrapped function, which is meant to be a seaborn-style function.

        Returns:
            ax, or optional return arguments from wrapped function (usually in the form of ax + extra).
        """
        kwargs = deepcopy(kwargs)

        # Filter df
        if conds is None:
            conds = {}
        else:
            conds = deepcopy(conds)
        df = filter_df(df, conds, attempt_index=attempt_index)

        # Context dependent plot settings
        if hue and palette:
            hue_keys = (
                sorted(df[hue].unique()) if hue in df.columns else df.index.unique(hue)
            )
            palette = palette_handler(palette, hue_keys)

            # Control hue order based on inputs
            if type(palette) is dict:
                kwargs["hue_order"] = list(palette.keys())
            else:
                kwargs["hue_order"] = hue_keys

        if single_block:  # If only plotting individual block
            kappa = df.index.unique("kappa")
            stim_type = df.index.unique("stimulus type")
            shape = df.index.unique("shape")
            if len(kappa) > 1 or len(stim_type) > 1 or len(shape) > 1:
                logger.debug(
                    f"length of kappa: {len(kappa)}, length of stim_type: {len(stim_type)}, length of shape: {len(shape)}"
                )
                raise Exception(
                    "Multiple experiment parameters found for single block. Make sure only single block is being supplied, or set collapse to True."
                )

            # For titling purposes, add block metadata
            conds["kappa"] = kappa[0]
            conds["stim type"] = stim_type[0]
            conds["shape"] = shape[0]

        # If plotting kappa on x-axis, create dummy column in order to plot kappa data evenly
        old_x = None
        if x == "kappa":
            df["stimulus reliability"] = pd.Series(
                df["kappa"].rank(method="dense") - 1, index=df.index
            )
            x = "stimulus reliability"
            old_x = True

        # Create ax if none
        fig, ax = fig_init(ax, **kwargs_handler(kwargs, "fig_kwargs"))

        # Pop any last keyword args not needed for seaborn here before running function
        legend_kwargs = kwargs_handler(kwargs, "legend_kwargs", dict(title=hue))
        title_kwargs = kwargs_handler(kwargs, "title_kwargs")
        xlabel_kwargs = kwargs_handler(kwargs, "xlabel_kwargs")
        ylabel_kwargs = kwargs_handler(kwargs, "ylabel_kwargs")

        # Run function, assuming seaborn plotting func
        if min_obs:
            if hue:
                ret = func(
                    df.groupby([x, hue], observed=True, as_index=False).filter(
                        lambda g: len(g) >= min_obs
                    ),
                    x=x,
                    hue=hue,
                    palette=palette,
                    ax=ax,
                    legend=legend,
                    **kwargs,
                )
            else:
                ret = func(
                    df.groupby(x, observed=True, as_index=False).filter(
                        lambda g: len(g) >= min_obs
                    ),
                    x=x,
                    ax=ax,
                    legend=legend,
                    **kwargs,
                )
        else:
            ret = func(
                df, x=x, hue=hue, palette=palette, ax=ax, legend=legend, **kwargs
            )

        # Adjust xticks to only show actual data
        if x == "stimulus reliability" and old_x:
            xticks = df.index.unique("kappa")
            [_ax.set_xticks(range(len(xticks)), xticks) for _ax in flatten(ax)]

        # Set title (if multiple axes, this does the first one)
        title = titler(title=title, conds=conds, title_override=title_override)
        _ax = np.atleast_1d(ax)
        if title:
            _ax[0].set_title(title, **title_kwargs)

        # Set units if specified
        if x_unit:
            _ax[0].set_xlabel(unitler(_ax[0].get_xlabel(), x_unit), **xlabel_kwargs)
        else:
            _ax[0].set_xlabel(_ax[0].get_xlabel(), **xlabel_kwargs)

        if y_unit:
            _ax[0].set_ylabel(unitler(_ax[0].get_ylabel(), y_unit), **ylabel_kwargs)
        else:
            _ax[0].set_ylabel(_ax[0].get_ylabel(), **ylabel_kwargs)

        # Modify legend
        if legend:
            for _ax in flatten(ax):
                if not _ax.get_legend_handles_labels() == ([], []):
                    _ax.legend(**legend_kwargs)
                # try:
                #     legend = _ax.get_legend()
                #     handles = legend.legend_handles
                #     _ax.legend(handles, box_labels, **legend_kwargs)
                # except Exception as e:
                #     print(e)
                #     _ax.legend(box_labels, **legend_kwargs)
        if ret is None:
            return ax
        return ret  # Assume there is usually an ax in here

    return wrapper


def multiplot(_func=None, figsize=MULTIPLOT_FIGSIZE):
    """
    A decorator to set figures containing multiple plots to a default figsize.

    Args:
        func: The plotting function to wrap.

    Returns:
        The wrapped function with adjusted figsize.
    """

    def _inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # add default figsize to kwargs
            fig_kwargs = kwargs_handler(kwargs, "fig_kwargs")
            if "figsize" not in fig_kwargs:
                fig_kwargs["figsize"] = figsize
            kwargs["fig_kwargs"] = fig_kwargs
            return func(*args, **kwargs)

        return wrapper

    if _func:
        return _inner(_func)
    return _inner


def legend_handler(_func=None, loc="upper left", bbox=(1, 1)):
    """
    A decorator to set the legend location to 'upper left' and bbox_to_anchor to (1, 1).

    Args:
        func: The plotting function to wrap.

    Returns:
        The wrapped function with the legend location set.
    """

    def _inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            ax = func(*args, **kwargs)
            for _ax in flatten(ax):
                if _ax.get_legend() is not None:
                    sns.move_legend(_ax, loc, bbox_to_anchor=bbox)
            return ax

        return wrapper

    if _func:
        return _inner(_func)
    return _inner


def _figure_handler(**kwargs):
    """
    Decorator for creating and closing figures. Use this to avoid memory leaks when creating multiple plots that can be drawn on the same figure object

    Args:
        **kwargs: keyword arguments for plt.subplots
    Returns:
        wrapped function inside decorator
    """

    def _inner(func):

        @wraps(func)
        def wrapper():

            # Create figure object
            fig, ax = plt.subplots(**kwargs)

            # Run plotting function
            func(fig, ax)

            # Release figure object from memory
            fig.clf()
            plt.close(fig)
            gc.collect()

        return wrapper

    return _inner


def _figure_saver(fig: plt.Figure, ax: plt.Axes, figure_path: str):
    """
    Save figure and clear it for later reuse

    Args:
        fig: figure to be drawn on
        ax: axis object to do drawing
        figure_path: path to save figure

    Returns:

    """
    Path(figure_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, facecolor="white")
    [x.clear() for x in flatten(ax)]


def per_block(
    func: Callable,
    df: pd.DataFrame,
    figure_dir: str,
    filename_prefix: str,
    conds: dict = None,
    fig_kwargs: dict = None,
    use_tqdm: bool = True,
    attempt_index: bool = True,
    by_subject: bool = False,
    **kwargs,
):
    """
    Generic wrapper for creating figures for a single block

    Args:
        func: the actual plotting routine to be executed on each block
        df: DataFrame of experiment data
        figure_dir: folder for figures to be created in
        filename_prefix: Prepended to filename of each block's figure
        conds: dictionary mapping level keys to values to be used to filter `df`
        fig_kwargs: dictionary of keyword arguments for plt.subplots
        use_tqdm: whether to use tqdm to display progress bar
        attempt_index: input argument to utils.data.filter_df
        by_subject: if True, then save figures in separate directories for each subject
        **kwargs: keyword arguments for func

    Returns:

    """
    if fig_kwargs is None:
        fig_kwargs = {}

    @_figure_handler(**fig_kwargs)
    def _inner(fig, ax):
        nonlocal conds
        nonlocal figure_dir
        old_figure_dir = figure_dir
        filtered_df = filter_df(df, conds, attempt_index=attempt_index)
        for subject in tqdm(filtered_df.index.unique("subject"), disable=not use_tqdm):
            if by_subject:
                figure_dir = os.path.join(old_figure_dir, f"subject {subject}")
            for sess_num in filtered_df.xs(subject, level="subject").index.unique(
                "session"
            ):
                for block_num in filtered_df.xs(
                    (subject, sess_num), level=("subject", "session")
                ).index.unique("block"):
                    conds = {
                        "subject": subject,
                        "session": sess_num,
                        "block": block_num,
                    }
                    try:
                        func(df=df, conds=conds, ax=ax, **kwargs)
                    except Exception as e:
                        logger.debug(
                            f"could not plot subject {subject} session {sess_num} block {block_num}"
                        )
                        logger.debug(e)
                        continue
                    figure_path = os.path.join(
                        figure_dir,
                        filename_prefix
                        + ",".join([k + "=" + str(v) for k, v in conds.items()])
                        + ".png",
                    )
                    _figure_saver(fig, ax, figure_path)

    _inner()


# todo: generalize iter_level to multiple levels listed in hierarchical order head first
def across_blocks(
    func: Callable,
    df: pd.DataFrame,
    figure_dir: str,
    filename_prefix: str,
    conds: dict = None,
    fig_kwargs: dict = None,
    iter_level: str = None,
    attempt_index: bool = True,
    **kwargs,
):
    """
    Generic wrapper for creating figures that summarize across multiple blocks grouped by conditions

    Args:
        func: the actual plotting routine to be executed on each session
        df: dataframe of experiment data
        figure_dir: folder for figures to be created in
        filename_prefix: Prepended to filename of each block's figure
        conds: dictionary mapping level keys to values to be used to filter dataframe
        fig_kwargs: dictionary of keyword arguments for plt.subplots
        iter_level: index level of dataframe across which plots will be generated, conditioned on each value of this level
        **kwargs: keyword arguments for func

    Returns:

    """

    @_figure_handler(**fig_kwargs)
    def _inner(fig, ax):
        nonlocal conds
        filtered_df = filter_df(df, conds, attempt_index=attempt_index)
        if iter_level is None:
            func(block_df=filtered_df, conds=conds, ax=ax, collapse=True, **kwargs)
            figure_path = os.path.join(
                figure_dir,
                filename_prefix
                + ",".join([k + "=" + str(v) for k, v in conds.items()])
                + ".png",
            )
            _figure_saver(fig, ax, figure_path)
        else:
            if conds is None:
                conds = {}
            for v in tqdm(filtered_df.index.unique(iter_level)):
                conds[iter_level] = v
                func(df=filtered_df, conds=conds, ax=ax, collapse=True, **kwargs)
                figure_path = os.path.join(
                    figure_dir,
                    filename_prefix
                    + ",".join([k + "=" + str(v) for k, v in conds.items()])
                    + ".png",
                )
                _figure_saver(fig, ax, figure_path)

    _inner()


def subject_plotter(subjects, plot_func, **kwargs):
    """
    Generates plots for each subject

    Args:
        subjects: iterable of subjects
        plot_func: plotting function
        **kwargs: keyword arguments to `plot_func`
            - if a dictionary containing subject names as keys, then the value of each key is a dictionary of keyword arguments to `plot_func`

    Returns:
        list of any returned output from `plot_func`
    """
    subj_kwargs = kwargs_handler(kwargs, "subj_kwargs")
    returns = []
    for i, subj in enumerate(subjects):
        if subj in subj_kwargs:
            ret = plot_func(i, subj, **(subj_kwargs[subj] | kwargs))
        else:
            ret = plot_func(i, subj, **kwargs)
        returns.append(ret)
    return returns


## Common routines
def enhanced_violinplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str = None,
    hue_order=None,
    palette=None,
    ax: plt.Axes = None,
    **kwargs,
) -> plt.Axes:
    """
    Plot a violinplot with mean + s.e. overlaid on top.

    Args:
        df: Dataframe
        x: x-axis
        y: y-axis
        hue: Hue variable
        hue_order: Order to assign hue
        palette: List of colors (same length as hue_order)
        ax: Axes to plot on. If none, a new figure and axes are created using plt.subplots. Specify keyword arguments in `fig_kwargs`.
        **kwargs: Keywords passed to seaborn's violinplot

    Returns:
        the axes
    """

    # Create ax if none
    fig, ax = fig_init(ax, **kwargs.pop("fig_kwargs", {}))
    sns.violinplot(
        df, x=x, y=y, hue=hue, hue_order=hue_order, palette=palette, ax=ax, **kwargs
    )

    # Plot means and se overlaid on violinplot
    groupers = [x]
    if hue and hue != x:
        groupers.append(hue)
    stats_df = df.groupby(groupers)[y].agg(["mean", "std", "count"]).reset_index()
    stats_df["se"] = stats_df["std"] / np.sqrt(stats_df["count"])

    # Calculate x-coordinates for the means
    violin_positions = []
    for patch in ax.collections:
        if isinstance(patch, PolyCollection):
            # Find the x position of the violin by averaging the x-values of the patch
            verts = patch.get_paths()[0].vertices
            x_pos = np.mean(verts[:, 0])
            violin_positions.append(x_pos)
    x_positions = violin_positions

    # Plot means and error bars for each subgroup with connecting lines
    if hue and hue != x:

        # Sort violins by hue_order or default ordering in seaborn
        if hue_order is None:
            if hue in df.columns:
                hue_order = df[hue].unique()
            else:
                hue_order = df.index.unique(hue)
        stats_df[hue] = pd.Categorical(
            stats_df[hue], categories=hue_order, ordered=True
        )
        stats_df = stats_df.sort_values(hue)
        for group_idx, group in enumerate(stats_df[x].unique()):

            subgroup_stats = stats_df[stats_df[x] == group]
            n_subgroups = subgroup_stats[hue].nunique()

            # Plot error bars and means
            group_x = x_positions[
                group_idx * n_subgroups : (group_idx + 1) * n_subgroups
            ]
            ax.errorbar(
                x=group_x,
                y=subgroup_stats["mean"],
                yerr=subgroup_stats["se"],
                fmt="o",
                color="black",
                capsize=5,
                capthick=2,
                elinewidth=2,
                markersize=8,
            )

            # Add connecting lines within subgroup
            ax.plot(group_x, subgroup_stats["mean"], color="black")
    else:
        # Plot error bars and means
        ax.errorbar(
            x=x_positions,
            y=stats_df["mean"],
            yerr=stats_df["se"],
            fmt="o",
            color="black",
            capsize=5,
            capthick=2,
            elinewidth=2,
            markersize=8,
        )

        # Add connecting lines within subgroup
        ax.plot(x_positions, stats_df["mean"], color="black")
    return ax


def plot_block_average_or_traces(
    df: pd.DataFrame,
    show_traces: bool,
    units: str = "block_id",
    **kwargs,
):
    kwargs = kwargs.copy()
    if show_traces:
        legend_flag = True
        if "legend" in kwargs:
            legend_flag = kwargs["legend"]
        kwargs["legend"] = False

        # Traces
        bp(sns.lineplot)(
            df,
            **kwargs,
            units=units,
            estimator=None,
            errorbar=None,
            alpha=0.2,
        )

        if "x_unit" in kwargs:
            kwargs.pop("x_unit")
        if "y_unit" in kwargs:
            kwargs.pop("y_unit")
        kwargs["legend"] = legend_flag

        # Average
        bp(sns.lineplot)(
            df,
            **kwargs,
            errorbar=None,
            lw=5,
        )

    else:
        bp(sns.lineplot)(df, **kwargs)


# Credit to https://stackoverflow.com/questions/22852244/how-to-get-the-numerical-fitting-results-when-plotting-a-regression-in-seaborn
def regplot(
    x: ArrayLike,
    y: ArrayLike,
    n_std: float = 1.96,
    n_pts: int = 100,
    ax: plt.Axes = None,
    **kwargs,
):
    """
    Plots a regression line along with confidence intervals and scatter points.

    Args:
        x: Independent variable data (array-like).
        y: Dependent variable data (array-like).
        n_std: Number of standard deviations for the confidence interval (default is 1.96, corresponding to a 95% confidence interval).
        n_pts: Number of points to generate for the prediction line (default is 100).
        ax: Matplotlib Axes object to plot on (optional). If not provided, a new one will be created.
        **kwargs: Additional keyword arguments passed to:
            - 'fig_kwargs': Parameters for creating the figure (passed to `plt.subplots`).
            - 'line_kwargs': Parameters for customizing the regression line plot.
            - 'ci_kwargs': Parameters for customizing the confidence interval shading.
            - 'scatter_kwargs': Parameters for customizing the scatter plot.

    Returns:
        fit_results: A statsmodels RegressionResults object containing the fitted regression results.
    """

    if ax is None:
        _, ax = plt.subplots(**kwargs_handler(kwargs, "fig_kwargs"))

    # Add constant to the x (for intercept in the regression)
    x_fit = sm.add_constant(x)

    # Fit the regression model
    fit_results = sm.OLS(y, x_fit).fit()

    # Generate predicted values over the range of x
    eval_x = sm.add_constant(np.linspace(np.min(x), np.max(x), n_pts))
    pred = fit_results.get_prediction(eval_x)

    # Draw the fit line and confidence interval
    ci_kws = kwargs_handler(kwargs, "ci_kwargs")
    if len(ci_kws) > 0:
        ax.fill_between(
            eval_x[:, 1],
            pred.predicted_mean - n_std * pred.se_mean,
            pred.predicted_mean + n_std * pred.se_mean,
            alpha=0.5,
            **ci_kws,
        )

    # Plot the regression line
    line_kwargs = kwargs_handler(kwargs, "line_kwargs", dict(color="black"))
    h = ax.plot(eval_x[:, 1], pred.predicted_mean, **line_kwargs)

    # Plot the scatter plot of the data
    scatter_kws = kwargs_handler(kwargs, "scatter_kwargs")
    if len(scatter_kws) > 0:
        ax.scatter(x, y, c=h[0].get_color(), **scatter_kws)

    return fit_results


def plot_variable_subplots(
    df: pd.DataFrame,
    func: Callable,
    row_cond: str,
    col_cond: str,
    axes: Iterable[plt.Axes] = None,
    legend: bool = True,
    simplify_row_title: bool = False,
    savefig: str = None,
    **kwargs,
):
    def _group_size(g, group):
        if group in g.columns:
            return g[group].nunique()
        return len(g.index.unique(group))

    max_cols = df.groupby(row_cond).apply(lambda g: _group_size(g, col_cond)).max()
    num_rows = len(df.groupby(row_cond))  # Compute rows needed
    fig_kwargs = kwargs_handler(
        kwargs,
        "fig_kwargs",
        dict(figsize=(5 * max_cols, 5 * num_rows), nrows=num_rows, ncols=max_cols),
    )
    fig, axes = fig_init(axes, **fig_kwargs)
    axes = np.atleast_2d(axes)
    for i, row_val in enumerate(df.groupby(row_cond).groups.keys()):
        df_row = filter_df(df, {row_cond: row_val})
        for j, col_val in enumerate(df_row.groupby(col_cond).groups.keys()):
            conds = {col_cond: col_val}
            df_group = filter_df(df_row, conds)
            func(df_group, conds=conds, ax=axes[i, j], **kwargs)
            if j > 0:
                axes[i, j].set_ylabel("")
            if simplify_row_title:
                axes[i, j].set_title(f"{col_cond}={col_val}")
        axes[i, 0].set_ylabel(row_val)
        for k in range(
            j + 1, max_cols
        ):  # intentially using the last j from previous loop
            fig.delaxes(axes[i, k])

    # Customize global legend
    # Get unique labels/handles across all subplots
    handles, labels = plt.gca().get_legend_handles_labels()  # Get from the current axis

    # Remove the automatic legend
    for ax in axes.flatten():
        try:
            ax.legend_.remove()
        except:
            continue

    # Add a custom legend with extracted artists
    if legend:
        legend_kwargs = kwargs_handler(
            kwargs,
            "legend_kwargs",
            dict(loc="upper right", bbox_to_anchor=(0.05, 0.05)),
        )
        lgd = fig.legend(handles, labels, **legend_kwargs)
    fig.tight_layout()

    if savefig:
        if legend:
            fig.savefig(
                savefig,
                bbox_extra_artists=(lgd,),
                facecolor="white",
                bbox_inches="tight",
            )
        else:
            fig.savefig(savefig, facecolor="white", bbox_inches="tight")

    return axes


def set_intelligent_xticks(ax, data, max_ticks=10):
    """
    Intelligently sets x-axis ticks based on data density.

    Args:
        ax: matplotlib Axes object
        data: array-like data to plot
        max_ticks: maximum number of ticks to show (default: 10)
    """
    # Get unique values and sort them
    unique_vals = np.sort(np.unique(data))

    # If we have fewer unique values than max_ticks, show them all
    if len(unique_vals) <= max_ticks:
        ax.set_xticks(unique_vals)
        return

    # Calculate the density of data points
    hist, bin_edges = np.histogram(data, bins=len(unique_vals))

    # Find the most populated bins
    sorted_bins = np.argsort(hist)[::-1]
    selected_bins = sorted_bins[:max_ticks]

    # Get the corresponding unique values
    selected_vals = unique_vals[selected_bins]

    # Set the ticks
    ax.set_xticks(selected_vals)


def stacked_barplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    palette: dict = None,
    ax: plt.Axes = None,
    **kwargs,
) -> plt.Axes:
    """
    Create a stacked barplot using seaborn.

    Args:
        df: DataFrame containing the data.
        x: Column name to be used for the x-axis.
        y: Column name to be used for the y-axis.
        hue: Column name to be used for the hue.
        ax: Axes to plot on. If none, a new figure and axes are created using plt.subplots.
        **kwargs: Additional keyword arguments passed to seaborn's barplot.

    Returns:
        The axes with the stacked barplot.
    """
    # Create ax if none
    fig, ax = fig_init(ax, **kwargs.pop("fig_kwargs", {}))
    df = df.copy().reset_index()
    hue_levels = df[hue].unique()
    if palette is None:
        palette = sns.color_palette(n_colors=len(df[hue].unique()))

    # Calculate the total sum for each x category
    legend_content = []
    for i, hue_level in enumerate(hue_levels[:]):
        total_df = df.groupby(x)[y].sum().reset_index()
        sns.barplot(data=total_df, x=x, y=y, color=palette[i], ax=ax, **kwargs)
        df = df[df[hue] != hue_level]
        legend_content.append(Patch(color=palette[i], label=hue_level))

    ax.legend(handles=legend_content, title=hue)
    return ax
