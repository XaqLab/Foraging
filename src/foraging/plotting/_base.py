import gc
import logging
import math
import os
from copy import deepcopy
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, Union

import ipywidgets as widgets
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from IPython.display import clear_output, display
from kneed import KneeLocator
from matplotlib import pyplot as plt
from matplotlib.artist import Artist
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from numpy.typing import ArrayLike
from scipy.optimize import curve_fit
from scipy.spatial.distance import euclidean
from tqdm import tqdm

from foraging import MULTIPLOT_FIGSIZE
from foraging.models import SuperDict
from foraging.models.experiment import Experiment
from foraging.utils import SupportsConds, flatten, kwargs_handler
from foraging.utils.data import filter_df
from foraging.utils.stats import moving_average

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@dataclass
class Embeddable:
    """A class that represents a plotting condition and its index."""

    name: str
    value: Any
    index: int


class SupportsEmbeddables(Protocol):
    """A protocol that defines a function that takes an iterable of Embeddables."""

    def __call__(self, embeddables: Iterable[Embeddable], **kwargs) -> Any: ...


def embeddable_to_conds(func: SupportsConds):
    """
    Wraps function that doesn't need embeddable functionality to be compatible with the Embeddable interface.

    Args:
        func: function to wrap

    Returns:
        wrapped function
    """

    @wraps(func)
    @legend_corrector
    def wrapper(
        embeddables: Iterable[Embeddable],
        **kwargs,
    ) -> Any:
        conds = {embeddable.name: embeddable.value for embeddable in embeddables}
        return func(conds=conds, **kwargs)

    return wrapper


def fig_init(ax: plt.Axes = None, **kwargs) -> tuple[plt.Figure, plt.Axes]:
    """
    Initialize a figure and axes. If `ax` is provided, return the figure and axes.
    """
    if ax is None:
        return plt.subplots(**kwargs)
    return ax.get_figure(), ax


def get_figure_from_axes(
    axes: plt.Axes | Iterable[plt.Axes],
) -> set[plt.Figure]:
    """
    Get the figure(s) from the axes.
    """
    figs = set()
    for ax in flatten(axes):
        figs.add(ax.figure)
    return figs


def titler(
    title: str = None, conds: dict[str, Any] = None, title_override: str = None
) -> str:
    """Creates title for plot, customizing it according to `conds`.

    Args:
        title: Main title for plot, concatenated with `conds` if specified.
        conds: Dictionary mapping keys to values that were relevant for generating the plot.
        title_override: Override title for plot, ignoring conds. Useful when embedded inside functions that take `conds` as input but you want to override the title.

    Returns:
        str: Title for plot.
    """
    if title_override is not None:
        return title_override
    if title is None:
        return None
    if conds is None or len(conds) == 0:
        return title

    # Customize title based on `conds`
    conds_str = ", ".join([k + " = " + str(v) for k, v in conds.items()])
    if len(title) > 0:
        return title + "\n" + conds_str
    return conds_str


def unitler(label: str, unit: str) -> str:
    """Adds unit to label if specified."""
    if unit is None:
        return label
    return label + " (" + unit + ")"


def format_yticks(axes: Iterable[plt.Axes], func: Callable) -> None:
    """Formats y-axis ticks using a function."""
    for ax in flatten(axes):
        ax.yaxis.set_major_formatter(FuncFormatter(func))


def get_bar_positions(
    ax: plt.Axes, hue_order: list = None, x_centers: ArrayLike = None
):
    """Gets the positions of the bars in the plot."""
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


def get_bar_heights(ax: plt.Axes, hue_order: list = None, x_centers: ArrayLike = None):
    """Gets the heights of the bars in the plot."""
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


def palette_corrector(palette: dict | list, categories: list) -> dict | list:
    """Corrects for any mismatch between `palette` and observed `categories`. If `palette` is list, return first `len(categories)` entries."""
    return (
        {k: v for k, v in palette.items() if k in categories}
        if type(palette) == dict
        else palette[: len(categories)]
    )


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


def legend_corrector(
    _func: Callable = None, loc: str = "upper left", bbox: tuple = (1, 1)
):
    """
    A decorator to set the legend location to 'upper left' and bbox_to_anchor to (1, 1).

    Args:
        func: The plotting function to wrap.
        loc: Location of the legend.
        bbox: Bounding box of the legend.
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


def update_legend(ax: plt.Axes, elements: Iterable[Artist]):
    """
    Update the legend of an axis with new elements.

    Args:
        ax: Axis to update the legend of.
        elements: Iterable of elements to add to the legend.
    """
    # Get existing legend handles and labels
    existing_legend = ax.get_legend()
    if existing_legend is not None:
        existing_handles = existing_legend.legend_handles
        existing_labels = [text.get_text() for text in existing_legend.get_texts()]

        # Combine existing and new handles/labels
        all_handles = existing_handles + elements
        all_labels = existing_labels + [e.get_label() for e in elements]

        # Remove existing legend and add combined one
        existing_legend.remove()
        ax.legend(handles=all_handles, labels=all_labels)
    else:
        # No existing legend, just add the new one
        ax.legend(handles=elements)


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


def _figure_saver(
    fig: plt.Figure,
    ax: plt.Axes,
    figure_path: str,
    bbox_inches: str = "tight",
    facecolor: str = "white",
):
    """
    Save figure and clear it for later reuse

    Args:
        fig: figure to be drawn on
        ax: axis object to do drawing
        figure_path: path to save figure
        bbox_inches: bbox_inches argument for plt.savefig
        facecolor: facecolor argument for plt.savefig
    """
    Path(figure_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, facecolor=facecolor, bbox_inches=bbox_inches)
    [x.clear() for x in flatten(ax)]


def across_conditions_plotter(
    cond_name: str,
    conditions: Iterable[Any],
    plot_func: SupportsEmbeddables,
    cond_kwargs: dict = None,
    embeddables: Iterable[Embeddable] = None,
    **kwargs,
) -> Iterable[Any]:
    """
    Generates plots across each condition

    Args:
        cond_name: name of the condition
        conditions: iterable of conditions
        plot_func: plotting function
        cond_kwargs: dictionary of keyword arguments to `plot_func` for each condition, where each key is a condition. Each condition-specific kwargs is merged with `kwargs`.
        embeddables: Pre-existing iterable of embeddables. If None, then a new iterable of embeddables is created.
        **kwargs: keyword arguments to `plot_func`
            - if a dictionary containing conditions as keys, then the value of each key is a dictionary of keyword arguments to `plot_func`

    Returns:
        list of any returned output from `plot_func`
    """
    if cond_kwargs is None:
        cond_kwargs = {}
    returns = []
    for i, cond in enumerate(conditions):
        embeddable = Embeddable(cond_name, cond, i)

        # Create new embeddables if none are provided, or update existing embeddables
        if embeddables is None:
            embeddables = [embeddable]
        else:
            embeddables.append(embeddable)

        if cond in cond_kwargs:
            ret = plot_func(embeddables=embeddables, **(cond_kwargs[cond] | kwargs))
        else:
            ret = plot_func(embeddables=embeddables, **kwargs)
        returns.append(ret)
    return returns


class BasePlotter:
    """Base class for all plotters."""

    def __init__(self, experiment: Experiment, config: dict | Iterable):
        self.experiment = experiment
        self.config = SuperDict(config)

    def get_config_value(self, key: str, default: Any = None) -> Any:
        return self.config.get(key.upper(), default)

    def _init_vars(self, **kwargs):
        """Initialize variables by populating with default values."""
        if "dataset" in kwargs and kwargs["dataset"] is None:
            kwargs["dataset"] = self.experiment
        if "conds" in kwargs and kwargs["conds"] is None:
            kwargs["conds"] = {}
        for key, value in self.config.items():
            key = key.lower()
            if key in kwargs and kwargs[key] is None:
                kwargs[key] = value
        return kwargs

    @legend_corrector
    @multiplot
    def _plot_conditions_grid(
        self,
        plot_func: SupportsEmbeddables,
        row_condition: str,
        col_condition: str,
        cond_kwargs: dict = None,
        fig_title: str = None,
        row_is_figure: bool = True,
        **kwargs,
    ) -> Iterable[Any]:
        """
        Apply a plotting function over a grid of conditions.

        One figure is created per row condition. Within each figure, there will be
        a row of subplots for each column condition.

        Args:
            plot_func: Function that draws into a 1D array of axes.
            row_conditions: Conditions enumerating figures.
            col_conditions: Conditions enumerating subplot rows within each figure.
            cond_kwargs: Dictionary of keyword arguments to `plot_func` for each condition, where each key is a condition. Each condition-specific kwargs is merged with `kwargs`.
            fig_title: Title for the figure.
            row_is_figure: If True, then each row is its own figure. If False, then each row is an axis in the same figure.
            **kwargs: Keyword arguments passed to seaborn. May also contain nested kwargs.
                - 'fig_kwargs': Dictionary to specify figure properties when creating a new figure (passed to `plt.subplots`).

        Returns:
            List of any returned output from `plot_func`.
        """
        kwargs = deepcopy(kwargs)
        if cond_kwargs is None:
            cond_kwargs = {}
        row_conditions = self.experiment.get_unique(row_condition)

        # Assume fixed number of columns
        if not row_is_figure:
            n_cols = len(self.experiment.get_unique(col_condition))
            n_rows = len(row_conditions)
            fig, axes = fig_init(
                **kwargs_handler(
                    kwargs, "fig_kwargs", {"ncols": n_cols, "nrows": n_rows}
                )
            )

        def _plot_row(embeddable: Embeddable, **kwargs):
            kwargs = deepcopy(kwargs)
            row_data = self.experiment.filter({row_condition: embeddable.value})
            col_conditions = row_data.get_unique(col_condition)
            if row_is_figure:
                fig, axes = fig_init(
                    **kwargs_handler(
                        kwargs, "fig_kwargs", {"ncols": len(col_conditions)}
                    )
                )
                embeddables = None
            else:
                axes = axes[embeddable.index]
                embeddables = [embeddable]

            cond_kwargs.update(
                {
                    col: {"legend": i == len(col_conditions) - 1, "ax": axes[i]}
                    for i, col in enumerate(col_conditions)
                }
            )

            # Plot each column
            res = across_conditions_plotter(
                col_condition,
                col_conditions,
                plot_func,
                cond_kwargs=cond_kwargs,
                dataset=row_data,
                embeddables=embeddables,
                **kwargs,
            )

            # Set row title
            if row_is_figure and fig_title is not None:
                fig.suptitle(f"{fig_title} for {row_condition}={embeddable.value}")
                fig.tight_layout()
            return res

        # Plot each row
        res = across_conditions_plotter(
            row_condition, row_conditions, _plot_row, cond_kwargs=cond_kwargs, **kwargs
        )
        if not row_is_figure:
            fig.tight_layout()
            if fig_title is not None:
                fig.suptitle(fig_title)
        return res

    def plot_quantity_across_block(
        self,
        x: str,
        y: str,
        y_name: str = None,
        x_name: str = None,
        dataset: Experiment = None,
        conds: dict[str, Any] = None,
        by_box: bool = False,
        show_traces: bool = False,
        auxiliary_plot: callable = None,
        **kwargs,
    ) -> plt.Axes:
        """Plot the quantity across the block.

        Args:
            x: Name of x variable in DataFrame.
            y: Name of y variable in DataFrame.
            y_name: Name of y variable in DataFrame. Defaults to  "mean" + y.
            x_name: Name of x variable in DataFrame. Defaults to "time".
            dataset: Experiment dataset. Defaults to self.experiment.
            conds: Dictionary to filter df.
            by_box: If True, then plot the quantity by the boxes.
            show_traces: If True, then show the traces.
            auxiliary_plot: Function to plot the auxiliary plot. Defaults to None.
            **kwargs: Additional keyword arguments.
                - smooth_kwargs: Dictionary to specify window properties for smoothing the reward rate (passed to `moving_average`).
                - additional keyword arguments get passed to `plot_average_or_traces`.

        Returns:
            The axes.
        """

        kwargs = deepcopy(kwargs)
        dataset = self._init_vars(dataset=dataset)
        data = dataset.filter(conds).df

        if y_name is None:
            y_name = "mean " + y

        if x_name is None:
            x_name = "time"

        # Average data over time
        groupers = ["block_id"]
        if by_box:
            groupers.append("box")

        # Plot reward rates
        kwargs.update({"x": x_name, "y": y_name, "x_unit": "s"})

        if by_box:
            kwargs.update({"hue": "box", "palette": self.get_config_value("palette")})
        else:
            kwargs.update({"color": "black"})

        smooth_kwargs = kwargs_handler(
            kwargs,
            "smooth_kwargs",
        )
        ma = moving_average(
            data,
            x=x,
            y=y,
            y_name=y_name,
            x_name=x_name,
            groupers=groupers,
            **smooth_kwargs,
        )
        if auxiliary_plot:
            auxiliary_plot(ma, **kwargs)
        return plot_average_or_traces(
            ma, show_traces=show_traces, conds=conds, **kwargs
        )

    @legend_corrector
    def plot_block_events(
        self,
        dataset: Experiment = None,
        conds: dict[str, Any] = None,
        x: str = "push times",
        y: str = "box position",
        x_unit: str = "s",
        y_unit: str = None,
        title: str = "Block activity",
        palette: dict[str, Any] = None,
        legend: bool = True,
        ax: plt.Axes = None,
        **kwargs,
    ) -> plt.Axes:
        """
        Plot the push-related variable in the block.

        Args:
            dataset: Experiment dataset.
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
        kwargs = deepcopy(kwargs)
        dataset, palette = self._init_vars(dataset=dataset, palette=palette)

        schedules = dataset.get_unique("assigned schedules")[0]

        # Create ax if none provided
        fig_kwargs = kwargs_handler(kwargs, "fig_kwargs")
        fig, ax = fig_init(ax, **fig_kwargs)

        # kappa = df_block.index.unique("kappa")
        # stim_type = df_block.index.unique("stimulus type")
        # shape = df_block.index.unique("shape")

        # if conds is None:
        #     conds = {}
        # else:
        #     conds = deepcopy(conds)
        # conds["kappa"] = kappa[0]
        # conds["stim type"] = stim_type[0]
        # conds["shape"] = shape[0]

        # Create switch segments (x, y) pairs for LineCollection
        x_vals = dataset.get(x)
        y_vals = dataset.get(y)
        colors = np.array(["black"] * (len(y_vals) - 1))
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
        colors = np.array([palette[i] for i in dataset.get("box")])
        mask = dataset.get("reward outcomes")
        ax.scatter(
            x_vals[mask], y_vals[mask], c=colors[mask], marker="^", s=80, zorder=2
        )
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
                kwargs,
                "legend_kwargs",
                {"loc": "upper left", "bbox_to_anchor": (1.05, 1)},
            )
            palette = palette_corrector(palette, dataset.get_unique("box"))
            legend_elements = [
                Line2D([0], [0], color=palette[j], linestyle="-", label=schedules[i])
                for i, j in enumerate(palette.keys())
            ] + [
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
            ax.legend(handles=legend_elements, **legend_kwargs)
        return ax


# TODO: these need to be refactored and moved inside a Plotter class later
def per_block(
    func: callable,
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
    func: callable,
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
    @legend_corrector
    def wrapper(
        df: pd.DataFrame = None,
        x: str = None,
        hue: str = None,
        palette: list = None,
        conds: dict = None,
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

        # Correct color settings
        if hue and palette:
            hue_keys = (
                sorted(df[hue].unique()) if hue in df.columns else df.index.unique(hue)
            )
            palette = palette_corrector(palette, hue_keys)

            # Control hue order based on inputs
            if type(palette) is dict:
                kwargs["hue_order"] = list(palette.keys())
            else:
                kwargs["hue_order"] = hue_keys

        # Create ax if none
        _, ax = fig_init(ax, **kwargs_handler(kwargs, "fig_kwargs"))

        # Pop any last keyword args not needed for seaborn here before running function
        legend_kwargs = kwargs_handler(kwargs, "legend_kwargs", dict(title=hue))
        title_kwargs = kwargs_handler(kwargs, "title_kwargs")
        xlabel_kwargs = kwargs_handler(kwargs, "xlabel_kwargs")
        ylabel_kwargs = kwargs_handler(kwargs, "ylabel_kwargs")

        # Run function, assuming seaborn plotting func
        if min_obs:

            def _filter_obs(df: pd.DataFrame) -> pd.DataFrame:
                return df.filter(lambda g: len(g) >= min_obs)

            if hue:
                ret = func(
                    _filter_obs(df.groupby([x, hue], observed=True, as_index=False)),
                    x=x,
                    hue=hue,
                    palette=palette,
                    ax=ax,
                    legend=legend,
                    **kwargs,
                )
            else:
                ret = func(
                    _filter_obs(df.groupby(x, observed=True, as_index=False)),
                    x=x,
                    ax=ax,
                    legend=legend,
                    **kwargs,
                )
        else:
            ret = func(
                df, x=x, hue=hue, palette=palette, ax=ax, legend=legend, **kwargs
            )

        # Set title (if multiple axes, this does the first one)
        title = titler(title=title, conds=conds, title_override=title_override)
        _ax = np.atleast_1d(ax)[0]
        if title:
            _ax.set_title(title, **title_kwargs)

        # Set units if specified
        if x_unit:
            _ax.set_xlabel(unitler(_ax.get_xlabel(), x_unit), **xlabel_kwargs)
        else:
            _ax.set_xlabel(_ax.get_xlabel(), **xlabel_kwargs)

        if y_unit:
            _ax.set_ylabel(unitler(_ax.get_ylabel(), y_unit), **ylabel_kwargs)
        else:
            _ax.set_ylabel(_ax.get_ylabel(), **ylabel_kwargs)

        # Modify legend
        if legend:
            for _ax in flatten(ax):
                if not _ax.get_legend_handles_labels() == (
                    [],
                    [],
                ):  # If there is a legend, modify it
                    _ax.legend(**legend_kwargs)

        if ret is None:
            return ax
        return ret  # Assume there is usually an ax in here

    return wrapper


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


def plot_average_or_traces(
    df: pd.DataFrame,
    show_traces: bool = False,
    show_average: bool = True,
    units: str = "block_id",
    alpha: float = 0.1,
    linewidth: float = 5,
    **kwargs,
) -> plt.Axes:
    kwargs = kwargs.copy()
    if show_traces:
        legend_flag = True
        if "legend" in kwargs:
            legend_flag = kwargs["legend"]
        kwargs["legend"] = False

        # Traces
        ax = bp(sns.lineplot)(
            df,
            **kwargs,
            units=units,
            estimator=None,
            errorbar=None,
            alpha=alpha,
        )

        if "x_unit" in kwargs:
            kwargs.pop("x_unit")
        if "y_unit" in kwargs:
            kwargs.pop("y_unit")
        kwargs["legend"] = legend_flag
        kwargs["ax"] = ax
        legend_content = [
            Line2D(
                [0],
                [0],
                color="black",
                alpha=alpha,
                linewidth=1,
                label="individual blocks",
            ),
        ]
        if show_average:
            legend_content.append(
                Line2D(
                    [0],
                    [0],
                    color="black",
                    linewidth=linewidth,
                    label="average over blocks",
                ),
            )

        # Average
        if show_average:
            res = bp(sns.lineplot)(df, **kwargs, errorbar=None, lw=linewidth)

        # Update legend to include traces and average
        if legend_flag:
            update_legend(
                ax,
                legend_content,
            )

    else:
        res = bp(sns.lineplot)(df, **kwargs)

    return res


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


# legacy function, will be removed later
def plot_variable_subplots(
    df: pd.DataFrame,
    func: callable,
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
        palette: Palette to be used for the hue.
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


# TODO: massively redo this
def toggle_plot(
    plot_func1: callable,
    plot_func2: callable,
    kwargs1: dict = None,
    kwargs2: dict = None,
    default_plot: int = 1,
    button_labels: tuple = ("Show Plot 1", "Show Plot 2"),
    inline: bool = False,
    cache_plots: bool = False,
):
    """
    Create a toggle widget to switch between two plotting functions, or render as inline figure.

    Parameters
    ----------
    plot_func1 : callable
        First plotting function to display (should return axes object or list of axes)
    plot_func2 : callable
        Second plotting function to display (should return axes object or list of axes)
    kwargs1 : dict, optional
        Keyword arguments for the first plotting function
    kwargs2 : dict, optional
        Keyword arguments for the second plotting function
    default_plot : int, default=1
        Which plot to show by default (1 or 2)
    button_labels : tuple of str, default=("Show Plot 1", "Show Plot 2")
        Labels for the toggle button (label alternates between these two)
    inline : bool, default=False
        If True, renders the default plot as a normal inline figure without widget behavior.
        Useful for converting notebooks to static HTML.
    cache_plots : bool, default=False
        #TODO: fix this
        If True, cache the rendered figures and redisplay them instead of re-rendering.
        This improves performance and ensures consistent ordering.

    Returns
    -------
    widget or axes
        Jupyter widget with toggle functionality if inline=False,
        or axes object if inline=True

    Examples
    --------
    >>> def plot1(data, color='blue'):
    ...     fig, ax = plt.subplots()
    ...     ax.scatter(data['x'], data['y'], color=color)
    ...     ax.set_title('Plot 1')
    ...     return ax
    >>> def plot2(data, bins=20):
    ...     fig, ax = plt.subplots()
    ...     ax.hist(data['x'], bins=bins)
    ...     ax.set_title('Plot 2')
    ...     return ax
    >>> # Interactive widget
    >>> toggle_plot(plot1, plot2,
    ...             kwargs1={'data': df, 'color': 'red'},
    ...             kwargs2={'data': df, 'bins': 30},
    ...             button_labels=("Show Plot 1", "Show Plot 2"))
    >>> # Static inline figure
    >>> toggle_plot(plot1, plot2,
    ...             kwargs1={'data': df, 'color': 'red'},
    ...             kwargs2={'data': df, 'bins': 30},
    ...             inline=True)
    """

    # TODO: best strategy is to keep the figure and cache the axes objects, swapping out the axes objects on the same figure objects
    kwargs1 = kwargs1 or {}
    kwargs2 = kwargs2 or {}

    # If inline mode is requested, just return the default plot as a normal figure
    if inline:
        if default_plot == 1:
            plot_func1(**kwargs1)
        else:
            plot_func2(**kwargs2)
        return

    # Otherwise, create the interactive widget
    output = widgets.Output()
    current_plot = 1 if default_plot == 1 else 2

    # Cache for storing display outputs
    cached_outputs = {1: None, 2: None}

    def render_and_cache_plot(plot_func, kwargs, plot_num):
        """Render a plot and cache the display output"""
        # Create a temporary output to capture the display
        temp_output = widgets.Output()
        with temp_output:
            axes = plot_func(**kwargs)
            figs = get_figure_from_axes(axes)
            if figs is not None:
                for fig in flatten(figs):
                    display(fig)
                    plt.close(fig)

        # Store the captured output
        cached_outputs[plot_num] = temp_output
        return axes

    def show_cached_plot(plot_num):
        """Display cached output without re-rendering"""
        with output:
            output.clear_output(wait=True)
            if cached_outputs[plot_num] is not None:
                # Display the cached output
                display(cached_outputs[plot_num])
            else:
                # Fallback: render if not cached
                if plot_num == 1:
                    render_and_cache_plot(plot_func1, kwargs1, 1)
                else:
                    render_and_cache_plot(plot_func2, kwargs2, 2)

    def show_plot(plot_func, kwargs, plot_num):
        """Show plot with optional caching"""
        if cache_plots:
            show_cached_plot(plot_num)
        else:
            # Original behavior: render fresh each time
            with output:
                output.clear_output(wait=True)
                axes = plot_func(**kwargs)
                figs = get_figure_from_axes(axes)
                if figs is not None:
                    for fig in flatten(figs):
                        display(fig)
                        plt.close(fig)
                return axes

    # Set initial button label
    if not button_labels or len(button_labels) != 2:
        button_labels = ("Show Plot 1", "Show Plot 2")
    label_map = {1: button_labels[0], 2: button_labels[1]}

    toggle_button = widgets.Button(
        description=label_map[current_plot],
        disabled=False,
        button_style="info",
        tooltip="Click to switch between plots",
        icon="refresh",
    )

    def on_button_click(b):
        nonlocal current_plot
        if current_plot == 1:
            show_plot(plot_func2, kwargs2, 2)
            current_plot = 2
        else:
            show_plot(plot_func1, kwargs1, 1)
            current_plot = 1
        toggle_button.description = label_map[current_plot]

    toggle_button.on_click(on_button_click)
    container = widgets.VBox([toggle_button, output])

    # Initial plot
    if current_plot == 1:
        show_plot(plot_func1, kwargs1, 1)
    else:
        show_plot(plot_func2, kwargs2, 2)

    return container
