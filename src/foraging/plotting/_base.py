import gc
import logging
import os
from copy import deepcopy
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterable

import math
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from kneed import KneeLocator
from matplotlib import pyplot as plt
from numpy.typing import ArrayLike
from scipy.optimize import curve_fit
from scipy.spatial.distance import euclidean
from tqdm import tqdm
from matplotlib.ticker import FuncFormatter

from foraging import utils
from foraging.utils import BOX_LABELS, flatten, kwargs_handler
from foraging.utils.data import filter_df
from foraging.plotting import BOX_COLORS

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def fig_init(ax: plt.Axes = None, **kwargs):
    if ax is None:
        return plt.subplots(**kwargs)
    return ax.get_figure(), ax

def titler(title_prefix: str = None, title: str = None, conds: dict = None):
    if conds is None:
        conds = {}
    if title_prefix is None:
        title_prefix = ''
    if title is None:
        title = ''
    return title_prefix + '\n' + ', '.join([k + ' = ' + str(v) for k, v in conds.items()]) if len(title) == 0 else title

def unitler(label: str, unit: str):
    if unit is None:
        return label
    return label+ ' ('+unit+')'

def format_yticks(axes, func):
    for ax in utils.flatten(axes):
        ax.yaxis.set_major_formatter(FuncFormatter(func))

def get_bar_positions(ax: plt.Axes, hue_order: list = BOX_LABELS, x_centers: ArrayLike = None):
    bars = ax.patches

    # Group bar patches by hue group
    bar_width = bars[0].get_width()  # Width of one bar
    n_groups = len(hue_order)
    if x_centers is None:
        x_centers = np.arange(len(bars)//n_groups)

    # Organize bar positions by hue group
    positions_by_group = {group: [] for group in hue_order}
    for x_center in x_centers:
        for i, group in enumerate(hue_order):
            # Leftmost group's left edge is offset furthest left
            offset = (i - (n_groups - 1) / 2) * bar_width
            x_left = x_center + offset
            positions_by_group[group].append(x_left)
    return {k: np.array(v) for k,v in positions_by_group.items()}

def get_bar_heights(ax: plt.Axes, hue_order: list = BOX_LABELS, x_centers: ArrayLike = None):
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
        x_centers = np.arange(len(bars)//n_groups)
    heights_by_group = {group: [] for group in hue_order}
    for x_center in x_centers:
        for i, group in enumerate(hue_order):
            # Leftmost group's left edge is offset furthest left
            offset = (i - (n_groups - 1) / 2) * bar_width
            x_left = x_center + offset
            idx = np.argmin(np.abs(np.array(proto_pos) - x_left))
            if math.isclose(proto_pos[idx], x_left, abs_tol=1e-4):
                heights_by_group[group].append(proto_heights[idx])
    return {k: np.array(v) for k,v in heights_by_group.items()}


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
    def wrapper(df: pd.DataFrame = None, x: str = None, conds: dict = None, accumulate: bool = False, palette: list = None, box_colors: list = BOX_COLORS, box_labels: list = BOX_LABELS, title: str = None, title_prefix: str = '', x_unit: str = None, y_unit: str = None, min_obs: int = None, attempt_index: bool = True, ax: plt.Axes = None, **kwargs) -> Any:
        """
        Convenience decorator that customizes figure in formulaic fashion

        Args:
            df: DataFrame of block(s) data.
            x: Name of x variable to be plotted.
            conds: Dictionary mapping level keys to values to be used to filter DataFrame. Necessary for setting title.
            accumulate: True indicates df consists of multiple blocks whose data should be accumulated.
            palette: List of colors. If None, defaults to `box_colors`.
            box_colors: List of colors for each box.
            box_labels: List of labels for each box.
            title: Title for figure (overrides `title_prefix`).
            title_prefix: Prefix string that precedes the string template that enumerates conditions for this block(s).
            x_unit: Unit of the x-axis. If None, then ignored.
            y_unit: Unit of the y-axis. If None, then ignored.
            min_obs: Threshold for min number of observations a bin must have to be displayed. Only used if not None.
            attempt_index: Refer to `filter_df` for more details.
            ax: Axis to plot on (not None if reusing premade figure and axis object).
            **kwargs: keyword arguments
                - fig_kwargs: keyword arguments to be passed to `plt.subplots`.
                - legend_kwargs: keyword arguments to be passed to `Axes.legend`.
                - title_kwargs: keyword arguments to be passed to `Axes.set_title`.
                - xlabel_kwargs: keyword arguments to be passed to `Axes.set_xlabel`.
                - ylabel_kwargs: keyword arguments to be passed to `Axes.set_ylabel`.
                - additional keyword arguments get passed to wrapped function, which is meant to be a seaborn-style function.

        Returns:
            ax, or optional return arguments from wrapped function usually in the form of ax + extra
        """

        # Filter df
        if conds is None:
            conds = {}
        else:
            conds = deepcopy(conds)
        df = filter_df(df, conds, attempt_index=attempt_index)

        # Context dependent plot settings
        if accumulate: # If plotting multiple blocks at once
            hue = kwargs.pop('hue', 'box')
            hue_order = kwargs.pop('hue_order', box_labels)
        else: # If only plotting individual block
            schedules = np.sort(df['schedule'].unique())
            kappa = df.index.unique('kappa')
            stim_type = df.index.unique('stimulus type')
            shape = df.index.unique('shape')
            if len(kappa) > 1 or len(stim_type) > 1 or len(shape) > 1:
                logger.debug(f"length of kappa: {len(kappa)}, length of stim_type: {len(stim_type)}, length of shape: {len(shape)}")
                raise Exception("Multiple experiment parameters found for single block. Make sure only single block is being supplied, or set collapse to True.")

            # For titling purposes, add block metadata
            conds['kappa'] = kappa[0]
            conds['stim type'] = stim_type[0]
            conds['shape'] = shape[0]

            hue = kwargs.pop('hue','schedule')
            hue_order = kwargs.pop('hue_order', schedules)
            box_labels = schedules

        palette = list(box_colors) if not palette else palette

        # If plotting kappa on x-axis, create dummy column in order to plot kappa data evenly
        if x == 'kappa':
            df['stimulus reliability'] = pd.Series(df['kappa'].rank(method ='dense') - 1, index = df.index)
            x = 'stimulus reliability'

        # Create ax if none
        fig, ax = fig_init(ax, **kwargs_handler(kwargs, 'fig_kwargs'))

        # Pop any last keyword args not needed for seaborn here before running function
        legend_kwargs = kwargs_handler(kwargs, 'legend_kwargs', dict(loc='upper right', title = 'schedule'))
        title_kwargs = kwargs_handler(kwargs, 'title_kwargs')
        xlabel_kwargs = kwargs_handler(kwargs, 'xlabel_kwargs')
        ylabel_kwargs = kwargs_handler(kwargs, 'ylabel_kwargs')

        # Run function, assuming seaborn plotting func
        if min_obs:
            ret = func(df.groupby([x, hue], as_index=False).filter(lambda g: len(g) >= min_obs), x = x, ax=ax, hue=hue, hue_order=hue_order, palette=palette, **kwargs)
        else:
            ret = func(df, x = x, ax=ax, hue=hue, hue_order=hue_order, palette=palette, **kwargs)

        # Adjust xticks to only show actual data
        if x == 'stimulus reliability':
            xticks = df.index.unique('kappa')
            [_ax.set_xticks(range(len(xticks)), xticks) for _ax in flatten(ax)]

        # Set title (if multiple axes, this does the first one)
        title = titler(title= title, title_prefix= title_prefix, conds = conds)
        _ax = np.atleast_1d(ax)
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
        if kwargs.pop('legend', True):
            for _ax in utils.flatten(ax):
                try:
                    legend = _ax.get_legend()
                    handles = legend.legend_handles
                    _ax.legend(handles, box_labels, **legend_kwargs)
                except Exception as e:
                    print(e)
                    _ax.legend(box_labels, **legend_kwargs)
        fig.tight_layout()
        if ret is None:
            return ax
        return ret # Assume there is usually an ax in here
    return wrapper

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
    fig.savefig(figure_path, facecolor = 'white')
    [x.clear() for x in utils.flatten(ax)]

def per_block(func: Callable, df: pd.DataFrame, figure_dir: str, filename_prefix: str, conds: dict = None, fig_kwargs: dict = None, use_tqdm: bool = True, attempt_index: bool = True, by_subject: bool = False, **kwargs):
    """
    Generic wrapper for creating figures for a single block

    Args:
        func: the actual plotting routine to be executed on each block
        df: dataframe of experiment data
        figure_dir: folder for figures to be created in
        filename_prefix: Prepended to filename of each block's figure
        conds: dictionary mapping level keys to values to be used to filter dataframe
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
        filtered_df = filter_df(df, conds, attempt_index = attempt_index)
        for subject in tqdm(filtered_df.index.unique('subject'), disable=not use_tqdm):
            if by_subject:
                figure_dir = os.path.join(old_figure_dir, f'subject {subject}')
            for sess_num in filtered_df.xs(subject, level = 'subject').index.unique('session'):
                for block_num in filtered_df.xs((subject, sess_num), level = ('subject','session')).index.unique('block'):
                    conds = {'subject': subject, 'session': sess_num, 'block': block_num}
                    try:
                        func(df = df, conds = conds, ax = ax, **kwargs)
                    except Exception as e:
                        logger.debug(f"could not plot subject {subject} session {sess_num} block {block_num}")
                        logger.debug(e)
                        continue
                    figure_path = os.path.join(figure_dir, filename_prefix+','.join([k + '=' + str(v) for k,v in conds.items()]) + ".png")
                    _figure_saver(fig, ax, figure_path)
    _inner()

#todo: generalize iter_level to multiple levels listed in hierarchical order head first
def across_blocks(func: Callable, df: pd.DataFrame, figure_dir: str, filename_prefix: str, conds: dict = None, fig_kwargs: dict = None, iter_level: str = None, attempt_index: bool = True, **kwargs):
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
        filtered_df = filter_df(df, conds, attempt_index= attempt_index)
        if iter_level is None:
            func(block_df = filtered_df, conds = conds, ax = ax, collapse = True, **kwargs)
            figure_path = os.path.join(figure_dir, filename_prefix + ','.join(
                [k + '=' + str(v) for k, v in conds.items()]) + ".png")
            _figure_saver(fig, ax, figure_path)
        else:
            if conds is None:
                conds = {}
            for v in tqdm(filtered_df.index.unique(iter_level)):
                conds[iter_level] = v
                func(df = filtered_df, conds = conds, ax = ax, collapse = True, **kwargs)
                figure_path = os.path.join(figure_dir, filename_prefix + ','.join(
                    [k + '=' + str(v) for k, v in conds.items()]) + ".png")
                _figure_saver(fig, ax, figure_path)
    _inner()

## common routines
def enhanced_violinplot(df: pd.DataFrame, x: str, y: str, hue: str = None, hue_order = None, palette = None, ax: plt.Axes = None, **kwargs) -> plt.Axes:
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
    fig, ax = fig_init(ax, **kwargs.pop('fig_kwargs', {}))
    sns.violinplot(df, x=x, y=y, hue = hue, hue_order = hue_order, palette = palette, ax=ax, **kwargs)

    # Plot means and se overlaid on violinplot
    groupers = [x]
    if hue:
        groupers.append(hue)
    stats_df = df.groupby(groupers)[y].agg(
        ['mean', 'std', 'count']).reset_index()
    stats_df['se'] = stats_df['std'] / np.sqrt(stats_df['count'])

    n_subgroups = stats_df[hue].nunique() if hue else 1
    violin_width = 0.8 / n_subgroups

    # Plot means and error bars for each subgroup with connecting lines
    for group_idx, group in enumerate(stats_df[x].unique()):
        subgroup_stats = stats_df[stats_df[x] == group]

        # Calculate x-coordinates for the means
        # For each condition, we need to center the subgroup means over their respective violins
        x_positions = group_idx + (violin_width * (n_subgroups - 1) / 2) - np.arange(n_subgroups)[::-1] * (
                    violin_width * (n_subgroups - 1) / 2)  # + (violin_width * subgroup_idx)

        # Plot error bars and means
        ax.errorbar(x=x_positions,
                    y=subgroup_stats['mean'],
                    yerr=subgroup_stats['se'],
                    fmt='o',
                    color='black',
                    capsize=5,
                    capthick=2,
                    elinewidth=2,
                    markersize=8)

        # Add connecting lines within subgroup
        ax.plot(x_positions,
                subgroup_stats['mean'],
                color='black')
    return ax


# Credit to https://stackoverflow.com/questions/22852244/how-to-get-the-numerical-fitting-results-when-plotting-a-regression-in-seaborn
def regplot(
        x: ArrayLike, y: ArrayLike, n_std: float = 1.96, n_pts: int = 100, ax: plt.Axes = None, **kwargs):
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
        _, ax = plt.subplots(**kwargs_handler(kwargs, 'fig_kwargs'))

    # Add constant to the x (for intercept in the regression)
    x_fit = sm.add_constant(x)

    # Fit the regression model
    fit_results = sm.OLS(y, x_fit).fit()

    # Generate predicted values over the range of x
    eval_x = sm.add_constant(np.linspace(np.min(x), np.max(x), n_pts))
    pred = fit_results.get_prediction(eval_x)

    # Draw the fit line and confidence interval
    ci_kws = kwargs_handler(kwargs, 'ci_kwargs')
    if len(ci_kws) > 0:
        ax.fill_between(
            eval_x[:, 1],
            pred.predicted_mean - n_std * pred.se_mean,
            pred.predicted_mean + n_std * pred.se_mean,
            alpha=0.5,
            **ci_kws,
        )

    # Plot the regression line
    line_kwargs = kwargs_handler(kwargs, 'line_kwargs', dict(color='black'))
    h = ax.plot(eval_x[:, 1], pred.predicted_mean, **line_kwargs)

    # Plot the scatter plot of the data
    scatter_kws = kwargs_handler(kwargs, 'scatter_kwargs')
    if len(scatter_kws) > 0:
        ax.scatter(x, y, c=h[0].get_color(), **scatter_kws)

    return fit_results


def plot_elbow(x: ArrayLike, y: ArrayLike, fit: bool=False, func: Callable=None, method: str='default', curve='concave', n_pts=1000, ax: plt.Axes = None, **kwargs):
    """
    Plots the "elbow" of a curve, often used for determining the optimal number of clusters or points in algorithms like K-means.

    Args:
        x: The x-values (independent variable).
        y: The y-values (dependent variable).
        fit: If True, fits the data using the given function.
        func: The function to fit the data (required if `fit=True`).
        method: The method used to locate the elbow. Options are 'default' (using KneeLocator) or 'triangle' (using the triangle method).
        curve: The shape of the curve, either 'concave' or 'convex'.
        n_pts: Number of points to use for plotting the fitted curve (default is 1000).
        ax: The matplotlib axes to plot on (optional). If not provided, a new plot is created.
        **kwargs: Additional arguments passed to `ax.axvline()` for plotting the elbow line.

    Returns:
        x_elbow: The x-coordinate of the elbow.
        y_elbow: The y-coordinate of the elbow.
        k_fit: The fitting parameters (if `fit=True`).
        ax: The matplotlib axes object with the elbow plot.
    """
    x_elbow, y_elbow, k_fit = 0, 0, None

    # Fit func if specified
    if fit and func is not None:
        params, _ = curve_fit(func, x, y, p0=[1])  # Fit the function to the data
        k_fit = params[0]  # The fit parameter
        x_fit = np.linspace(0, max(x), n_pts)
        y_fit = func(x_fit, k_fit)
        x, y = x_fit, y_fit  # Use fitted data for further analysis

    # Default method (KneeLocator)
    if method == 'default':
        knee_locator = KneeLocator(x, y, curve=curve, S=3)
        x_elbow = knee_locator.knee
        y_elbow = knee_locator.knee_y

    # Triangle method
    if method == 'triangle':
        # Define the line between the first and last points
        p1, p2 = np.array([x[0], y[0]]), np.array([x[-1], y[-1]])

        # Compute the perpendicular distance of each point from the line
        distances = np.array([np.abs(np.cross(p2 - p1, p1 - np.array([x[i], y[i]]))) / euclidean(p1, p2) for i in range(len(x))])

        # Find the elbow (point with max distance)
        elbow_idx = np.argmax(distances)
        x_elbow = x[elbow_idx]
        y_elbow = y[elbow_idx]

    # Plot the elbow
    if ax is None:
        _, ax = plt.subplots(**kwargs_handler(kwargs, 'fig_kwargs'))
    ax.axvline(x_elbow, **kwargs)
    return x_elbow, y_elbow, k_fit, ax


def plot_variable_subplots(df: pd.DataFrame, func: Callable, row_cond: str, col_cond: str, axes: Iterable[plt.Axes] = None, legend: bool = True, savefig: str = None, **kwargs):
    def _group_size(g, group):
        if group in g.columns:
            return g[group].nunique()
        return len(g.index.unique(group))
    max_cols = df.groupby(row_cond).apply(lambda g: _group_size(g, col_cond)).max()
    num_rows = len(df.groupby(row_cond))  # Compute rows needed
    fig_kwargs = kwargs_handler(kwargs, 'fig_kwargs', dict(figsize=(5 * max_cols, 5 * num_rows), nrows=num_rows, ncols=max_cols))
    fig, axes = fig_init(axes, **fig_kwargs)
    axes = np.atleast_2d(axes)
    for i, row_val in enumerate(df.groupby(row_cond).groups.keys()):
        df_row = filter_df(df, {row_cond: row_val})
        # axes[i, 0].set_ylabel(row_val)
        for j, col_val in enumerate(df_row.groupby(col_cond).groups.keys()):
            conds = {col_cond: col_val}
            df_group = filter_df(df_row, conds)
            func(df_group, conds=conds, ax=axes[i,j], **kwargs)
            if j > 0:
                axes[i, j].set_ylabel("")
            # axes[i, j].set_title(f'{col_cond}={col_val}')
        for k in range(j + 1, max_cols):
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
        legend_kwargs = kwargs_handler(kwargs, 'legend_kwargs', dict(loc='upper right', bbox_to_anchor=(0.05, 0.05)))
        lgd = fig.legend(handles, labels, **legend_kwargs)
    fig.tight_layout()

    if savefig:
        if legend:
            fig.savefig(savefig, bbox_extra_artists=(lgd,), facecolor = 'white', bbox_inches='tight')
        else:
            fig.savefig(savefig, facecolor = 'white', bbox_inches='tight')

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


