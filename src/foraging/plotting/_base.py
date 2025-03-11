import gc
import logging
import os
from functools import wraps
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from kneed import KneeLocator
from matplotlib import pyplot as plt
from numpy.typing import ArrayLike
from scipy.optimize import curve_fit
from scipy.spatial.distance import euclidean
from tqdm import tqdm

from foraging import utils
import utils.data
from foraging.plotting import BOX_COLORS, BOX_LABELS

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

## wrappers
def bp(func):
    """
    Args:
        func: function to wrap

    Returns:
        wrapped function
    """
    @wraps(func)
    def wrapper(df: pd.DataFrame = None, conds: dict = None, collapse: bool = False, palette: list = None, box_colors: list = BOX_COLORS, box_labels: list = BOX_LABELS, title: str = None, title_prefix: str = '', ax: plt.Axes = None, **kwargs) -> Any:
        """
        Convenience decorator that customizes figure in formulaic fashion

        Args:
            collapse:
            df: dataframe of block(s) data
            conds: dictionary mapping level keys to values to be used to filter dataframe. Necessary for setting title
            box_colors: list of colors for each box
            box_labels: list of labels for each box
            ax: axis to plot on (not none if reusing premade figure and axis object)
            **kwargs: keyword arguments to be passed to func

        Returns:
            ax, or optional return arguments from wrapped function usually in the form of ax + extra
        """

        # filter df
        if conds is None:
            conds = {}
        df = utils.data.filter_df(df, conds)

        # context dependent plot settings
        if collapse:
            hue = 'box rank'
            hue_order = range(len(box_colors))
        else:
            schedules = -np.sort(-df['schedule'].unique())
            kappa = df.index.unique('kappa')
            stim_type = df.index.unique('stimulus type')
            shape = df.index.unique('shape')
            if len(kappa) > 1 or len(stim_type) > 1 or len(shape) > 1:
                logger.debug(f"length of kappa: {len(kappa)}, length of stim_type: {len(stim_type)}, length of shape: {len(shape)}")
                raise Exception("Multiple experiment parameters found for single block. Make sure only single block is being supplied, or set collapse to True.")

            # for titling purposes
            conds['kappa'] = kappa[0]
            conds['stim type'] = stim_type[0]
            conds['shape'] = shape[0]

            hue = 'schedule'
            hue_order = schedules
            box_labels = schedules

        palette = list(box_colors) if not palette else palette

        # if plotting kappa on x-axis, create dummy column in order to plot kappa data evenly
        if kwargs.get('x', False) == 'kappa':
            df['stimulus reliability'] = pd.Series(df['kappa'].rank(method ='dense') - 1, index = df.index)
            kwargs['x'] = 'stimulus reliability'

        # Create ax if none
        if ax is None:
            _, ax = plt.subplots(**kwargs.pop('fig_kwargs', {}))

        # Pop any last keyword args not needed for seaborn here before running function
        legend_kwargs = kwargs.pop('legend_kwargs', {
            'loc': 'upper right'
        })

        # run function, assuming seaborn plotting func
        ret = func(df, ax=ax, hue=hue, hue_order=hue_order, palette=palette, **kwargs)

        # adjust xticks to only show actual data
        if kwargs.get('x', False) == 'stimulus reliability':
            xticks = df.index.unique('kappa')
            [_ax.set_xticks(range(len(xticks)), xticks) for _ax in utils.flatten(ax)]

        # set title (if multiple axes, this does the first one)
        title_str = title_prefix + '\n' + ', '.join([k+' = '+str(v) for k,v in conds.items()]) if not title else title
        _ax = np.atleast_1d(ax)
        _ax[0].set_title(title_str)
        fig = _ax[0].get_figure()
        fig.tight_layout()

        # modify legend
        if kwargs.pop('legend', True):
            for _ax in utils.flatten(ax):
                try:
                    legend = _ax.get_legend()
                    legend.set_title('schedule')
                    handles = legend.legend_handles
                    # [text.set_text(box_labels[i]) for i, text in enumerate(legend.get_texts())]
                    _ax.legend(handles, box_labels, **legend_kwargs)
                    # try:
                    #     # markerscale = legend_kwargs.pop('markerscale', 20)
                    #     # [handle.set_markersize(markerscale) for handle in legend.legendHandles]
                    #     pass
                    # except:
                    #     pass
                except Exception as e:
                    print(e)
                    _ax.legend(box_labels, **legend_kwargs)
        if ret is None:
            return ax
        return ret
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

def _figure_saver(fig: plt.Figure, ax: plt.Axes, plot_dir: str, save_prefix: str, conds: dict):
    """
    Routine for saving figure and clearing it for later reuse

    Args:
        fig: figure to be drawn on
        ax: axis object to do drawing
        plot_dir: folder for figures to be created in
        save_prefix: prefix to uniquely identify plots produced by func
        conds: dictionary mapping level keys to values that was used to filter dataframe. Can be None but must be specified to properly name file

    Returns:

    """

    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    if conds:
        fname = ','.join([k+'='+str(v) for k,v in conds.items()])
    else:
        fname = ''
    save_path = os.path.join(plot_dir, save_prefix + fname + ".png")
    fig.savefig(save_path)
    [x.clear() for x in utils.flatten(ax)]

def _per_block(func, df: pd.DataFrame, plot_dir: str, save_prefix: str, fig_kwargs, conds: dict = None, **kwargs):
    """
    Generic wrapper for creating figures for a single block

    Args:
        func: the actual plotting routine to be executed on each block
        df: dataframe of experiment data
        plot_dir: folder for figures to be created in
        save_prefix: prefix to uniquely identify plots produced by func
        fig_kwargs: dictionary of keyword arguments for plt.subplots
        conds: dictionary mapping level keys to values to be used to filter dataframe
        **kwargs: keyword arguments for func

    Returns:

    """
    @_figure_handler(**fig_kwargs)
    def _inner(fig, ax):
        nonlocal conds
        filtered_df = utils.data.filter_df(df, conds)
        for subject in tqdm(filtered_df.index.unique('subject')):
            for sess_num in tqdm(filtered_df.xs(subject, level = 'subject').index.unique('session')):
                for block_num in filtered_df.xs((subject, sess_num), level = ('subject','session')).index.unique('block'):
                    conds = {'subject': subject, 'session': sess_num, 'block': block_num}
                    try:
                        func(df = df, conds = conds, ax = ax, **kwargs)
                    except:
                        logger.debug(f"could not plot subject {subject} session {sess_num} block {block_num}")
                        continue
                    _figure_saver(fig, ax, plot_dir, save_prefix, conds)
    _inner()

#todo: generalize iter_level to multiple levels listed in hierarchical order head first
def _across_blocks(func, df: pd.DataFrame, plot_dir: str, save_prefix: str, fig_kwargs, conds: dict = None, iter_level: str = None, **kwargs):
    """
    Generic wrapper for creating figures that summarize across multiple blocks grouped by conditions

    Args:
        func: the actual plotting routine to be executed on each session
        df: dataframe of experiment data
        save_prefix: prefix to uniquely identify plots produced by func
        plot_dir: folder for figures to be created in
        conds: dictionary mapping level keys to values to be used to filter dataframe
        fig_kwargs: dictionary of keyword arguments for plt.subplots
        iter_level: index level of dataframe across which plots will be generated conditioned on each value of this level
        **kwargs: keyword arguments for func

    Returns:

    """

    @_figure_handler(**fig_kwargs)
    def _inner(fig, ax):
        nonlocal conds
        filtered_df = utils.data.filter_df(df, conds)
        if iter_level is None:
            func(block_df = filtered_df, conds = conds, ax = ax, collapse = True, **kwargs)
            _figure_saver(fig, ax, plot_dir, save_prefix, conds)
        else:
            if conds is None:
                conds = {}
            for v in tqdm(filtered_df.index.unique(iter_level)):
                conds[iter_level] = v
                func(df = filtered_df, conds = conds, ax = ax, collapse = True, **kwargs)
                _figure_saver(fig, ax, plot_dir, save_prefix, conds)
    _inner()

## common routines
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
        _, ax = plt.subplots(**kwargs.pop('fig_kwargs', {}))

    # Add constant to the x (for intercept in the regression)
    x_fit = sm.add_constant(x)

    # Fit the regression model
    fit_results = sm.OLS(y, x_fit).fit()

    # Generate predicted values over the range of x
    eval_x = sm.add_constant(np.linspace(np.min(x), np.max(x), n_pts))
    pred = fit_results.get_prediction(eval_x)

    # Draw the fit line and confidence interval
    ci_kws = kwargs.pop('ci_kwargs', None)
    if ci_kws:
        ax.fill_between(
            eval_x[:, 1],
            pred.predicted_mean - n_std * pred.se_mean,
            pred.predicted_mean + n_std * pred.se_mean,
            alpha=0.5,
            **ci_kws,
        )

    # Plot the regression line
    line_kwargs = kwargs.pop('line_kwargs', {})
    h = ax.plot(eval_x[:, 1], pred.predicted_mean, **line_kwargs)

    # Plot the scatter plot of the data
    scatter_kws = kwargs.pop('scatter_kwargs', None)
    if scatter_kws:
        ax.scatter(x, y, c=h[0].get_color(), **scatter_kws)

    return fit_results

def plot_elbow(x: ArrayLike, y: ArrayLike, fit=False, func=None, method='default', curve='concave', n_pts=1000, ax: plt.Axes = None, **kwargs):
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
        _, ax = plt.subplots(**kwargs.pop('fig_kwargs', {}))
    ax.axvline(x_elbow, **kwargs)

    return x_elbow, y_elbow, k_fit, ax
