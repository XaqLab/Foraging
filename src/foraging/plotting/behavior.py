import logging
from typing import Optional

import numpy as np
import pandas as pd
import seaborn as sns
import utils
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D


from foraging.plotting import BOX_COLORS, BOX_LABELS
from ._base import regplot, bp

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def plot_push_intervals(df: pd.DataFrame, x: str = 'push times', title: str = "Push intervals",
                        box_colors: list = BOX_COLORS, box_labels: list = BOX_LABELS,
                        legend: bool = True, ax: Optional[plt.Axes] = None, **kwargs) -> plt.Axes:
    """
    Plots the push intervals for each box, with different colors and line styles based on the stay/switch behavior,
    and displays reward outcomes with markers. Optionally adds a custom legend.

    Args:
        df: DataFrame containing session data with 'push times', 'consecutive push intervals', 'box rank',
            'stay/switch' (indicating whether the push was a switch or stay), and 'reward outcomes'.
        x: Column name to use for the x-axis (typically 'push times').
        title: Title of the plot.
        box_colors: List of colors to use for each box.
        box_labels: List of labels corresponding to each box for the legend.
        legend: If True, a custom legend is added to the plot.
        ax: Optional, existing matplotlib Axes object. If None, a new one will be created.
        kwargs: Additional keyword arguments.
            - 'fig_kwargs': Dictionary to specify figure properties when creating a new figure (passed to `plt.subplots`).
            - 'legend_kwargs': Dictionary of keyword arguments for customizing the legend (passed to `ax.legend`).

    Returns:
        ax: The Axes object with the plot.
    """
    df = df.reset_index()
    x_name = x
    x = df[x_name].values
    y = df['consecutive push intervals'].values
    colors = [box_colors[i] for i in df['box rank'].values]
    styles = ['dashed' if x else 'solid' for x in df['stay/switch'].values]

    # Create segments (x, y) pairs for LineCollection
    segments = [[(0, 0), (x[0], y[0])]] + [[(x[i], y[i]), (x[i + 1], y[i + 1])] for i in range(len(x) - 1)]

    # Create the LineCollection
    lc = LineCollection(segments, colors=colors, linestyles=styles, linewidth=2)

    # Create ax if none provided
    if ax is None:
        fig_kwargs = kwargs.pop('fig_kwargs', {})
        _, ax = plt.subplots(**fig_kwargs)
    ax.add_collection(lc)
    ax.autoscale()
    ax.grid(False)
    ax.set_title(title)
    ax.set_ylabel("Push intervals")
    ax.set_xlabel(x_name)

    # Add reward outcomes with green (rewarded) and red (not rewarded) markers
    mask = df['reward outcomes'] == True
    ax.scatter(x[mask], y[mask], c='g', marker='^')
    ax.scatter(x[~mask], y[~mask], c='r', marker='v')

    # Create legend manually with proxy artists
    if legend:
        legend_kwargs = kwargs.pop('legend_kwargs', {'loc': 'upper right'})

        legend_elements = ([Line2D([0], [0], color=box_colors[i], linestyle='-', label=box_labels[i]) for i in range(len(box_colors))]
                           + [Line2D([0], [0], color='black', linestyle='-', label='stay pushes'),
                              Line2D([0], [0], color='black', linestyle='--', label='switch times')]
                           + [Line2D([0], [0], color='green', linestyle='', marker='^', label='rewarded'),
                              Line2D([0], [0], color='red', linestyle='', marker='v', label='no reward')])

        ax.legend(handles=legend_elements, **legend_kwargs)
    return ax

def plot_push_intervals_vs_reward_intervals(df: pd.DataFrame, title_prefix: str = "Push intervals vs reward intervals", **kwargs) -> (plt.Axes, float, float):
    """
    Plot linear regression of push intervals against reward intervals in a block

    Args:
        df: dataframe of experiment data for a given block
        ax: axis to plot on (not none if reusing premade figure and axis object)
        **kwargs: keyword arguments for seaborn

    Returns:
        axes, r-squared and slope from regression
    """

    # Remove first push from each box, since reward time is messed up for first pushes
    df = df.copy()
    n_boxes = df['box'].nunique()
    [df.drop(df.loc[df['box'] == i + 1].index[0], inplace=True) for i in range(n_boxes)]
    ax = bp(sns.scatterplot)(df, x='reward intervals', y='push intervals', title_prefix = title_prefix, **kwargs)

    # Filter df here or inside regplot using conds
    conds = kwargs.pop('conds', None)
    df = utils.filter_df(df, conds)
    fit_results = regplot(df['reward intervals'].to_numpy(), df['push intervals'].to_numpy(), line_kws={'color': 'green'},
                          **kwargs)

    return ax, fit_results.rsquared, fit_results.params[1]

def plot_experiment_parameters(df: pd.DataFrame, conds: dict, title: str = "Experiment parameters by session",
                               ax: Optional[plt.Axes] = None, **kwargs) -> plt.Axes:
    """
    Plots the distribution of experiment parameters (kappa, stimulus type, shape) across different sessions.
    Displays the number of blocks associated with each parameter and session.

    Args:
        df: DataFrame containing experiment session data with hierarchical index ('session', 'stimulus type', 'shape', 'kappa').
        conds: Dictionary of conditions used to filter the DataFrame before plotting.
        title: Title of the plot.
        ax: Optional, existing matplotlib Axes object. If None, a new one will be created.
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
        fig_kwargs = kwargs.pop('fig_kwargs', {})
        _, ax = plt.subplots(**fig_kwargs)

    # Get all unique experiment parameters
    kappas = df.index.unique('kappa').sort_values()
    stim_types = df.index.unique('stimulus type').sort_values()
    shapes = df.index.unique('shape').sort_values()
    n_params = len(kappas) + len(stim_types) + len(shapes)

    # Filter df according to conditions
    df = utils.data.filter_df(df, conds)
    sessions = df.index.unique('session').sort_values()
    y_labels = [str(s) for s in shapes] + [str(s) for s in stim_types] + [str(k) for k in kappas]
    a, b, c = 2, 2, 1  # Constants to control spacing
    v_offset = 0.25  # Vertical offset for annotation alignment
    h_offset = 0.05  # Horizontal offset for annotation alignment
    shape_ticks = [0, 1 * a]
    stim_type_ticks = [2 * b, 3 * b]
    kappa_ticks = [i * c + max(stim_type_ticks) + 1 for i in range(1, len(kappas) + 1)]

    ax.scatter(np.ones(n_params + 1), np.arange(n_params + 1), alpha=0)  # Generate n_param + 1 ticks
    for i in sessions:
        # Count parameter values for the current session
        kappa_counts = df.xs(i, level='session').reset_index().groupby('kappa')['block'].nunique()
        stim_types_counts = df.xs(i, level='session').reset_index().groupby('stimulus type')['block'].nunique()
        shapes_counts = df.xs(i, level='session').reset_index().groupby('shape')['block'].nunique()

        # Determine y-coordinates for parameter value annotations
        y_kappas = np.searchsorted(kappas, kappa_counts.index.values)
        y_stim_types = np.searchsorted(stim_types, stim_types_counts.index.values) + len(shapes)
        y_shapes = np.searchsorted(shapes, shapes_counts.index.values)

        # Annotate the count of blocks associated with each parameter value
        [ax.annotate(shapes_counts.values[j], (i - 1 - h_offset, y * a - v_offset), c='c', fontsize=10) for j, y in enumerate(y_shapes)]
        [ax.annotate(stim_types_counts.values[j], (i - 1 - h_offset, y * b - v_offset), c='m', fontsize=10) for j, y in enumerate(y_stim_types)]
        [ax.annotate(kappa_counts.values[j], (i - 1 - h_offset, (y + 1) + max(stim_type_ticks) + 1 - v_offset), c='g', fontsize=10) for j, y in enumerate(y_kappas)]

    ax.set_xticks(sessions - 1, sessions)
    ax.set_yticks(shape_ticks + stim_type_ticks + kappa_ticks, y_labels, fontsize=10)
    ax.set_ylabel("kappa\nstim type\nshape", rotation='horizontal', labelpad=55, multialignment='left', va='center',
                  linespacing=7, fontsize=15)
    ax.set_ylim(-1, max(kappa_ticks) + 1)
    ax.set_title(title)
    return ax

def plot_continuous3d_dict(continuous_data: dict, list_blocks: list, x: str, title: str = None, color_key: str = 'time',
                           ax: Optional[plt.Axes] = None, **kwargs) -> tuple:
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
        fig_kwargs = kwargs.pop('fig_kwargs', {})
        fig = plt.figure(**fig_kwargs)
        ax = fig.add_subplot(projection='3d')

    plt_kwargs = kwargs.pop('plt_kwargs', {})
    p = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=c, **plt_kwargs)
    if title:
        ax.set_title(title)

    # Configure plot
    cbar_kwargs = {'label': color_key} | kwargs.pop('cbar_kwargs', {})
    plt.colorbar(p, ax=ax, **cbar_kwargs)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.set_ticks([])
    view_kwargs = {'elev': 0, 'azim': -45} | kwargs.pop('view_kwargs', {})
    ax.view_init(**view_kwargs)

    return ax, p

def plot_continuous3d_df(df: pd.DataFrame, continuous_data: dict, x: str, title: str = None,
                          color_key: str = 'time since last push (s)', color_filter = None, ax: Optional[plt.Axes] = None,
                          **kwargs) -> tuple:
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
        end = block[1]['push times']
        start = end - block[1]['consecutive push intervals']
        key = (block[0][0], block[0][1], block[0][2])
        if key in continuous_data:
            continuous_data_block = continuous_data[key]
            start_idx, end_idx = np.searchsorted(continuous_data_block['time'], [start, end])
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
            elif color_key == 'time since last push (s)':
                t_vec = continuous_data_block['time'][start_idx:end_idx][not_nans]
                t_vec -= continuous_data_block['time'][start_idx]
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
        fig_kwargs = kwargs.pop('fig_kwargs', {})
        fig = plt.figure(**fig_kwargs)
        ax = fig.add_subplot(projection='3d')

    plt_kwargs = kwargs.pop('plt_kwargs', {})
    p = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=c, **plt_kwargs)
    if title:
        ax.set_title(title)

    # Configure color bar
    cbar_kwargs = {'label': color_key} | kwargs.pop('cbar_kwargs', {})
    cbar = plt.colorbar(p, ax=ax, **cbar_kwargs)
    cbar.solids.set_alpha(1.0)

    # Remove axis ticks
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.set_ticks([])

    # Set 3D view parameters
    view_kwargs = {'elev': 0, 'azim': -45} | kwargs.pop('view_kwargs', {})
    ax.view_init(**view_kwargs)
    return ax, p

def plot_continuous2d_df(df: pd.DataFrame, continuous_data: dict, x: str, dims: tuple = (0, 1),
                          title: str = None, color_key: str = 'time since last push (s)', color_filter = None,
                          ax: Optional[plt.Axes] = None, **kwargs) -> tuple:
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
        end = block[1]['push times']
        start = end - block[1]['consecutive push intervals']
        key = (block[0][0], block[0][1], block[0][2])
        if key in continuous_data:
            continuous_data_block = continuous_data[key]
            start_idx, end_idx = np.searchsorted(continuous_data_block['time'], [start, end])
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
            elif color_key == 'time since last push (s)':
                t_vec = continuous_data_block['time'][start_idx:end_idx][not_nans]
                t_vec -= continuous_data_block['time'][start_idx]
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
        fig_kwargs = kwargs.pop('fig_kwargs', {})
        fig, ax = plt.subplots(**fig_kwargs)

    plt_kwargs = kwargs.pop('plt_kwargs', {})
    p = ax.scatter(points[:, 0], points[:, 1], c=c, **plt_kwargs)
    if title:
        ax.set_title(title)

    cbar_kwargs = {'label': color_key} | kwargs.pop('cbar_kwargs', {})
    cbar = plt.colorbar(p, ax=ax, **cbar_kwargs)
    cbar.solids.set_alpha(1.0)

    # Remove axis ticks
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_ticks([])

    return ax, p
