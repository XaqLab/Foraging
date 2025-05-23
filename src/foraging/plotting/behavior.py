import logging
from copy import deepcopy

import numpy as np
import pandas as pd
import seaborn as sns
from foraging import utils
from foraging.utils import data
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from scipy.spatial.distance import jensenshannon

from foraging.config.constants import BOX_LABELS, BOX_COLORS, BOX_POSITIONS
from foraging.plotting import PALETTE
from foraging.utils import INDEX, MIN_INDEX, kwargs_handler
from foraging.utils.data import get_blocks, filter_df
from foraging.plotting._base import fig_init, titler, unitler, bp, regplot, get_bar_heights, palette_handler

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def plot_experiment_overview(df: pd.DataFrame, conds: dict = None, title: str = '', title_prefix: str = '', palette: dict = PALETTE, label_rotation: float = 35, annotate_block: bool = False, ax: plt.Axes = None, **kwargs) -> plt.Axes:
    """
    Plot the pushes over all blocks in the experiment, organized by sessions. This assumes one subject is specified in the `conds` dictionary.

    Args:
        df: DataFrame.
        conds: Dictionary to filter df.
        title: Title of figure. If specified, overrides `title_prefix`.
        title_prefix: Prefix of title that is used to construct the contents of the title together with conds. See `titler` for more details. Ignored if `title` is specified.
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
    x_offset = get_blocks(df)['duration'].last()
    x_offset.iloc[1:] = x_offset.groupby(['subject', 'session']).cumsum().iloc[:-1]
    session_start = x_offset.reset_index(level='block').groupby(['subject','session'])['block'].first()
    for idx, x in session_start.items(): # Make sure each row (session) starts from 0 on the x-axis
        x_offset.loc[idx + (x,)] = 0
    df_temp = df.join(x_offset, rsuffix = '_offset', on=INDEX[:MIN_INDEX-1])
    df_temp['x'] = df_temp['push times'] + df_temp['duration_offset']

    # Offset y-coord
    session_order = sorted(df_temp.index.unique('session'))
    session_offsets = {session: i for i, session in enumerate(session_order)}
    box_order = sorted(df_temp['box position'].unique())
    box_offsets = {box: box - 1 for box in box_order}

    df_temp['y_offset_1'] = df_temp['box position'].map(box_offsets)
    df_temp['y_offset_2'] = df_temp.index.map(lambda x: session_offsets[x[INDEX.index('session')]])

    # Change multiplier to control spacing between sessions and rows
    y_offset_1_factor = 1
    y_offset_2_factor = 6
    df_temp['y'] = 1 + y_offset_1_factor * df_temp['y_offset_1'] + y_offset_2_factor * df_temp['y_offset_2']

    # Create ax if none provided
    fig_kwargs = kwargs_handler(kwargs, 'fig_kwargs')
    fig, ax = fig_init(ax, **fig_kwargs)

    legend = True
    for session in session_order:
        bp(sns.scatterplot)(
            filter_df(df_temp, {'session': session}),
            x='x',
            y='y',
            marker = '|',
            s = 100,
            hue='box',
            palette=palette,
            title = None,
            legend=legend,
            ax=ax,
            **kwargs
        )
        legend = False

    # Annotate block parameters
    y_text_offset = 0.5
    if annotate_block:
        for session in session_order:
            df_session = filter_df(df_temp, {'session': session})
            y_text = df_session['y'].max() + y_text_offset
            blocks = df_session.index.get_level_values('block')
            kappas = df_session.index.get_level_values('kappa')
            kappas = kappas[np.insert(blocks[1:] != blocks[:-1], 0, True)]
            shapes = df_session.index.get_level_values('shape')
            shapes = shapes[np.insert(blocks[1:] != blocks[:-1], 0, True)]
            x_text = df_session['duration_offset'].unique()
            for i in range(len(kappas)):
                ax.text(x_text[i], y_text, rf'$\kappa$={kappas[i]},$\alpha$={shapes[i]}')

    # Demarcate blocks
    for session in session_order:
        df_session = filter_df(df_temp, {'session': session})
        x_text = df_session['duration_offset'].unique()[1:]
        ax.vlines(x_text, y_offset_2_factor * df_session['y_offset_2'].unique()[0] + 0.5, y_offset_2_factor * df_session['y_offset_2'].unique()[0] + 3.5, linestyles = 'dotted', colors = 'black')

    # Tidy up axes
    ax.set_yticks(
        [y_offset_2_factor * offset + 0.5 for offset in sorted(df_temp['y_offset_2'].unique())],
        [str(s) for s in session_order]
    )
    ax.tick_params(axis = 'y', labelrotation = label_rotation)
    ax.set_xlabel('time in block (s)')
    ax.set_ylabel('session')
    ax.set_title(titler(title=title, title_prefix=title_prefix, conds=conds))
    fig.tight_layout()
    return ax


def plot_block_events(df: pd.DataFrame, conds: dict = None, x: str = 'push times', y: str = 'box position', x_unit: str = 's', y_unit: str = None, title: str = '', title_prefix: str = 'Block activity',
                      palette: dict = PALETTE, legend: bool = True, ax: plt.Axes = None, **kwargs) -> plt.Axes:
    """
    Plot the push-related variable in the block.

    Args:
        df: DataFrame.
        conds: Dictionary to filter df.
        x: Name of x variable in DataFrame. Defaults to `push times`.
        y: Name of y variable in DataFrame. Defaults to `box rank`.
        x_unit: Unit to assign to x. Defaults to `s` for seconds. Ignored if None.
        y_unit: Unit to assign to y. Defaults to None. Ignored if None.
        title: Title of figure. If specified, overrides `title_prefix`.
        title_prefix: Prefix of title that is used to construct the contents of the title together with conds. See `titler` for more details. Ignored if `title` is specified.
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
    fig_kwargs = kwargs_handler(kwargs, 'fig_kwargs')
    fig, ax = fig_init(ax, **fig_kwargs)

    # Get block data and metadata
    df_block = utils.data.filter_df(df, conds)
    schedules = sorted(df_block['schedule'].unique())
    kappa = df_block.index.unique('kappa')
    stim_type = df_block.index.unique('stimulus type')
    shape = df_block.index.unique('shape')

    if conds is None:
        conds = {}
    else:
        conds = deepcopy(conds)
    conds['kappa'] = kappa[0]
    conds['stim type'] = stim_type[0]
    conds['shape'] = shape[0]

    # Create switch segments (x, y) pairs for LineCollection
    x_vals = df_block[x].values
    y_vals = df_block[y].values
    colors = np.array(['black'] * (len(y_vals) - 1))
    # styles = ['dashed' if x else 'solid' for x in df_block['stay/switch'].values[1:]]
    segments = [[(x_vals[i], y_vals[i]), (x_vals[i + 1], y_vals[i + 1])] for i in range(len(x_vals) - 1)]

    # Create the LineCollection
    line_kwargs = kwargs_handler(kwargs, 'line_kwargs', dict(linestyles='--', linewidth = 1, zorder = 0))
    lc = LineCollection(segments, colors = colors, **line_kwargs)

    # Set labels
    ax.add_collection(lc)
    ax.autoscale()
    title = titler(title = title, title_prefix = title_prefix, conds = conds)
    ax.set_title(title)
    ax.set_ylabel(unitler(y, y_unit))
    ax.set_xlabel(unitler(x, x_unit))

    # Add reward outcomes with shaded (rewarded) and empty (not rewarded) markers
    colors = np.array([palette[i] for i in df_block['box'].values])
    mask = df_block['reward outcomes'] == True
    ax.scatter(x_vals[mask], y_vals[mask], c=colors[mask], marker='^', s = 80, zorder = 2)
    ax.scatter(x_vals[~mask], y_vals[~mask], edgecolors=colors[~mask], marker='v', s = 80, zorder = 2, facecolors ="none")

    # Create legend manually with proxy artists
    if legend:
        legend_kwargs = kwargs_handler(kwargs, 'legend_kwargs', dict(loc='upper right'))
        palette = palette_handler(palette, df_block['box'].unique())
        legend_elements = ([Line2D([0], [0], color=palette[j], linestyle='-', label=schedules[i]) for i, j in enumerate(palette.keys())]
                           + [Line2D([0], [0], color='black', linestyle='', marker='^', label='rewarded'),
                              Line2D([0], [0], color='black', linestyle='', marker='v', markerfacecolor = 'none', label='no reward')])
        ax.legend(handles=legend_elements, **legend_kwargs)
    return ax


def plot_pushes(df: pd.DataFrame, conds: dict = None, title: str = '', title_prefix: str = 'Pushes for ',
                      palette: dict = PALETTE, box_labels: list = BOX_POSITIONS,
                      legend: bool = True, ax: plt.Axes = None, **kwargs) -> plt.Axes:

    ax = plot_block_events(df, conds= conds, title= title, title_prefix= title_prefix, palette= palette, legend= legend, ax= ax, **kwargs)

    # Custom plotting logic
    df_block = utils.data.filter_df(df, conds).reset_index()
    ax.set_xlim([0, df_block['push times'].max() + 1])
    box_labels = [box_labels[i] for i in sorted(df_block['box position'].unique())]
    ax.set_yticks(range(len(box_labels)), box_labels, rotation = 90, va = 'center')
    ax.set_ylabel('')
    return ax


def plot_frequencies_over_experiment(df: pd.DataFrame, category: str, conds: dict = None, title: str = None, title_prefix: str = None, palette: list = BOX_COLORS, label_rotation: float = 35, ax: plt.Axes = None, **kwargs):

    # Get frequencies for specified category
    visit_freqs = filter_df(get_blocks(df)[category].value_counts(normalize=True).to_frame(), conds=conds).reset_index()

    # Define horizontal offset for each session
    session_order = sorted(visit_freqs['session'].unique())
    session_offsets = {session: i for i, session in enumerate(session_order)}
    visit_freqs['y_offset'] = visit_freqs['session'].map(session_offsets)
    visit_freqs['y'] = 1 - visit_freqs['proportion'] + visit_freqs['y_offset']

    # Create ax if none provided
    fig_kwargs = kwargs_handler(kwargs, 'fig_kwargs')
    fig, ax = fig_init(ax, **fig_kwargs)
    legend = True
    category_order = sorted(df[category].unique())
    for session in session_order:
        sns.lineplot(
            data=visit_freqs[visit_freqs['session'] == session].sort_values(by='block'),
            x='block',
            y='y',
            hue=category,
            hue_order=category_order,
            palette=list(palette),
            ax=ax,
            legend=legend,
            **kwargs
        )
        legend = False

    # Tidy up axes
    ax.set_yticks(
        [offset + 0.5 for offset in session_offsets.values()],
        [str(s) for s in session_order]
    )
    ax.tick_params(axis = 'y', labelrotation = label_rotation)
    ax.set_xlabel('blocks')
    ax.set_ylabel('session')
    ax.set_title(titler(title=title, title_prefix=title_prefix, conds=conds))

    # Draw the scale bar
    scale_length = 1
    x_start = visit_freqs.loc[visit_freqs['session'] == session_order[0], 'block'].max()  # x-position of the scale
    y_start = 0  # y-position of the scale bar
    x_offset = 0.05
    ax.plot([x_start + x_offset, x_start + x_offset], [y_start, y_start + scale_length], color='black', lw=2)
    ax.text(x_start, y_start, '1', ha='center', va='bottom')
    ax.text(x_start, y_start + scale_length + 0.5, '0', ha='center', va='bottom')
    ax.invert_yaxis()
    ax.legend(loc='upper right', title=category)
    fig.tight_layout()
    return ax




def plot_runlengths(df: pd.DataFrame, null_model: bool = False, disp_js: bool = False) -> plt.Axes:
    # Identify consecutive pushes and when they switch
    x = df.index.get_level_values('push #')
    consecutive_mask = x[1:] - x[:-1] == 1
    change_mask = (df['stay/switch'] == 'switch') & np.insert(consecutive_mask, 0, True)
    push_nums = utils.data.get_blocks(df)["push times"].rank().astype(
        int)  # Calculate from scratch in case pushes got dropped
    change_mask[push_nums == 1] = True

    # Count the runlengths at different boxes
    group_labels = change_mask.cumsum()
    labeled_lengths = pd.DataFrame(
        {"group": group_labels, "box": df['box'],
         "next box": utils.data.get_blocks(df['box']).shift(-1).fillna('missing')}
    ).set_index(df.index)
    labeled_lengths_all = labeled_lengths.groupby(['subject', 'kappa', 'box', 'group']).size().to_frame().rename(
        columns={0: 'length'})
    labeled_lengths_all = labeled_lengths_all[
        (labeled_lengths_all['length'] > 1) & (labeled_lengths_all['length'] <= 10)]

    # Calculate the distribution of runlengths under a dice that is rolled by visitation frequencies
    visit_freqs = df.groupby(['subject', 'kappa'])['box'].value_counts(normalize=True).to_frame()

    # Contrast low and high kappa conditions
    kappas = {
        'marco': (0.01, 0.1),
        'dylan': (0.01, 0.1),
        'humans': (0.0, 0.1),
        'viktor': (0.0, 0.1)
    }
    subjects = df.index.unique('subject')
    fig, axes = plt.subplots(len(subjects), 2, figsize=(15, 5 * len(subjects)))
    for i, subj in enumerate(subjects):
        for j, kappa in enumerate(kappas[subj]):
            bp(sns.histplot)(labeled_lengths_all, x='length', accumulate=True, conds={'subject': subj, 'kappa': kappa},
                             title_prefix="Distribution of runlengths", discrete=True, stat='probability',
                             common_norm=True, multiple='dodge', ax=axes[i, j])

            # Overlay random dice probabilities from geometric distribution
            if null_model:
                bars = axes[i,j].patches
                bar_width = bars[0].get_width()  # Width of one bar
                probs = visit_freqs.loc[(subj, kappa)] # Visit probabilities
                boxes = sorted(probs.index.unique('box'))
                handles = axes[i, j].get_legend().legend_handles
                labels = [t.get_text() for t in axes[i, j].get_legend().get_texts()]
                for b, box in enumerate(boxes):
                    try:
                        p = probs.loc[box].iloc[0]
                        run_lengths = labeled_lengths_all.loc[(subj, kappa, box), 'length'].sort_values().unique()
                        geom = p ** run_lengths * (1 - p)
                        offset = (b - (len(boxes) - 1) / 2) * bar_width
                        x = run_lengths + offset
                        axes[i, j].plot(x, geom, c=BOX_COLORS[BOX_LABELS.index(box)], label = 'random dice')
                        if disp_js:
                            bar_heights = get_bar_heights(axes[i, j], x_centers=run_lengths)
                            # for k, bar in enumerate(bar_heights[box]):
                            #     axes[i, j].text(x[k], bar, f'{jensenshannon(geom, bar):.1f}', ha='center', va='bottom', fontsize = 7)
                            axes[i, j].set_title(axes[i, j].get_title() + f'\nJS-distance = {jensenshannon(geom, bar_heights[box])}')
                            # print(f"Jensen-shannon distance between empirical distribution and null distribution of ({subj}, {kappa}, {box}): {jensenshannon(geom, bar_heights[box])}")
                    except:
                        continue
                axes[i, j].legend(handles = handles + [Line2D([0], [0], color='black', linestyle='-', label='random dice')], labels = labels + ['random dice'] )
            axes[i, j].sharex(axes[i, 0])
            axes[i, j].sharey(axes[i, 0])
    fig.tight_layout()
    return axes


# def plot_runlengths_by_box(df: pd.DataFrame) -> plt.Axes:
#     # Identify consecutive pushes and when they switch
#     x = df.index.get_level_values('push #')
#     consecutive_mask = x[1:] - x[:-1] == 1
#     change_mask = (df['stay/switch'] == 'switch') & np.insert(consecutive_mask, 0, True)
#     push_nums = utils.data.get_blocks(df)["push times"].rank().astype(
#         int)  # Calculate from scratch in case pushes got dropped
#     change_mask[push_nums == 1] = True
#
#     # Count the runlengths at different boxes
#     group_labels = change_mask.cumsum()
#     labeled_lengths = pd.DataFrame(
#         {"group": group_labels, "box": df['box'],
#          "next box": utils.data.get_blocks(df['box']).shift(-1).fillna('missing')}
#     ).set_index(df.index)
#     labeled_lengths['next box'] = labeled_lengths.groupby('group')['next box'].transform('last')
#
#     labeled_lengths_by_box = labeled_lengths.groupby(
#         ['subject', 'kappa', 'next box', 'box label', 'group']).size().to_frame().rename(columns={0: 'length'})
#     labeled_lengths_by_box = labeled_lengths_by_box[
#         (labeled_lengths_by_box['length'] > 1) & (labeled_lengths_by_box['length'] <= 10)]
#
#     # Contrast low and high kappa conditions
#     kappas = {
#         'marco': (0.01, 0.1),
#         'dylan': (0.01, 0.1),
#         'humans': (0.0, 0.1),
#         'viktor': (0.0, 0.1)
#     }
#     subjects = df.index.unique('subject')
#     fig, axes = plt.subplots(len(subjects), len(BOX_LABELS) * 2, figsize=(25, 4 * len(subjects)))
#     for i, subj in enumerate(subjects):
#         for j, box in enumerate(BOX_LABELS):
#             for k, kappa in enumerate(kappas[subj]):
#                 actual_j = 2 * j + k % 2
#                 bp(sns.histplot)(labeled_lengths_by_box, x='length', accumulate=True,
#                                  conds={'subject': subj, 'kappa': kappa, 'next box': box},
#                                  palette=[BOX_COLORS[i] for i in range(len(BOX_COLORS)) if BOX_LABELS[i] != box],
#                                  box_labels=[b for b in BOX_LABELS if b != box],
#                                  title_prefix="Distribution of runlengths", title_kwargs={'fontsize': 10},
#                                  discrete=True, stat='probability', common_norm=True, multiple='dodge',
#                                  ax=axes[i, actual_j])
#
#                 # Overlay random dice probabilities from geometric distribution
#                 probs = visit_freqs.loc[(subj, kappa)]
#                 for b in probs.index.unique('box label'):
#                     try:
#                         if b != box:
#                             p = probs.loc[b].iloc[0]
#                             run_lengths = labeled_lengths_all.loc[(subj, kappa, b), 'length'].sort_values().unique()
#                             geom = p ** run_lengths * (1 - p)
#                             axes[i, actual_j].plot(run_lengths, geom, c=BOX_COLORS[BOX_LABELS.index(b)])
#                     except:
#                         continue
#
#                 axes[i, actual_j].sharex(axes[i, 0])
#                 axes[i, actual_j].sharey(axes[i, 0])
#     fig.tight_layout()

def plot_fisher(df: pd.DataFrame, conds: dict = None, title: str = None, title_prefix: str = 'Fisher info for',
                        box_colors: list = BOX_COLORS, box_labels: list = None,
                        legend: bool = True, specific: bool = False, ax: plt.Axes = None, **kwargs) -> plt.Axes:
    """
    Plots the push intervals for each box, with different colors and line styles based on the stay/switch behavior,
    and displays reward outcomes with markers. Optionally adds a custom legend.

    Args:
        df: Dataframe
        conds: Dictionary to filter df. Should specify a block.
        x: x-axis
        title: Title of figure. If specified, overrides `title_prefix`.
        title_prefix: Prefix of title that is used to construct the contents of the title together with conds. See `titler` for more details. Ignored if `title` is specified.
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
    fig, ax = fig_init(ax, **kwargs.pop('fig_kwargs', {}))

    # Get data from block
    x = 'push times'
    df_block = utils.data.filter_df(df, conds)
    x_vals = df_block[x].values
    y = 'specific fisher' if specific else 'fisher'
    y_vals = df_block[y].values
    colors = np.array(['black'] * len(y_vals))

    # Create segments (x, y) pairs for LineCollection
    segments = [[(0, 0), (x_vals[0], y_vals[0])]] + [[(x_vals[i], y_vals[i]), (x_vals[i + 1], y_vals[i + 1])] for i in range(len(x_vals) - 1)]

    # Create the LineCollection
    lc = LineCollection(segments, colors=colors, linewidth=2)

    # Create ax if none provided
    if ax is None:
        fig_kwargs = kwargs.pop('fig_kwargs', {})
        _, ax = plt.subplots(**fig_kwargs)
    ax.add_collection(lc)
    ax.autoscale()

    # Get block metadata
    schedules = np.sort(df_block['schedule'].unique())
    kappa = df_block.index.unique('kappa')
    stim_type = df_block.index.unique('stimulus type')
    shape = df_block.index.unique('shape')

    if conds is None:
        conds = {}
    else:
        conds = deepcopy(conds)
    conds['kappa'] = kappa[0]
    conds['stim type'] = stim_type[0]
    conds['shape'] = shape[0]
    box_labels = box_labels if box_labels else schedules

    title = titler(title=title, title_prefix=title_prefix, conds=conds)
    ax.set_title(title)
    ax.set_ylabel('Fisher information')
    ax.set_xlabel(unitler(x, 's'))

    # Add reward outcomes with shaded (rewarded) and empty (not rewarded) markers
    colors = np.array([box_colors[i] for i in df_block['box rank'].values])
    mask = df_block['reward outcomes'] == True
    ax.scatter(x_vals[mask], y_vals[mask], c=colors[mask], marker='^', s = 80, zorder = 2)
    ax.scatter(x_vals[~mask], y_vals[~mask], edgecolors=colors[~mask], marker='v', s = 80, zorder = 2, facecolors ="none")

    # Create legend manually with proxy artists
    if legend:
        legend_kwargs = kwargs.pop('legend_kwargs', {'loc': 'upper right'})
        legend_elements = ([Line2D([0], [0], color=box_colors[j], linestyle='-', label=box_labels[i]) for i, j in enumerate(sorted(df_block['box rank'].unique()))]
                           + [Line2D([0], [0], color='black', linestyle='', marker='^', label='rewarded'),
                              Line2D([0], [0], color='black', linestyle='', marker='v', markerfacecolor = 'none', label='no reward')])
        ax.legend(handles=legend_elements, **legend_kwargs)
    return ax


def plot_push_intervals_vs_reward_intervals(df: pd.DataFrame, y = 'same-box push intervals', title: str = None, title_prefix: str = "Push intervals vs reward intervals", unity = True, **kwargs) -> (plt.Axes, float, float):
    """
    Plot linear regression of push intervals against reward intervals in a block

    Args:
        df: DataFrame of experiment data for a given block
        y: Name of y-variable
        title: Title of figure. If specified, overrides `title_prefix`.
        title_prefix: Prefix of title that is used to construct the contents of the title together with conds. See `titler` for more details. Ignored if `title` is specified.
        **kwargs: keyword arguments for seaborn

    Returns:
        axes, r-squared and slope from regression
    """

    # Remove first push from each box, since reward time is messed up for first pushes
    df = df.copy()
    n_boxes = df['box position'].nunique()
    [df.drop(df.loc[df['box position'] == i].index[0], inplace=True) for i in range(n_boxes)]
    ax = bp(sns.scatterplot)(df, x='reward intervals', y=y, title_prefix = title_prefix, title = title, **kwargs)

    # Filter df here or inside regplot using conds
    conds = kwargs.pop('conds', None)
    df = utils.data.filter_df(df, conds)
    kwargs['ax'] = ax
    fit_results = regplot(df['reward intervals'].to_numpy(), df[y].to_numpy(), line_kws={'color': 'black'},
                          **kwargs)

    if unity:
        x_max = max(ax.get_xlim()[1], ax.get_ylim()[1])
        x = np.arange(x_max)
        ax.plot([0, x_max], [0, x_max], linestyle='dashed', color='black')
        ax.fill_between(x, x, x_max, color="green", alpha=0.1)
        ax.fill_between(x, x, color="red", alpha=0.1)
    return ax, fit_results.rsquared, fit_results.params[1]

def plot_experiment_parameters(df: pd.DataFrame, conds: dict, title: str = "Experiment parameters by session",
                               label_rotation: float = 35, ax: plt.Axes = None, **kwargs) -> plt.Axes:
    """
    Plots the distribution of experiment parameters (kappa, stimulus type, shape) across different sessions.
    Displays the number of blocks associated with each parameter and session.

    Args:
        df: DataFrame containing experiment session data with hierarchical index ('session', 'stimulus type', 'shape', 'kappa').
        conds: Dictionary of conditions used to filter the DataFrame before plotting.
        title: Title of the plot.
        label_rotation: degrees to rotate xtick labels.
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

    # Generate n_param + 1 y-ticks
    ax.scatter(np.ones(n_params + 1), np.arange(n_params + 1), alpha=0)
    for i, sess in enumerate(sessions):
        # Count parameter values for the current session
        kappa_counts = df.xs(sess, level='session').reset_index().groupby('kappa')['block'].nunique()
        stim_types_counts = df.xs(sess, level='session').reset_index().groupby('stimulus type')['block'].nunique()
        shapes_counts = df.xs(sess, level='session').reset_index().groupby('shape')['block'].nunique()

        # Determine y-coordinates for parameter value annotations
        y_kappas = np.searchsorted(kappas, kappa_counts.index.values)
        y_stim_types = np.searchsorted(stim_types, stim_types_counts.index.values) + len(shapes)
        y_shapes = np.searchsorted(shapes, shapes_counts.index.values)

        # Annotate the count of blocks associated with each parameter value
        [ax.annotate(shapes_counts.values[j], (i - h_offset, y * a - v_offset), c='c', fontsize=10) for j, y in enumerate(y_shapes)]
        [ax.annotate(stim_types_counts.values[j], (i - h_offset, y * b - v_offset), c='m', fontsize=10) for j, y in enumerate(y_stim_types)]
        [ax.annotate(kappa_counts.values[j], (i - h_offset, (y + 1) + max(stim_type_ticks) + 1 - v_offset), c='g', fontsize=10) for j, y in enumerate(y_kappas)]

    ax.set_xticks(range(len(sessions)), sessions)
    ax.tick_params(axis = 'x', labelrotation = label_rotation)
    ax.set_yticks(shape_ticks + stim_type_ticks + kappa_ticks, y_labels, fontsize=10)
    ax.set_ylabel("kappa\nstim type\nshape", rotation='horizontal', labelpad=55, multialignment='left', va='center',
                  linespacing=7, fontsize=15)
    ax.set_ylim(-1, max(kappa_ticks) + 1)
    ax.set_title(title)
    return ax

def plot_continuous3d_dict(continuous_data: dict, list_blocks: list, x: str, title: str = None, color_key: str = 'time',
                           ax: plt.Axes = None, **kwargs) -> tuple:
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
                          color_key: str = 'time since last push (s)', color_filter = None, ax: plt.Axes = None,
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
                          ax: plt.Axes = None, **kwargs) -> tuple:
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
