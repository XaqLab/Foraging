import logging
from copy import deepcopy
from typing import Callable

from cycler import cycler
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from scipy.spatial.distance import jensenshannon
from scipy.stats import kstest, expon, fit
from scipy.optimize import minimize_scalar
from scipy.stats import gamma

from foraging.config.constants import BOX_COLORS, BOX_POSITIONS, KAPPA_LEVELS, SEED
from foraging.plotting import PALETTE, PALETTE_DARK, fig_init, titler, unitler, bp, enhanced_violinplot, subject_plotter, legend_handler
from foraging.utils import INDEX, MIN_INDEX, kwargs_handler
from foraging.utils.data import get_blocks, filter_df, bin_data, get_continuous_from_df_to_dict
from foraging.plotting._base import regplot, get_bar_heights, palette_handler

import pickle
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

@legend_handler
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
        ax.vlines(x_text, y_offset_2_factor * df_session['y_offset_2'].unique()[0] - 0.5, y_offset_2_factor * df_session['y_offset_2'].unique()[0] + 2.5, linestyles = 'dotted', colors = 'black')

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

def plot_push_percentiles(df: pd.DataFrame, percentiles: dict = None, **kwargs):
    def _plot(i, subj, **kwargs):
        # Plot each push's percentiles
        ax = bp(sns.scatterplot)(df, x='consecutive push intervals', y='push percentiles', conds={'subject': subj}, title_prefix="Percentiles of consecutive push intervals", x_unit='s', legend = False, **kwargs)
        y = df.loc[(subj,), 'push percentiles'].sort_values()
        x = df.loc[(subj,), 'consecutive push intervals'].sort_values()

        # Mark specific percentile
        if percentiles:
            percentile = percentiles[subj]
            perc_idx = np.argmin((y - percentile)**2)
            push_perc = x.iloc[perc_idx]
            print(f'{subj}\'s {round(percentile * 100, 2)}% push {push_perc}')
            ax.axvline(push_perc, color='black', linestyle='dashed', label=f'{round(percentile * 100, 2)}% push')
            ax.legend(loc = 'upper right')
    subject_plotter(df.index.unique('subject'), _plot, **kwargs)


def plot_long_push_blocks(df: pd.DataFrame, top_n: int, figsize: tuple[float, float] = (20, 2.5), **kwargs):
    # Sort data by magnitude of push interval in descending order
    df_sorted = df.sort_values(by='consecutive push intervals', ascending=False)

    def _plot(i, subj, **kwargs):
        fig, axes = plt.subplots(1, top_n, figsize=figsize)
        df_subject = filter_df(df_sorted, {'subject': subj})

        # Plot each block containing the `top_n` pushes
        for i, (idx, g) in enumerate(get_blocks(df_subject, sort=False)):
            if i >= top_n:
                break
            conds = dict(zip(INDEX[:MIN_INDEX - 1], idx))
            plot_pushes(g.sort_index(), conds=conds, title_prefix='', legend=False, ax=axes[i], **kwargs)
            axes[i].set_title(f"session {conds['session']} block {conds['block']}")
        fig.suptitle(subj)
        fig.tight_layout()
    subject_plotter(df.index.unique('subject'), _plot, **kwargs)

@legend_handler
def plot_reward_outcomes_long_vs_medium_pushes(df: pd.DataFrame, percentiles: dict, n_samples: int = 5000, window: float = 30, long_label: str = 'long push intervals', medium_label: str = 'medium push intervals', seed: int = SEED, **kwargs):
    df_rate_content = {'subject': [], 'push intervals': [], '% rewarded': [], 'previous wait time (s)': [],
                       'previous reward outcome': [], 'type': []}
    rng = np.random.default_rng(seed)
    def _helper(subject, percentile: float = None):
        df_subject = filter_df(df, {'subject': subject})
        if percentile:
            df_subject = df_subject[df_subject['push percentiles'] >= percentile]
        else:
            df_subject = df_subject[(df_subject['push percentiles'] >= 0.25) & (df_subject['push percentiles'] <= 0.75)]

        for row in df_subject.sample(n_samples, replace=True, random_state=rng).itertuples():
            idx = row.Index
            # Get push immediately before current push
            push_num = idx[INDEX.index('push #')]
            new_idx = df.index.get_loc(idx) - 1
            if push_num == 1:
                continue

            # Calculate reward fraction within a given time window of the previous push
            new_row = df.iloc[new_idx]
            if new_row['push percentiles'] > 0.75:  # if previous push is also outlier, then considering this push is redundant
                continue

            df_block = df.loc[idx[:MIN_INDEX - 1]]
            df_window = df_block.loc[(df_block['push times'] <= new_row['push times']) & (
                    df_block['push times'] >= new_row['push times'] - window)]
            row = df.loc[idx]
            # Populate data arrays
            if len(df_window) > 0:
                df_rate_content['subject'].append(subject)
                df_rate_content['push intervals'].append(row['consecutive push intervals'])
                df_rate_content['% rewarded'].append(
                    df_window['reward outcomes'].sum() / len(df_window['reward outcomes']))
                df_rate_content['previous wait time (s)'].append(new_row['wait times'])
                df_rate_content['previous reward outcome'].append(new_row['rewarded?'])
                df_rate_content['type'].append(long_label) if percentile else df_rate_content['type'].append(medium_label)

    # For each subject, find the top x% pushes and middle 50% pushes and calculate the reward fraction of the past `window` seconds prior to each push's onset
    for subject in df.index.unique('subject'):
        _helper(subject, percentile = percentiles[subject])
        _helper(subject)

    fig_kwargs = kwargs_handler(kwargs, 'fig_kwargs', dict(figsize=(20, 10)))
    fig, axes = plt.subplots(2, 1, **fig_kwargs)
    df_rate = pd.DataFrame(df_rate_content).set_index('subject').dropna()
    df_rate = df_rate[df_rate['previous wait time (s)'] <= 30]
    enhanced_violinplot(df_rate, x='subject', y='% rewarded', hue='type', hue_order=[medium_label, long_label],
                        ax=axes[0], inner=None, cut=0)
    axes[0].set_title(f"% Rewarded pushes within {window} s of previous push (Long vs Medium Push Intervals)")
    sns.move_legend(axes[0], "upper left", bbox_to_anchor=(1, 1))

    df_rate['outcomes'] = df_rate['previous reward outcome'].map({'Yes': 'Rewarded', 'No': 'Unrewarded'}) + ', ' + df_rate['type']
    enhanced_violinplot(df_rate, x='subject', y='previous wait time (s)', hue='outcomes', ax=axes[1],
                        palette=['#89ff87', '#28ab26', '#84ebf7', '#1cb7c9'],
                        hue_order=['Unrewarded, ' + medium_label, 'Unrewarded, ' + long_label, 'Rewarded, ' + medium_label,
                                   'Rewarded, ' + long_label], inner=None, cut=0)
    axes[1].set_title(f"Previous wait time and outcome (Long vs Medium Push Intervals)")
    fig.tight_layout()
    return axes

@legend_handler
def plot_preceding_pushes_long_vs_medium_pushes(df: pd.DataFrame, percentiles: dict, n_samples: int = 5000, n_preceding: int = 5, long_label: str = 'long push intervals', medium_label: str = 'medium push intervals', seed: int = SEED, **kwargs):
    """
    Plot the n pushes preceding long and medium push intervals to analyze their co-occurrence patterns.
    
    Args:
        df: DataFrame containing push data
        percentiles: Dictionary mapping subjects to their percentile thresholds for long pushes
        n_preceding: Number of preceding pushes to analyze (default: 5)
    """
    # Function to get preceding pushes
    def get_preceding_pushes(push_idx, n):
        preceding = []
        current_idx = df.index.get_loc(push_idx)
        for i in range(n):
            if current_idx - i - 1 >= 0:
                preceding.append(df.iloc[current_idx - i - 1])
        return preceding

    df_rate_content = {'subject': [], 'previous push intervals': [], 'type': []}
    rng = np.random.default_rng(seed)
    def _helper(subject, percentile: float = None):
        df_subject = filter_df(df, {'subject': subject})
        if percentile:
            df_subject = df_subject[df_subject['push percentiles'] >= percentile]
        else:
            df_subject = df_subject[(df_subject['push percentiles'] >= 0.25) & (df_subject['push percentiles'] <= 0.75)]

        for row in df_subject.sample(n_samples, replace=True, random_state=rng).itertuples():
            idx = row.Index
            # Get push immediately before current push
            push_num = idx[INDEX.index('push #')]
            new_idx = df.index.get_loc(idx) - 1
            if push_num == 1:
                continue

            preceding = get_preceding_pushes(idx, n_preceding)
            for p in preceding:
                df_rate_content['subject'].append(subject)
                df_rate_content['previous push intervals'].append(p['consecutive push intervals'])
                df_rate_content['type'].append(long_label) if percentile else df_rate_content['type'].append(medium_label)

    # For each subject, find the top x% pushes and middle 50% pushes and calculate the reward fraction of the past `window` seconds prior to each push's onset
    for subject in df.index.unique('subject'):
        _helper(subject, percentile = percentiles[subject])
        _helper(subject)

    fig_kwargs = kwargs_handler(kwargs, 'fig_kwargs')
    fig, axes = plt.subplots(**fig_kwargs)
    df_rate = pd.DataFrame(df_rate_content).set_index('subject').dropna()
    enhanced_violinplot(df_rate, x='subject', y='previous push intervals', hue='type', hue_order=[medium_label, long_label],
                        ax=axes, inner=None, cut=0, log_scale=True)
    axes.set_title(f"Duration of past {n_preceding} push intervals (Long vs Medium Push Intervals)")
    fig.tight_layout()
    return axes

@legend_handler
def plot_session_onsets_long_vs_medium_pushes(df: pd.DataFrame, percentiles: dict, long_label: str = 'long push intervals', medium_label: str = 'medium push intervals'):
    subjects = df.index.unique('subject')

    # Get time of each push in the session, not just block
    x_offset = get_blocks(df)['duration'].last()
    x_offset.iloc[1:] = x_offset.groupby(['subject', 'session']).cumsum().iloc[:-1]
    session_start = x_offset.reset_index(level='block').groupby(['subject', 'session'])['block'].first()
    for idx, x in session_start.items():  # Make sure each row (session) starts from 0 on the x-axis
        x_offset.loc[idx + (x,)] = 0
    df_temp = df.join(x_offset, rsuffix='_offset', on=INDEX[:MIN_INDEX - 1])
    df_temp['push time in session'] = df_temp['push times'] + df_temp['duration_offset']
    df_temp['onset (s)'] = get_blocks(df_temp['push time in session']).shift().fillna(0)

    # Compare top pushes and middle pushes
    df_temp_content = []
    for i, subject in enumerate(subjects):
        percentile = percentiles[subject]
        df_subject = filter_df(df_temp, {'subject': subject})
        df_subject = df_subject[(df_subject['push percentiles'] >= percentile) | (
                    (df_subject['push percentiles'] >= 0.25) & (df_subject['push percentiles'] <= 0.75))]
        df_subject.loc[df_subject['push percentiles'] >= percentile, 'type'] = long_label
        df_subject.loc[
            (df_subject['push percentiles'] >= 0.25) & (df_subject['push percentiles'] <= 0.75), 'type'] = medium_label
        df_temp_content.append(df_subject)
    df_temp = pd.concat(df_temp_content)
    ax = enhanced_violinplot(df_temp, x='subject', y='onset (s)', hue='type', hue_order=[medium_label, long_label],
                             cut=0, inner=None)
    ax.set_title('Onset of push in session')
    ax.get_figure().tight_layout()
    return ax


def plot_vertical_position_in_block(df: pd.DataFrame, conds: dict, data_dir: str):
    # Get vertical data for block specified by `conds`
    df_block = filter_df(df, conds).copy()
    continuous_data, errors = get_continuous_from_df_to_dict(df_block, data_dir)
    time = continuous_data[tuple(conds.values())]['time']
    vertical = continuous_data[tuple(conds.values())]['position'][:, 2]

    # Plot all vertical positions
    fig, ax = plt.subplots()
    ax.plot(time, vertical, c='grey', linewidth=1)

    # Plot vertical position at time of push
    idx = np.abs(time[None, :] - df_block['push times'].values[:, None]).argmin(axis=1)
    df_block['z-coordinate'] = vertical[idx]
    ax = plot_block_events(df_block, conds=conds, y='z-coordinate', y_unit='mm',
                           title_prefix='Vertical position in block', ax=ax)
    ax.axhline(y=500, label='screen', linestyle=':', c='black')
    ax.set_xlabel("time (s)")
    ax.set_ylabel("vertical position (mm)")
    ax.set_title("Vertical position in block")



def plot_vertical_position_long_vs_medium_pushes(df: pd.DataFrame, data_dir: str, percentiles: dict = None, **kwargs):
    continuous_data, _ = get_continuous_from_df_to_dict(df, data_dir)
    dfs_cont = []

    # Only keep blocks that have real position data
    for block, data in continuous_data.items():
        if np.any(data['position'] != np.nan):
            dfs_cont.append(df.xs(block, level=('subject', 'session', 'block'), drop_level=False))
    df = pd.concat(dfs_cont)

    @legend_handler
    def _plot(i, subject, **kwargs):
        df_subject = filter_df(df, {'subject': subject})
        df_vertical = {'push intervals': [], 'average vertical position': [], 'push percentiles': []}
        # Find average vertical position over each push interval
        for idx, row in df_subject.iterrows():
            try:
                conds = dict(subject=subject, session=idx[INDEX.index('session')], block=idx[INDEX.index('block')])
                time = continuous_data[tuple(conds.values())]['time']
                vertical = continuous_data[tuple(conds.values())]['position'][:, 2]
                push_interval_end = row['push times']
                push_interval_start = push_interval_end - row['consecutive push intervals']
                x = np.abs(time[None, :] - np.array([[push_interval_start], [push_interval_end]])).argmin(axis=1)
                v = vertical[x[0]:x[1]]
                mean_vertical = v[~np.isnan(v)].mean()
                if not np.isnan(mean_vertical):
                    df_vertical['average vertical position'].append(mean_vertical)
                    df_vertical['push intervals'].append(row['consecutive push intervals'])
                    df_vertical['push percentiles'].append(row['push percentiles'])
            except:
                continue
        fig_kwargs = kwargs_handler(kwargs, 'fig_kwargs')
        legend_kwargs = kwargs_handler(kwargs, 'legend_kwargs')
        fig, ax = plt.subplots(**fig_kwargs)
        sns.scatterplot(df_vertical, x='push percentiles', y='average vertical position', ax = ax)
        ax.axvline(x=percentiles[subject], linestyle='--', color='black', label = 'long push intervals')
        ax.legend(**legend_kwargs)
        ax.set_title(f"{subject}'s average vertical position\n occupied during push interval")
        return ax
    # Plot each subject's vertical position distribution
    subject_plotter(df.index.unique('subject'), _plot, **kwargs)

#TODO: visualize xy-position in block in 2d and super-impose block activity
#TODO: rework aggregating xy statistics-- is it velocity or position?
def plot_xy_velocity_long_vs_medium_pushes(df: pd.DataFrame, data_dir: str, percentiles: dict = None, n_samples: int = 5000, long_label: str = 'long push intervals', medium_label: str = 'medium push intervals', **kwargs):
    continuous_data, _ = get_continuous_from_df_to_dict(df, data_dir)
    dfs_cont = []
    rng = np.random.default_rng(SEED)
    # Only keep blocks that have real position data
    for block, data in continuous_data.items():
        if np.any(data['position'] != np.nan):
            dfs_cont.append(df.xs(block, level=('subject', 'session', 'block'), drop_level=False))
    df = pd.concat(dfs_cont)

    def _plot(i, subject):
        df_subject = filter_df(df, {'subject': subject})
        xy_pos = {'x': [], 'y': [], 'type': []}

        # Find x-y positions for medium pushes
        for idx, row in df_subject.sample(frac=1, random_state=rng).iterrows():
            if row['push percentiles'] >= 0.25 and row['push percentiles'] <= 0.75:
                conds = dict(subject=subject, session=idx[INDEX.index('session')], block=idx[INDEX.index('block')])
                time = continuous_data[tuple(conds.values())]['time']
                xy = continuous_data[tuple(conds.values())]['position'][:, :2]
                mask = ~np.isnan(xy).any(axis=1)
                xy_pos['x'] += list(xy[mask][:, 0])
                xy_pos['y'] += list(xy[mask][:, 1])
                xy_pos['type'] += [medium_label] * len(xy[mask])
                if len(xy_pos['x']) >= n_samples:
                    break

        # Find x-y positions for medium pushes
        for idx, row in df_subject.sample(frac=1, random_state=rng).iterrows():
            if row['push percentiles'] >= percentiles[subject]:
                conds = dict(subject=subject, session=idx[INDEX.index('session')], block=idx[INDEX.index('block')])
                time = continuous_data[tuple(conds.values())]['time']
                xy = continuous_data[tuple(conds.values())]['position'][:, :2]
                mask = ~np.isnan(xy).any(axis=1)
                xy_pos['x'] += list(xy[mask][:, 0])
                xy_pos['y'] += list(xy[mask][:, 1])
                xy_pos['type']+= [long_label] * len(xy[mask])
                if len(xy_pos['x']) >= 2 * n_samples:
                    break
        fig, axes = plt.subplots(1, 2,**kwargs)
        xy_pos = pd.DataFrame(xy_pos)
        
        sns.histplot(xy_pos[xy_pos['type'] == medium_label], x='x', y='y', ax = axes[0])
        sns.histplot(xy_pos[xy_pos['type'] == long_label], x='x', y='y', ax = axes[1])
        axes[0].set_title(f"{subject}'s xy positions occupied\n during medium push interval")
        axes[1].set_title(f"{subject}'s xy positions occupied\n during long push interval")
        fig.tight_layout()
    # Plot each subject's vertical position distribution
    subject_plotter(df.index.unique('subject'), _plot)

@legend_handler(bbox = (1.1, 1))
def plot_hmm_probabilities_in_block(df: pd.DataFrame, filepath: str, block_idx: int = 3):


    # Load HMM probabilities
    with open(filepath, 'rb') as f:
        saved = pickle.load(f)

    subject, kappa, block_ids = saved['subject'], saved['kappa'], saved['block_ids']
    pos, gaze = saved['pos'], saved['gaze']
    push, move, look = saved['push'], saved['move'], saved['look']
    p_policy = saved['p_policy']

    # Overlay probabilities on top of a specific block
    session_id, block = block_ids[block_idx]
    hmm_probs = p_policy[block_idx]
    ax = plot_pushes(df, conds=dict(subject=subject, session=int(session_id), block=block + 1))
    ax2 = ax.twinx()

    # Set up custom plotting
    cycle = cycler(color=['darkgreen', 'lime'], linestyle=['-', '-'])
    ax2.set_prop_cycle(cycle)
    p = ax2.plot(hmm_probs, label = ['HMM policy 1', 'HMM policy 2'])
    ax2.set_ylabel('HMM Policy Probability')

    # Get existing legend handles and labels
    legend = ax.get_legend()
    existing_handles = legend.legend_handles
    existing_labels = [text.get_text() for text in legend.get_texts()]    

    # Combine existing handles with custom lines
    existing_handles.extend(p)
    existing_labels.extend(['HMM policy 1', 'HMM policy 2'])
    legend.remove()
    ax2.legend(existing_handles, existing_labels)
    return ax2


# @legend_handler
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
    df_block = filter_df(df, conds)
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
    mask = df_block['reward outcomes'].values
    ax.scatter(x_vals[mask], y_vals[mask], c=colors[mask], marker='^', s = 80, zorder = 2)
    ax.scatter(x_vals[~mask], y_vals[~mask], edgecolors=colors[~mask], marker='v', s = 80, zorder = 2, facecolors ="none")
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Create legend manually with proxy artists
    if legend:
        legend_kwargs = kwargs_handler(kwargs, 'legend_kwargs')
        palette = palette_handler(palette, df_block['box'].unique())
        legend_elements = ([Line2D([0], [0], color=palette[j], linestyle='-', label=schedules[i]) for i, j in enumerate(palette.keys())]
                           + [Line2D([0], [0], color='black', linestyle='', marker='^', label='rewarded'),
                              Line2D([0], [0], color='black', linestyle='', marker='v', markerfacecolor = 'none', label='no reward')])
        ax.legend(handles=legend_elements, **legend_kwargs)
    return ax


def plot_pushes(df: pd.DataFrame, conds: dict = None, title: str = '', title_prefix: str = 'Pushes for ',
                      palette: dict = PALETTE, box_labels: list = BOX_POSITIONS,
                      legend: bool = True, ax: plt.Axes = None, **kwargs) -> plt.Axes:
    """
    Plot the pushes in the block by the box they occur at.

    Args:
        df: DataFrame.
        conds: Dictionary to filter df.
        title: Title of figure. If specified, overrides `title_prefix`.
        title_prefix: Prefix of title that is used to construct the contents of the title together with conds. See `titler` for more details. Ignored if `title` is specified.
        palette: Dictionary mapping box schedules to colors. Can also be a list of just colors.
        box_labels: Labels on y-axis for each box.
        legend: If True, display legend. Specify keyword arguments in `legend_kwargs`.
        ax: Axes to plot on. If None, a new figure and axes are created using plt.subplots. Specify keyword arguments in `fig_kwargs`.
        **kwargs: Additional keyword arguments passed to `plot_block_events`.

    Returns:
        The axes.
    """

    ax = plot_block_events(df, conds= conds, title= title, title_prefix= title_prefix, palette= palette, legend= legend, ax= ax, **kwargs)

    # Custom plotting logic
    df_block = filter_df(df, conds).reset_index()
    ax.set_xlim([0, df_block['push times'].max() + 1])
    box_labels = [box_labels[i] for i in sorted(df_block['box position'].unique())]
    ax.set_yticks(range(len(box_labels)), box_labels, rotation = 90, va = 'center')
    ax.set_ylabel('')
    return ax


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
        fig_kwargs = kwargs.pop('fig_kwargs', {})
        _, ax = plt.subplots(**fig_kwargs)

    # Get all unique experiment parameters
    kappas = df.index.unique('kappa').sort_values()
    stim_types = df.index.unique('stimulus type').sort_values()
    shapes = df.index.unique('shape').sort_values()
    n_params = len(kappas) + len(stim_types) + len(shapes)

    # Filter df according to conditions
    df = filter_df(df, conds)
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


def plot_wait_times_by_sessions(df: pd.DataFrame, label_rotation: float = 35, **kwargs):

    # Examine monkeys separately from humans
    monkey_subjects = ['dylan', 'marco', 'viktor']
    conds = {'subject': monkey_subjects}
    df_monkey = filter_df(df, conds).copy()

    # Reformat time data
    df_monkey['session id'] = pd.to_datetime(df_monkey.index.get_level_values('session'), format='%Y%m%d')
    df_ref = df_monkey.groupby('subject')['session id'].min()
    df_monkey = df_monkey.join(df_ref, how='left', rsuffix='_ref', on=['subject'])
    df_monkey['day'] = (df_monkey['session id'] - df_monkey['session id_ref']).dt.days
    fig_kwargs = kwargs_handler(kwargs, 'fig_kwargs', dict(figsize=(25, 10), height_ratios=[1, 2], sharex = True))
    def _plot(i, subj, **kwargs):
        # Plot experiment overview and wait time distribution
        fig, axes = plt.subplots(2, 1, **fig_kwargs)
        plot_experiment_parameters(df_monkey, conds={'subject': subj}, ax=axes[0])
        bp(sns.swarmplot)(df_monkey, x="day", y="consecutive push intervals", conds={'subject': subj}, hue='box',
                          palette=PALETTE, title="Push intervals across sessions", size=1, log_scale=True,
                          legend = False, ax=axes[1], **kwargs) # legend_kwargs={'markerscale': 5}

        # Add weekday labels
        labels = axes[1].get_xticklabels()
        days = df_monkey.xs(subj, level='subject').reset_index().groupby(['session'], as_index=False)['week day'].max()[
            'week day']
        for j, l in enumerate(labels):
            tmp = l.get_text()
            labels[j] = tmp + '\n' + days[j]
        axes[1].set_xticklabels(labels)
        axes[1].tick_params(axis = 'x', labelrotation = label_rotation)
        fig.suptitle(f"Wait times for {subj}")
        fig.tight_layout()
    subject_plotter(monkey_subjects, _plot, **kwargs)

def plot_wait_times(df: pd.DataFrame, stim_reliabilities: dict = KAPPA_LEVELS, palette: dict = PALETTE, palette_dark: dict = PALETTE_DARK, **kwargs) -> plt.Axes:
    """

    Args:
        df:
        stim_reliabilities:
        palette:
        palette_dark:
        stim_reliabilities:
        **kwargs:

    Returns:

    """
    fig_kwargs = kwargs_handler(kwargs, 'fig_kwargs')
    def _plot(i, subj, **kwargs):
        fig, ax = fig_init(None, **fig_kwargs)
        df_subj = filter_df(df, {'subject': subj})
        swarm_kwargs = kwargs_handler(kwargs, 'swarm_kwargs', dict(size=0.5, log_scale=True, dodge=True))
        bp(sns.swarmplot)(df_subj, x='stimulus reliability', order = stim_reliabilities[subj].keys(),
                        y='wait times', hue='box', palette=palette_dark, legend=False, ax=ax, **swarm_kwargs)
        bp(enhanced_violinplot)(df_subj, x='stimulus reliability', order = stim_reliabilities[subj].keys(), y='wait times', hue='box', palette=palette,
                                y_unit='s', cut=0, inner=None,
                                log_scale=True, common_norm=True, ax=ax, **kwargs)
        fig.suptitle(f'Wait times for {subj}')
        fig.tight_layout()
    subject_plotter(df.index.unique('subject'), _plot, **kwargs)


def plot_stay_switch_pushes(df: pd.DataFrame, palette: dict = PALETTE, palette_dark: dict = PALETTE_DARK, null_model: bool = False, **kwargs) -> plt.Axes:
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
        The axes.
    """

    n_boxes = len(palette)
    box_labels = palette.keys()
    fig_kwargs = kwargs_handler(kwargs, 'fig_kwargs', dict(nrows = n_boxes, ncols = n_boxes, figsize=(12, 10), sharex=True, sharey=True))

    def _plot(i, subj, **kwargs):
        fig, axes = fig_init(None, **fig_kwargs)
        df_subj = filter_df(df, {'subject': subj})
        for i, source in enumerate(box_labels):
            for j, dest in enumerate(box_labels):
                ax = axes[i, j]
                subset = df_subj[(df_subj['prev box'] == source) & (df_subj['box'] == dest)]
                color_fill = palette[dest]
                color_fill_dark = palette_dark[dest]
                color_outline = palette[source]
                if not subset.empty:
                    sns.kdeplot(data=subset, x="consecutive push intervals", fill=True, ax=ax, color=color_fill, alpha=0.6)
                    if source != dest:
                        sns.kdeplot(data=subset, x="consecutive push intervals", fill=True, ax=ax, color=color_fill_dark,
                                    alpha=0.6)
                        sns.kdeplot(data=subset, x="consecutive push intervals", fill=False, ax=ax, color=color_outline,
                                    linewidth=2)
                    if null_model:
                        delta = subset['consecutive push intervals'].min()
                        fit_res = fit(expon, subset['consecutive push intervals'] - delta)
                        print(f"Fitting geometric distribution to ({source} -> {dest}) push intervals with offset {delta}", fit_res.params)
                        res = kstest(subset['consecutive push intervals'] - delta, expon.cdf, fit_res.params)
                        print(f"KS-test of ({source} -> {dest}) push intervals", res.pvalue)
                        x = np.sort(subset['consecutive push intervals'].unique())
                        ax.plot(x, expon.pdf(x - delta, fit_res.params[0]), color = 'black', linestyle = '-')

                ax.set_xlabel("")
                if i == 0:
                    ax.set_title(dest, fontsize = 15)
                else:
                    ax.set_title("")

                if j == 0:
                    ax.set_ylabel(source, fontsize = 15)
                    if i == n_boxes - 1:
                        ax.set_xlabel("push interval (s)")
                else:
                    ax.set_ylabel("")
        fig.suptitle(f'Stay and switch push intervals for {subj}', y = 1)
        fig.text(0.5, 0.95, 'TO', ha='center')
        fig.text(0.0, 0.5, 'FROM', va='center', rotation='vertical')
        fig.tight_layout()
    subject_plotter(df.index.unique('subject'), _plot, **kwargs)


def plot_runlengths(df: pd.DataFrame, palette: dict = PALETTE, stim_reliabilities: dict = KAPPA_LEVELS, null_model: bool = False, disp_js: bool = False, **kwargs) -> plt.Axes:
    """
    Plot the

    Args:
        df:
        palette:
        stim_reliabilities:
        title:
        null_model:
        disp_js:
        ax:
        **kwargs: Additional keyword arguments.
            - 'fig_kwargs': Dictionary to specify figure properties when creating a new figure (passed to `plt.subplots`).

    Returns:

    """

    @legend_handler
    def _plot(i, subj, **kwargs):   
        # Identify consecutive pushes and when they switch
        df_subj = filter_df(df, {'subject': subj})
        x = df_subj.index.get_level_values('push #')
        consecutive_mask = x[1:] - x[:-1] == 1
        change_mask = (df_subj['stay/switch'] == 'switch') & np.insert(consecutive_mask, 0, True)
        push_nums = get_blocks(df_subj)["push times"].rank().astype(
            int)  # Calculate from scratch in case pushes got dropped
        change_mask[push_nums == 1] = True

        # Count the runlengths at different boxes
        group_labels = change_mask.cumsum()
        labeled_lengths = pd.DataFrame(
            {"group": group_labels, "box": df_subj['box'],
            "next box": get_blocks(df_subj['box']).shift(-1).fillna('missing')}
        ).set_index(df_subj.index)
        labeled_lengths_all = labeled_lengths.groupby(['stimulus reliability', 'box', 'group']).size().to_frame().rename(
            columns={0: 'length'})
        labeled_lengths_all = labeled_lengths_all[
            (labeled_lengths_all['length'] > 1) & (labeled_lengths_all['length'] <= 10)]

        # Calculate the distribution of runlengths under a dice that is rolled by visitation frequencies
        visit_freqs = df_subj.groupby(['stimulus reliability'])['box'].value_counts(normalize=True).to_frame()
        kappas = stim_reliabilities[subj].keys()
        fig_kwargs = kwargs_handler(kwargs, 'fig_kwargs', {'ncols': len(kappas), 'sharey': True, 'sharex': True})
        fig, ax = fig_init(None, **fig_kwargs)
        for i, kappa in enumerate(kappas):
            try:
                bp(sns.histplot)(labeled_lengths_all.reset_index(), x='length', conds={'stimulus reliability': kappa}, hue = 'box', palette = PALETTE,
                            discrete=True,  common_norm=True, multiple='stack', legend = i == len(kappas) - 1, ax=ax[i], **kwargs)
            except:
                continue

            # Overlay random dice probabilities from geometric distribution
            if null_model:
                bars = ax[i].patches
                bar_width = bars[0].get_width()  # Width of one bar
                probs = visit_freqs.loc[kappa] # Visit probabilities
                boxes = sorted(probs.index.unique('box'))
                handles = ax[i].get_legend().legend_handles
                labels = [t.get_text() for t in ax[i].get_legend().get_texts()]
                for b, box in enumerate(boxes):
                    try:
                        p = probs.loc[box].iloc[0]
                        run_lengths = labeled_lengths_all.loc[(kappa, box), 'length'].sort_values().unique()
                        geom = p ** run_lengths * (1 - p)
                        offset = (b - (len(boxes) - 1) / 2) * bar_width
                        x = run_lengths + offset
                        ax[i].plot(x, geom, c=palette[box], label = 'random dice')
                        if disp_js:
                            bar_heights = get_bar_heights(ax[i], x_centers=run_lengths)
                            # for k, bar in enumerate(bar_heights[box]):
                            #     axes[i, j].text(x[k], bar, f'{jensenshannon(geom, bar):.1f}', ha='center', va='bottom', fontsize = 7)
                            ax[i].set_title(ax[i].get_title() + f'\nJS-distance = {jensenshannon(geom, bar_heights[box])}')
                            # print(f"Jensen-shannon distance between empirical distribution and null distribution of ({subj}, {kappa}, {box}): {jensenshannon(geom, bar_heights[box])}")
                    except:
                        continue
                ax[i].legend(handles = handles + [Line2D([0], [0], color='black', linestyle='-', label='random dice')], labels = labels + ['random dice'] )
        fig.suptitle(f'Runlengths for {subj}')
        fig.tight_layout()
        return ax
    subject_plotter(df.index.unique('subject'), _plot, **kwargs)

def plot_push_intervals_vs_reward_intervals(df: pd.DataFrame, palette: dict = PALETTE, stim_reliabilities: dict = KAPPA_LEVELS, unity = True, annotate_reg: bool = False, **kwargs) -> plt.Axes:
    """
    Plot linear regression of push intervals against reward intervals in a block

    Args:
        df: DataFrame of experiment data for a given block.
        palette:
        stim_reliabilities:
        title: Title of figure. If specified, overrides `title_prefix`.
        unity: If True, then display data in square and color region by reward outcome.
        **kwargs: keyword arguments for seaborn

    Returns:
        axes
    """

    # Remove first push from each box, since reward time is messed up for first pushes
    # df = df.drop(df[df['push # by box'] == 1].index)
    fig_kwargs = kwargs_handler(kwargs, 'fig_kwargs', {'sharey': True, 'sharex': True})
    @legend_handler
    def _plot(i, subj, **kwargs):
        kappas = stim_reliabilities[subj].keys()
        fig_kwargs['ncols'] = len(kappas)
        fig, axes = fig_init(None, **fig_kwargs)
        fit_results = []
        max_x = 0
        df_subj = filter_df(df, {'subject': subj})
        for i, kappa in enumerate(kappas):
            bp(sns.scatterplot)(df_subj, x='reward intervals', y='wait times', conds={'stimulus reliability': kappa}, hue = 'box', palette = palette, ax = axes[i], legend = i == len(kappas) - 1, **kwargs)
            df_subset = filter_df(df_subj, conds={'stimulus reliability': kappa})
            fit_results.append(regplot(df_subset['reward intervals'].to_numpy(), df_subset['wait times'].to_numpy(),
                                line_kws={'color': 'black'}, ax=axes[i], **kwargs))
            max_x = max(max_x, axes[i].get_xlim()[1], axes[i].get_ylim()[1])

        # Add some aesthetics
        for i in range(len(kappas)):
            if unity:
                # max_x = max(axes[i].get_xlim()[1], axes[i].get_ylim()[1])
                x = np.arange(max_x)
                axes[i].plot([0, max_x], [0, max_x], linestyle='dashed', color='black')
                axes[i].fill_between(x, x, max_x, color="green", alpha=0.1)
                axes[i].fill_between(x, x, color="red", alpha=0.1)

            if annotate_reg:
                axes[i].text(0.75, 0.1, f'slope={fit_results[i].params[1]:.2f}',
                            transform=axes[i].transAxes,
                            fontsize = 10)

        fig.suptitle(f"Push intervals vs reward intervals for {subj}")
        fig.tight_layout()
        return axes
    subject_plotter(df.index.unique('subject'), _plot, **kwargs)

def plot_next_push_surprise(df: pd.DataFrame, stim_reliabilities: dict = KAPPA_LEVELS, palette: dict = PALETTE, palette_dark: dict = PALETTE_DARK, **kwargs):
    """Plot the change in wait time after each push.

    Args:
        df: DataFrame containing experiment data
        stim_reliabilities: Dictionary mapping subjects to their stimulus reliability levels
        palette: Dictionary mapping box schedules to colors
        palette_dark: Dictionary mapping box schedules to darkened colors
        **kwargs: Additional keyword arguments
    """

    df = df.copy()
    push_deltas = df.groupby(['subject', 'session', 'block', 'box'])
    df['consecutive wait'] = push_deltas['push # by box'].diff().fillna(1)
    df = df.loc[df['consecutive wait'] == 1]
    df['change in next wait time'] = -push_deltas['wait times'].diff(-1)
    df['rewarded'] = df['reward outcomes'].map({
        True: 'yes',
        False: 'no'
    })
    df['stay/switch'] = df['stay/switch'].shift(-1)

    fig_kwargs = kwargs_handler(kwargs, 'fig_kwargs', {'nrows': 2, 'sharey': True, 'sharex': True})
    @legend_handler
    def _plot(i, subj, **kwargs):
        df_subj = filter_df(df, {'subject': subj})
        kappas = stim_reliabilities[subj].keys()
        fig_kwargs['ncols'] = len(kappas)
        fig, axes = fig_init(None, **fig_kwargs)
        cnt = 0
        for i, ro in enumerate(df_subj['rewarded'].unique()):
            for j, kappa in enumerate(kappas):
                cnt += 1
                bp(sns.scatterplot)(df_subj, x='wait times', y='change in next wait time', conds={'stimulus reliability': kappa, 'rewarded': ro}, hue='box', palette=palette if ro == 'yes' else palette_dark,
                            style='stay/switch', alpha=0.5, ax=axes[i][j], legend = cnt == len(kappas), **kwargs)
                axes[i][j].hlines(0, 0, axes[i][j].get_xlim()[1], linestyles='dashed', colors='black')
                axes[i][j].set_xlim([0, 40])
                axes[i][j].set_ylim([-40, 40])
        fig.suptitle(f'Change in waiting time as a function of reward outcome for {subj}')
        fig.tight_layout()
        return axes
    subject_plotter(df.index.unique('subject'), _plot, **kwargs)

def plot_stay_probabilities(df: pd.DataFrame, stim_reliabilities: dict = KAPPA_LEVELS,  bin_width: float = 10, **kwargs):
    df = df.copy()
    df['rewarded'] = df['reward outcomes'].map({
        True: 'yes',
        False: 'no'
    })
    df['time'] = bin_data(df, 'wait times', bin_width = bin_width)
    df['P(stay)'] = df['stay/switch'].shift(-1).map({
        'stay': 1,
        'switch': 0
    })

    fig_kwargs = kwargs_handler(kwargs, 'fig_kwargs', {'sharey': True, 'sharex': True})
    @legend_handler
    def _plot(i, subj, **kwargs):
        df_subj = filter_df(df, {'subject': subj})
        kappas = stim_reliabilities[subj].keys()
        fig_kwargs['ncols'] = len(kappas)
        fig, ax = fig_init(None, **fig_kwargs)
        for i, kappa in enumerate(kappas):
            bp(sns.lineplot)(df_subj, conds = {'stimulus reliability': kappa}, x='time', y='P(stay)', x_unit = 's', hue='rewarded', hue_order=['no', 'yes'],
                                errorbar='se', ax = ax[i], **kwargs)
        fig.suptitle(f"P(stay) as a function of wait time for {subj}")
        fig.tight_layout()
        return ax
    subject_plotter(df.index.unique('subject'), _plot, **kwargs)


def plot_reward_rates_across_blocks(df: pd.DataFrame, stim_reliabilities: dict = KAPPA_LEVELS, palette: dict = PALETTE, by_box: bool = False, **kwargs):
    """

    Args:
        df:
        stim_reliabilities:
        palette:
        by_box:
        title:
        ax:
        **kwargs:

    Returns:

    """
    # Bin time
    x_bins = 'time'
    df = df.copy()
    bin_kwargs = kwargs_handler(kwargs, 'bin_kwargs', dict(bin_width = 60, strategy = 'full'))
    df[x_bins] = bin_data(df, 'push times', **bin_kwargs)

    # Aggregate rewards by box or across boxes
    if by_box:
        rr = df.groupby(['subject', 'session', 'stimulus reliability', 'block', 'time', 'box'], observed = True)['reward outcomes'].sum()
    else:
        rr = df.groupby(['subject', 'session', 'stimulus reliability', 'block', 'time'], observed = True)['reward outcomes'].sum()

    rr = rr.to_frame().reset_index()
    rr['reward rate'] = rr['reward outcomes'] / rr['time'].apply(lambda x: x.length)
    rr['time'] = rr['time'].apply(lambda x: float(x.left))
    fig_kwargs = kwargs_handler(kwargs, 'fig_kwargs', {'sharey': True, 'sharex': True})
    @legend_handler
    def _plot(i, subj, **kwargs):
        rr_subj = filter_df(rr, {'subject': subj})
        kappas = stim_reliabilities[subj].keys()
        fig_kwargs['ncols'] = len(kappas)
        fig, ax = fig_init(None, **fig_kwargs)
        for i, kappa in enumerate(kappas):
            schedules = filter_df(df, {'stimulus reliability': 'high', 'subject': 'viktor'})[['box', 'schedule']].value_counts().index.tolist()
            if by_box:
                bp(sns.lineplot)(rr_subj, conds = {'stimulus reliability': kappa}, x = 'time', y='reward rate', hue='box', palette=palette, ax = ax[i], legend = i == len(kappas) - 1, **kwargs)
            else:
                bp(sns.lineplot)(rr_subj, conds = {'stimulus reliability': kappa}, x = 'time', y='reward rate', ax = ax[i], legend = i == len(kappas) - 1, **kwargs)
        fig.suptitle(f"Reward rate for {subj}")
        fig.tight_layout()
        return ax
    subject_plotter(df.index.unique('subject'), _plot, **kwargs)


def plot_quantity_in_block(df: pd.DataFrame, y: str = None, stim_reliabilities: list = KAPPA_LEVELS, palette: dict = PALETTE, auxiliary_plot: Callable = None, **kwargs):

    # Bin time
    x_bins = 'time'
    df.copy()
    bin_kwargs = kwargs_handler(kwargs, 'bin_kwargs', dict(bin_width = 60))
    df[x_bins] = bin_data(df, 'push times', **bin_kwargs)

    # Plot quantity of interest over time in the block
    fig_kwargs = kwargs_handler(kwargs, 'fig_kwargs', {'sharey': True, 'sharex': True})
    @legend_handler    
    def _plot(i, subj, **kwargs):
        df_subj = filter_df(df, {'subject': subj})
        kappas = stim_reliabilities[subj].keys()
        fig_kwargs['ncols'] = len(kappas)
        fig, ax = fig_init(None, **fig_kwargs)
        for i, kappa in enumerate(kappas):
            bp(sns.lineplot)(df_subj, conds={'stimulus reliability': kappa}, x='time', y=y, hue='box',
                            palette=palette, ax=ax[i], legend = i == len(kappas) - 1, **kwargs)
            if auxiliary_plot:
                auxiliary_plot(df, conds={'stimulus reliability': kappa, 'subject': subj}, palette = palette, ax = ax[i], **kwargs)
        fig.suptitle(f"{y} for {subj}")
        fig.tight_layout()
        return ax
    subject_plotter(df.index.unique('subject'), _plot, **kwargs)


def plot_matching_law(df: pd.DataFrame, stim_reliabilities: list = KAPPA_LEVELS, palette: dict = PALETTE, **kwargs):
    x_bins = 'time'
    df = df.copy()
    bin_kwargs = kwargs_handler(kwargs, 'bin_kwargs', dict(bin_width = 60, strategy = 'full'))
    df[x_bins] = bin_data(df, 'push times', **bin_kwargs)
    grouped = df.groupby(['subject', 'session', 'stimulus reliability', 'block', 'time', 'box'], observed=True)
    rr = grouped['reward outcomes'].sum()
    rr = rr.to_frame().reset_index()
    time_bins = rr['time'].apply(lambda x: x.length)
    rr['reward rate'] = rr['reward outcomes']
    rr['push rate'] = (grouped.size().reset_index()[0]) #.reset_index()[0]
    rr['ratio'] = rr['reward rate'] / rr['push rate'] / time_bins
    rr['time'] = rr['time'].apply(lambda x: float(x.left))

    fig_kwargs = kwargs_handler(kwargs, 'fig_kwargs', {'sharey': True, 'sharex': True})
    @legend_handler
    def _plot(i, subj, **kwargs):
        kappas = stim_reliabilities[subj].keys()
        fig_kwargs['ncols'] = len(kappas)
        fig, ax = fig_init(None, **fig_kwargs)
        rr_subj = filter_df(rr, {'subject': subj})
        for i, kappa in enumerate(kappas):
            bp(sns.lineplot)(rr_subj, conds = {'stimulus reliability': kappa}, x = 'time', y='ratio', hue='box', palette=palette, ax = ax[i], legend = i == len(kappas) - 1, **kwargs)
            if i == 0:
                ax[i].set_ylabel('efficiency')
        fig.suptitle(f"Matching law for {subj}")
        fig.tight_layout()
        return ax
    subject_plotter(df.index.unique('subject'), _plot, **kwargs)


def plot_theoretical_fisher_info(t, schedules = [7, 14, 21], alpha = 10, figsize = (10, 5)):
    def _fisher_info(t, schedule, alpha):
        scale = schedule / alpha
        cdf = gamma.cdf(t, a = alpha, scale = scale)
        return (t/schedule)**2*gamma.pdf(t, a = alpha, scale = scale)**2/(cdf*(1-cdf))

    schedules = [7, 14, 21]
    alpha = 10
    optimal_fisher_t = {}
    optimal_fisher_info = {}
    for schedule in schedules:
        print(f'Finding optimal t for schedule {schedule}')
        res = minimize_scalar(lambda x: -_fisher_info(x, schedule, alpha))
        # res = minimize_scalar(lambda x: -gamma.pdf(x, a = alpha, scale = schedule / alpha))

        # Result
        x_max = res.x
        f_max = _fisher_info(x_max, schedule, alpha)

        print(f"Argmax: {x_max:.4f}, Max value: {f_max:.4f}")
        optimal_fisher_info[schedule] = f_max
        optimal_fisher_t[schedule] = x_max

    fig, ax = plt.subplots(figsize = figsize)
    fishers = []
    for schedule in schedules:
        fishers.append(_fisher_info(t, schedule, alpha))
    [ax.plot(t, fishers[j], label = schedules[j], color = BOX_COLORS[j]) for j in range(len(schedules))]
    ax.set_ylabel("fisher information")
    ax.set_xlabel("push time (s)")
    ax.vlines(x = optimal_fisher_t.values(), ymin = 0, ymax = 0.15, color = BOX_COLORS, linestyle = '--', label = r'$t^*=\underset{t}{\mathrm{argmax}}J_{\lambda}(t)$')# ax.legend()



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

