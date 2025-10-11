"""
This module contains functions used to load and process the data, including useful data manipulations to perform on experiment data.
"""

import fnmatch
import logging
import os
import pickle
from datetime import datetime
from functools import wraps
from typing import Any, Callable

import h5py
import numpy as np
import pandas as pd
from IPython.core.display import Markdown
from IPython.core.display_functions import DisplayHandle, display
from IPython.display import HTML
from numpy.typing import ArrayLike
from pandas.core.groupby import DataFrameGroupBy
from scipy.io import loadmat
from tqdm import tqdm

from foraging.config.experiments import (
    AngelakiExperimentConstants,
    ValentinExperimentConstants,
)
from foraging.models import Experiment, HashableDict

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def open_pickle_file(path: str) -> Any:
    """
    Loads data from a pickle file.

    Args:
        path: The file path of the pickle file to load.

    Returns:
        Contents of pickle file.
    """

    # Open the pickle file and load its contents
    with open(path, "rb") as f:
        ds = pickle.load(f)

    # Return all contents
    return ds


def display_df(df: pd.DataFrame, cols: list[str]) -> DisplayHandle:
    """
    Pretty prints dataframe's specified columns.

    Args:
        df: The dataframe to print.
        cols: The columns of dataframe to print.

    Returns:
        A display object (HTML or Markdown)
    """
    # Option 1: Use HTML formatting (no external dependencies)
    # return display(HTML(df.head()[cols].to_html(classes='table table-striped')))

    # Option 2: Simple markdown without tabulate (uncomment to use)
    table_data = df.head().reset_index()[cols]
    markdown_lines = []
    markdown_lines.append("| " + " | ".join(cols) + " |")
    markdown_lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in table_data.iterrows():
        markdown_lines.append("| " + " | ".join(str(val) for val in row) + " |")
    return display(Markdown("\n".join(markdown_lines)))


def map_box_positions_to_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map box positions to ranks. Usually `df` is a block.
    """
    return (
        df[["box position", "box rank"]]
        .drop_duplicates()
        .set_index("box position")
        .sort_index()
    )


def filter_df(
    df: pd.DataFrame, conds: dict[str, Any] = None, attempt_index: bool = False
) -> pd.DataFrame:
    """
    Filter a DataFrame according to conditions specified in a dictionary. If a condition does not exist, it is ignored.

    Args:
        df: DataFrame containing experiment data.
        conds: Dictionary mapping column or index level names to values to filter on.
        attempt_index: Flag indicating whether to attempt converting `conds` to MultiIndex key. Defaults to False. Turn
        on if you want slight performance boost in exchange for dropping the filtered index levels using .loc on `conds` as MultiIndex key.

    Returns:
        Filtered DataFrame.
    """

    if conds is not None:
        mask = np.full(len(df), True)
        if attempt_index:
            keys = conds.keys()
            if set(keys) == set(df.index.names[: len(keys)]):
                try:
                    return df.loc[
                        tuple(conds.values())
                    ]  # Bear in mind this will drop levels
                except:
                    pass
        for k, v in conds.items():
            v = np.atleast_1d(v)
            if k in df.index.names:  # Filter on index levels
                mask &= df.index.get_level_values(k).isin(v)
            elif k in df.columns:  # Filter on columns
                mask &= df[k].isin(v)
        return df[mask]
    return df


def bin_data(
    x: ArrayLike | pd.Series,
    bins: int | list[float] = 10,
    bin_width: float = None,
    include_lowest: bool = True,
    strategy: str = "right",
    remove_unused_categories: bool = True,
    **kwargs,
) -> pd.Series:
    """
    Bins data in the specified column of the DataFrame, with support for different binning strategies. This function
    performs binning on the specified column (`x`) and labels the bins according to the selected strategy. It supports
    defining the number of bins or specific bin edges.

    Args:
        x: Array-like data or pandas Series to bin.
        bins: Number of bins or list of bin edges.
            If an integer is provided, the data will be divided into that number of equal-width bins.
            If a list of floats is provided, it will specify the bin edges.
            Defaults to 20.
        bin_width: If specified, this is the width of each bin. Bins will be determined by dividing the range into equal-sized bins of this width.
        strategy: Labeling strategy for the bins.
            - 'full': Labels the bins using the full interval (i.e., both left and right edges).
            - 'left': Labels the bins using only the left edge.
            - 'right': Labels the bins using only the right edge.
            - 'center': Labels the bins using the center of the bin.
            Defaults to 'right'.

    Returns:
        A pandas Series containing the binned data.

    Example:
        df = pd.DataFrame({'value': np.random.randn(100)})
        df['binned'] = bin_data(df['value'], bins=5, strategy='right')
    """
    # Perform initial binning based on n_bins or custom bin edges
    if bin_width:
        bins = np.arange(start=x.min(), stop=x.max() + bin_width, step=bin_width)
    binned = pd.cut(x, bins=bins, include_lowest=include_lowest, **kwargs)
    if len(binned) == 2:
        binned, bins = binned
    dtype = x.dtype  # Get the dtype of the column to maintain consistency in bin labels

    cats = None
    if hasattr(binned, "categories"):
        cats = binned.categories
    elif hasattr(binned, "cat"):
        cats = binned.cat.categories
    else:
        raise ValueError("Input is not a numpy array or pandas series")

    # Select the appropriate bin labels based on the strategy
    match strategy:
        case "full":
            bin_edges = cats
        case "right":
            bin_edges = cats.right.astype(dtype)
        case "left":
            bin_edges = cats.left.astype(dtype)
        case _:
            bin_edges = ((cats.left + cats.right) / 2).astype(dtype)

    # Apply the bin labels to the original data
    binned = pd.cut(
        x, bins=bins, include_lowest=include_lowest, labels=bin_edges, **kwargs
    )
    ret_bins = None
    if len(binned) == 2:
        binned, ret_bins = binned

    if remove_unused_categories:
        if hasattr(binned, "cat"):
            binned = binned.cat.remove_unused_categories()
        else:
            binned = binned.remove_unused_categories()

    if ret_bins is not None:
        return binned, ret_bins
    else:
        return binned


## Convenience scripts for specific experiments
def get_continuous_from_block(f: h5py.File, session: int, block: int) -> dict:
    """
    Extract continuous data from a specified session and block in an HDF5 file.

    Args:
        f: The HDF5 file object containing the session data.
        session: The session number from which to extract data (1-indexed).
        block: The block number within the session to extract data from (1-indexed).

    Returns:
        A dictionary containing the continuous data arrays, including:
            - 'eye_arena_int': gaze locations in arena (e.g., 3D coordinates)
            - 'eye_vertical': Vertical eye position data
            - 'eye_horizontal': Horizontal eye position data
            - 'head_dir_vec': Head direction vector data
            - 'visual_cue': Visual cue signal data
            - 'position': Position data (e.g., 3D coordinates)
            - 'time': Time data for the recorded events
        If the data is inconsistent, returns an empty dictionary.
    """

    try:
        # Retrieve data from the HDF5 file for the specified session and block
        eye_arena_int = f[
            f[f["session"]["block"][session - 1, 0]]["continuous"][block - 1, 0]
        ].get("eyeArenaInt")[:]
        eye_vertical = (
            f[f[f["session"]["block"][session - 1, 0]]["continuous"][block - 1, 0]]
            .get("eyeV")[:]
            .flatten()
        )
        eye_horizontal = (
            f[f[f["session"]["block"][session - 1, 0]]["continuous"][block - 1, 0]]
            .get("eyeH")[:]
            .flatten()
        )
        head_dir_vec = f[
            f[f["session"]["block"][session - 1, 0]]["continuous"][block - 1, 0]
        ].get("headDirVec")[:]
        visual_cue = f[
            f[f["session"]["block"][session - 1, 0]]["continuous"][block - 1, 0]
        ].get("visualCueSignal")
        visual_cue_dict = {k: visual_cue[k][:].flatten() for k in visual_cue.keys()}
        pos = f[
            f[f["session"]["block"][session - 1, 0]]["continuous"][block - 1, 0]
        ].get("position")[:]
        t = (
            f[f[f["session"]["block"][session - 1, 0]]["continuous"][block - 1, 0]]
            .get("t")[:]
            .flatten()
        )

        # Ensure all data arrays have the same length and are in the correct format
        if (
            len(t) == len(pos)
            and pos.ndim > 1
            and pos.shape[1] == 3
            and len(t) == len(head_dir_vec)
            and len(t) == len(eye_horizontal)
            and len(t) == len(eye_vertical)
            and len(t) == len(eye_arena_int)
        ):
            return {
                "eye_arena_int": eye_arena_int,
                "eye_vertical": eye_vertical,
                "eye_horizontal": eye_horizontal,
                "head_dir_vec": head_dir_vec,
                "visual_cue": visual_cue_dict,
                "position": pos,
                "time": t,
            }
    except Exception as e:
        # If there is an error in data extraction, return an empty result
        logger.debug(f"Could not process (session {session}, block {block}): {str(e)}")
    return {}


# This works for angelaki experiment
def get_continuous_from_df_to_dict(
    experiment: Experiment, data_dir: str
) -> tuple[dict, set]:
    """
    Extract continuous data from blocks and return it as a dictionary.

    Args:
        experiment: The Experiment dataset containing session and block information.
        data_dir: The directory path where the subject files are located.

    Returns:
        A tuple containing:
            - data: A dictionary with extracted continuous data from the blocks. Keys are the block key from the Experiment dataset.
            - errors: A set of block keys of blocks causing errors encountered during the extraction process.
    """
    # Retrieve the list of subjects and open their respective files
    subjects = experiment.get_unique("subject")
    files = {subj: open_angelaki_subject_file(subj, data_dir) for subj in subjects}

    def _inner(experiment: Experiment, block_key: dict, block: pd.DataFrame):
        """
        Helper function to extract continuous data from a specific block and session for a given block key.

        Args:
            experiment: The Experiment dataset containing session and block information.
            block_key: The key of the current block, assumed to contain subject and block.
            block: The block DataFrame.

        Returns:
            dict: The extracted continuous data for the specified block and session.
        """

        _sess = block["_session"].iloc[0]
        return get_continuous_from_block(
            files[block_key["subject"]], _sess, block_key["block"]
        )

    # Process the blocks using the helper function
    data, errors = experiment.process_blocks(_inner)

    # Filter out blocks that don't have any data
    data = {k: v for k, v in data.items() if len(v) > 0}

    # Close all subject files
    [f.close() for f in files.values()]

    return data, errors


def open_angelaki_subject_file(subject: str, path: str = ".") -> h5py.File:
    """
    Open an HDF5 subject file. Remember to close this on your own as you would any regular file.

    Args:
        subject: Subject identifier.
        path: Path to the directory containing the file. Defaults to '.'.

    Returns:
        Opened HDF5 file.
    """

    return h5py.File(os.path.join(path, f"data_{subject}.mat"), "r")


def open_angelaki_meta_file(subject: str, path: str = ".") -> pd.DataFrame:
    """
    Open and preprocess an Excel metadata file for a given subject.

    Args:
        subject: Subject identifier.
        path: Path to the directory containing the file. Defaults to '.'.

    Returns:
        Processed metadata DataFrame.
    """

    df_meta = pd.read_excel(os.path.join(path, f"table_{subject}.xlsx"))
    return df_meta


def make_angelaki_experiment(
    path: str,
    config: AngelakiExperimentConstants = None,
) -> Experiment:
    """
    Given experiment matfiles and metadata, construct a DataFrame.

    Args:
        path: Path to the folder containing the experiment data.
        config: Configuration object for the experiment.

    Returns:
        An Experiment dataset containing a configuration and DataFrame.
    """
    if config is None:
        config = AngelakiExperimentConstants()

    # Identify all subjects in the given directory
    subject_files = fnmatch.filter(os.listdir(path), "data_*.mat")
    subjects = [subject_file.split(".")[0][5:] for subject_file in subject_files]
    df_dict = {
        "subject": [],
        "session": [],
        "_session": [],  # the original session number in sequential order inside the matlab struct
        "weekday": [],
        "n boxes": [],
        "block": [],
        "schedule": [],
        "assigned schedules": [],
        "box rank": [],
        "shape": [],
        "stimulus type": [],
        "kappa": [],
        "stimulus reliability": [],
        "stimulus reliability order": [],
        "eye tracking": [],
        "position tracking": [],
        "duration": [],
        "box position": [],
        "push times": [],
        "push intervals": [],
        "reward outcomes": [],
        "reward intervals": [],
    }

    day_to_week = {0: "M", 1: "T", 2: "W", 3: "R", 4: "F", 5: "S", 6: "U"}
    box_labels = dict(zip(range(len(config.BOX_LABELS)), config.BOX_LABELS))
    box_pos_labels = dict(zip(range(len(config.BOX_POSITIONS)), config.BOX_POSITIONS))
    for subject in subjects:

        # Load MATLAB file
        with open_angelaki_subject_file(subject, path) as f:

            # Load metafile
            df_meta = open_angelaki_meta_file(subject, path)
            all_sess_data = f["session"]["block"]
            all_sess_data_cue = (
                f["session"]["blocks"] if "blocks" in f["session"] else None
            )
            n_sessions = len(
                all_sess_data
            )  # For humans, sessions = individual subjects
            week_day = None

            # For each session
            for sess_idx in range(n_sessions):

                # Get session id, depending on type of subject
                sess_id = (
                    f[f["session"]["id"][sess_idx, 0]][0, 0]
                    if subject == "humans"
                    else "".join(
                        [chr(c) for c in f[f["session"]["id"][sess_idx, 0]][:, 0]]
                    )
                )

                # If session is date, get weekday
                if subject != "humans":
                    week_day = day_to_week[
                        datetime.strptime(sess_id, "%Y%m%d").weekday()
                    ]
                sess_id = int(sess_id)

                sess_data = f[all_sess_data[sess_idx, 0]]["events"]
                param_data = f[all_sess_data[sess_idx, 0]]["params"]
                param_data_cue = (
                    f[all_sess_data_cue[sess_idx, 0]]["params"]
                    if all_sess_data_cue
                    else None
                )

                # For each block in session
                n_blocks = len(sess_data)
                for block_idx in range(n_blocks):
                    try:

                        # Get block's metadata
                        block_meta = df_meta.loc[
                            (df_meta["sessionId"] == sess_id)
                            & (df_meta["blockId"] == block_idx + 1)
                        ].iloc[0]

                        schedules = tuple(
                            [float(x) for x in block_meta["scheduleMean"].split(",")]
                        )
                        box_ranks = np.argsort(
                            np.argsort(schedules)
                        )  # fast, medium, slow

                        kappa = block_meta["stimulusNoise"]
                        stim_reliability = config.KAPPA_LEVELS[subject][kappa]
                        stim_reliability_order = config.KAPPA_LEVELS_ORDER[
                            stim_reliability
                        ]
                        shape = block_meta["GammaShape"]
                        stim_type = block_meta["stimulusCueType"]
                        eye_tracking = block_meta["eyeTracking"]
                        position_tracking = block_meta["position/head variables"]
                        duration = (
                            f[sess_data[block_idx, 0]].get("tEndBeh")[0, 0]
                            - f[sess_data[block_idx, 0]].get("tStartBeh")[0, 0]
                            if subject != "humans"
                            else f[sess_data[block_idx, 0]].get("tEnd")[0, 0]
                            - f[sess_data[block_idx, 0]].get("tStart")[0, 0]
                        )

                        # Parse box data
                        staging_dict = {k: [] for k in df_dict.keys()}
                        for i in range(len(schedules)):
                            box = "box" + str(i + 1)
                            push_times = np.atleast_1d(
                                f[sess_data[block_idx, 0]].get("tPush/" + box)
                            ).ravel()  # Make sure data is an array before attempting to flatten it

                            reward_outcomes = (
                                np.atleast_1d(
                                    f[sess_data[block_idx, 0]].get("pushLogical/" + box)
                                )
                                .astype(bool)
                                .ravel()
                            )

                            # Populate metadata
                            n_events = len(push_times)
                            staging_dict["subject"].extend(
                                [subject for _ in range(n_events)]
                            )
                            staging_dict["session"].extend(
                                [sess_id for _ in range(n_events)]
                            )
                            staging_dict["_session"].extend(
                                [sess_idx + 1 for _ in range(n_events)]
                            )
                            staging_dict["weekday"].extend(
                                [week_day for _ in range(n_events)]
                            )
                            staging_dict["block"].extend(
                                [block_idx + 1 for _ in range(n_events)]
                            )
                            staging_dict["n boxes"].extend(
                                [len(schedules) for _ in range(n_events)]
                            )
                            staging_dict["schedule"].extend(
                                [schedules[i] for _ in range(n_events)]
                            )
                            staging_dict["assigned schedules"].extend(
                                [schedules for _ in range(n_events)]
                            )
                            staging_dict["box rank"].extend(
                                [box_ranks[i] for _ in range(n_events)]
                            )
                            staging_dict["kappa"].extend(
                                [kappa for _ in range(n_events)]
                            )
                            staging_dict["stimulus reliability"].extend(
                                [stim_reliability for _ in range(n_events)]
                            )
                            staging_dict["stimulus reliability order"].extend(
                                [stim_reliability_order for _ in range(n_events)]
                            )
                            staging_dict["shape"].extend(
                                [shape for _ in range(n_events)]
                            )
                            staging_dict["eye tracking"].extend(
                                [eye_tracking for _ in range(n_events)]
                            )
                            staging_dict["position tracking"].extend(
                                [position_tracking for _ in range(n_events)]
                            )
                            staging_dict["duration"].extend(
                                [duration for _ in range(n_events)]
                            )
                            staging_dict["box position"].extend(
                                [i for _ in range(n_events)]
                            )
                            staging_dict["stimulus type"].extend(
                                [stim_type for _ in range(n_events)]
                            )

                            # Populate push-specific data
                            staging_dict["push times"].extend(push_times)
                            staging_dict["push intervals"].extend(
                                np.insert(
                                    push_times[1:] - push_times[:-1], 0, push_times[0]
                                )
                            )
                            staging_dict["reward outcomes"].extend(reward_outcomes)

                            # Populate reward interval for each push
                            color_cue = (
                                f[param_data_cue[block_idx, 0]]["rewardWaitTime"]
                                if param_data_cue
                                else f[param_data[block_idx, 0]]["rewardWaitTime"]
                            )
                            if color_cue:
                                staging_dict["reward intervals"].extend(
                                    np.atleast_1d(color_cue.get(box)).ravel()[
                                        : len(push_times)
                                    ]
                                )
                            else:
                                staging_dict["reward intervals"].extend(
                                    -np.ones(len(push_times))
                                )

                        # Check all fields are equal length
                        length = None
                        for k, v in staging_dict.items():
                            if length is None:
                                length = len(v)
                            if len(v) != length:
                                logger.debug(
                                    f"Could not parse ({subject}, {sess_id}, {block_idx + 1}) due to incorrect size of arrays"
                                )
                                length = False
                                break
                        if length:
                            [v.extend(staging_dict[k]) for k, v in df_dict.items()]
                    except Exception as e:  # If block causes issue, skip it
                        logger.debug(
                            f"Could not parse ({subject}, {sess_id}, {block_idx + 1})"
                        )
                        logger.debug(e)

    # Make DataFrame and sort by relevant fields
    df = pd.DataFrame(df_dict).sort_values(
        by=["subject", "session", "block", "push times"]
    )

    # Add some more columns
    df["box"] = df["box rank"].map(box_labels)
    df["box position label"] = df["box position"].map(box_pos_labels)
    df["prev box"] = df.groupby(["subject", "session", "block"], observed=True)[
        "box"
    ].shift(1)
    df["normalized pushes"] = df["push intervals"] / df["schedule"]
    df["consecutive push intervals"] = df["push times"].diff()
    df["eye tracking"] = df["eye tracking"].map({"TRUE": True, "FALSE": False})
    df["position tracking"] = df["position tracking"].map(
        {"TRUE": True, "FALSE": False}
    )
    df["rewarded?"] = df["reward outcomes"].map({True: "Yes", False: "No"})
    df["push #"] = (
        df.groupby(["subject", "session", "block"])["push times"].rank().astype(int)
    )
    df["push # by box"] = (
        df.groupby(["subject", "session", "block", "box rank"])["push times"]
        .rank()
        .astype(int)
    )
    df["stay/switch"] = (
        df["box rank"].diff().astype(bool).map({False: "stay", True: "switch"})
    )

    # Convenience columns
    df["time"] = df["push times"]
    df["_time"] = pd.to_datetime(df["time"], unit="s")
    df[r"$\kappa$"] = df["kappa"]
    df[r"$\alpha$"] = df["shape"]

    # Correct some columns
    df["shape"] = df["shape"].astype(int)
    df["subject"] = df["subject"].astype(str)
    df.loc[df["push #"] == 1, "consecutive push intervals"] = df.loc[
        df["push #"] == 1, "push times"
    ]  # Make sure the first push interval of each block is just the time of that push
    df.loc[df["push #"] == 1, "stay/switch"] = (
        "stay"  # Count first push of each block as 'stay' push, so that stay pushes are a subset of same-box push intervals
    )

    # Finally, denote the label order for some columns
    label_order = {
        "stimulus reliability": "stimulus reliability order",
        "box": "box rank",
    }

    # Drop all push intervals with value 0 as these are bad data
    df = df[df["consecutive push intervals"] > 0]
    return Experiment(
        df,
        config.to_dict(),
        ["subject", "session", "block"],
        ["push #"],
        ["stimulus reliability", "stimulus type", "kappa", "shape"],
        ["assigned schedules", "weekday"],
        label_order=label_order,
    )


def angelaki_exclusion_criteria(experiment: Experiment, data_dir: str) -> Experiment:
    """
    Applies exclusion criteria to filter out data based on specific conditions.

    This function performs several exclusions based on the following criteria:
    1. Excludes blocks with fewer than 10 pushes.
    2. Excludes blocks with a schedule value of 80.
    3. Excludes rows where consecutive push intervals are greater than 30 s.
    4. Excludes rows where the average vertical position exceeds 750 mm.

    Args:
        experiment: The Experiment dataset containing the data to be filtered.
        data_dir: The directory path where the subject files are located.

    Returns:
        A filtered dataset that has had the exclusion criteria applied.
    """
    # Count the number of pushes per block
    n_pushes_per_block = experiment.blocks.size().reset_index(name="n pushes per block")
    df = experiment.df

    # Exclude blocks with fewer than 10 pushes
    df_filtered = df.drop(
        n_pushes_per_block.loc[n_pushes_per_block["n pushes per block"] < 10]
        .set_index(experiment.block_identifiers)
        .index
    )

    # Exclude blocks where the schedule is 80
    df_filtered = df_filtered.drop(
        (df_filtered[df_filtered["schedule"] == 80])
        .groupby(experiment.block_identifiers)
        .size()
        .index
    )

    # Exclude pushes where the consecutive push intervals are greater than 30 s
    df_filtered = df_filtered.drop(
        df_filtered[df_filtered["consecutive push intervals"] > 30].index
    )

    # Get blocks with valid position data
    continuous_data, _ = get_continuous_from_df_to_dict(
        experiment.wrap(df_filtered), data_dir
    )
    df_cont_content = []
    for block, data in continuous_data.items():
        if np.any(
            data["position"] != np.nan
        ):  # If a single position is not nan, then the block has valid position data
            df_cont_content.append(filter_df(df_filtered, block))

    # Exclude pushes where the average vertical position exceeds the height of the boxes (750 mm)
    df_cont = pd.concat(df_cont_content)
    sess_idx = experiment.block_identifiers.index("session")
    block_idx = experiment.block_identifiers.index("block")
    for subject in df_cont.index.unique("subject"):
        df_cont_subject = filter_df(df_cont, {"subject": subject})
        # For each push interval, check if the average vertical position exceeds the height of the boxes (750 mm)
        for idx, row in df_cont_subject.iterrows():
            try:
                conds = HashableDict(
                    dict(
                        subject=subject,
                        session=idx[sess_idx],
                        block=idx[block_idx],
                    )
                )
                time = continuous_data[conds]["time"]
                vertical = continuous_data[conds]["position"][:, 2]

                # Identify push interval time window
                push_interval_end = row["push times"]
                push_interval_start = (
                    push_interval_end - row["consecutive push intervals"]
                )
                x = np.abs(
                    time[None, :]
                    - np.array([[push_interval_start], [push_interval_end]])
                ).argmin(axis=1)

                # Calculate average vertical position over push interval
                v = vertical[x[0] : x[1]]
                mean_vertical = v[~np.isnan(v)].mean()
                if not np.isnan(mean_vertical):
                    # If the average vertical position exceeds the height of the boxes (750 mm), exclude the push interval
                    if mean_vertical > 750:
                        df_filtered = df_filtered.drop(idx)
            except:
                continue
    return experiment.wrap(df_filtered)


# TODO: combine all subject files
def make_valentin_experiment(
    path: str, config: ValentinExperimentConstants = None
) -> Experiment:
    """
    Given experiment matfiles and metadata, construct a DataFrame.

    Args:
        path: Path to the folder containing the experiment data.
        config: Configuration object for the experiment.

    Returns:
        An Experiment dataset containing a configuration and DataFrame.
    """
    if config is None:
        config = ValentinExperimentConstants()

    df_dict = {
        "subject": [],
        "session": [],
        "block": [],
        "_session": [],  # the original session number in sequential order inside the matlab struct
        "n boxes": [],
        "schedule": [],
        "assigned schedules": [],
        "box position": [],
        "push times": [],
        "reward outcomes": [],
        "reward intervals": [],
        "duration": [],
        "eye tracking": [],
        "position tracking": [],
    }
    mat_filenames = fnmatch.filter(os.listdir(path), "*.mat")
    mat_files = [loadmat(os.path.join(path, _file)) for _file in mat_filenames]

    box_labels = dict(zip(range(len(config.BOX_LABELS)), config.BOX_LABELS))
    box_pos_labels = dict(zip(range(len(config.BOX_POSITIONS)), config.BOX_POSITIONS))
    for mat_file in mat_files:
        session_names = mat_file["sessionName"][:, 0]
        sessions = []
        for s in session_names:
            if not "MonitorsOff" in s.item():
                sessions.append(int(s.item()[:8]))

        for sess_idx, sess in enumerate(sessions):
            sess_data = mat_file["data"][:, sess_idx].item()[0]
            push_times = sess_data[:, 0]
            n_pushes = len(push_times)
            # push_intervals = np.diff(push_times)
            # push_intervals = np.insert(push_intervals, 0, push_times[0])
            box_positions = sess_data[:, 1].astype(int)
            schedules = sess_data[:, 2]
            n_boxes = len(np.unique(schedules))
            reward_outcomes = sess_data[:, 5].astype(bool)
            # TODO: check if reward intervals are correct
            reward_intervals = sess_data[:, 3:5]
            reward_intervals = reward_intervals[np.arange(n_pushes), box_positions - 1]

            assigned_schedules = tuple(
                [schedules[box_positions == 1][0], schedules[box_positions == 2][0]]
            )

            df_dict["subject"].extend(["Rum" for _ in range(n_pushes)])
            df_dict["session"].extend([sess for _ in range(n_pushes)])
            df_dict["block"].extend([1 for _ in range(n_pushes)])
            df_dict["_session"].extend([sess_idx for _ in range(n_pushes)])
            df_dict["n boxes"].extend([n_boxes for _ in range(n_pushes)])
            df_dict["schedule"].extend(schedules)
            df_dict["assigned schedules"].extend(
                [assigned_schedules for _ in range(n_pushes)]
            )
            df_dict["box position"].extend(box_positions)
            df_dict["push times"].extend(push_times)
            df_dict["reward outcomes"].extend(reward_outcomes)
            df_dict["reward intervals"].extend(reward_intervals)
            df_dict["duration"].extend(
                [push_times[-1] + 30 for _ in range(n_pushes)]
            )  # placeholder for duration

            df_dict["eye tracking"].extend([False for _ in range(n_pushes)])
            df_dict["position tracking"].extend([True for _ in range(n_pushes)])

    df = pd.DataFrame(df_dict).sort_values(by=["subject", "session", "push times"])
    df["reward outcomes"] = df["reward outcomes"].astype(bool)
    df["push #"] = df.groupby(["subject", "session"])["push times"].rank().astype(int)
    df["push intervals"] = df.groupby(["subject", "session", "box position"])[
        "push times"
    ].diff()
    df.loc[df["push #"] == 1, "push intervals"] = df.loc[
        df["push #"] == 1, "push times"
    ]  # Make sure the first push interval of each block is just the time of that push
    df["box rank"] = (
        df.groupby(["subject", "session"])["schedule"].rank(method="dense").astype(int)
        - 1
    )
    df["box"] = df["box rank"].map(box_labels)
    df["box position"] = df["box position"].astype(int) - 1
    df["box position label"] = df["box position"].map(box_pos_labels)
    df["prev box"] = df.groupby(["subject", "session"], observed=True)["box"].shift(1)
    df["normalized pushes"] = df["push intervals"] / df["schedule"]
    df["consecutive push intervals"] = df["push times"].diff()
    df["rewarded?"] = df["reward outcomes"].map({True: "Yes", False: "No"})
    df["push # by box"] = (
        df.groupby(["subject", "session", "box rank"])["push times"].rank().astype(int)
    )
    df["stay/switch"] = (
        df["box rank"].diff().astype(bool).map({False: "stay", True: "switch"})
    )

    # Add stimulus reliability column
    df_meta = pd.read_excel(
        os.path.join(path, "ForagingData_FileMasterlist.xlsx"), "session_reliability"
    )
    df = df.merge(df_meta, on="session", how="left")
    df["stimulus reliability order"] = df["stimulus reliability"].map(
        config.KAPPA_LEVELS_ORDER
    )
    df[r"$\kappa$"] = df["stimulus reliability"]

    # Convenience columns
    df["time"] = df["push times"]
    df["_time"] = pd.to_datetime(df["time"], unit="s")

    # Correct some columns
    df["subject"] = df["subject"].astype(str)
    df.loc[df["push #"] == 1, "consecutive push intervals"] = df.loc[
        df["push #"] == 1, "push times"
    ]  # Make sure the first push interval of each block is just the time of that push
    df.loc[df["push #"] == 1, "stay/switch"] = (
        "stay"  # Count first push of each block as 'stay' push, so that stay pushes are a subset of same-box push intervals
    )

    # Drop all push intervals with value 0 as these are bad data
    df = df[df["consecutive push intervals"] > 0]

    # Finally, denote the label order for some columns
    label_order = {
        "stimulus reliability": "stimulus reliability order",
        "box": "box rank",
    }

    return Experiment(
        df,
        config.to_dict(),
        ["subject", "session", "block"],
        ["push #"],
        ["stimulus reliability"],
        ["assigned schedules"],
        label_order=label_order,
    )
