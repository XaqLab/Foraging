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
from pandas.core.groupby import DataFrameGroupBy
from tqdm import tqdm

from foraging.config.constants import (
    BOX_LABELS,
    BOX_POSITIONS,
    KAPPA_CATEGORIES,
    KAPPA_LEVELS,
)
from foraging.utils import INDEX

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# import polars as pl
# TODO: consider whether its worth encapsulating functions into a class, like in rl.py
def get_subjects(path: str) -> list[str]:
    """
    Retrieve the names of all subjects from experiment matfiles.

    Args:
        path: Path to experiment data directory.

    Returns:
        A list of subject names extracted from file names.
    """

    subject_files = fnmatch.filter(os.listdir(path), "data_*.mat")
    subjects = [subject_file.split(".")[0][5:] for subject_file in subject_files]
    return subjects


def open_subject_file(subject: str, path: str = ".") -> h5py.File:
    """
    Open an HDF5 subject file. Remember to close this on your own as you would any regular file.

    Args:
        subject: Subject identifier.
        path: Path to the directory containing the file. Defaults to '.'.

    Returns:
        Opened HDF5 file.
    """

    return h5py.File(os.path.join(path, f"data_{subject}.mat"), "r")


def open_meta_file(subject: str, path: str = ".") -> pd.DataFrame:
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


def make_df(
    path: str,
    box_labels: list[str] = BOX_LABELS,
    box_positions: list[str] = BOX_POSITIONS,
) -> pd.DataFrame:
    """
    Given experiment matfiles and metadata, construct a DataFrame.

    Args:
        path: Path to the folder containing the experiment data to load into a DataFrame.

    Returns:
        A DataFrame where each row represents a push and each column encodes experiment variables, such as session, block, reliablity conditions, etc.
    """

    # Identify all subjects in the given directory
    subjects = get_subjects(path)
    df_dict = {
        "subject": [],
        "session": [],
        "_session": [],  # the original session number in sequential order inside the matlab struct
        "week day": [],
        "n boxes": [],
        "block": [],
        "schedule": [],
        "box rank": [],
        "shape": [],
        "stimulus type": [],
        "kappa": [],
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
    box_labels = dict(zip(range(len(box_labels)), box_labels))
    box_pos_labels = dict(zip(range(len(box_positions)), box_positions))
    for subject in subjects:

        # Load MATLAB file
        with open_subject_file(subject, path) as f:

            # Load metafile
            df_meta = open_meta_file(subject, path)
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

                        schedules = [
                            float(x) for x in block_meta["scheduleMean"].split(",")
                        ]
                        box_ranks = np.argsort(np.argsort(schedules))

                        kappa = block_meta["stimulusNoise"]
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
                            staging_dict["week day"].extend(
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
                            staging_dict["box rank"].extend(
                                [box_ranks[i] for _ in range(n_events)]
                            )
                            staging_dict["kappa"].extend(
                                [kappa for _ in range(n_events)]
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

    # Add some more columns based on block-dependent statistics
    # df["box rank"] = (
    #     df.groupby(["subject", "session", "block"])["schedule"].rank(
    #         method="dense"
    #     )
    #     - 1
    # ).astype(int) # ranks boxes fast --> medium --> slow
    df["box"] = df["box rank"].map(box_labels)
    df["box position label"] = df["box position"].map(box_pos_labels)
    df["prev box"] = get_blocks(df)["box"].shift(1)
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

    # Correct some columns
    df["shape"] = df["shape"].astype(int)
    df["subject"] = df["subject"].astype(str)
    df.loc[df["push #"] == 1, "consecutive push intervals"] = df.loc[
        df["push #"] == 1, "push times"
    ]  # Make sure the first push interval of each block is just the time of that push
    df.loc[df["push #"] == 1, "stay/switch"] = (
        "stay"  # Count first push of each block as 'stay' push, so that stay pushes are a subset of same-box push intervals
    )

    # Finally, drop all push intervals with value 0 as these are bad data
    df = df[df["consecutive push intervals"] > 0]

    # Categorize stimulus reliabilities
    for subject, kappa_filter in KAPPA_LEVELS.items():
        for label, values in kappa_filter.items():
            df_filter = filter_df(df, {"subject": subject, "kappa": values})
            df.loc[df_filter.index, "stimulus reliability"] = label

    # Create unique block ID as hash of subject, session, and block
    df["block_id"] = (
        df["subject"].astype(str)
        + "_"
        + df["session"].astype(str)
        + "_"
        + df["block"].astype(str)
    ).apply(hash)

    # Set index, refer to INDEX definition in utils.__init__.py
    df.set_index(list(INDEX), inplace=True)
    df.sort_index(inplace=True)
    return df


def display_df(df: pd.DataFrame, cols: list[str]) -> DisplayHandle:
    """
    Pretty prints dataframe's specified columns.

    Args:
        df: The dataframe to print.
        cols: The columns of dataframe to print.

    Returns:
        A markdown display object
    """

    return display(Markdown(df.head()[cols].to_markdown()))


def map_box_positions_to_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map box positions to ranks.
    """
    return df[["box position", "box rank"]].drop_duplicates().set_index("box position")


def filter_df(
    df: pd.DataFrame, conds: dict[str, Any] = None, attempt_index: bool = False
) -> pd.DataFrame:
    """
    Filter a DataFrame according to conditions specified in a dictionary.

    Args:
        df: DataFrame containing experiment data.
        conds: Dictionary mapping column or index level names to values to filter on.
        attempt_index: Flag indicating whether to attempt converting `conds` to MultiIndex key. Defaults to False. Turn
        on if you want slight performance boost in exchange for dropping the filtered index levels using .loc on `conds` as MultiIndex key.

    Returns:
        Filtered DataFrame.
    """

    if conds:
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


def get_blocks(
    df: pd.DataFrame, groupers: list = None, observed: bool = True, **kwargs
) -> DataFrameGroupBy:
    """
    Group DataFrame by subject, session, and block.

    Args:
        df: DataFrame containing experiment data.
        groupers: List of additional columns to group by.
        **kwargs: Keyword arguments to `DataFrame.groupby`.

    Returns:
        Grouped DataFrame object.
    """
    if groupers is None:
        groupers = []
    return df.groupby(
        ["subject", "session", "block"] + groupers, observed=observed, **kwargs
    )


def process_block_safely(func: Callable) -> Callable:
    """
    Decorator to safely process each block, catching exceptions and logging errors.

    Args:
        func: The function to apply to each block.

    Returns:
        Wrapped function with error handling.
    """

    @wraps(func)
    def wrapper(df: pd.DataFrame, index: tuple, *args, **kwargs):
        try:
            return func(df, index, *args, **kwargs)
        except Exception as e:
            logger.debug(f"Could not process ({index}): {str(e)}")
            return None

    return wrapper


def process_blocks(
    df: pd.DataFrame,
    compute_function: Callable,
    *args,
    use_tqdm: bool = False,
    **kwargs,
) -> tuple[dict, set]:
    """
    Apply a function to each block in DataFrame and aggregate results in a dictionary, where each key is a block index from the DataFrame.

    Args:
        df: DataFrame containing hierarchical data.
        compute_function: Function to apply to each block. Takes in df, index, *args, **kwargs.
        use_tqdm: Whether to display a progress bar. Defaults to False.
        *args: Additional arguments for `compute_function`.
        **kwargs: Additional keyword arguments for `compute_function`.

    Returns:
        A dictionary of results and a set of error blocks.
    """
    results = {}
    err_blocks = set()
    for index, block in tqdm(get_blocks(df), disable=not use_tqdm):
        result = compute_function(df, index, *args, **kwargs)
        if result is None:
            err_blocks.add(index)
        else:
            results[index] = result
    return results, err_blocks


def select_from_ranges(df: pd.DataFrame, quantiles: dict, key: str) -> pd.DataFrame:
    """
    Select rows from a DataFrame based on specified ranges of a given key.

    Args:
        df: The input DataFrame containing the data.
        quantiles: A dictionary where each key is a subject and each value is another dictionary
                          with box numbers as keys and ranges (either a list of two values or a single value)
                          as values for the specified key. If single value, then this is treated as a lower bound.
        key: The column name in the DataFrame on which to filter the data.

    Returns:
        A DataFrame containing only the rows that fall within the specified ranges for each subject and box.
    """

    # Save the dataframe of selected pushes
    df_selected = df.iloc[:0, :].copy()
    for subj, v in quantiles.items():
        for box, q_vals in v.items():
            # Extract data for the given subject
            subj_df = df.xs(subj, level="subject", drop_level=False)
            subj_df = subj_df[subj_df["box rank"] == box]
            try:
                # If q_vals is a range (list with two values)
                iter(q_vals)
                df_selected = pd.concat(
                    [
                        df_selected,
                        subj_df[
                            (subj_df[key] >= q_vals[0]) & (subj_df[key] <= q_vals[1])
                        ],
                    ]
                )
            except:
                # If q_vals is a single value, select rows where the key is greater than or equal to it
                df_selected = pd.concat([df_selected, subj_df[subj_df[key] >= q_vals]])
    return df_selected


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


def get_continuous_from_df_to_dict(df: pd.DataFrame, data_dir: str) -> (dict, list):
    """
    Extract continuous data from blocks in the DataFrame and return it as a dictionary.

    Args:
        df: The DataFrame containing session and block information.
        data_dir: The directory path where the subject files are located.

    Returns:
        A tuple containing:
            - data: A dictionary with extracted continuous data from the blocks. Keys are the block index from the DataFrame.
            - errors: A list of blocks causing errors encountered during the extraction process.
    """

    # Retrieve the list of subjects and open their respective files
    subjects = df.index.unique("subject").values
    files = {subj: open_subject_file(subj, data_dir) for subj in subjects}

    @process_block_safely
    def _inner(df: pd.DataFrame, index: tuple):
        """
        Helper function to extract continuous data from a specific block and session for a given index.

        Args:
            df: The input DataFrame.
            index: The index of the current block, containing subject, session, and block.

        Returns:
            dict: The extracted continuous data for the specified block and session.
        """

        _sess = df.loc[index, "_session"].iloc[0]
        return get_continuous_from_block(
            files[index[INDEX.index("subject")]], _sess, index[INDEX.index("block")]
        )

    # Process the blocks using the helper function
    data, errors = process_blocks(df, _inner)

    # Filter out blocks that don't have any data
    data = {k: v for k, v in data.items() if len(v) > 0}

    # Close all subject files
    [f.close() for f in files.values()]

    return data, errors


def get_continuous3d_from_df_to_df(
    df: pd.DataFrame, data_dir: str, key: str = "position"
) -> pd.DataFrame:
    """
    Extract continuous data for each push interval from the provided DataFrame and return it as a new DataFrame.

    Args:
        df: The input DataFrame containing session, block, and push interval information.
        data_dir: The directory path where the subject files are located.
        key: continuous data variable to extract.

    Returns:
        A DataFrame with continuous data (x, y, z) and corresponding time (t),
                      indexed by subject, session, block, and push interval.
    """

    # Initialize empty arrays to store valid position and time data
    p_valid = np.empty((0, 3))  # Array to store valid positions (x, y, z)
    t_valid = np.empty(0)  # Array to store valid time points
    new_index = []  # List to store new index values for the DataFrame

    # Retrieve subjects and open their respective files
    subjects = get_subjects(data_dir)
    files = {subj: open_subject_file(subj, data_dir) for subj in subjects}

    # Iterate over each block of data from the DataFrame
    for (subj, sess, block), block_data in get_blocks(df):
        # Get continuous data (position and time) for the current block
        res = get_continuous_from_block(
            files[subj], block_data["_session"].iloc[0], block
        )
        p, t = res[key], res["time"]
        # Skip block if no valid data was retrieved
        if p is None:
            continue

        # For each push interval, find the start and end times, and locate the nearest positions
        end_t = block_data["push times"]
        start_t = np.insert(
            end_t[:-1], 0, 0
        )  # Insert 0 for the first push interval start time
        start_idx, end_idx = np.searchsorted(t, [start_t, end_t])

        # Construct continuous data and a new index based on the discrete push data
        old_index = (subj, sess, block)
        for i, (s, e) in enumerate(zip(start_idx, end_idx)):
            if len(p[s:e]) == len(t[s:e]):
                p_valid = np.concatenate((p_valid, p[s:e]), axis=0)
                t_valid = np.concatenate((t_valid, t[s:e]))
                idx = [old_index + block_data.index[i]] * len(t[s:e])
                new_index += idx

    # Close all opened subject files
    [f.close() for f in files.values()]

    # Create a DataFrame from the valid position and time data
    new_df = pd.DataFrame(
        {"x": p_valid[:, 0], "y": p_valid[:, 1], "z": p_valid[:, 2], "t": t_valid},
        index=new_index,
    )

    return new_df


# def get_continuous3d_from_df_to_polars(
#     df: pl.DataFrame, data_dir: str, key: str = "position"
# ) -> pl.DataFrame:
#     """
#     Extract continuous data for each push interval from the provided DataFrame and return it as a new Polars DataFrame.
#
#     Args:
#         df: The input Polars DataFrame containing session, block, and push interval information.
#         data_dir: The directory path where the subject files are located.
#         key: continuous data variable to extract.
#
#     Returns:
#         A Polars DataFrame with continuous data (x, y, z) and corresponding time (t),
#                       indexed by subject, session, block, and push interval.
#     """
#
#     # Initialize empty arrays to store valid position and time data
#     p_valid = np.empty((0, 3))  # Array to store valid positions (x, y, z)
#     t_valid = np.empty(0)  # Array to store valid time points
#     new_index = []  # List to store new index values for the DataFrame
#
#     # Retrieve subjects and open their respective files
#     subjects = get_subjects(data_dir)
#     files = {subj: open_subject_file(subj, data_dir) for subj in subjects}
#
#     # Iterate over each block of data from the DataFrame
#     for (subj, sess, block), block_data in get_blocks(df):
#         # Get continuous data (position and time) for the current block
#         res = get_continuous_from_block(files[subj], block_data["_session"][0], block)
#         p, t = res[key], res["time"]
#
#         # Skip block if no valid data was retrieved
#         if p is None:
#             continue
#
#         # For each push interval, find the start and end times, and locate the nearest positions
#         end_t = block_data["push times"]
#         start_t = np.insert(
#             end_t[:-1], 0, 0
#         )  # Insert 0 for the first push interval start time
#         start_idx, end_idx = np.searchsorted(t, [start_t, end_t])
#
#         # Construct continuous data and a new index based on the discrete push data
#         old_index = (subj, sess, block)
#         for i, (s, e) in enumerate(zip(start_idx, end_idx)):
#             if len(p[s:e]) == len(t[s:e]):
#                 p_valid = np.concatenate((p_valid, p[s:e]), axis=0)
#                 t_valid = np.concatenate((t_valid, t[s:e]))
#                 idx = [old_index + (block_data["push times"][i],)] * len(t[s:e])
#                 new_index += idx
#
#     # Convert the new index list to a NumPy array
#     new_index = np.array(new_index)
#
#     # Close all opened subject files
#     [f.close() for f in files.values()]
#
#     # Create a Polars DataFrame from the valid position and time data
#     new_df = pl.DataFrame(
#         {"x": p_valid[:, 0], "y": p_valid[:, 1], "z": p_valid[:, 2], "t": t_valid},
#         schema_overrides={"t": pl.Float64},
#     )
#
#     # Add additional columns for subject, session, block, and push times
#     return new_df.with_columns(
#         pl.Series("subject", new_index[:, 0]),
#         pl.Series("session", new_index[:, 1]),
#         pl.Series("block", new_index[:, 2]),
#         pl.Series("push times", new_index[:, 3]),
#     )


def populate_busyness(df: pd.DataFrame) -> (dict, set):
    """
    Add a 'push busyness' column to the DataFrame, representing the busyness level of pushes in each box.

    Args:
        df: The input DataFrame with push data, including push times and box rank.

    Returns:
        The original DataFrame with the 'push busyness' column populated.
    """

    # Initialize 'push busyness' column with NaN values
    df["push busyness"] = np.nan

    @process_block_safely
    def _inner(df: pd.DataFrame, index: tuple):
        """
        Helper function to process each block of the DataFrame and calculate 'push busyness' for each box.

        Args:
            df (pd.DataFrame): The DataFrame to process.
            index (tuple): The index of the current block, containing subject, session, and block.

        Returns:
            bool: Always returns True after processing the block.
        """
        # Extract the subset of data for the current block based on the index
        df_block = df.xs(index, level=("subject", "session", "block"), drop_level=False)

        # Iterate over each box rank in the block
        for box in df_block["box rank"].unique():
            # Filter the DataFrame for the current box
            box_df = filter_df(df_block, {"box rank": box})

            # Calculate 'push busyness' using pd.cut to categorize the push times and calculate frequency
            df.loc[box_df.index, "push busyness"] = (
                pd.cut(
                    df_block["push times"],
                    bins=np.insert(
                        box_df["push times"].values, 0, 0
                    ),  # Create bins from push times, starting from 0
                    include_lowest=True,
                    right=False,
                )
                .value_counts()
                .sort_index()
                .values
            )  # Count values and sort by index to align with original order

        return True  # Dummy return variable to conform with wrapper

    # Process each block using the helper function _inner
    return process_blocks(df, _inner)


def extend_df(
    df: pd.DataFrame, blocks_dict: dict, col_name: str, by_box: bool = False
) -> (dict, set):
    """
    Extend the DataFrame by adding a new column with values from the provided blocks dictionary.
    The new column is filled with values corresponding to each box rank in the DataFrame.

    Args:
        df: The input DataFrame to extend with a new column.
        blocks_dict: A dictionary containing block data for each subject, session, and block.
                     If by_box is False (default), the dictionary should have the format:
                     { (subject, session, block): values }
                     where 'values' is a list of events aligned to that block.

                     If by_box is set to True, the dictionary should have the format:
                     { (subject, session, block): {box: values} }
                     where 'box' corresponds to box's ordered by schedules and 'values' is a list of values for each box.
        col_name: The name of the new column to add to the DataFrame.
        by_box: If true, then blocks_dict is formatted as separate lists for each box and thus each list needs to be matched to its corresponding box in the dataframe. Otherwise, it is assumed that each row of an item in blocks_dict is aligned to each event in that block.

    Returns:
        The original DataFrame with the new column added and filled with data from blocks_dict.
    """

    # Initialize the new column with NaN values
    df[col_name] = np.nan

    @process_block_safely
    def _inner(df: pd.DataFrame, index: tuple):
        """
        Helper function to process each block and assign values from blocks_dict to the corresponding rows.

        Args:
            df (pd.DataFrame): The DataFrame to process.
            index (tuple): The index of the current block, containing subject, session, and block.

        Returns:
            bool: Always returns True after processing the block.
        """
        # Retrieve block data from the blocks_dict for the current index
        block_data = blocks_dict[index]

        # Extract the subset of the DataFrame for the current block
        df_block = df.xs(index, level=("subject", "session", "block"), drop_level=False)

        if by_box:
            # Iterate over each box rank in the block
            for box in df_block["box rank"].unique():
                # Filter the DataFrame for the current box
                box_df = filter_df(df_block, {"box rank": box})

                # Assign the corresponding values from blocks_dict to the new column
                df.loc[box_df.index, col_name] = pd.Series(
                    block_data[box][: len(box_df)], index=box_df.index
                )
        else:
            # Assume each row of block_data is matched to each event in df
            df.loc[df_block.index, col_name] = pd.Series(
                block_data, index=df_block.index
            )
        return True  # Return True after processing the block

    # Process each block using the helper function _inner
    return process_blocks(df, _inner)


def exclusion_criteria(df: pd.DataFrame, data_dir: str) -> pd.DataFrame:
    """
    Applies exclusion criteria to filter out data based on specific conditions.

    This function performs several exclusions based on the following criteria:
    1. Excludes blocks with fewer than 10 pushes.
    2. Excludes blocks with a schedule value of 80.
    3. Excludes rows where consecutive push intervals are greater than 30 s.
    4. Excludes rows where the average vertical position exceeds 750 mm.

    Args:
        df: The DataFrame containing the data to be filtered.
        data_dir: The directory path where the subject files are located.

    Returns:
        A filtered DataFrame that has had the exclusion criteria applied.
    """

    # Count the number of pushes per block
    n_pushes_per_block = get_blocks(df).size().reset_index(name="n pushes per block")

    # Exclude blocks with fewer than 10 pushes
    df_filtered = df.drop(
        n_pushes_per_block.loc[n_pushes_per_block["n pushes per block"] < 10]
        .set_index(["subject", "session", "block"])
        .index
    )

    # Exclude blocks where the schedule is 80
    df_filtered = df_filtered.drop(
        get_blocks(df_filtered[df_filtered["schedule"] == 80]).size().index
    )

    # Exclude pushes where the consecutive push intervals are greater than 30 s
    df_filtered = df_filtered.drop(
        df_filtered[df_filtered["consecutive push intervals"] > 30].index
    )

    # Get blocks with valid position data
    continuous_data, _ = get_continuous_from_df_to_dict(df_filtered, data_dir)
    df_cont_content = []

    for block, data in continuous_data.items():
        if np.any(data["position"] != np.nan):
            df_cont_content.append(
                df_filtered.xs(
                    block, level=("subject", "session", "block"), drop_level=False
                )
            )

    # Exclude pushes where the average vertical position exceeds the height of the boxes (750 mm)
    df_cont = pd.concat(df_cont_content)
    for subject in df_cont.index.unique("subject"):
        df_cont_subject = filter_df(df_cont, {"subject": subject})
        for idx, row in df_cont_subject.iterrows():
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
                    if mean_vertical > 750:
                        df_filtered = df_filtered.drop(idx)
            except:
                continue
    return df_filtered


def bin_data(
    df: pd.DataFrame,
    x: str,
    bins: int | list[float] = 20,
    bin_width: float = None,
    strategy: str = "center",
) -> pd.Series:
    """
    Bins data in the specified column of the DataFrame, with support for different binning strategies. This function
    performs binning on the specified column (`x`) and labels the bins according to the selected strategy. It supports
    defining the number of bins or specific bin edges.

    Args:
        df: Input DataFrame containing the data.
        x: Name of the column to bin.
        bins: Number of bins or list of bin edges.
            If an integer is provided, the data will be divided into that number of equal-width bins.
            If a list of floats is provided, it will specify the bin edges.
            Defaults to 30.
        bin_width: If specified, this is the width of each bin. Bins will be determined by dividing the range into equal-sized bins of this width.
        strategy: Labeling strategy for the bins.
            - 'full': Labels the bins using the full interval (i.e., both left and right edges).
            - 'left': Labels the bins using only the left edge.
            - 'right': Labels the bins using only the right edge.
            - 'center': Labels the bins using the center of the bin.
            Defaults to 'left'.

    Returns:
        A pandas Series containing the binned data.

    Example:
        df = pd.DataFrame({'value': np.random.randn(100)})
        df['binned'] = bin_data(df, 'value', bins=5, strategy='right')
    """
    # Perform initial binning based on n_bins or custom bin edges
    if bin_width:
        bins = np.arange(
            start=df[x].min(), stop=df[x].max() + bin_width, step=bin_width
        )
    _bins = pd.cut(df[x], bins=bins, include_lowest=True)
    dtype = df[
        x
    ].dtype  # Get the dtype of the column to maintain consistency in bin edges

    # Select the appropriate bin edges based on the strategy
    match strategy:
        case "full":
            bin_edges = _bins.cat.categories
        case "right":
            bin_edges = _bins.cat.categories.right.astype(dtype)
        case "left":
            bin_edges = _bins.cat.categories.left.astype(dtype)
        case _:
            bin_edges = (
                (_bins.cat.categories.left + _bins.cat.categories.right) / 2
            ).astype(dtype)

    # Apply the bin labels to the original data
    return pd.cut(
        df[x], bins=bins, include_lowest=True, labels=bin_edges
    ).cat.remove_unused_categories()
