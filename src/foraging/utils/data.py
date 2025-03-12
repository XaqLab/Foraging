import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import numpy as np
import pandas as pd
import polars as pl

from functools import wraps
import logging
import os
import fnmatch
import h5py
import pickle
from datetime import datetime
from typing import Callable, Optional

from tqdm import tqdm

from foraging.utils import INDEX
from ._base import date_to_integer

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def get_subjects(path: str) -> list[str]:
    """
    Retrieve the names of all subjects from experiment matfiles.

    Args:
        path (str): Path to experiment data directory.

    Returns:
        list[str]: A list of subject names extracted from file names.
    """
    subject_files = fnmatch.filter(os.listdir(path), "data_*.mat")
    subjects = [subject_file.split(".")[0][5:] for subject_file in subject_files]
    return subjects


def open_subject_file(subject: str, path: str = ".") -> h5py.File:
    """
    Open an HDF5 subject file.

    Args:
        subject (str): Subject identifier.
        path (str, optional): Path to the directory containing the file. Defaults to '.'.

    Returns:
        h5py.File: Opened HDF5 file.
    """
    return h5py.File(os.path.join(path, f"data_{subject}.mat"), "r")


def open_meta_file(subject: str, path: str = ".") -> pd.DataFrame:
    """
    Open and preprocess an Excel metadata file for a given subject.

    Args:
        subject (str): Subject identifier.
        path (str, optional): Path to the directory containing the file. Defaults to '.'.

    Returns:
        pd.DataFrame: Processed metadata DataFrame.
    """
    df_meta = pd.read_excel(os.path.join(path, f"table_{subject}.xlsx"))
    df_meta["sessionId"] = df_meta["sessionId"].astype(int).sort_values()
    return df_meta


def make_dataframe(path: str) -> pd.DataFrame:
    """
    Given experiment matfiles and metadata, construct a DataFrame.

    Args:
        path (str): Path to the experiment data to load into a DataFrame.

    Returns:
        pd.DataFrame: A DataFrame where each row represents a push and each column encodes experiment context, such as session, block, experiment conditions, etc.
    """

    # Identify all subjects in the given directory
    subjects = get_subjects(path)
    df_dict = {
        "subject": [],
        "session id": [],
        "_session": [],
        "week day": [],
        "block": [],
        "schedule": [],
        "shape": [],
        "stimulus type": [],
        "kappa": [],
        "box": [],
        "push times": [],
        "same-box push intervals": [],
        "reward outcomes": [],
        "reward intervals": [],
    }
    day_to_week = {0: "M", 1: "T", 2: "W", 3: "R", 4: "F", 5: "S", 6: "U"}

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

            for sess_idx in range(n_sessions):
                sess_id = (
                    int(f[f["session"]["id"][sess_idx, 0]][0, 0])
                    if subject == "humans"
                    else datetime.strptime(
                        "".join(
                            [chr(c) for c in f[f["session"]["id"][sess_idx, 0]][:, 0]]
                        ),
                        "%Y%m%d",
                    )
                )
                if isinstance(sess_id, datetime):
                    week_day = day_to_week[sess_id.weekday()]
                    sess_id = date_to_integer(sess_id)

                sess_data = f[all_sess_data[sess_idx, 0]]["events"]
                param_data = f[all_sess_data[sess_idx, 0]]["params"]
                param_data_cue = (
                    f[all_sess_data_cue[sess_idx, 0]]["params"]
                    if all_sess_data_cue
                    else None
                )
                n_blocks = len(sess_data)

                for block_idx in range(n_blocks):
                    try:
                        block_meta = df_meta.loc[
                            (df_meta["sessionId"] == sess_id)
                            & (df_meta["blockId"] == block_idx + 1)
                        ]
                        schedules = [
                            float(x)
                            for x in block_meta["scheduleMean"].iloc[0].split(",")
                        ]

                        kappa = block_meta["stimulusNoise"].iloc[0]
                        shape = block_meta["GammaShape"].iloc[0]
                        stim_type = block_meta["stimulusCueType"].iloc[0]

                        # Parse boxes
                        staging_dict = {
                            k: [] for k in df_dict.keys() if k != "push busyness"
                        }
                        for i in range(len(schedules)):
                            box = "box" + str(i + 1)
                            push_times = np.atleast_1d(
                                f[sess_data[block_idx, 0]].get("tPush/" + box)
                            ).ravel()
                            reward_outcomes = (
                                np.atleast_1d(
                                    f[sess_data[block_idx, 0]].get("pushLogical/" + box)
                                )
                                .astype(bool)
                                .ravel()
                            )

                            # Populate the fields that are the same for all pushes in a box
                            n_events = len(push_times)
                            staging_dict["subject"].extend(
                                [subject for _ in range(n_events)]
                            )
                            staging_dict["session id"].extend(
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
                            staging_dict["schedule"].extend(
                                [schedules[i] for _ in range(n_events)]
                            )
                            staging_dict["kappa"].extend(
                                [kappa for _ in range(n_events)]
                            )
                            staging_dict["shape"].extend(
                                [shape for _ in range(n_events)]
                            )
                            staging_dict["box"].extend([i + 1 for _ in range(n_events)])
                            staging_dict["stimulus type"].extend(
                                [stim_type for _ in range(n_events)]
                            )

                            # Populate push-specific data
                            staging_dict["push times"].extend(push_times)
                            staging_dict["same-box push intervals"].extend(
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
                                    f"Could not parse ({subject}, {sess_idx + 1}, {block_idx + 1}) due to incorrect size of arrays"
                                )
                                length = False
                                break
                        if length:
                            [v.extend(staging_dict[k]) for k, v in df_dict.items()]
                    except Exception as e:  # If block causes issue, skip it
                        logger.debug(
                            f"Could not parse ({subject}, {sess_idx + 1}, {block_idx + 1})"
                        )
                        logger.debug(e)

    # Make DataFrame and sort by relevant fields
    df = pd.DataFrame(df_dict).sort_values(
        by=["subject", "session id", "block", "push times"]
    )

    # Add some more columns based on block-dependent statistics
    df["box rank"] = (
        df.groupby(["subject", "session id", "block"])["schedule"].rank(
            method="dense"
        )
        - 1
    ).astype(int) # ranks boxes fast --> medium --> slow
    df["normalized pushes"] = df["same-box push intervals"] / df["schedule"]
    df["consecutive push intervals"] = df["push times"].diff()

    # Drop all push intervals with value 0, these are bad data
    df = df[df["consecutive push intervals"] > 0]

    df["push #"] = (
        df.groupby(["subject", "session id", "block"])["push times"].rank().astype(int)
    )
    df["push # by box"] = (
        df.groupby(["subject", "session id", "block", "box rank"])["push times"]
        .rank()
        .astype(int)
    )
    df["session"] = df.groupby("subject")["session id"].rank(method="dense").astype(int)
    df["stay/switch"] = df["box rank"].diff().astype(bool)

    # Correct some columns
    df["shape"] = df["shape"].astype(int)
    df["subject"] = df["subject"].astype(str)
    df.loc[df["push #"] == 1, "consecutive push intervals"] = df.loc[
        df["push #"] == 1, "push times"
    ]
    df.loc[df["push #"] == 1, "stay/switch"] = False

    # Set index, refer to INDEX definition at top of this file
    df.set_index(INDEX, inplace=True)
    df.sort_index(inplace=True)

    return df


def filter_df(
    df: pd.DataFrame, conds: Optional[dict[str, list]] = None
) -> pd.DataFrame:
    """
    Filter a DataFrame according to conditions specified in a dictionary.

    Args:
        df (pd.DataFrame): DataFrame containing experiment data.
        conds (Optional[Dict[str, list]]): Dictionary mapping column or index level names to values to filter on.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    if conds:
        mask = np.full(len(df), True)
        for k, v in conds.items():
            v = np.atleast_1d(v)
            if k in df.index.names:  # Filter on index levels
                mask &= df.index.get_level_values(k).isin(v)
            elif k in df.columns:  # Filter on columns
                mask &= df[k].isin(v)
        return df[mask]
    return df


def process_block_safely(func: Callable) -> Callable:
    """
    Decorator to safely process each block, catching exceptions and logging errors.

    Args:
        func (Callable): The function to apply to each block.

    Returns:
        Callable: Wrapped function with error handling.
    """

    @wraps(func)
    def wrapper(df: pd.DataFrame, index: tuple, *args, **kwargs):
        try:
            return func(df, index, *args, **kwargs)
        except Exception as e:
            logger.debug(f"Could not process ({index}): {str(e)}")
            return None

    return wrapper


def get_blocks(df: pd.DataFrame):
    """
    Group DataFrame by subject, session, and block.

    Args:
        df (pd.DataFrame): DataFrame containing experiment data.

    Returns:
        DataFrameGroupBy: Grouped DataFrame object.
    """
    return df.groupby(["subject", "session", "block"])


def process_blocks(
    df: pd.DataFrame,
    compute_function: Callable,
    *args,
    use_tqdm: bool = False,
    **kwargs,
) -> (dict, set):
    """
    Apply a function to each block in a hierarchical dataset and aggregate results.

    Args:
        df (pd.DataFrame): DataFrame containing hierarchical data.
        compute_function (Callable): Function to apply to each block.
        use_tqdm (bool, optional): Whether to display a progress bar. Defaults to False.
        *args: Additional arguments for `compute_function`.
        **kwargs: Additional keyword arguments for `compute_function`.

    Returns:
        Tuple[Dict, Set]: Dictionary of results and a set of error blocks.
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
        df (pd.DataFrame): The input DataFrame containing the data.
        quantiles (dict): A dictionary where each key is a subject and each value is another dictionary
                          with box numbers as keys and quantile ranges (either a list of two values or a single value)
                          as values for the specified key.
        key (str): The column name in the DataFrame on which to filter the data.

    Returns:
        pd.DataFrame: A DataFrame containing only the rows that fall within the specified ranges for each subject and box.
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
        f (h5py.File): The HDF5 file object containing the session data.
        session (int): The session number from which to extract data (1-indexed).
        block (int): The block number within the session to extract data from (1-indexed).

    Returns:
        dict: A dictionary containing the continuous data arrays, including:
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
        pass

    return {}


def get_continuous_from_df_to_dict(df: pd.DataFrame, data_dir: str) -> (dict, list):
    """
    Extract continuous data from blocks in the DataFrame and return it as a dictionary.

    Args:
        df (pd.DataFrame): The DataFrame containing session and block information.
        data_dir (str): The directory path where the subject files are located.

    Returns:
        tuple: A tuple containing:
            - data (dict): A dictionary with extracted continuous data from the blocks.
            - errors (list): A list of blocks causing errors encountered during the extraction process.
    """
    # Retrieve the list of subjects and open their respective files
    subjects = df.index.unique('subject').values
    files = {subj: open_subject_file(subj, data_dir) for subj in subjects}

    @process_block_safely
    def _inner(df: pd.DataFrame, index: tuple):
        """
        Helper function to extract continuous data from a specific block and session for a given index.

        Args:
            df (pd.DataFrame): The input DataFrame.
            index (tuple): The index of the current block, containing subject, session, and block.

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
        df (pd.DataFrame): The input DataFrame containing session, block, and push interval information.
        data_dir (str): The directory path where the subject files are located.
        key (str): continuous data variable to extract.

    Returns:
        pd.DataFrame: A DataFrame with continuous data (x, y, z) and corresponding time (t),
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


def get_continuous3d_from_df_to_polars(
    df: pl.DataFrame, data_dir: str, key: str = "position"
) -> pl.DataFrame:
    """
    Extract continuous data for each push interval from the provided DataFrame and return it as a new Polars DataFrame.

    Args:
        df (pl.DataFrame): The input Polars DataFrame containing session, block, and push interval information.
        data_dir (str): The directory path where the subject files are located.
        key (str): continuous data variable to extract.

    Returns:
        pl.DataFrame: A Polars DataFrame with continuous data (x, y, z) and corresponding time (t),
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
        res = get_continuous_from_block(files[subj], block_data["_session"][0], block)
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
                idx = [old_index + (block_data["push times"][i],)] * len(t[s:e])
                new_index += idx

    # Convert the new index list to a NumPy array
    new_index = np.array(new_index)

    # Close all opened subject files
    [f.close() for f in files.values()]

    # Create a Polars DataFrame from the valid position and time data
    new_df = pl.DataFrame(
        {"x": p_valid[:, 0], "y": p_valid[:, 1], "z": p_valid[:, 2], "t": t_valid},
        schema_overrides={"t": pl.Float64},
    )

    # Add additional columns for subject, session, block, and push times
    return new_df.with_columns(
        pl.Series("subject", new_index[:, 0]),
        pl.Series("session", new_index[:, 1]),
        pl.Series("block", new_index[:, 2]),
        pl.Series("push times", new_index[:, 3]),
    )


def populate_busyness(df: pd.DataFrame) -> tuple[dict, set]:
    """
    Add a 'push busyness' column to the DataFrame, representing the busyness level of pushes in each box.

    Args:
        df (pd.DataFrame): The input DataFrame with push data, including push times and box rank.

    Returns:
        pd.DataFrame: The original DataFrame with the 'push busyness' column populated.
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


def extend_df(df: pd.DataFrame, blocks_dict: dict, col_name: str, by_box: bool = False) -> tuple[dict, set]:
    """
    Extend the DataFrame by adding a new column with values from the provided blocks dictionary.
    The new column is filled with values corresponding to each box rank in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame to extend with a new column.
        blocks_dict (dict): A dictionary containing block data for each subject, session, and block.
                             The dictionary should have the format:
                             { (subject, session, block): {box: values} }
                             where 'box' corresponds to box's in order of schedules and 'values' is a list of values for each box.
        col_name (str): The name of the new column to add to the DataFrame.
        by_box (bool): If true, then blocks_dict is formatted as separate lists for each box and thus each list needs to be matched to its corresponding box in the dataframe. Otherwise, it is assumed that each row of an item in blocks_dict is aligned to each event in that block's dataframe

    Returns:
        pd.DataFrame: The original DataFrame with the new column added and filled with data from blocks_dict.
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
            df.loc[df_block.index, col_name] = pd.Series(block_data, index = df_block.index)
        return True  # Return True after processing the block

    # Process each block using the helper function _inner
    return process_blocks(df, _inner)


def exclusion_criteria(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies exclusion criteria to filter out data based on specific conditions.

    This function performs several exclusions based on the following criteria:
    1. Excludes blocks with fewer than 10 pushes.
    2. Excludes blocks with a schedule value of 80.
    3. Excludes rows where consecutive push intervals are greater than 50.

    Args:
        df: The DataFrame containing the data to be filtered.

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

    # Exclude rows where consecutive push intervals are greater than 50
    df_filtered = df_filtered.drop(
        df_filtered[df_filtered["consecutive push intervals"] > 50].index
    )
    return df_filtered


def load_pickled_data(path: str) -> dict:
    """
    Loads data from a pickle file and returns its contents as a dictionary.

    This function loads any data stored in the pickle file and returns it as a
    dictionary, where the keys are the names of the stored elements and the
    values are the corresponding data.

    Args:
        path: The file path of the pickle file to load.

    Returns:
        A dictionary where keys are the names of the elements in the pickle file
        and the values are the corresponding data.
    """
    # Open the pickle file and load its contents
    with open(path, "rb") as f:
        ds = pickle.load(f)

    # Return all contents as a dictionary
    return ds


def bin_data(
    df: pd.DataFrame,
    x: str,
    n_bins: Optional[int | list[float]] = 20,
    strategy: str = 'left'
) -> list[float]:
    """
    Bins data in the specified column of the DataFrame, with support for different binning strategies.

    This function performs binning on the specified column (`x`) and labels the bins
    according to the selected strategy. It supports defining the number of bins or
    specific bin edges. Optionally, it can return the binned data as a new column
    in the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data.
        x (str): Name of the column to bin.
        n_bins (Optional[Union[int, list[float]]]): Number of bins or list of bin edges.
            If an integer is provided, the data will be divided into that number of equal-width bins.
            If a list of floats is provided, it will specify the bin edges.
            Defaults to 20.
        strategy (str): Labeling strategy for the bins.
            - 'full': Labels the bins using the full interval (i.e., both left and right edges).
            - 'left': Labels the bins using only the left edge.
            - 'right': Labels the bins using only the right edge.
            Defaults to 'left'.

    Returns:
        pd.Series: A pandas Series containing the binned data.

    Example:
        df = pd.DataFrame({'value': np.random.randn(100)})
        df['binned'] = bin_data(df, 'value', n_bins=5, strategy='right')

    Notes:
        If `n_bins` is an integer, the binning is done by dividing the data into equal-width bins.
        If `n_bins` is a list of floats, the exact bin edges are used.
    """

    # Perform initial binning based on n_bins or custom bin edges
    bins = pd.cut(df[x], bins=n_bins, include_lowest=True)

    dtype = df[x].dtype  # Get the dtype of the column to maintain consistency in bin edges

    # Select the appropriate bin edges based on the strategy
    match strategy:
        case 'full':
            bin_edges = bins.cat.categories.astype(dtype)
        case 'right':
            bin_edges = bins.cat.categories.right.astype(dtype)
        case _:
            bin_edges = bins.cat.categories.left.astype(dtype)

    # Apply the bin labels to the original data
    return pd.cut(df[x], bins=n_bins, include_lowest=True, labels=bin_edges)