import logging
from datetime import datetime
from typing import Any
from typing import Optional, Callable

import numpy as np
from numpy.typing import ArrayLike

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# Credit: https://gist.github.com/gosuto-inzasheru/b6deccd3fd5fefbabb72759c74040745
def flatten(li: list | tuple | set | range | np.ndarray):
    """
    Recursively flattens a nested list, tuple, set, or NumPy array.

    Args:
        li (list | tuple | set | range | np.ndarray): The input iterable to be flattened.

    Yields:
        Individual elements in a flattened sequence.
    """
    if isinstance(li, (list, tuple, set, range)):
        for item in li:
            yield from flatten(item)
    elif isinstance(li, np.ndarray):
        yield from li.flatten()
    else:
        yield li


def discrete_time(
        arr: ArrayLike,
        obs_ts: ArrayLike,
        dt: float,
        end_t: float,
        fill_time_func: Optional[Callable[[ArrayLike, int, int, dict[str, Any]], ArrayLike]] = None,
        **kwargs
) -> np.ndarray:
    """
    Converts an observation-based array into a time-based representation.

    Args:
        arr (ArrayLike): The array containing observed values.
        obs_ts (ArrayLike): The corresponding array of observation timestamps.
        dt (float): Temporal resolution for the time-based representation.
        end_t (float): Total duration for the output time-based array.
        fill_time_func (Optional[Callable[[ArrayLike, int, int], ArrayLike]]): A function to fill time gaps dynamically. Defaults to None.
        **kwargs: Additional arguments for the `fill_time_func` if provided.

    Returns:
        np.ndarray: A time-based version of `arr` aligned with `dt` resolution.
    """
    # Convert input array to NumPy array format
    arr = np.asarray(arr)

    # Compute the number of time steps based on resolution
    num_timesteps = int(end_t / dt)

    # Initialize output array with the appropriate shape
    if arr.ndim < 2:
        new_arr = np.zeros(num_timesteps)
    else:
        new_arr = np.zeros((num_timesteps,) + arr.shape[1:])

    num_obs = len(obs_ts)
    old_t = 0

    for t in range(num_obs):
        obs_t = int(obs_ts[t] / dt)

        # Fill in "dead" time unless a dynamics function is specified
        if not fill_time_func:
            new_arr[old_t:obs_t] = arr[t]
        else:
            new_arr[old_t:obs_t] = fill_time_func(arr, t, t - 1, **kwargs)

        # Update the last processed timestamp
        old_t = obs_t

        # Fill in the remaining time at the end
    new_arr[int(obs_ts[-1] / dt):] = arr[-1]
    return new_arr


def date_to_integer(dt_time: datetime) -> int:
    """
    Converts a datetime object into an integer format (YYYYMMDD).

    Args:
        dt_time (datetime): The datetime object to convert.

    Returns:
        int: The date represented as an integer in YYYYMMDD format.
    """
    # Compute integer representation of date
    return 10000 * dt_time.year + 100 * dt_time.month + dt_time.day
