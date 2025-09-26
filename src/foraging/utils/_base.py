import logging
from collections.abc import Iterable
from datetime import datetime
from typing import Any, Callable, Optional, Protocol

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from foraging.models.experiment import Experiment

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# Credit: https://gist.github.com/gosuto-inzasheru/b6deccd3fd5fefbabb72759c74040745
def flatten(x: Iterable):
    """
    Recursively flattens a nested list, tuple, set, or NumPy array.

    Args:
        x (Iterable): The input iterable to be flattened.

    Yields:
        Individual elements in a flattened sequence.
    """
    try:
        for item in x:
            yield from flatten(item)
    except TypeError:
        if isinstance(x, np.ndarray):
            yield from x.flatten()
        else:
            yield x


def kwargs_handler(kwargs: dict, key: str, default: dict = None) -> dict:
    """
    Intelligently extract keyword arguments from kwargs object, merging with default arguments specified inside the function. This is helpful when a single function accepts a nested kwargs dictionary that may contain the
    kwargs for several different subroutines.

    Args:
        kwargs: Keyword arguments object to extract arguments from.
        key: The keyword argument to extract. Assumed to be a dictionary.
        default: The default behavior of the argument

    Returns:
        The extracted keyword argument merged with default, if specified.
    """
    if default is None:
        default = {}
    result = kwargs.pop(key, {})
    if not type(result) == dict:
        raise ValueError("Keyword argument is not dictionary.")
    return default | result


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


class SupportsConds(Protocol):
    """
    A protocol for a function that takes in any dataset and dictionary of conditions to filter by.
    """

    def __call__(
        self, conds: dict[str, Any], dataset: pd.DataFrame | Experiment, **kwargs
    ) -> Any: ...
