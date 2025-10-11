import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.special import beta, betainc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from foraging import BIN_WIDTH, STEP, WINDOW_SIZE
from foraging.models.experiment import Experiment
from foraging.utils import kwargs_handler
from foraging.utils.data import bin_data


def moving_average(
    dataset: Experiment,
    x: str,
    y: str,
    y_name: str = None,
    x_name: str = None,
    groupers: list = None,
    bin_func: callable = None,
    bin_width: float = BIN_WIDTH,
    window_size: float = WINDOW_SIZE,
    win_type: str = None,
    step: float = STEP,
    center: bool = True,
    rate: bool = False,
    fill_value: float = 0,
    **kwargs,
):
    """
    Calculate a moving average of time series data by binning and applying rolling window operations.

    Args:
        dataset: Experiment object containing the data to analyze
        x: Column name for the time/independent variable to bin
        y: Column name for the dependent variable to average
        y_name: Name for the output averaged variable (default: "mean " + y)
        x_name: Name for the output binned time variable (default: "time")
        groupers: List of column names to group by during processing
        bin_func: Function to apply within each time bin (default: mean)
        bin_width: Width of each time bin for discretization
        window_size: Size of the rolling window for smoothing
        win_type: Type of window for rolling operation (e.g., "gaussian")
        step: Step size for the rolling window
        center: Whether to center the rolling window
        rate: Whether to convert result to rate by dividing by bin_width
        fill_value: Value to fill missing bins with (default: 0)
        **kwargs: Additional arguments passed to rolling operations

    Returns:
        DataFrame with binned time series data and moving averages, indexed by block identifiers,
        groupers, and time bins.
    """
    if bin_func is None:
        bin_func = lambda x: x[y].mean()

    if y_name is None:
        y_name = "mean " + y

    if x_name is None:
        x_name = "time"

    agg_kwargs = kwargs_handler(kwargs, "agg_kwargs")
    window = int(window_size / bin_width)
    step = int(step / bin_width)
    if win_type == "gaussian":
        agg_kwargs["std"] = agg_kwargs.pop("std", window) / 2

    # Create time bins
    df = dataset.df
    df = df.copy()
    bins = bin_data(df[x], bin_width=bin_width, remove_unused_categories=False)
    df[x_name] = bins
    dataset = dataset.wrap(df)

    # Assign data to full grid of time bins-- time bins should be fine enough to "binarize" the time series
    # Fill value should be 0 for rates, so that the moving average gives the % time-bins containing a value
    binned_data = dataset.get_blocks(groupers=groupers).apply(
        lambda x: bin_func(x.groupby(x_name, observed=True))
        .reindex(bins.cat.categories, fill_value=fill_value)
        .reset_index(),
        include_groups=False,
    )

    # Smooth the time series by calculating the moving average
    rolled_data = (
        binned_data.groupby(dataset.block_identifiers + groupers)
        .apply(
            lambda x: x.set_index("index")
            .rolling(
                window=window, step=step, win_type=win_type, center=center, **kwargs
            )
            .mean(**agg_kwargs)
        )
        .reset_index(level="index")
    )

    rolled_data = rolled_data.rename(columns={0: y, "index": x_name}).set_index(
        x_name, append=True
    )
    rolled_data[y_name] = rolled_data[y]
    if rate:
        rolled_data[y_name] /= bin_width
        # if 'win_type' in rolling_kwargs and rolling_kwargs['win_type'] == 'gaussian':
        #     rolled_data[y_name] *= compute_gaussian_correction_factor(window_size, agg_kwargs['std'], bin_width=bin_width)
    return rolled_data
