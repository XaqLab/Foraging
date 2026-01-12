from abc import abstractmethod
from dataclasses import dataclass
from typing import Generic, Iterable, Protocol, TypeVar, Any
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from tqdm import tqdm
import logging
from foraging.models import HashableDict, SuperDict

# Lazilyy import filter_df to avoid circular dependency
_filter_df = None

def _get_filter_df():
    global _filter_df
    if _filter_df is None:
        from foraging.utils.data import filter_df
        _filter_df = filter_df
    return _filter_df

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class SupportsDataFrame(Protocol):
    """
    A protocol for being compliant with the Dataframe Interchange Protocol.
    """
    @abstractmethod
    def __dataframe__(self, nan_as_null: bool = False, allow_copy: bool = True): ...


class SupportsBlocks:
    """
    A protocol for a function that takes in a `dataset`, `block_key`, `block`, and returns a result.
    `block_key` and `block` are the outputs of `Experiment.blocks`.
    """
    def __call__(self, dataset: "Experiment", block_key: dict[str, Any], block: pd.DataFrame, *args, **kwargs) -> Any: ...


class Experiment:
    """
    Dataset to store experiment data and metadata. 
    """
    def __init__(self, data: "SupportsDataFrame", constants: dict | Iterable, block_identifiers: Iterable[str], within_block_identifiers: Iterable[str], experiment_conditions: Iterable[str], block_metadata: Iterable[str] = None, skip_index: bool = False, label_order: dict[str, str] = None):
        """
        Args:
            data: DataFrame containing experiment data.
            constants: Constants for the experiment.
            block_identifiers: Names of the columns that identify the blocks.
            within_block_identifiers: Names of the columns that identify datapoints inside each block.
            experiment_conditions: Names of columns that identify experiment conditions.
            block_metadata: Names of columns that identify the block metadata.
            skip_index: Whether to skip creating a multi-index. Set to True if `data` is already configured correctly.
            label_order: Dictionary mapping columns to the columns containing the ordering of their values (useful for plotting categorical variables).
        """

        self.df = pd.api.interchange.from_dataframe(data)
        self.constants = SuperDict(constants)
        self.block_identifiers = block_identifiers
        self.within_block_identifiers = within_block_identifiers
        self.experiment_conditions = experiment_conditions
        self.block_metadata = block_metadata
        self.label_order = label_order

        # Create multi-index for faster queries
        if "block_id" not in self.df.index.names:
            self.df["block_id"] = (
                self.df[block_identifiers]
                .astype(str)
                .agg("_".join, axis=1)
                .apply(hash)
            )

        self.index = block_identifiers + ["block_id"] + within_block_identifiers + experiment_conditions
        if block_metadata is not None:
            self.index += block_metadata
        if not skip_index:
            self.df.set_index(self.index, inplace=True)
            self.df.sort_index(inplace=True)

    @property
    def blocks(self) -> pd.core.groupby.DataFrameGroupBy:
        return self.get_blocks()

    def get_blocks(self, groupers: list = None, observed: bool = True, **kwargs) -> pd.core.groupby.DataFrameGroupBy:
        """Return DataFrame grouped by blocks and additonal variables specified in `groupers`."""
        if groupers is None:
            groupers = []
        return self.df.groupby(
            self.block_identifiers + groupers, observed=observed, **kwargs
        )


    def get(self, name: str | Iterable[str]) -> ArrayLike:
        """Get values of an identifier or a column from the DataFrame."""
        if name in self.index:
            return self.df.index.get_level_values(name).values
        if name in self.df.columns:
            return self.df[name].values
        return None


    def get_unique(self, name: str | Iterable[str], order: bool = True) -> ArrayLike:
        """Get the unique values of an identifier or a column from the DataFrame."""
        if order and self.label_order is not None and name in self.label_order:
            return self.df.reset_index()[[name, self.label_order[name]]].drop_duplicates().sort_values(self.label_order[name])[name].tolist()
        if name in self.index:
            return self.df.index.unique(name)
        if name in self.df.columns:
            return self.df[name].unique()
        return None

    def wrap(self, df: pd.DataFrame) -> "Experiment":
        """Wrap a DataFrame into an Experiment. Usually this is because you did some operations to the DataFrame of the original Experiment instance and want to maintain the Experiment semantics."""
        return Experiment(df, self.constants, self.block_identifiers, self.within_block_identifiers, self.experiment_conditions, self.block_metadata, True, self.label_order)


    def filter(self, conds: dict[str, Any]) -> "Experiment":
        """Filter the experiment according to conditions specified in a dictionary."""
        return self.wrap(_get_filter_df()(self.df, conds))
    

    def extend(
        self, data_dict: dict, col_name: str
    ):
        """
        Extend the DataFrame in place by adding a new column with values from the provided dictionary.

        Args:
            data_dict: A dictionary mapping keys to values. Keys should amenable to being passed to `filter_df`.
            col_name: The name of the new column to add to the DataFrame.
        """

        # Initialize the new column with NaN values
        self.df[col_name] = np.nan

        # Populate new column with values from the provided dictionary
        for key, value in data_dict.items():
            _df = _get_filter_df()(self.df, key)
            self.df.loc[_df.index, col_name] = value


    def process_blocks(
        self,
        compute_function: "SupportsBlocks",
        *args,
        use_tqdm: bool = False,
        **kwargs,
    ) -> tuple[dict, set]:
        """
        Apply a function to each block in DataFrame and aggregate results in a dictionary, where each key is a block identifier from the DataFrame.

        Args:
            dataset: Dataset containing experiment data.
            compute_function: Function to apply to each block. Takes in dataset, index, block, *args, **kwargs.
            use_tqdm: Whether to display a progress bar. Defaults to False.
            *args: Additional arguments for `compute_function`.
            **kwargs: Additional keyword arguments for `compute_function`.

        Returns:
            A dictionary of results and a set of error blocks.
        """
        results = {}
        err_blocks = set()
        for index, block in tqdm(self.blocks, disable=not use_tqdm):
            block_key = HashableDict(dict(zip(self.block_identifiers, index)))
            try:
                results[block_key] = compute_function(self, block_key, block, *args, **kwargs)
            except Exception as e:
                logger.debug(f"Could not process ({block_key}): {str(e)}")
                err_blocks.add(block_key)
        return results, err_blocks


    def __dataframe__(self, nan_as_null: bool = False, allow_copy: bool = True):
        return self.df.reset_index()


    def __len__(self):
        return len(self.df)