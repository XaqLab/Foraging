# Set names of index
# Subject to change but always keep minimal index set needed to uniquely identify a push at the beginning and the rest such as experiment parameters at the end
from ._base import discrete_time, flatten, kwargs_handler

INDEX = (
    "subject",
    "session",
    "block",
    "push #",
    "block_id",
    "stimulus type",
    "shape",
    "stimulus reliability",
    "kappa",
    "week day",
)
MIN_INDEX = 4  # Marks the end of the minimal index
