# Set names of index
# Subject to change but always keep minimal index set needed to uniquely identify a push at the beginning and the rest such as experiment parameters at the end
INDEX = ['subject', 'session', 'block', 'push #', 'stimulus type', 'shape', 'kappa', 'week day']
MIN_INDEX = 4 # Marks the end of the minimal index

from ._base import flatten
from .data import make_dataframe, filter_df