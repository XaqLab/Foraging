import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import os
from foraging.config.constants import BOX_COLORS, BOX_COLORS_DARK, BOX_LABELS

# Constants
PALETTE = dict(zip(BOX_LABELS, BOX_COLORS))
PALETTE_DARK = dict(zip(BOX_LABELS, BOX_COLORS_DARK))
FIGSIZE = (15, 10)

# Get the current directory
current_dir = Path(__file__).resolve().parent

# Configure matplotlib
plt.style.use(current_dir.parent / os.getenv('PLOTCONFIG_PATH'))

from ._base import fig_init, titler, unitler, get_bar_positions, bp, enhanced_violinplot, plot_variable_subplots, format_yticks, per_block, across_blocks