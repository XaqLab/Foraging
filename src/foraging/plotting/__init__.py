import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import os
from foraging.utils import BOX_LABELS

# Constants
# BOX_COLORS = np.array([(0, 113.98, 188.95), (216.75, 82.87, 24.99), (236.89, 176.97, 31.87)]) / 255  # blue, yellow, orange
BOX_COLORS = np.array([(111, 255, 0), (255, 131, 0), (255, 0, 0)]) / 255  # green, orange, red
BOX_COLORS_DARK = np.array([(60, 138, 0), (166, 87, 0), (158, 0, 0)]) / 255
PALETTE = dict(zip(BOX_LABELS, BOX_COLORS))
PALETTE_DARK = dict(zip(BOX_LABELS, BOX_COLORS_DARK))
FIGSIZE = (15, 10)

# Get the current directory
current_dir = Path(__file__).resolve().parent

# Configure matplotlib
plt.style.use(current_dir.parent / os.getenv('PLOTCONFIG_PATH'))

from ._base import fig_init, titler, unitler, get_bar_positions, bp, enhanced_violinplot, plot_elbow, plot_variable_subplots, format_yticks, per_block, across_blocks