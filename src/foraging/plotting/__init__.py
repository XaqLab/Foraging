from pathlib import Path
from matplotlib import pyplot as plt
import seaborn as sns
import os
from foraging.config.constants import BOX_COLORS, BOX_COLORS_DARK, BOX_LABELS

# Get the current directory
current_dir = Path(__file__).resolve().parent

# Configure matplotlib and seaborn
sns.set_theme(style = 'white', rc={"axes.grid": False})
plt.style.use(current_dir.parent / os.getenv('PLOTCONFIG_PATH'))

# Constants
PALETTE = dict(zip(BOX_LABELS, BOX_COLORS))
PALETTE_DARK = dict(zip(BOX_LABELS, BOX_COLORS_DARK))

from ._base import fig_init, titler, unitler, get_bar_positions, bp, subject_plotter, enhanced_violinplot, plot_variable_subplots, format_yticks, per_block, across_blocks