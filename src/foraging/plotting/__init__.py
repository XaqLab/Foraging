from pathlib import Path
from matplotlib import pyplot as plt
import seaborn as sns
import os
from foraging.config.constants import BOX_COLORS, BOX_COLORS_DARK, BOX_LABELS

# Get the current directory
current_dir = Path(__file__).resolve().parent

# Configure matplotlib and seaborn
# First apply the custom style
plt.style.use(current_dir.parent / os.getenv('PLOTCONFIG_PATH'))

# Then apply seaborn style with specific overrides that won't conflict
# sns.set_theme(style='white', rc={
#     "axes.grid": False,  # Ensure grid is off
#     "axes.facecolor": (0,0,0,0),  # Match your style
#     "figure.facecolor": (0,0,0,0),  # Match your style
#     "axes.spines.right": False,
#     "axes.spines.top": False,
#     "axes.spines.left": False,
#     "axes.spines.bottom": False,
# })

# Constants
PALETTE = dict(zip(BOX_LABELS, BOX_COLORS))
PALETTE_DARK = dict(zip(BOX_LABELS, BOX_COLORS_DARK))

from ._base import fig_init, titler, unitler, legend_handler, bp, subject_plotter, enhanced_violinplot, plot_variable_subplots, format_yticks, per_block, across_blocks