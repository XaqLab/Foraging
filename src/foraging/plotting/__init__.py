import os
from pathlib import Path

import seaborn as sns
from matplotlib import pyplot as plt

from foraging.config.constants import BOX_COLORS, BOX_COLORS_DARK, BOX_LABELS

from ._base import (
    across_blocks,
    bp,
    enhanced_violinplot,
    fig_init,
    format_yticks,
    legend_handler,
    per_block,
    plot_variable_subplots,
    subject_plotter,
    titler,
    unitler,
)

# Get the current directory
current_dir = Path(__file__).resolve().parent

# Configure matplotlib and seaborn
# First apply the custom style
plt.style.use(current_dir.parent / os.getenv("PLOTCONFIG_PATH"))

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
HEATMAP_PALETTE = dict(zip(BOX_LABELS, ["Blues_r", "Oranges_r", "Reds_r"]))
