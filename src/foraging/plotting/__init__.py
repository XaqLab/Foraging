import os
from pathlib import Path

import seaborn as sns
from matplotlib import pyplot as plt

from foraging.config.constants import BOX_COLORS, BOX_COLORS_DARK, BOX_LABELS

from ._base import (
    across_blocks,
    across_conditions_plotter,
    bp,
    enhanced_violinplot,
    fig_init,
    format_yticks,
    legend_handler,
    multiplot,
    per_block,
    plot_block_average_or_traces,
    plot_quantity_across_block,
    plot_variable_subplots,
    titler,
    toggle_plot,
    unitler,
)

# Get the current directory
current_dir = Path(__file__).resolve().parent

# Configure matplotlib and seaborn
# First apply the custom style
plt.style.use(current_dir.parent / os.getenv("PLOTCONFIG_PATH"))
