import os
from pathlib import Path

import seaborn as sns
from matplotlib import pyplot as plt

from foraging.plotting._base import (
    BasePlotter,
    across_blocks,
    across_conditions_plotter,
    bp,
    enhanced_violinplot,
    fig_init,
    format_yticks,
    get_figure_from_axes,
    legend_corrector,
    multiplot,
    per_block,
    plot_average_or_traces,
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
