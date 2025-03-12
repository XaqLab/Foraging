import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import os

# Constants
BOX_LABELS = ['fast','medium','slow']
BOX_COLORS = np.array([(0, 113.98, 188.95), (216.75, 82.87, 24.99), (236.89, 176.97, 31.87)]) / 255  # yellow, orange, blue
FIGSIZE = (15, 10)

# Get the current directory
current_dir = Path(__file__).resolve().parent

# Configure matplotlib
plt.style.use(current_dir.parent / os.getenv('PLOTCONFIG_PATH'))
from ._base import fig_init, bp, regplot, plot_elbow
