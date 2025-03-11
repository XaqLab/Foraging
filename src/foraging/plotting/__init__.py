import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import os

# Constants
BOX_LABELS = ['slow', 'medium', 'fast']
BOX_COLORS = np.array([(236.89, 176.97, 31.87), (216.75, 82.87, 24.99), (0, 113.98, 188.95)]) / 255  # yellow, orange, blue
FIGSIZE = (15, 10)

# Get the current directory
current_dir = Path(__file__).resolve().parent

# Configure matplotlib
plt.style.use(current_dir.parent / os.getenv('PLOTCONFIG_PATH'))
from ._base import plot_elbow, regplot, bp
