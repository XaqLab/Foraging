import json
import logging.config
import os
from pathlib import Path

from dotenv import load_dotenv

# Load config
current_dir = Path(__file__).resolve().parent
load_dotenv(current_dir / ".env")
with open(current_dir / os.getenv("LOGCONFIG_PATH"), "rb") as fp:
    logging.config.dictConfig(json.load(fp))

# Define top-level constants
# Plotting parameters
MULTIPLOT_FIGSIZE: tuple[int, int] = (20, 5)

# Time series analysis parameters
BIN_WIDTH: float = 0.5  # seconds
WINDOW_SIZE: int = 60  # seconds
STEP: int = 5  # seconds

# Random seed for reproducible results
SEED: int = 42

# Version information
__version__ = "0.1.0"
__author__ = "Foraging Research Team"

# Jupyter Book rendering configuration
TO_HTML: bool = False  # Set to True when building HTML documentation
