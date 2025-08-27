"""
Configuration constants for the foraging package.

This module contains all package-wide constants including:
- Experimental parameters and stimuli
- Plotting configurations and color schemes
- Data processing parameters
- Default values for analysis functions
- Package metadata and versioning

All constants should be documented with clear descriptions of their purpose
and expected data types. Use type hints where possible for better IDE support.
"""

from typing import Any, Dict, List, Tuple, Union

import numpy as np

# ============================================================================
# EXPERIMENTAL PARAMETERS
# ============================================================================
# TODO: eventually, all of these fields will need to be dynamically determined based on dataset ie. angelaki vs valentin. ideally, this can accommodate any dataset adhering to a common interface
# Stimulus reliability categories and levels for different subjects
KAPPA_CATEGORIES: List[str] = ["low", "medium", "high"]

# Kappa levels (stimulus reliability) for each subject
# Format: {subject: {category: (values,)}}
KAPPA_LEVELS: Dict[str, Dict[str, Tuple[float, ...]]] = {
    "dylan": {"low": (0.01, 0.04), "high": (0.07, 0.1)},
    "marco": {"low": (0.01,), "high": (0.1, 0.2)},
    "humans": {
        "low": (0.0, 0.02),
        "medium": (0.03, 0.04, 0.06),
        "high": (0.07, 0.08, 0.1),
    },
    "viktor": {
        "low": (0.0, 0.01, 0.02),
        "medium": (0.03, 0.04, 0.05),
        "high": (0.07, 0.08, 0.1),
    },
}

# Box positions in the experimental arena
BOX_POSITIONS: List[str] = ["S", "NE", "NW"]
BOX_POSITIONS_ORDER: List[int] = [2, 0, 1]  # NW S NE ordering for plotting

# ============================================================================
# PLOTTING AND VISUALIZATION
# ============================================================================

# Color schemes for different box types
BOX_LABELS: List[str] = ["fast", "medium", "slow"]

# RGB colors (normalized to 0-1 range)
BOX_COLORS: List[Tuple[float, float, float]] = [
    (0, 169, 252),  # blue
    (255, 131, 0),  # orange
    (255, 0, 0),  # red
]
BOX_COLORS = [tuple(np.array(color) / 255) for color in BOX_COLORS]

# Darker variants for
# TODO: would be nice if defaults can be automatically determined
BOX_COLORS_DARK: List[Tuple[float, float, float]] = [
    (0, 109, 163),  # dark blue
    (207, 107, 0),  # dark orange
    (207, 0, 0),  # dark red
]
BOX_COLORS_DARK = [tuple(np.array(color) / 255) for color in BOX_COLORS_DARK]

# Alternative color scheme (commented out)
# BOX_COLORS_ALT = [(111, 255, 0), (255, 131, 0), (255, 0, 0)]  # green, orange, red
# BOX_COLORS_ALT = [tuple(np.array(color) / 255) for color in BOX_COLORS_ALT]

# Color palettes for different plotting contexts
PALETTE: Dict[str, Tuple[float, float, float]] = dict(zip(BOX_LABELS, BOX_COLORS))
PALETTE_DARK: Dict[str, Tuple[float, float, float]] = dict(
    zip(BOX_LABELS, BOX_COLORS_DARK)
)
HEATMAP_PALETTE: Dict[str, str] = dict(
    zip(BOX_LABELS, ["Blues_r", "Oranges_r", "Reds_r"])
)  # Useful for belief heatmaps

# Default figure sizes
MULTIPLOT_FIGSIZE: Tuple[int, int] = (15, 5)
PSYCHOPHYSICS_IMAGE_SIZE: Tuple[int, int] = (16, 16)

# ============================================================================
# DATA PROCESSING PARAMETERS
# ============================================================================

# Time series analysis parameters
BIN_WIDTH: float = 0.5  # seconds
WINDOW_SIZE: int = 60  # seconds
STEP: int = 5  # seconds

# Random seed for reproducible results
SEED: int = 42

# ============================================================================
# PACKAGE METADATA
# ============================================================================

# Version information
__version__ = "0.1.0"
__author__ = "Foraging Research Team"

# ============================================================================
# DEVELOPMENT AND BUILD CONFIGURATION
# ============================================================================

# Jupyter Book rendering configuration
TO_HTML: bool = False  # Set to True when building HTML documentation

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def validate_config() -> bool:
    """
    Validate that all configuration constants are properly defined.

    Returns:
        bool: True if all validations pass, False otherwise
    """
    try:
        # Check that all subjects have kappa levels defined
        for subject, levels in KAPPA_LEVELS.items():
            assert isinstance(
                levels, dict
            ), f"Kappa levels for {subject} must be a dict"
            for category, values in levels.items():
                assert isinstance(
                    values, tuple
                ), f"Kappa values for {subject}.{category} must be a tuple"
                assert all(
                    isinstance(v, (int, float)) for v in values
                ), f"Kappa values must be numeric"

        # Check color consistency
        assert (
            len(BOX_LABELS) == len(BOX_COLORS) == len(BOX_COLORS_DARK)
        ), "Color arrays must have same length"
        assert (
            len(BOX_LABELS) == len(PALETTE) == len(PALETTE_DARK)
        ), "Palette dicts must have same length"

        # Check position consistency
        assert len(BOX_POSITIONS) == len(
            BOX_POSITIONS_ORDER
        ), "Position arrays must have same length"
        assert max(BOX_POSITIONS_ORDER) < len(
            BOX_POSITIONS
        ), "Position order indices out of bounds"

        return True

    except AssertionError as e:
        print(f"Configuration validation failed: {e}")
        return False
