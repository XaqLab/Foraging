import numpy as np

# Plotting constants
# Palette
BOX_COLORS = list(
    np.array([(0, 169, 252), (255, 131, 0), (255, 0, 0)]) / 255
)  # blue, orange, red
BOX_COLORS_DARK = list(np.array([(0, 109, 163), (207, 107, 0), (207, 0, 0)]) / 255)
# BOX_COLORS = list(np.array([(111, 255, 0), (255, 131, 0), (255, 0, 0)]) / 255)  # green, orange, red
# BOX_COLORS_DARK = list(np.array([(60, 138, 0), (207, 107, 0), (207, 0, 0)]) / 255)

BOX_LABELS = ["fast", "medium", "slow"]

PALETTE = dict(zip(BOX_LABELS, BOX_COLORS))
PALETTE_DARK = dict(zip(BOX_LABELS, BOX_COLORS_DARK))
HEATMAP_PALETTE = dict(zip(BOX_LABELS, ["Blues_r", "Oranges_r", "Reds_r"]))

# Stimulus reliability
KAPPA_CATEGORIES = ["low", "medium", "high"]
KAPPA_LEVELS = {
    "dylan": dict(zip(["low", "high"], [(0.01, 0.04), (0.07, 0.1)])),
    "marco": dict(zip(["low", "high"], [(0.01,), (0.1, 0.2)])),
    "humans": dict(
        zip(
            ["low", "medium", "high"],
            [(0.0, 0.02), (0.03, 0.04, 0.06), (0.07, 0.08, 0.1)],
        )
    ),
    "viktor": dict(
        zip(
            ["low", "medium", "high"],
            [(0.0, 0.01, 0.02), (0.03, 0.04, 0.05), (0.07, 0.08, 0.1)],
        )
    ),
}

# Misc
MULTIPLOT_FIGSIZE = (15, 5)
BIN_WIDTH = 0.5
WINDOW_SIZE = 60
STEP = 5
BOX_POSITIONS = ["S", "NE", "NW"]
BOX_POSITIONS_ORDER = [2, 0, 1]  # NW S NE ordering
PSYCHOPHYSICS_IMAGE_SIZE = (16, 16)
SEED = 42
