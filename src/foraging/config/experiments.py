"""
Configurations for different experiments.
"""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class Config:
    def to_dict(self):
        """Export as dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Export as JSON string."""
        import json

        return json.dumps(self.to_dict(), indent=2)


@dataclass
class AngelakiExperimentConfig(Config):
    KAPPA_CATEGORIES: list[str]
    KAPPA_LEVELS: dict[str, dict[str, dict[float, str]]]
    BOX_POSITIONS: list[str]
    BOX_POSITIONS_ORDER: list[int]
    BOX_LABELS: list[str]

    def __init__(self, box_labels: list[str] = None):
        if box_labels is None:
            box_labels = ["fast", "medium", "slow"]

        self.BOX_POSITIONS = ["S", "NE", "NW"]  # This is the ordering in the data
        self.BOX_POSITIONS_ORDER = [
            2,
            0,
            1,
        ]  # Ths is the ordering for plotting (NW S NE)
        self.BOX_LABELS = box_labels
        self.KAPPA_CATEGORIES = ["low", "medium", "high"]

        # Kappa levels (stimulus reliability) for each subject
        # Format: {subject: {value: category}}
        self.KAPPA_LEVELS = {
            "dylan": {
                0.01: "low",
                0.04: "low",
                0.07: "high",
                0.1: "high",
            },
            "marco": {
                0.01: "low",
                0.1: "high",
                0.2: "high",
            },
            "humans": {
                0.0: "low",
                0.02: "low",
                0.03: "medium",
                0.04: "medium",
                0.06: "medium",
                0.07: "high",
                0.08: "high",
                0.1: "high",
            },
            "viktor": {
                0.0: "low",
                0.01: "low",
                0.02: "low",
                0.03: "medium",
                0.04: "medium",
                0.05: "medium",
                0.07: "high",
                0.08: "high",
                0.1: "high",
                0.2: "high",
            },
        }

        self.KAPPA_LEVELS_ORDER = {
            "low": 0,
            "medium": 1,
            "high": 2,
        }


@dataclass
class AngelakiPlottingConfig(Config):
    BOX_LABELS: list[str]
    BOX_COLORS: list[tuple[float, float, float]]
    BOX_COLORS_DARK: list[tuple[float, float, float]]
    PALETTE: dict[str, tuple[float, float, float]]
    PALETTE_DARK: dict[str, tuple[float, float, float]]
    HEATMAP_PALETTE: dict[str, str]

    def __init__(
        self,
        box_labels: list[str] = None,
        box_colors: list[tuple[float, float, float]] = None,
        box_colors_dark: list[tuple[float, float, float]] = None,
        heatmaps: list[str] = None,
    ):
        if box_labels is None:
            box_labels = ["fast", "medium", "slow"]
        self.BOX_LABELS = box_labels

        if box_colors is None:
            box_colors = [
                (0, 169, 252),  # blue
                (255, 131, 0),  # orange
                (255, 0, 0),  # red
            ]
        self.BOX_COLORS = [tuple(np.array(color) / 255) for color in box_colors]

        if box_colors_dark is None:
            box_colors_dark = [
                (0, 109, 163),  # dark blue
                (207, 107, 0),  # dark orange
                (207, 0, 0),  # dark red
            ]
        self.BOX_COLORS_DARK = [
            tuple(np.array(color) / 255) for color in box_colors_dark
        ]

        if heatmaps is None:
            heatmaps = ["Blues_r", "Oranges_r", "Reds_r"]

        self.BOX_POSITIONS = ["S", "NE", "NW"]
        self.BOX_POSITIONS_ORDER = [2, 0, 1]  # NW S NE ordering for plotting
        self.PALETTE = dict(zip(self.BOX_LABELS, self.BOX_COLORS))
        self.PALETTE_DARK = dict(zip(self.BOX_LABELS, self.BOX_COLORS_DARK))
        self.HEATMAP_PALETTE = dict(zip(self.BOX_LABELS, heatmaps))


@dataclass
class ValentinExperimentConfig(Config):
    BOX_POSITIONS: list[str]
    BOX_POSITIONS_ORDER: list[int]
    BOX_LABELS: list[str]

    def __init__(self, box_labels: list[str] = None):
        if box_labels is None:
            box_labels = ["fast", "slow"]

        self.BOX_LABELS = box_labels
        self.BOX_POSITIONS = ["1", "2"]
        self.BOX_POSITIONS_ORDER = [0, 1]
        self.KAPPA_LEVELS_ORDER = {
            "low": 0,
            "medium": 1,
            "high": 2,
        }


@dataclass
class ValentinPlottingConfig(Config):
    BOX_LABELS: list[str]
    BOX_COLORS: list[tuple[float, float, float]]
    BOX_COLORS_DARK: list[tuple[float, float, float]]
    PALETTE: dict[str, tuple[float, float, float]]
    PALETTE_DARK: dict[str, tuple[float, float, float]]
    HEATMAP_PALETTE: dict[str, str]

    def __init__(
        self,
        box_labels: list[str] = None,
        box_colors: list[tuple[float, float, float]] = None,
        box_colors_dark: list[tuple[float, float, float]] = None,
        heatmaps: list[str] = None,
    ):
        if box_labels is None:
            box_labels = ["fast", "slow"]
        self.BOX_LABELS = box_labels

        if box_colors is None:
            box_colors = [
                (0, 169, 252),  # blue
                (255, 0, 0),  # red
            ]
        self.BOX_COLORS = [tuple(np.array(color) / 255) for color in box_colors]

        if box_colors_dark is None:
            box_colors_dark = [
                (0, 109, 163),  # dark blue
                (207, 0, 0),  # dark red
            ]
        self.BOX_COLORS_DARK = [
            tuple(np.array(color) / 255) for color in box_colors_dark
        ]

        if heatmaps is None:
            heatmaps = ["Blues_r", "Reds_r"]

        self.BOX_POSITIONS = ["1", "2"]
        self.BOX_POSITIONS_ORDER = [0, 1]  # NW S NE ordering for plotting
        self.PALETTE = dict(zip(self.BOX_LABELS, self.BOX_COLORS))
        self.PALETTE_DARK = dict(zip(self.BOX_LABELS, self.BOX_COLORS_DARK))
        self.HEATMAP_PALETTE = dict(zip(self.BOX_LABELS, heatmaps))
