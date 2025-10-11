"""
Frame to Video Converter for Jupyter Notebooks

This module converts arrays of frames to videos that can be embedded in Jupyter notebooks.
"""

import io
import os
import warnings
from typing import Any, Callable, List, Optional, Tuple, Union

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML, display
from matplotlib.patches import Rectangle
from PIL import Image


class FrameVideoConverter:
    """
    Converts arrays of frames to videos for Jupyter notebook display.
    """

    def __init__(self, fps: int = 30, figsize: Tuple[int, int] = (10, 8)):
        """
        Initialize the frame video converter.

        Args:
            fps: Frames per second for the video
            figsize: Figure size for the video display
        """
        self.fps = fps
        self.figsize = figsize
        self.fig = None
        self.ax = None

    def frames_to_animation(
        self, frames: np.ndarray, duration: Optional[float] = None
    ) -> animation.FuncAnimation:
        """
        Convert frames array to matplotlib animation.

        Args:
            frames: Array of frames (shape: [n_frames, height, width, channels] or [n_frames, height, width])
            duration: Video duration in seconds (if None, calculated from fps)

        Returns:
            Matplotlib animation object
        """
        if frames.ndim == 3:
            # Grayscale frames: [n_frames, height, width]
            frames = frames[..., np.newaxis]  # Add channel dimension
        elif frames.ndim != 4:
            raise ValueError(
                f"Frames must be 3D (grayscale) or 4D (color), got {frames.ndim}D"
            )

        n_frames, height, width, channels = frames.shape

        # Calculate duration if not provided
        if duration is None:
            duration = n_frames / self.fps

        # Calculate interval between frames
        interval = 1000 / self.fps  # milliseconds

        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(0, height)
        self.ax.set_aspect("equal")
        self.ax.axis("off")

        # Create image object
        if channels == 1:
            # Grayscale
            img = self.ax.imshow(frames[0], cmap="gray", vmin=0, vmax=1)
        else:
            # Color (RGB)
            img = self.ax.imshow(frames[0])

        def animate(frame_idx):
            """Animation function."""
            if channels == 1:
                img.set_array(frames[frame_idx])
            else:
                img.set_array(frames[frame_idx])
            return [img]

        # Create animation
        anim = animation.FuncAnimation(
            self.fig,
            animate,
            frames=n_frames,
            interval=interval,
            blit=True,
            repeat=True,
        )

        return anim

    def frames_to_html_video(
        self, frames: np.ndarray, duration: Optional[float] = None
    ) -> HTML:
        """
        Convert frames array to HTML5 video for Jupyter display.

        Args:
            frames: Array of frames
            duration: Video duration in seconds

        Returns:
            HTML video object
        """
        anim = self.frames_to_animation(frames, duration)
        html_video = anim.to_html5_video()
        return HTML(html_video)

    def display_frames(self, frames: np.ndarray, duration: Optional[float] = None):
        """
        Display frames as video in Jupyter notebook.

        Args:
            frames: Array of frames
            duration: Video duration in seconds
        """
        video = self.frames_to_html_video(frames, duration)
        display(video)

    def save_video(
        self, frames: np.ndarray, filename: str, duration: Optional[float] = None
    ):
        """
        Save frames as video file.

        Args:
            frames: Array of frames
            filename: Output filename
            duration: Video duration in seconds
        """
        anim = self.frames_to_animation(frames, duration)

        # Save animation
        Writer = animation.writers["ffmpeg"]
        writer = Writer(
            fps=self.fps, metadata=dict(artist="FrameVideoConverter"), bitrate=1800
        )

        anim.save(filename, writer=writer)
        plt.close(self.fig)

    def preview_frames(self, frames: np.ndarray, n_preview: int = 5):
        """
        Display a preview of frames as static images.

        Args:
            frames: Array of frames
            n_preview: Number of preview frames to show
        """
        n_frames = len(frames)
        step = max(1, n_frames // n_preview)

        fig, axes = plt.subplots(1, n_preview, figsize=(2 * n_preview, 2))
        if n_preview == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            frame_idx = i * step
            if frame_idx < n_frames:
                if frames.ndim == 3:
                    # Grayscale
                    ax.imshow(frames[frame_idx], cmap="gray", vmin=0, vmax=1)
                else:
                    # Color
                    ax.imshow(frames[frame_idx])
                ax.set_title(f"Frame {frame_idx}")
            ax.axis("off")

        plt.tight_layout()
        plt.show()


# Convenience functions
def frames_to_video(
    frames: np.ndarray, fps: int = 30, duration: Optional[float] = None
) -> HTML:
    """
    Quick function to convert frames to video.

    Args:
        frames: Array of frames
        fps: Frames per second
        duration: Video duration in seconds

    Returns:
        HTML video object
    """
    converter = FrameVideoConverter(fps=fps)
    return converter.frames_to_html_video(frames, duration)


def display_frames_video(
    frames: np.ndarray, fps: int = 30, duration: Optional[float] = None
):
    """
    Quick function to display frames as video.

    Args:
        frames: Array of frames
        fps: Frames per second
        duration: Video duration in seconds
    """
    converter = FrameVideoConverter(fps=fps)
    converter.display_frames(frames, duration)


def save_frames_video(
    frames: np.ndarray, filename: str, fps: int = 30, duration: Optional[float] = None
):
    """
    Quick function to save frames as video file.

    Args:
        frames: Array of frames
        filename: Output filename
        fps: Frames per second
        duration: Video duration in seconds
    """
    converter = FrameVideoConverter(fps=fps)
    converter.save_video(frames, filename, duration)


def frames_to_rgb(frames: np.ndarray, cmap, **cmap_kwargs) -> np.ndarray:
    """
    Convert frames to RGB using a colormap function.

    Args:
        frames: Array of frames (shape: [n_frames, height, width])
        cmap: Colormap function that takes a value and returns RGB color
        **cmap_kwargs: Additional arguments for cmap

    Returns:
        RGB frames array (shape: [n_frames, height, width, 3])
    """
    if frames.ndim != 3:
        raise ValueError(f"Frames must be 3D (grayscale), got {frames.ndim}D")

    n_frames, height, width = frames.shape

    # Initialize RGB array
    frames_rgb = np.zeros((n_frames, height, width, 3))

    # Convert each frame
    for frame_idx in range(n_frames):
        frame = frames[frame_idx]

        # Convert each pixel value to RGB
        for i in range(height):
            for j in range(width):
                value = frame[i, j]
                try:
                    color = cmap(value, **cmap_kwargs)
                    # Ensure color is in RGB format (0-1)
                    if isinstance(color, (list, tuple, np.ndarray)):
                        if len(color) == 3:
                            frames_rgb[frame_idx, i, j] = [float(c) for c in color]
                        elif len(color) == 4:  # RGBA
                            frames_rgb[frame_idx, i, j] = [float(c) for c in color[:3]]
                        else:
                            # Fallback to grayscale
                            frames_rgb[frame_idx, i, j] = [value, value, value]
                    else:
                        # Fallback to grayscale
                        frames_rgb[frame_idx, i, j] = [value, value, value]
                except Exception as e:
                    # Fallback to grayscale if colormap fails
                    frames_rgb[frame_idx, i, j] = [value, value, value]

    return frames_rgb
