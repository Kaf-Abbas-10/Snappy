"""
Snappy — Base filter class.

All face filters inherit from BaseFilter and implement the apply() method.
This provides a consistent interface for the main loop to call filters
without knowing their implementation details.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseFilter(ABC):
    """
    Abstract base class for all Snappy face filters.

    Each filter receives the current video frame and MediaPipe face landmarks,
    and returns the modified frame with the filter applied.

    Subclasses must implement:
        - name (property): Human-readable filter name for the HUD display.
        - apply(frame, landmarks, frame_shape): Apply the filter to the frame.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the filter, displayed on the HUD."""
        pass

    @abstractmethod
    def apply(self, frame: np.ndarray, landmarks, frame_shape: tuple) -> np.ndarray:
        """
        Apply the filter effect to the video frame.

        Args:
            frame (np.ndarray): The current BGR video frame (modified in-place).
            landmarks: MediaPipe face landmarks for the detected face.
            frame_shape (tuple): Shape of the frame (height, width, channels).

        Returns:
            np.ndarray: The modified frame with the filter applied.
        """
        pass
