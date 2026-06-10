"""
Snappy — Wobble Face Filter.

Applies a sine-wave warp to the face region, creating a rippling jelly-like effect.
The distortion waves flow horizontally across the face in real time.
"""

import numpy as np
import math
import time

from .base import BaseFilter
from utils import get_landmark_point


class WobbleFilter(BaseFilter):
    """
    Wobble Face filter.

    Applies a sine-wave distortion to the face region, making it ripple
    like jelly. The effect is procedural and updates each frame.
    """

    def __init__(self):
        """Initialize the wobble filter."""
        self._start_time = time.time()

    @property
    def name(self) -> str:
        return "🌊 Wobble Face"

    def apply(self, frame: np.ndarray, landmarks, frame_shape: tuple) -> np.ndarray:
        """
        Apply sine-wave distortion to the face region.

        Args:
            frame: Current BGR video frame.
            landmarks: MediaPipe face landmarks.
            frame_shape: Shape of the frame.

        Returns:
            The frame with wobble effect applied.
        """
        h, w = frame_shape[:2]
        
        # Get face bounding box from landmarks
        nose_tip = get_landmark_point(landmarks, 1, frame_shape)  # Nose tip
        forehead = get_landmark_point(landmarks, 10, frame_shape)  # Forehead
        chin = get_landmark_point(landmarks, 152, frame_shape)  # Chin
        left_ear = get_landmark_point(landmarks, 234, frame_shape)  # Left ear
        right_ear = get_landmark_point(landmarks, 454, frame_shape)  # Right ear
        
        # Calculate bounding box
        x_min = max(0, int(left_ear[0]) - 10)
        x_max = min(w, int(right_ear[0]) + 10)
        y_min = max(0, int(forehead[1]) - 10)
        y_max = min(h, int(chin[1]) + 10)
        
        # Extract the face region
        face_region = frame[y_min:y_max, x_min:x_max].copy()
        
        # Create output region
        output = face_region.copy()
        
        # Apply sine wave distortion
        elapsed = time.time() - self._start_time
        amplitude = 8  # Pixels
        frequency = 0.02  # Cycles per pixel
        speed = 3  # Pixels per second
        
        for y in range(face_region.shape[0]):
            # Calculate wave offset for this row
            offset = int(amplitude * math.sin(frequency * (y + elapsed * speed)))
            
            # Shift pixels horizontally with wrap-around
            x_indices = (np.arange(face_region.shape[1]) + offset) % face_region.shape[1]
            output[y] = face_region[y, x_indices]
        
        # Place modified region back
        frame[y_min:y_max, x_min:x_max] = output
        
        return frame
