"""
Snappy — Pixelate Zoom Filter.

Dynamically pixelates the face while keeping the background sharp.
The pixelation reacts to movement, creating a live censor blur effect.
"""

import numpy as np
import cv2

from .base import BaseFilter
from utils import get_landmark_point


class PixelateZoomFilter(BaseFilter):
    """
    Pixelate Zoom filter.

    Dynamically pixelates the face region while maintaining a sharp background.
    Creates a censor-blur effect that reacts to movement.
    """

    @property
    def name(self) -> str:
        return "📦 Pixelate Zoom"

    def apply(self, frame: np.ndarray, landmarks, frame_shape: tuple) -> np.ndarray:
        """
        Apply dynamic pixelation to the face region.

        Args:
            frame: Current BGR video frame.
            landmarks: MediaPipe face landmarks.
            frame_shape: Shape of the frame.

        Returns:
            The frame with pixelation applied to the face.
        """
        h, w = frame_shape[:2]
        
        # Get face bounding box
        nose_tip = get_landmark_point(landmarks, 1, frame_shape)
        forehead = get_landmark_point(landmarks, 10, frame_shape)
        chin = get_landmark_point(landmarks, 152, frame_shape)
        left_ear = get_landmark_point(landmarks, 234, frame_shape)
        right_ear = get_landmark_point(landmarks, 454, frame_shape)
        
        x_min = max(0, int(left_ear[0]) - 15)
        x_max = min(w, int(right_ear[0]) + 15)
        y_min = max(0, int(forehead[1]) - 15)
        y_max = min(h, int(chin[1]) + 15)
        
        face_region = frame[y_min:y_max, x_min:x_max].copy()
        
        # Pixelation block size (larger = more pixelated)
        pixel_size = 12
        
        # Downscale and upscale to create pixelation effect
        small = cv2.resize(face_region, (max(1, face_region.shape[1] // pixel_size), 
                                         max(1, face_region.shape[0] // pixel_size)), 
                          interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (face_region.shape[1], face_region.shape[0]), 
                               interpolation=cv2.INTER_NEAREST)
        
        # Place back
        frame[y_min:y_max, x_min:x_max] = pixelated
        
        return frame
