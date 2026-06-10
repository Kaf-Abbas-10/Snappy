"""
Snappy — Astronaut Helmet Filter.

Creates a transparent glass dome over the entire head region using PNG asset.
Features proper positioning and sizing based on facial landmarks.
"""

import os
import numpy as np

from .base import BaseFilter
from utils import get_landmark_point, get_face_width, get_face_angle, overlay_image_rotated, load_asset


class AstronautHelmetFilter(BaseFilter):
    """
    Astronaut Helmet filter.

    Overlays a glass dome over the head using PNG asset with proper scaling.
    """

    def __init__(self):
        """Load the astronaut helmet PNG asset."""
        assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
        self._helmet = load_asset(os.path.join(assets_dir, "astronaut_helmet.png"))

    @property
    def name(self) -> str:
        return "👨‍🚀 Astronaut Helmet"

    def apply(self, frame: np.ndarray, landmarks, frame_shape: tuple) -> np.ndarray:
        """
        Apply astronaut helmet overlay.

        Args:
            frame: Current BGR video frame.
            landmarks: MediaPipe face landmarks.
            frame_shape: Shape of the frame.

        Returns:
            The frame with astronaut helmet.
        """
        # Get facial landmarks for positioning
        forehead = get_landmark_point(landmarks, 10, frame_shape)
        chin = get_landmark_point(landmarks, 152, frame_shape)
        left_ear = get_landmark_point(landmarks, 234, frame_shape)
        right_ear = get_landmark_point(landmarks, 454, frame_shape)
        
        face_width = get_face_width(landmarks, frame_shape)
        face_angle = get_face_angle(landmarks, frame_shape)
        
        # Calculate helmet position and size
        face_center_x = (left_ear[0] + right_ear[0]) // 2
        face_center_y = (forehead[1] + chin[1]) // 2
        helmet_center_x = face_center_x
        helmet_center_y = face_center_y
        
        # Scale helmet to fit head - make it much larger
        helmet_width = int(face_width * 1.8)
        helmet_height = int(helmet_width * 1.1)  # Slightly taller for dome
        
        # Apply helmet with rotation
        frame = overlay_image_rotated(
            frame, self._helmet,
            helmet_center_x, helmet_center_y,
            helmet_width, helmet_height,
            face_angle
        )
        
        return frame

