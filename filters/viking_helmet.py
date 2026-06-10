"""
Snappy — Viking Helmet & Beard Filter.

Overlays a detailed viking warrior helmet and beard using PNG asset.
Positioned based on face width and chin landmarks.
"""

import os
import numpy as np

from .base import BaseFilter
from utils import get_landmark_point, get_face_width, get_face_angle, overlay_image_rotated, load_asset


class VikingHelmetFilter(BaseFilter):
    """
    Viking Helmet & Beard filter.

    Overlays a warrior helmet and beard using PNG asset with proper positioning.
    """

    def __init__(self):
        """Load the viking helmet PNG asset."""
        assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
        self._helmet = load_asset(os.path.join(assets_dir, "viking_helmet.png"))

    @property
    def name(self) -> str:
        return "⚔️ Viking Helmet & Beard"

    def apply(self, frame: np.ndarray, landmarks, frame_shape: tuple) -> np.ndarray:
        """
        Apply Viking helmet and beard overlays.

        Args:
            frame: Current BGR video frame.
            landmarks: MediaPipe face landmarks.
            frame_shape: Shape of the frame.

        Returns:
            The frame with Viking helmet and beard.
        """
        # Get facial landmarks for positioning
        forehead = get_landmark_point(landmarks, 10, frame_shape)
        nose_tip = get_landmark_point(landmarks, 1, frame_shape)
        chin = get_landmark_point(landmarks, 152, frame_shape)
        left_ear = get_landmark_point(landmarks, 234, frame_shape)
        right_ear = get_landmark_point(landmarks, 454, frame_shape)
        
        face_width = get_face_width(landmarks, frame_shape)
        face_angle = get_face_angle(landmarks, frame_shape)
        
        # Calculate helmet position and size
        # Use face center rather than just forehead
        face_center_y = (forehead[1] + chin[1]) // 2
        helmet_center_x = (left_ear[0] + right_ear[0]) // 2
        helmet_center_y = face_center_y - int(face_width * 0.05)
        
        # Scale helmet relative to face width - make it larger
        helmet_width = int(face_width * 1.6)
        helmet_height = int(helmet_width * 0.75)  # Helmet proportions
        
        # Apply helmet with rotation
        frame = overlay_image_rotated(
            frame, self._helmet,
            helmet_center_x, helmet_center_y,
            helmet_width, helmet_height,
            face_angle
        )
        
        return frame

