"""
Snappy — Masquerade Mask Filter.

An ornate venetian half-mask overlay covering the nose bridge and eye region.
Rotation-aware to follow face tilt. Uses PNG asset for realistic appearance.
"""

import os
import numpy as np

from .base import BaseFilter
from utils import get_landmark_point, get_face_width, get_face_angle, overlay_image_rotated, load_asset


class MasqueradeMaskFilter(BaseFilter):
    """
    Masquerade Mask filter.

    Overlays an ornate venetian-style half-mask on the eyes and nose,
    with rotation awareness using PNG asset.
    """

    def __init__(self):
        """Load the masquerade mask PNG asset."""
        assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
        self._mask = load_asset(os.path.join(assets_dir, "masquerade_mask.png"))

    @property
    def name(self) -> str:
        return "🎭 Masquerade Mask"

    def apply(self, frame: np.ndarray, landmarks, frame_shape: tuple) -> np.ndarray:
        """
        Apply masquerade mask overlay.

        Args:
            frame: Current BGR video frame.
            landmarks: MediaPipe face landmarks.
            frame_shape: Shape of the frame.

        Returns:
            The frame with masquerade mask.
        """
        # Get facial landmarks for positioning
        nose_bridge = get_landmark_point(landmarks, 6, frame_shape)
        left_eye = get_landmark_point(landmarks, 33, frame_shape)
        right_eye = get_landmark_point(landmarks, 263, frame_shape)
        
        face_width = get_face_width(landmarks, frame_shape)
        face_angle = get_face_angle(landmarks, frame_shape)
        
        # Calculate mask position and size
        mask_center_x = (left_eye[0] + right_eye[0]) // 2
        mask_center_y = (left_eye[1] + right_eye[1]) // 2
        
        # Scale mask relative to face width
        mask_width = int(face_width * 1.2)
        mask_height = int(mask_width * 0.65)  # Mask proportions
        
        # Apply mask with rotation
        frame = overlay_image_rotated(
            frame, self._mask,
            mask_center_x, mask_center_y,
            mask_width, mask_height,
            face_angle
        )
        
        return frame

