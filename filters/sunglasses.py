"""
Snappy — Sunglasses Filter.

Overlays stylish aviator sunglasses aligned to the user's eyes.
Uses eye corner landmarks for precise positioning and rotates the
sunglasses to match the face tilt angle.

Landmark reference:
    - 33: Left eye outer corner
    - 263: Right eye outer corner
    - 133: Left eye inner corner
    - 362: Right eye inner corner
    - 159: Left eye top
    - 145: Left eye bottom
"""

import os
import math
import numpy as np

from .base import BaseFilter
from utils import overlay_image_rotated, get_landmark_point, get_face_width, get_face_angle, load_asset


class SunglassesFilter(BaseFilter):
    """
    Sunglasses filter.

    Positions aviator sunglasses over the user's eyes, scaling to match
    face width and rotating to follow face tilt. The glasses are sized
    to extend slightly beyond the face for a natural look.
    """

    def __init__(self):
        """Load the sunglasses PNG asset."""
        assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
        self._glasses = load_asset(os.path.join(assets_dir, "sunglasses.png"))

    @property
    def name(self) -> str:
        return "😎 Sunglasses"

    def apply(self, frame: np.ndarray, landmarks, frame_shape: tuple) -> np.ndarray:
        """
        Apply sunglasses overlay to the frame.

        The sunglasses are centered between the two eyes and scaled to
        ~1.15x face width. They rotate to match the face tilt angle.

        Args:
            frame: Current BGR video frame.
            landmarks: MediaPipe face landmarks.
            frame_shape: Frame dimensions (h, w, c).

        Returns:
            Frame with sunglasses applied.
        """
        if self._glasses is None:
            return frame

        # Get eye positions for centering
        left_eye_outer = get_landmark_point(landmarks, 33, frame_shape)
        right_eye_outer = get_landmark_point(landmarks, 263, frame_shape)

        # Center point between the eyes
        center_x = (left_eye_outer[0] + right_eye_outer[0]) // 2
        center_y = (left_eye_outer[1] + right_eye_outer[1]) // 2

        # Scale glasses to face width (slightly wider than face)
        face_w = get_face_width(landmarks, frame_shape)
        glasses_w = int(face_w * 1.15)
        glasses_h = int(glasses_w * self._glasses.shape[0] / self._glasses.shape[1])

        # Get face rotation angle
        angle = get_face_angle(landmarks, frame_shape)

        # Overlay with rotation
        frame = overlay_image_rotated(
            frame, self._glasses, center_x, center_y,
            glasses_w, glasses_h, angle
        )

        return frame
