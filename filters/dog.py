"""
Snappy — Dog Ears & Nose Filter.

Places cute cartoon dog ears on top of the user's head and a dog nose
over their nose tip. Uses MediaPipe Face Mesh landmarks for precise
positioning and scaling.

Landmark reference:
    - 10: Forehead top center (used to position ears above the head)
    - 234: Left ear (face width reference)
    - 454: Right ear (face width reference)
    - 1: Nose tip (used to position the dog nose)
    - 33, 263: Eye corners (used for face angle)
"""

import os
import numpy as np

from .base import BaseFilter
from utils import overlay_image_rotated, get_landmark_point, get_face_width, get_face_angle, load_asset


class DogFilter(BaseFilter):
    """
    Dog Ears & Nose filter.

    Overlays cartoon dog ears above the forehead and a cartoon dog nose
    on the nose tip. Both overlays scale proportionally with face width
    and rotate to match face tilt.
    """

    def __init__(self):
        """Load the dog ears and dog nose PNG assets."""
        assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
        self._ears = load_asset(os.path.join(assets_dir, "dog_ears.png"))
        self._nose = load_asset(os.path.join(assets_dir, "dog_nose.png"))

    @property
    def name(self) -> str:
        return "🐶 Dog Ears & Nose"

    def apply(self, frame: np.ndarray, landmarks, frame_shape: tuple) -> np.ndarray:
        """
        Apply dog ears and nose overlays to the frame.

        The ears are placed above the forehead and scaled to ~1.2x face width.
        The nose is placed on the nose tip and scaled to ~0.25x face width.

        Args:
            frame: Current BGR video frame.
            landmarks: MediaPipe face landmarks.
            frame_shape: Frame dimensions (h, w, c).

        Returns:
            Frame with dog ears and nose applied.
        """
        face_w = get_face_width(landmarks, frame_shape)
        angle = get_face_angle(landmarks, frame_shape)

        # --- Dog Ears ---
        if self._ears is not None:
            # Position ears above the forehead (landmark 10 = top of forehead)
            forehead = get_landmark_point(landmarks, 10, frame_shape)

            # Scale ears to ~1.3x face width for a cute oversized look
            ear_w = int(face_w * 1.3)
            ear_h = int(ear_w * self._ears.shape[0] / self._ears.shape[1])

            # Center ears above the forehead point, shifted up
            ear_cx = forehead[0]
            ear_cy = forehead[1] - int(ear_h * 0.45)

            frame = overlay_image_rotated(frame, self._ears, ear_cx, ear_cy, ear_w, ear_h, angle)

        # --- Dog Nose ---
        if self._nose is not None:
            # Position nose on the nose tip (landmark 1)
            nose_tip = get_landmark_point(landmarks, 1, frame_shape)

            # Scale nose to ~0.3x face width
            nose_w = int(face_w * 0.3)
            nose_h = int(nose_w * self._nose.shape[0] / self._nose.shape[1])

            frame = overlay_image_rotated(frame, self._nose, nose_tip[0], nose_tip[1], nose_w, nose_h, angle)

        return frame
