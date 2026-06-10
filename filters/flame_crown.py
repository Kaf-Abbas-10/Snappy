"""
Snappy — Flame Crown Filter (Unique / Procedural).

A unique filter that draws an animated flame/energy crown floating
above the user's head. Unlike the other filters, this one is entirely
procedurally generated using OpenCV drawing primitives — no asset
files required.

The effect features:
- Multiple flickering flame tongues with randomized heights
- A warm color palette (orange, yellow, red) with glow effects
- Frame-based animation for a dynamic, living feel
- Semi-transparent blending for a magical aura look

Landmark reference:
    - 10: Forehead top center
    - 234: Left ear (face width)
    - 454: Right ear (face width)
    - 109: Left forehead
    - 338: Right forehead
"""

import cv2
import numpy as np
import math
import random

from .base import BaseFilter
from utils import get_landmark_point, get_face_width, get_face_angle


class FlameCrownFilter(BaseFilter):
    """
    Procedural Flame Crown filter.

    Draws animated flames above the user's head using OpenCV primitives.
    Each frame, the flame shapes are slightly randomized to create a
    flickering effect. Uses alpha blending for a glow-like appearance.
    """

    def __init__(self):
        """Initialize frame counter and random seed for animation."""
        self._frame_count = 0
        # Pre-generate random offsets for consistent-but-varied flame shapes
        # Re-seeded every N frames for flickering effect
        self._flame_offsets = self._generate_offsets()
        self._update_interval = 4  # Update flame shape every N frames

    @property
    def name(self) -> str:
        return "🔥 Flame Crown"

    def _generate_offsets(self):
        """Generate random height offsets for each flame tongue."""
        return [random.uniform(0.6, 1.4) for _ in range(12)]

    def apply(self, frame: np.ndarray, landmarks, frame_shape: tuple) -> np.ndarray:
        """
        Draw an animated flame crown above the user's head.

        The crown consists of multiple elliptical flame shapes arranged
        in an arc above the forehead. Each flame has a randomized height
        that changes every few frames to create a flickering effect.

        A Gaussian blur is applied to create a glow effect, which is
        then alpha-blended onto the original frame.

        Args:
            frame: Current BGR video frame.
            landmarks: MediaPipe face landmarks.
            frame_shape: Frame dimensions (h, w, c).

        Returns:
            Frame with flame crown applied.
        """
        self._frame_count += 1

        # Regenerate flame offsets periodically for flickering
        if self._frame_count % self._update_interval == 0:
            self._flame_offsets = self._generate_offsets()

        face_w = get_face_width(landmarks, frame_shape)
        angle = get_face_angle(landmarks, frame_shape)

        # Get forehead position
        forehead = get_landmark_point(landmarks, 10, frame_shape)
        left_forehead = get_landmark_point(landmarks, 109, frame_shape)
        right_forehead = get_landmark_point(landmarks, 338, frame_shape)

        # Crown center is above the forehead
        crown_cx = forehead[0]
        crown_cy = forehead[1] - int(face_w * 0.2)

        # Crown dimensions based on face width
        crown_width = int(face_w * 1.1)
        base_flame_height = int(face_w * 0.4)

        # Create a transparent overlay for the flames
        overlay = np.zeros_like(frame, dtype=np.uint8)

        # Define warm flame colors (BGR format)
        flame_colors = [
            (0, 70, 255),     # Deep orange-red
            (0, 120, 255),    # Orange
            (0, 180, 255),    # Bright orange
            (0, 220, 255),    # Yellow-orange
            (0, 255, 255),    # Yellow
            (80, 255, 255),   # Bright yellow
        ]

        num_flames = len(self._flame_offsets)
        angle_rad = math.radians(angle)

        # Draw flame tongues along the crown arc
        for i in range(num_flames):
            # Position each flame along the crown arc
            t = (i / (num_flames - 1)) - 0.5  # -0.5 to 0.5
            spread = crown_width * 0.5

            # Base position along the arc (with face rotation)
            base_x = int(crown_cx + t * spread * 2 * math.cos(angle_rad))
            base_y = int(crown_cy + t * spread * 2 * math.sin(angle_rad))

            # Flame height with random offset for flickering
            flame_h = int(base_flame_height * self._flame_offsets[i])

            # Flame width (narrower at edges, wider in center)
            center_factor = 1.0 - abs(t) * 0.6
            flame_w = int(face_w * 0.08 * center_factor + face_w * 0.04)

            # Flame tip position (above base, rotated with face)
            tip_x = int(base_x - flame_h * math.sin(angle_rad))
            tip_y = int(base_y - flame_h * math.cos(angle_rad))

            # Draw the flame as a filled ellipse + triangle combination
            # Outer glow (larger, more transparent)
            color_idx = i % len(flame_colors)
            outer_color = flame_colors[max(0, color_idx - 1)]

            # Draw elongated ellipse for the flame body
            ellipse_center = ((base_x + tip_x) // 2, (base_y + tip_y) // 2)
            ellipse_axes = (max(flame_w, 3), max(flame_h // 2, 3))
            ellipse_angle = -angle + 90  # Perpendicular to face tilt

            cv2.ellipse(overlay, ellipse_center, ellipse_axes,
                        ellipse_angle, 0, 360, outer_color, -1)

            # Inner flame (smaller, brighter)
            inner_axes = (max(flame_w // 2, 2), max(flame_h // 3, 2))
            inner_color = flame_colors[min(color_idx + 2, len(flame_colors) - 1)]
            cv2.ellipse(overlay, ellipse_center, inner_axes,
                        ellipse_angle, 0, 360, inner_color, -1)

        # Apply Gaussian blur for glow effect
        overlay = cv2.GaussianBlur(overlay, (21, 21), 10)

        # Also draw sharper flames on top for definition
        sharp_overlay = np.zeros_like(frame, dtype=np.uint8)
        for i in range(num_flames):
            t = (i / (num_flames - 1)) - 0.5
            spread = crown_width * 0.5

            base_x = int(crown_cx + t * spread * 2 * math.cos(angle_rad))
            base_y = int(crown_cy + t * spread * 2 * math.sin(angle_rad))

            flame_h = int(base_flame_height * self._flame_offsets[i] * 0.7)
            center_factor = 1.0 - abs(t) * 0.6
            flame_w = int(face_w * 0.05 * center_factor + face_w * 0.02)

            tip_x = int(base_x - flame_h * math.sin(angle_rad))
            tip_y = int(base_y - flame_h * math.cos(angle_rad))

            ellipse_center = ((base_x + tip_x) // 2, (base_y + tip_y) // 2)
            ellipse_axes = (max(flame_w, 2), max(flame_h // 2, 2))
            ellipse_angle = -angle + 90

            color_idx = i % len(flame_colors)
            bright_color = flame_colors[min(color_idx + 3, len(flame_colors) - 1)]
            cv2.ellipse(sharp_overlay, ellipse_center, ellipse_axes,
                        ellipse_angle, 0, 360, bright_color, -1)

        sharp_overlay = cv2.GaussianBlur(sharp_overlay, (7, 7), 3)

        # Blend the glow overlay onto the frame
        # Use additive blending for a natural glow effect
        glow_mask = np.any(overlay > 0, axis=2)
        frame[glow_mask] = cv2.addWeighted(
            frame[glow_mask], 0.6,
            overlay[glow_mask], 0.8,
            0
        )

        # Add sharp flames on top
        sharp_mask = np.any(sharp_overlay > 0, axis=2)
        frame[sharp_mask] = cv2.addWeighted(
            frame[sharp_mask], 0.4,
            sharp_overlay[sharp_mask], 0.9,
            0
        )

        return frame
