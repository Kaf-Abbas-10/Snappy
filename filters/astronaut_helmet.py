"""
Snappy — Astronaut Helmet Filter.

Creates a transparent glass dome over the entire head region with
a subtle reflection sheen effect.
"""

import numpy as np
import cv2
import math

from .base import BaseFilter
from utils import get_landmark_point, get_face_width


class AstronautHelmetFilter(BaseFilter):
    """
    Astronaut Helmet filter.

    Draws a transparent glass dome over the head with reflection highlights.
    """

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
        h, w = frame_shape[:2]
        
        # Get facial reference points
        nose_tip = get_landmark_point(landmarks, 1, frame_shape)
        forehead = get_landmark_point(landmarks, 10, frame_shape)
        chin = get_landmark_point(landmarks, 152, frame_shape)
        left_ear = get_landmark_point(landmarks, 234, frame_shape)
        right_ear = get_landmark_point(landmarks, 454, frame_shape)
        
        face_width = get_face_width(landmarks, frame_shape)
        
        # Calculate dome center and size
        center_x = (left_ear[0] + right_ear[0]) // 2
        center_y = (forehead[1] + chin[1]) // 2
        
        # Dome radius (encompasses the entire head)
        dome_radius = int(face_width * 0.7)
        
        # Create overlay for the dome with transparency
        overlay = frame.copy()
        
        # Draw the glass dome (semi-transparent cyan/light blue)
        cv2.circle(overlay, (center_x, center_y), dome_radius,
                  (200, 180, 100), 2)  # Outline
        
        # Fill with semi-transparent color
        cv2.circle(overlay, (center_x, center_y), dome_radius,
                  (200, 180, 100), -1)
        
        # Blend the overlay to make it semi-transparent
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        
        # Draw the dome outline more prominently
        cv2.circle(frame, (center_x, center_y), dome_radius,
                  (180, 160, 80), 3)
        
        # ─── Add Reflection Shine ───
        # Top-left reflection (bright)
        reflection_offset = int(dome_radius * 0.3)
        reflection_x = center_x - reflection_offset
        reflection_y = center_y - reflection_offset
        reflection_size = int(dome_radius * 0.25)
        
        # Draw reflection arc (gradient effect with multiple circles)
        for i in range(reflection_size, 0, -2):
            alpha = int(200 * (1 - i / reflection_size))
            cv2.circle(frame, (reflection_x, reflection_y), i,
                      (255, 255, 220), 1)
        
        # Secondary reflection (subtle)
        sec_reflection_x = center_x + int(dome_radius * 0.4)
        sec_reflection_y = center_y + int(dome_radius * 0.3)
        sec_reflection_size = int(dome_radius * 0.15)
        
        for i in range(sec_reflection_size, 0, -2):
            cv2.circle(frame, (sec_reflection_x, sec_reflection_y), i,
                      (200, 220, 255), 1)
        
        # Dome connector ring (at the base/neck)
        connector_y = center_y + int(dome_radius * 0.8)
        connector_width = int(dome_radius * 0.5)
        cv2.ellipse(frame, (center_x, connector_y),
                   (connector_width, int(connector_width * 0.3)),
                   0, 0, 180, (150, 140, 120), 2)
        
        return frame
