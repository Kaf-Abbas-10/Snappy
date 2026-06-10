"""
Snappy — Blush Reactor Filter.

Soft animated blush circles appear on cheeks, intensifying with smile.
Uses lip corner landmarks for smile detection.
"""

import numpy as np
import cv2
import math

from .base import BaseFilter
from utils import get_landmark_point, get_face_width


class BlushReactorFilter(BaseFilter):
    """
    Blush Reactor filter.

    Displays animated blush circles on cheeks that respond to smile intensity.
    """

    @property
    def name(self) -> str:
        return "🌸 Blush Reactor"

    def apply(self, frame: np.ndarray, landmarks, frame_shape: tuple) -> np.ndarray:
        """
        Apply blush reactor effect.

        Args:
            frame: Current BGR video frame.
            landmarks: MediaPipe face landmarks.
            frame_shape: Shape of the frame.

        Returns:
            The frame with blush effect.
        """
        h, w = frame_shape[:2]
        
        # Get facial landmarks
        nose_tip = get_landmark_point(landmarks, 1, frame_shape)
        left_cheek = get_landmark_point(landmarks, 226, frame_shape)    # Left cheek
        right_cheek = get_landmark_point(landmarks, 446, frame_shape)   # Right cheek
        
        # Mouth corners (for smile detection)
        mouth_left = get_landmark_point(landmarks, 61, frame_shape)
        mouth_right = get_landmark_point(landmarks, 291, frame_shape)
        mouth_top = get_landmark_point(landmarks, 13, frame_shape)
        mouth_bottom = get_landmark_point(landmarks, 14, frame_shape)
        
        # Calculate smile width (horizontal distance between mouth corners)
        smile_width = abs(mouth_right[0] - mouth_left[0])
        
        # Calculate mouth height (openness)
        mouth_height = abs(mouth_bottom[1] - mouth_top[1])
        
        # Calculate smile intensity (0.0 to 1.0)
        # Takes into account both mouth width and vertical lift
        base_smile_width = get_face_width(landmarks, frame_shape) * 0.6
        smile_intensity = min(1.0, (smile_width / (base_smile_width + 1)) + (mouth_height / 20))
        
        # Clamp smile intensity
        smile_intensity = max(0.0, min(1.0, smile_intensity))
        
        # Blush parameters
        blush_radius_base = 15
        blush_radius = int(blush_radius_base * (0.5 + smile_intensity * 0.5))
        
        # Blush intensity (alpha for transparency)
        blush_alpha = int(100 + smile_intensity * 150)
        
        # Blush color (soft pink)
        blush_color = (180, 100, 150)
        
        # Draw left blush
        left_blush_x = max(0, min(w - 1, left_cheek[0]))
        left_blush_y = max(0, min(h - 1, left_cheek[1]))
        
        # Create overlay for blush with transparency
        blush_overlay = frame.copy()
        cv2.circle(blush_overlay, (left_blush_x, left_blush_y), blush_radius,
                  blush_color, -1)
        
        # Blend with varying intensity based on smile
        alpha = blush_alpha / 255.0
        cv2.addWeighted(blush_overlay, alpha * 0.4, frame, 1 - alpha * 0.4, 0, frame)
        
        # Draw right blush
        right_blush_x = max(0, min(w - 1, right_cheek[0]))
        right_blush_y = max(0, min(h - 1, right_cheek[1]))
        
        blush_overlay = frame.copy()
        cv2.circle(blush_overlay, (right_blush_x, right_blush_y), blush_radius,
                  blush_color, -1)
        cv2.addWeighted(blush_overlay, alpha * 0.4, frame, 1 - alpha * 0.4, 0, frame)
        
        # Add subtle highlight in center of blush
        if smile_intensity > 0.3:
            highlight_radius = int(blush_radius * 0.3)
            highlight_color = (220, 150, 180)
            
            cv2.circle(frame, (left_blush_x - blush_radius // 3, left_blush_y - blush_radius // 3),
                      highlight_radius, highlight_color, -1)
            cv2.circle(frame, (right_blush_x - blush_radius // 3, right_blush_y - blush_radius // 3),
                      highlight_radius, highlight_color, -1)
        
        return frame
