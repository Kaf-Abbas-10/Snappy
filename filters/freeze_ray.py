"""
Snappy — Freeze Ray Filter.

When you open your mouth wide, a frost/crystallize effect spreads across your face.
Creates an icy transformation animation.
"""

import numpy as np
import cv2
import math
import random

from .base import BaseFilter
from utils import get_landmark_point, get_face_width


class FreezeRayFilter(BaseFilter):
    """
    Freeze Ray filter.

    Detects wide mouth opening and applies a spreading crystalline/frost effect.
    """

    def __init__(self):
        """Initialize freeze state."""
        self._freeze_progress = 0.0

    @property
    def name(self) -> str:
        return "❄️ Freeze Ray"

    def apply(self, frame: np.ndarray, landmarks, frame_shape: tuple) -> np.ndarray:
        """
        Apply freeze ray effect.

        Args:
            frame: Current BGR video frame.
            landmarks: MediaPipe face landmarks.
            frame_shape: Shape of the frame.

        Returns:
            The frame with freeze effect.
        """
        h, w = frame_shape[:2]
        
        # Get facial landmarks
        forehead = get_landmark_point(landmarks, 10, frame_shape)
        nose_tip = get_landmark_point(landmarks, 1, frame_shape)
        chin = get_landmark_point(landmarks, 152, frame_shape)
        left_ear = get_landmark_point(landmarks, 234, frame_shape)
        right_ear = get_landmark_point(landmarks, 454, frame_shape)
        
        # Mouth points
        mouth_top = get_landmark_point(landmarks, 13, frame_shape)
        mouth_bottom = get_landmark_point(landmarks, 14, frame_shape)
        mouth_left = get_landmark_point(landmarks, 61, frame_shape)
        mouth_right = get_landmark_point(landmarks, 291, frame_shape)
        
        # Calculate mouth openness
        mouth_openness = abs(mouth_bottom[1] - mouth_top[1])
        mouth_width = abs(mouth_right[0] - mouth_left[0])
        mouth_aspect_ratio = mouth_openness / (mouth_width + 1)
        
        # Detect mouth wide open (trigger freeze)
        if mouth_aspect_ratio > 0.16:
            # Activate freeze
            self._freeze_progress = min(1.0, self._freeze_progress + 0.1)
        else:
            # Deactivate freeze
            self._freeze_progress = max(0.0, self._freeze_progress - 0.05)
        
        # Draw frost effect if active
        if self._freeze_progress > 0.1:
            face_center_x = (left_ear[0] + right_ear[0]) // 2
            face_center_y = (forehead[1] + chin[1]) // 2
            face_width = get_face_width(landmarks, frame_shape)
            
            # Calculate frost spread radius
            max_radius = int(face_width * 0.8)
            frost_radius = int(max_radius * self._freeze_progress)
            
            # Draw crystalline ice patterns
            num_crystals = int(20 * self._freeze_progress)
            
            for _ in range(num_crystals):
                # Random crystal position within spread radius
                angle = random.uniform(0, 2 * math.pi)
                distance = random.uniform(0, frost_radius)
                
                crystal_x = int(face_center_x + distance * math.cos(angle))
                crystal_y = int(face_center_y + distance * math.sin(angle))
                
                # Clamp to frame
                crystal_x = max(0, min(w - 1, crystal_x))
                crystal_y = max(0, min(h - 1, crystal_y))
                
                # Draw ice crystal (star/snowflake-like shape)
                crystal_size = random.randint(3, 12)
                
                # Draw lines radiating from center (snowflake pattern)
                for angle_offset in [0, 60, 120]:
                    angle_rad = math.radians(angle_offset)
                    end_x = int(crystal_x + crystal_size * math.cos(angle_rad))
                    end_y = int(crystal_y + crystal_size * math.sin(angle_rad))
                    
                    end_x = max(0, min(w - 1, end_x))
                    end_y = max(0, min(h - 1, end_y))
                    
                    cv2.line(frame, (crystal_x, crystal_y), (end_x, end_y),
                            (200, 240, 255), 1)
            
            # Draw icy glow effect (blue tint overlay)
            overlay = frame.copy()
            cv2.circle(overlay, (face_center_x, face_center_y), frost_radius,
                      (180, 220, 255), -1)
            
            # Blend with frame
            alpha = 0.15 * self._freeze_progress
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            
            # Draw frost edge ring
            cv2.circle(frame, (face_center_x, face_center_y), frost_radius,
                      (180, 220, 255), 2)
        
        return frame
