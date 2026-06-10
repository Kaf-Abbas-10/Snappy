"""
Snappy — Ink Splash Filter.

Animated ink blobs that spread from the forehead down.
Resets on significant face movement.
"""

import numpy as np
import cv2
import time
import random

from .base import BaseFilter
from utils import get_landmark_point, get_face_width


class InkSplashFilter(BaseFilter):
    """
    Ink Splash filter.

    Generates animated ink blob spread effect from the forehead.
    Resets when face movement is detected.
    """

    def __init__(self):
        """Initialize ink splash animation state."""
        self._start_time = time.time()
        self._last_nose_pos = None
        self._animation_offset = 0

    @property
    def name(self) -> str:
        return "🖤 Ink Splash"

    def apply(self, frame: np.ndarray, landmarks, frame_shape: tuple) -> np.ndarray:
        """
        Apply ink splash effect.

        Args:
            frame: Current BGR video frame.
            landmarks: MediaPipe face landmarks.
            frame_shape: Shape of the frame.

        Returns:
            The frame with ink splash effect.
        """
        h, w = frame_shape[:2]
        
        # Get facial landmarks
        forehead = get_landmark_point(landmarks, 10, frame_shape)
        nose_tip = get_landmark_point(landmarks, 1, frame_shape)
        chin = get_landmark_point(landmarks, 152, frame_shape)
        left_ear = get_landmark_point(landmarks, 234, frame_shape)
        right_ear = get_landmark_point(landmarks, 454, frame_shape)
        
        face_width = get_face_width(landmarks, frame_shape)
        
        # Detect significant face movement to reset animation
        if self._last_nose_pos is not None:
            movement = abs(nose_tip[0] - self._last_nose_pos[0]) + abs(nose_tip[1] - self._last_nose_pos[1])
            if movement > face_width * 0.1:  # Reset on large movement
                self._start_time = time.time()
        
        self._last_nose_pos = nose_tip
        
        # Animation progress
        elapsed = time.time() - self._start_time
        progress = min(1.0, elapsed / 2.0)  # Animation lasts 2 seconds
        
        # Draw ink blobs spreading downward
        num_blobs = 5
        ink_color = (50, 50, 50)  # Dark ink
        
        for i in range(num_blobs):
            # Blob position (spreads from forehead downward)
            blob_y = forehead[1] + int((chin[1] - forehead[1]) * progress * (i / num_blobs))
            blob_x = nose_tip[0] + random.randint(-int(face_width * 0.2), int(face_width * 0.2))
            
            # Blob size (larger blobs appear first, then fade)
            base_size = int(face_width * 0.1)
            size_variation = random.randint(int(base_size * 0.5), int(base_size * 1.5))
            
            # Fade effect - early blobs fade out as animation progresses
            blob_age = progress - (i / num_blobs)
            if 0 <= blob_age <= 1.0:
                fade = int(200 * (1 - blob_age * 0.5))
                
                # Draw main ink blob
                cv2.circle(frame, (blob_x, blob_y), size_variation,
                          ink_color, -1)
                
                # Draw splatter effect (smaller circles around main blob)
                for _ in range(3):
                    splatter_x = blob_x + random.randint(-size_variation, size_variation)
                    splatter_y = blob_y + random.randint(-size_variation, size_variation)
                    splatter_size = random.randint(2, int(size_variation * 0.4))
                    
                    splatter_x = max(0, min(w - 1, splatter_x))
                    splatter_y = max(0, min(h - 1, splatter_y))
                    
                    cv2.circle(frame, (splatter_x, splatter_y), splatter_size,
                              ink_color, -1)
        
        # Draw ink trails (dripping effect)
        for i in range(int(progress * 8)):
            trail_x = nose_tip[0] + random.randint(-int(face_width * 0.15), int(face_width * 0.15))
            trail_y1 = forehead[1] + int((chin[1] - forehead[1]) * (i / 8))
            trail_y2 = trail_y1 + random.randint(10, 30)
            
            trail_x = max(0, min(w - 1, trail_x))
            trail_y2 = min(h - 1, trail_y2)
            
            cv2.line(frame, (trail_x, trail_y1), (trail_x, trail_y2),
                    ink_color, 2)
        
        return frame
