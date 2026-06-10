"""
Snappy — Matrix Rain Filter.

Green cascading characters fall within the face silhouette,
leaving background untouched. Creates a hacker aesthetic.
"""

import numpy as np
import cv2
import time
import random

from .base import BaseFilter
from utils import get_landmark_point, get_face_width


class MatrixRainFilter(BaseFilter):
    """
    Matrix Rain filter.

    Procedurally generates falling green characters (matrix-style)
    constrained to the face region.
    """

    def __init__(self):
        """Initialize matrix rain with random columns."""
        self._start_time = time.time()
        self._columns = []

    @property
    def name(self) -> str:
        return "💚 Matrix Rain"

    def apply(self, frame: np.ndarray, landmarks, frame_shape: tuple) -> np.ndarray:
        """
        Apply matrix rain effect to face.

        Args:
            frame: Current BGR video frame.
            landmarks: MediaPipe face landmarks.
            frame_shape: Shape of the frame.

        Returns:
            The frame with matrix rain effect.
        """
        h, w = frame_shape[:2]
        
        # Get face bounding region
        forehead = get_landmark_point(landmarks, 10, frame_shape)
        chin = get_landmark_point(landmarks, 152, frame_shape)
        left_ear = get_landmark_point(landmarks, 234, frame_shape)
        right_ear = get_landmark_point(landmarks, 454, frame_shape)
        
        face_top = int(forehead[1]) - 20
        face_bottom = int(chin[1]) + 20
        face_left = int(left_ear[0]) - 20
        face_right = int(right_ear[0]) + 20
        
        face_width = face_right - face_left
        
        # Initialize columns if needed
        if not self._columns:
            num_columns = max(2, face_width // 15)
            self._columns = [{'y': random.randint(-100, face_top), 'speed': random.uniform(2, 5)} 
                           for _ in range(num_columns)]
        
        # Matrix characters (Japanese-style)
        chars = "ｦｧｨｩｪｫｬｭｮｯﾀﾁﾂﾃﾄﾅﾆﾇﾈﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾗﾘﾙﾚﾜﾝ0123456789"
        
        elapsed = time.time() - self._start_time
        
        # Draw and update columns
        for i, col in enumerate(self._columns):
            x = face_left + int((i / len(self._columns)) * face_width)
            
            # Update column position
            col['y'] += col['speed']
            
            # Reset if it falls off screen
            if col['y'] > face_bottom + 50:
                col['y'] = face_top - 50
                col['speed'] = random.uniform(2, 5)
            
            # Draw multiple characters in the column (trail effect)
            trail_length = 8
            for j in range(trail_length):
                char_y = int(col['y'] - j * 15)
                
                if face_top <= char_y <= face_bottom:
                    # Random character
                    char = random.choice(chars)
                    
                    # Fade effect for trail
                    fade = int(255 * (1 - j / trail_length))
                    color = (0, fade, 0)  # Green
                    
                    # Draw character
                    cv2.putText(frame, char, (x, char_y),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                              color, 1, cv2.LINE_AA)
        
        return frame
