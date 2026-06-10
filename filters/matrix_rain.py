"""
Snappy — Matrix Rain Filter (Improved).

Green cascading characters fall within the face silhouette.
Features authentic matrix aesthetic with glows, glitches, and dynamic trails.
"""

import numpy as np
import cv2
import time
import random
import math

from .base import BaseFilter
from utils import get_landmark_point, get_face_width


class MatrixRainFilter(BaseFilter):
    """
    Matrix Rain filter (Enhanced).

    Procedurally generates falling green characters (matrix-style)
    constrained to the face region with glows, glitches, and dynamic effects.
    """

    def __init__(self):
        """Initialize matrix rain with enhanced column system."""
        self._start_time = time.time()
        self._columns = []
        # Character sets for variety
        self._katakana = "ｦｧｨｩｪｫｬｭｮｯﾀﾁﾂﾃﾄﾅﾆﾇﾈﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾗﾘﾙﾚﾜﾝ"
        self._numbers = "0123456789"
        self._symbols = "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン"
        self._all_chars = self._katakana + self._numbers

    @property
    def name(self) -> str:
        return "💚 Matrix Rain"

    def _get_random_char(self):
        """Get a random matrix character."""
        # 80% katakana, 20% numbers
        if random.random() < 0.8:
            return random.choice(self._katakana)
        else:
            return random.choice(self._numbers)

    def apply(self, frame: np.ndarray, landmarks, frame_shape: tuple) -> np.ndarray:
        """
        Apply enhanced matrix rain effect to face.

        Args:
            frame: Current BGR video frame.
            landmarks: MediaPipe face landmarks.
            frame_shape: Shape of the frame.

        Returns:
            The frame with matrix rain effect.
        """
        h, w = frame_shape[:2]
        
        # Get face bounding region with better constraints
        forehead = get_landmark_point(landmarks, 10, frame_shape)
        chin = get_landmark_point(landmarks, 152, frame_shape)
        left_cheek = get_landmark_point(landmarks, 226, frame_shape)
        right_cheek = get_landmark_point(landmarks, 446, frame_shape)
        left_ear = get_landmark_point(landmarks, 234, frame_shape)
        right_ear = get_landmark_point(landmarks, 454, frame_shape)
        
        # Better bounds
        face_top = int(forehead[1]) - 30
        face_bottom = int(chin[1]) + 20
        face_left = max(0, int(left_ear[0]) - 25)
        face_right = min(w, int(right_ear[0]) + 25)
        
        face_width = face_right - face_left
        face_height = face_bottom - face_top
        
        # Initialize columns on first run
        if not self._columns:
            num_columns = max(3, face_width // 12)
            self._columns = []
            for i in range(num_columns):
                self._columns.append({
                    'x': face_left + int((i / num_columns) * face_width),
                    'y': random.randint(-150, face_top),
                    'speed': random.uniform(2.5, 6.0),
                    'trail_length': random.randint(6, 15),
                    'chars': [self._get_random_char() for _ in range(20)],
                    'char_index': 0,
                    'intensity_variation': random.uniform(0.7, 1.0),  # Column brightness variation
                    'glitch_counter': random.randint(0, 100),
                })
        
        elapsed = time.time() - self._start_time
        
        # Draw and update columns
        for col in self._columns:
            col['glitch_counter'] -= 1
            
            # Update column position
            col['y'] += col['speed']
            
            # Reset if it falls off screen
            if col['y'] > face_bottom + 100:
                col['y'] = face_top - 100
                col['speed'] = random.uniform(2.5, 6.0)
                col['trail_length'] = random.randint(6, 15)
                col['intensity_variation'] = random.uniform(0.7, 1.0)
                col['glitch_counter'] = random.randint(0, 100)
            
            # Draw trail of characters
            for j in range(col['trail_length']):
                char_y = int(col['y'] - j * 14)
                
                # Check if character is in face bounds
                if face_top - 20 <= char_y <= face_bottom + 20:
                    # Glitch effect (occasional character jumps)
                    char_x = col['x']
                    if col['glitch_counter'] == 0 and j < 3:
                        char_x += random.randint(-8, 8)
                    
                    # Get character
                    col['char_index'] = (col['char_index'] + 1) % len(col['chars'])
                    if j == 0 or random.random() < 0.3:
                        col['chars'][col['char_index']] = self._get_random_char()
                    
                    char = col['chars'][col['char_index']]
                    
                    # Fade effect for trail (smoother, exponential decay)
                    trail_ratio = j / max(1, col['trail_length'])
                    fade = int(255 * (1 - trail_ratio ** 1.5))
                    
                    # Apply intensity variation
                    fade = int(fade * col['intensity_variation'])
                    
                    # Bright "head" of the trail
                    if j == 0:
                        fade = int(255 * col['intensity_variation'])
                        # Add glow effect (brighter head)
                        glow_color = (0, 255, 100)
                        cv2.putText(frame, char, (char_x, char_y),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                  glow_color, 2, cv2.LINE_AA)
                    
                    # Main character
                    color = (0, fade, 0)  # Green with fade
                    cv2.putText(frame, char, (char_x, char_y),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                              color, 1, cv2.LINE_AA)
                    
                    # Occasional bright flashes (scanline effect)
                    if j < 4 and random.random() < 0.15:
                        flash_brightness = int(200 + 55 * math.sin(elapsed * 5 + j))
                        flash_color = (0, flash_brightness, 0)
                        cv2.putText(frame, char, (char_x - 1, char_y),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                                  flash_color, 1, cv2.LINE_AA)
        
        # Occasional screen glitch effect (subtle)
        if random.random() < 0.05:
            glitch_y = random.randint(face_top, face_bottom)
            glitch_height = random.randint(1, 3)
            cv2.rectangle(frame, (face_left, glitch_y), (face_right, glitch_y + glitch_height),
                         (0, 255, 0), -1)
        
        return frame

