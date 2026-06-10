"""
Snappy — Mood Meter Filter.

Uses mouth openness (jaw landmarks) and eyebrow raise height to display
an emoji + mood label above the head in real time.
"""

import numpy as np
import cv2
import math

from .base import BaseFilter
from utils import get_landmark_point, get_face_width


class MoodMeterFilter(BaseFilter):
    """
    Mood Meter filter.

    Analyzes facial expressions (mouth openness, eyebrow height) and
    displays a mood emoji + text label in real time.
    """

    @property
    def name(self) -> str:
        return "😊 Mood Meter"

    def apply(self, frame: np.ndarray, landmarks, frame_shape: tuple) -> np.ndarray:
        """
        Apply mood meter overlay.

        Args:
            frame: Current BGR video frame.
            landmarks: MediaPipe face landmarks.
            frame_shape: Shape of the frame.

        Returns:
            The frame with mood meter display.
        """
        h, w = frame_shape[:2]
        
        # Get facial landmarks for mood detection
        forehead = get_landmark_point(landmarks, 10, frame_shape)
        nose_tip = get_landmark_point(landmarks, 1, frame_shape)
        
        # Mouth points (for openness)
        mouth_top = get_landmark_point(landmarks, 13, frame_shape)      # Upper lip center
        mouth_bottom = get_landmark_point(landmarks, 14, frame_shape)   # Lower lip center
        mouth_left = get_landmark_point(landmarks, 61, frame_shape)     # Left mouth corner
        mouth_right = get_landmark_point(landmarks, 291, frame_shape)   # Right mouth corner
        
        # Eyebrow points (for raise detection)
        left_eyebrow_top = get_landmark_point(landmarks, 105, frame_shape)    # Left eyebrow
        right_eyebrow_top = get_landmark_point(landmarks, 334, frame_shape)   # Right eyebrow
        left_eye = get_landmark_point(landmarks, 33, frame_shape)
        right_eye = get_landmark_point(landmarks, 263, frame_shape)
        
        # Calculate mouth openness (vertical distance)
        mouth_openness = abs(mouth_bottom[1] - mouth_top[1])
        mouth_width = abs(mouth_right[0] - mouth_left[0])
        mouth_aspect_ratio = mouth_openness / (mouth_width + 1)
        
        # Calculate eyebrow raise (distance from eye to eyebrow)
        left_brow_raise = abs(left_eye[1] - left_eyebrow_top[1])
        right_brow_raise = abs(right_eye[1] - right_eyebrow_top[1])
        avg_brow_raise = (left_brow_raise + right_brow_raise) / 2
        
        # Determine mood
        mood_emoji = "😐"
        mood_label = "Neutral"
        mood_color = (100, 100, 255)
        
        # Thresholds for mood detection
        if mouth_aspect_ratio > 0.15:
            # Mouth wide open = Happy/Surprised
            if avg_brow_raise > 20:
                mood_emoji = "😲"
                mood_label = "Surprised"
                mood_color = (100, 200, 255)
            else:
                mood_emoji = "😄"
                mood_label = "Happy"
                mood_color = (100, 255, 100)
        elif avg_brow_raise > 25:
            # Raised brows = Curious/Confused
            mood_emoji = "🤔"
            mood_label = "Curious"
            mood_color = (255, 200, 100)
        elif mouth_aspect_ratio < 0.02 and avg_brow_raise < 10:
            # Closed mouth, low brows = Sad
            mood_emoji = "😢"
            mood_label = "Sad"
            mood_color = (255, 100, 100)
        
        # Draw mood indicator above head
        display_y = max(20, forehead[1] - 60)
        display_x = nose_tip[0]
        
        # Draw semi-transparent background box
        box_width = 140
        box_height = 50
        box_x1 = display_x - box_width // 2
        box_y1 = display_y - 10
        box_x2 = display_x + box_width // 2
        box_y2 = display_y + box_height - 10
        
        cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2),
                     mood_color, -1)
        cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2),
                     (255, 255, 255), 2)
        
        # Draw mood emoji
        cv2.putText(frame, mood_emoji, (display_x - 15, display_y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                   (255, 255, 255), 2, cv2.LINE_AA)
        
        # Draw mood label
        cv2.putText(frame, mood_label, (display_x - 30, display_y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                   (255, 255, 255), 1, cv2.LINE_AA)
        
        # Draw mouth and eyebrow metrics (optional debug info)
        # You can remove these for cleaner UI
        
        return frame
