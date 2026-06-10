"""
Snappy — Viking Helmet & Beard Filter.

Adds a warrior getup with a helmet and flowing beard.
Uses face width for helmet sizing and chin landmarks for beard placement.
"""

import numpy as np
import cv2

from .base import BaseFilter
from utils import get_landmark_point, get_face_width, get_face_angle


class VikingHelmetFilter(BaseFilter):
    """
    Viking Helmet & Beard filter.

    Overlays a warrior helmet (using OpenCV shapes) on the head and
    a flowing beard using the chin landmarks.
    """

    @property
    def name(self) -> str:
        return "⚔️ Viking Helmet & Beard"

    def apply(self, frame: np.ndarray, landmarks, frame_shape: tuple) -> np.ndarray:
        """
        Apply Viking helmet and beard overlays.

        Args:
            frame: Current BGR video frame.
            landmarks: MediaPipe face landmarks.
            frame_shape: Shape of the frame.

        Returns:
            The frame with Viking helmet and beard.
        """
        h, w = frame_shape[:2]
        
        # Get key facial points
        nose_tip = get_landmark_point(landmarks, 1, frame_shape)
        forehead = get_landmark_point(landmarks, 10, frame_shape)
        chin = get_landmark_point(landmarks, 152, frame_shape)
        left_ear = get_landmark_point(landmarks, 234, frame_shape)
        right_ear = get_landmark_point(landmarks, 454, frame_shape)
        
        face_width = get_face_width(landmarks, frame_shape)
        face_angle = get_face_angle(landmarks, frame_shape)
        
        # ─── Draw Helmet ───
        helmet_center = (nose_tip[0], int(forehead[1]) - int(face_width * 0.2))
        helmet_width = int(face_width * 1.1)
        helmet_height = int(face_width * 0.6)
        
        # Helmet outline (dark metal color)
        cv2.ellipse(frame, helmet_center, (helmet_width // 2, helmet_height // 2),
                   face_angle, 0, 180, (50, 60, 100), 3)
        
        # Helmet top ridge
        ridge_top = (helmet_center[0], helmet_center[1] - helmet_height // 2)
        ridge_left = (helmet_center[0] - helmet_width // 3, helmet_center[1])
        ridge_right = (helmet_center[0] + helmet_width // 3, helmet_center[1])
        
        pts = np.array([ridge_top, ridge_left, ridge_right], np.int32)
        cv2.polylines(frame, [pts], True, (100, 120, 180), 2)
        
        # Horns
        horn_length = int(face_width * 0.3)
        horn_angle_offset = 30
        
        left_horn_base = (helmet_center[0] - helmet_width // 3, helmet_center[1] - helmet_height // 3)
        right_horn_base = (helmet_center[0] + helmet_width // 3, helmet_center[1] - helmet_height // 3)
        
        # Left horn
        left_horn_end = (int(left_horn_base[0] - horn_length * np.cos(np.radians(face_angle + horn_angle_offset))),
                        int(left_horn_base[1] - horn_length * np.sin(np.radians(face_angle + horn_angle_offset))))
        cv2.line(frame, left_horn_base, left_horn_end, (200, 200, 100), 3)
        
        # Right horn
        right_horn_end = (int(right_horn_base[0] + horn_length * np.cos(np.radians(face_angle - horn_angle_offset))),
                         int(right_horn_base[1] - horn_length * np.sin(np.radians(face_angle - horn_angle_offset))))
        cv2.line(frame, right_horn_base, right_horn_end, (200, 200, 100), 3)
        
        # ─── Draw Beard ───
        beard_start = chin
        beard_length = int(face_width * 0.4)
        beard_width = int(face_width * 0.3)
        
        # Beard is wavy brown triangular shape
        beard_bottom = (beard_start[0], min(h - 1, beard_start[1] + beard_length))
        beard_left = (max(0, beard_start[0] - beard_width // 2), beard_start[1] + int(beard_length * 0.6))
        beard_right = (min(w - 1, beard_start[0] + beard_width // 2), beard_start[1] + int(beard_length * 0.6))
        
        beard_pts = np.array([beard_start, beard_left, beard_bottom, beard_right], np.int32)
        cv2.fillPoly(frame, [beard_pts], (100, 100, 150))
        cv2.polylines(frame, [beard_pts], True, (60, 60, 100), 2)
        
        # Beard texture (wavy lines)
        for i in range(3):
            y_offset = int(i * beard_length / 3)
            x_offset = int(10 * np.sin(i))
            cv2.line(frame,
                    (beard_start[0] + x_offset, beard_start[1] + y_offset),
                    (beard_start[0] - x_offset, beard_start[1] + y_offset + int(beard_length * 0.15)),
                    (50, 50, 80), 1)
        
        return frame
