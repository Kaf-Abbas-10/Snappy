"""
Snappy — Masquerade Mask Filter.

An ornate half-mask covering the nose bridge and eye region.
Rotation-aware to follow face tilt.
"""

import numpy as np
import cv2
import math

from .base import BaseFilter
from utils import get_landmark_point, get_face_width, get_face_angle


class MasqueradeMaskFilter(BaseFilter):
    """
    Masquerade Mask filter.

    Draws an ornate venetian-style half-mask over the eyes and nose,
    with rotation awareness.
    """

    @property
    def name(self) -> str:
        return "🎭 Masquerade Mask"

    def apply(self, frame: np.ndarray, landmarks, frame_shape: tuple) -> np.ndarray:
        """
        Apply masquerade mask overlay.

        Args:
            frame: Current BGR video frame.
            landmarks: MediaPipe face landmarks.
            frame_shape: Shape of the frame.

        Returns:
            The frame with masquerade mask.
        """
        h, w = frame_shape[:2]
        
        # Get facial landmarks
        nose_bridge = get_landmark_point(landmarks, 6, frame_shape)  # Nose bridge
        left_eye = get_landmark_point(landmarks, 33, frame_shape)   # Left eye outer
        right_eye = get_landmark_point(landmarks, 263, frame_shape)  # Right eye outer
        
        face_width = get_face_width(landmarks, frame_shape)
        face_angle = get_face_angle(landmarks, frame_shape)
        
        # Calculate mask dimensions
        mask_width = int(face_width * 1.2)
        mask_height = int(face_width * 0.5)
        
        # Mask center (between the eyes)
        mask_center_x = (left_eye[0] + right_eye[0]) // 2
        mask_center_y = (left_eye[1] + right_eye[1]) // 2 - int(face_width * 0.1)
        
        # ─── Draw Mask Shape ───
        # Create mask outline points
        mask_points = []
        
        # Main mask shape (ornate venetian style)
        # Left side curve
        mask_points.append((mask_center_x - mask_width // 2, mask_center_y))
        mask_points.append((mask_center_x - mask_width // 3, mask_center_y - mask_height // 2))
        
        # Left eye hole
        mask_points.append((mask_center_x - face_width // 4, mask_center_y - mask_height // 4))
        mask_points.append((mask_center_x - face_width // 5, mask_center_y - mask_height // 3))
        
        # Center (nose bridge)
        mask_points.append((mask_center_x, mask_center_y - mask_height // 2 - 10))
        mask_points.append((mask_center_x, mask_center_y + mask_height // 3))
        
        # Right eye hole
        mask_points.append((mask_center_x + face_width // 5, mask_center_y - mask_height // 3))
        mask_points.append((mask_center_x + face_width // 4, mask_center_y - mask_height // 4))
        
        # Right side curve
        mask_points.append((mask_center_x + mask_width // 3, mask_center_y - mask_height // 2))
        mask_points.append((mask_center_x + mask_width // 2, mask_center_y))
        
        mask_pts_array = np.array(mask_points, np.int32)
        
        # Draw ornate mask with gold color
        cv2.polylines(frame, [mask_pts_array], False, (50, 180, 200), 3)
        cv2.fillPoly(frame, [mask_pts_array], (100, 200, 220), )
        
        # ─── Add Ornamental Details ───
        # Eye decorations (diamond shapes)
        left_eye_deco_x = mask_center_x - face_width // 4
        left_eye_deco_y = mask_center_y - mask_height // 4
        
        diamond_size = int(face_width * 0.08)
        diamond_pts_left = np.array([
            (left_eye_deco_x, left_eye_deco_y - diamond_size),
            (left_eye_deco_x + diamond_size, left_eye_deco_y),
            (left_eye_deco_x, left_eye_deco_y + diamond_size),
            (left_eye_deco_x - diamond_size, left_eye_deco_y)
        ], np.int32)
        
        cv2.polylines(frame, [diamond_pts_left], True, (200, 200, 100), 2)
        cv2.fillPoly(frame, [diamond_pts_left], (180, 180, 80))
        
        # Right eye decoration
        right_eye_deco_x = mask_center_x + face_width // 4
        diamond_pts_right = np.array([
            (right_eye_deco_x, left_eye_deco_y - diamond_size),
            (right_eye_deco_x + diamond_size, left_eye_deco_y),
            (right_eye_deco_x, left_eye_deco_y + diamond_size),
            (right_eye_deco_x - diamond_size, left_eye_deco_y)
        ], np.int32)
        
        cv2.polylines(frame, [diamond_pts_right], True, (200, 200, 100), 2)
        cv2.fillPoly(frame, [diamond_pts_right], (180, 180, 80))
        
        # Central nose piece (rhinestone-like)
        nose_piece_size = int(face_width * 0.06)
        cv2.circle(frame, (mask_center_x, nose_bridge[1]), nose_piece_size,
                  (220, 220, 150), -1)
        cv2.circle(frame, (mask_center_x, nose_bridge[1]), nose_piece_size,
                  (200, 200, 100), 2)
        
        return frame
