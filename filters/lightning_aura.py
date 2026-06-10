"""
Snappy — Lightning Aura Filter.

Procedural electric arcs that spark outward from the face outline.
Creates a dynamic energy effect each frame.
"""

import numpy as np
import cv2
import time
import random
import math

from .base import BaseFilter
from utils import get_landmark_point, get_face_width


class LightningAuraFilter(BaseFilter):
    """
    Lightning Aura filter.

    Generates procedural electric arc effects radiating from the face.
    Refreshes each frame for dynamic sparking effect.
    """

    @property
    def name(self) -> str:
        return "⚡ Lightning Aura"

    def apply(self, frame: np.ndarray, landmarks, frame_shape: tuple) -> np.ndarray:
        """
        Apply lightning aura effect.

        Args:
            frame: Current BGR video frame.
            landmarks: MediaPipe face landmarks.
            frame_shape: Shape of the frame.

        Returns:
            The frame with lightning aura.
        """
        h, w = frame_shape[:2]
        
        # Get face center and size
        nose_tip = get_landmark_point(landmarks, 1, frame_shape)
        forehead = get_landmark_point(landmarks, 10, frame_shape)
        chin = get_landmark_point(landmarks, 152, frame_shape)
        left_ear = get_landmark_point(landmarks, 234, frame_shape)
        right_ear = get_landmark_point(landmarks, 454, frame_shape)
        
        face_center_x = (left_ear[0] + right_ear[0]) // 2
        face_center_y = (forehead[1] + chin[1]) // 2
        face_width = get_face_width(landmarks, frame_shape)
        
        # Draw multiple lightning bolts radiating outward
        num_bolts = random.randint(3, 6)
        
        for _ in range(num_bolts):
            # Random angle around the face
            angle = random.uniform(0, 2 * math.pi)
            
            # Lightning starts at face edge
            start_dist = int(face_width * 0.6)
            start_x = int(face_center_x + start_dist * math.cos(angle))
            start_y = int(face_center_y + start_dist * math.sin(angle))
            
            # Lightning extends outward
            end_dist = start_dist + random.randint(40, 80)
            end_x = int(face_center_x + end_dist * math.cos(angle))
            end_y = int(face_center_y + end_dist * math.sin(angle))
            
            # Clamp to frame boundaries
            start_x = max(0, min(w - 1, start_x))
            start_y = max(0, min(h - 1, start_y))
            end_x = max(0, min(w - 1, end_x))
            end_y = max(0, min(h - 1, end_y))
            
            # Draw main bolt (bright cyan/white)
            cv2.line(frame, (start_x, start_y), (end_x, end_y), (255, 255, 100), 2)
            
            # Draw multiple branches from the main bolt
            num_branches = random.randint(2, 4)
            for _ in range(num_branches):
                # Branch point along the main bolt
                t = random.uniform(0.3, 0.8)
                branch_start_x = int(start_x + t * (end_x - start_x))
                branch_start_y = int(start_y + t * (end_y - start_y))
                
                # Branch direction (perpendicular deviation)
                branch_angle = angle + random.uniform(-math.pi/3, math.pi/3)
                branch_len = random.randint(20, 50)
                
                branch_end_x = int(branch_start_x + branch_len * math.cos(branch_angle))
                branch_end_y = int(branch_start_y + branch_len * math.sin(branch_angle))
                
                # Clamp branch
                branch_end_x = max(0, min(w - 1, branch_end_x))
                branch_end_y = max(0, min(h - 1, branch_end_y))
                
                # Draw branch (slightly dimmer)
                cv2.line(frame, (branch_start_x, branch_start_y),
                        (branch_end_x, branch_end_y), (200, 200, 50), 1)
        
        # Draw glowing effect at face center
        glow_radius = int(face_width * 0.4)
        cv2.circle(frame, (face_center_x, face_center_y), glow_radius, (100, 200, 255), 1)
        
        return frame
