"""
Snappy — Utility functions for face filter overlays.

Provides helper functions for:
- Alpha-blended image overlay onto video frames
- MediaPipe landmark coordinate extraction
- Face angle calculation for rotation-aware filters
"""

import cv2
import numpy as np
import math


def overlay_image(frame, overlay, x, y, w, h):
    """
    Overlay a PNG image with alpha channel onto a video frame.

    This function handles:
    - Resizing the overlay to the target dimensions (w x h)
    - Alpha blending for smooth transparency
    - Boundary clipping so overlays near frame edges don't crash

    Args:
        frame (np.ndarray): The BGR video frame to draw on (modified in-place).
        overlay (np.ndarray): The BGRA overlay image (must have alpha channel).
        x (int): X-coordinate of the top-left corner for placement.
        y (int): Y-coordinate of the top-left corner for placement.
        w (int): Desired width of the overlay.
        h (int): Desired height of the overlay.

    Returns:
        np.ndarray: The frame with the overlay applied.
    """
    if overlay is None or w <= 0 or h <= 0:
        return frame

    # Resize overlay to target dimensions
    overlay_resized = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_AREA)

    # Get frame dimensions
    frame_h, frame_w = frame.shape[:2]

    # Calculate the region of interest, clipping to frame boundaries
    # Top-left corner of the overlay on the frame
    x1 = max(x, 0)
    y1 = max(y, 0)
    # Bottom-right corner of the overlay on the frame
    x2 = min(x + w, frame_w)
    y2 = min(y + h, frame_h)

    # Corresponding region in the overlay image
    ox1 = x1 - x
    oy1 = y1 - y
    ox2 = ox1 + (x2 - x1)
    oy2 = oy1 + (y2 - y1)

    # Check if there's any visible region
    if x1 >= x2 or y1 >= y2 or ox1 >= ox2 or oy1 >= oy2:
        return frame

    # Extract the overlay region and its alpha channel
    overlay_crop = overlay_resized[oy1:oy2, ox1:ox2]

    if overlay_crop.shape[2] == 4:
        # Separate BGR and alpha channels
        overlay_bgr = overlay_crop[:, :, :3]
        alpha = overlay_crop[:, :, 3].astype(float) / 255.0

        # Expand alpha to 3 channels for broadcasting
        alpha_3ch = np.stack([alpha] * 3, axis=-1)

        # Extract the frame region
        frame_region = frame[y1:y2, x1:x2].astype(float)

        # Alpha blend: result = overlay * alpha + frame * (1 - alpha)
        blended = overlay_bgr.astype(float) * alpha_3ch + frame_region * (1.0 - alpha_3ch)
        frame[y1:y2, x1:x2] = blended.astype(np.uint8)
    else:
        # No alpha channel — just paste directly
        frame[y1:y2, x1:x2] = overlay_crop[:, :, :3]

    return frame


def overlay_image_rotated(frame, overlay, center_x, center_y, w, h, angle_deg):
    """
    Overlay a PNG image with alpha channel, rotated by a given angle.

    Useful for filters that need to follow the tilt of the face (e.g., sunglasses).

    Args:
        frame (np.ndarray): The BGR video frame to draw on (modified in-place).
        overlay (np.ndarray): The BGRA overlay image (must have alpha channel).
        center_x (int): X-coordinate of the center point for placement.
        center_y (int): Y-coordinate of the center point for placement.
        w (int): Desired width of the overlay before rotation.
        h (int): Desired height of the overlay before rotation.
        angle_deg (float): Rotation angle in degrees (positive = counter-clockwise).

    Returns:
        np.ndarray: The frame with the rotated overlay applied.
    """
    if overlay is None or w <= 0 or h <= 0:
        return frame

    # Resize overlay to target dimensions
    overlay_resized = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_AREA)

    # Get rotation matrix centered on the overlay
    ow, oh = w, h
    rot_mat = cv2.getRotationMatrix2D((ow // 2, oh // 2), angle_deg, 1.0)

    # Calculate new bounding box size after rotation
    cos_a = abs(rot_mat[0, 0])
    sin_a = abs(rot_mat[0, 1])
    new_w = int(oh * sin_a + ow * cos_a)
    new_h = int(oh * cos_a + ow * sin_a)

    # Adjust the rotation matrix for the new dimensions
    rot_mat[0, 2] += (new_w - ow) / 2
    rot_mat[1, 2] += (new_h - oh) / 2

    # Rotate the overlay (with alpha preserved)
    rotated = cv2.warpAffine(
        overlay_resized, rot_mat, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )

    # Place using the non-rotated overlay function, centering on the given point
    x = center_x - new_w // 2
    y = center_y - new_h // 2

    return overlay_image(frame, rotated, x, y, new_w, new_h)


def get_landmark_point(landmarks, index, frame_shape):
    """
    Convert a normalized MediaPipe face landmark to pixel coordinates.

    MediaPipe returns landmarks as normalized (0.0-1.0) coordinates.
    This function converts them to actual pixel positions in the frame.

    Args:
        landmarks: MediaPipe face landmarks object.
        index (int): The landmark index (0-467 for Face Mesh).
        frame_shape (tuple): Shape of the frame (height, width, channels).

    Returns:
        tuple: (x, y) pixel coordinates as integers.
    """
    h, w = frame_shape[:2]
    # Support both old (solutions) and new (tasks) MediaPipe API formats
    # New API: landmarks is a list of NormalizedLandmark
    # Old API: landmarks has .landmark attribute
    if hasattr(landmarks, 'landmark'):
        lm = landmarks.landmark[index]
    else:
        lm = landmarks[index]
    return int(lm.x * w), int(lm.y * h)


def get_face_width(landmarks, frame_shape):
    """
    Calculate the width of the face in pixels using ear-to-ear landmarks.

    Uses landmarks 454 (right ear) and 234 (left ear) — the widest points
    of the face mesh.

    Args:
        landmarks: MediaPipe face landmarks object.
        frame_shape (tuple): Shape of the frame.

    Returns:
        int: Face width in pixels.
    """
    left = get_landmark_point(landmarks, 234, frame_shape)
    right = get_landmark_point(landmarks, 454, frame_shape)
    return int(math.dist(left, right))


def get_face_angle(landmarks, frame_shape):
    """
    Calculate the tilt angle of the face in degrees.

    Uses the left and right eye outer corner landmarks to determine
    the roll angle of the face. Returns the angle in degrees where:
    - 0° = face is level
    - Positive = tilted counter-clockwise
    - Negative = tilted clockwise

    Args:
        landmarks: MediaPipe face landmarks object.
        frame_shape (tuple): Shape of the frame.

    Returns:
        float: Face tilt angle in degrees.
    """
    # Left eye outer corner (landmark 33) and right eye outer corner (landmark 263)
    left_eye = get_landmark_point(landmarks, 33, frame_shape)
    right_eye = get_landmark_point(landmarks, 263, frame_shape)

    # Calculate angle between the two eyes
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]

    angle = math.degrees(math.atan2(dy, dx))

    # Negate because the webcam frame is horizontally flipped (mirrored),
    # which inverts the tilt direction for overlays
    return -angle


def load_asset(path):
    """
    Load a PNG image with its alpha channel preserved.

    Args:
        path (str): Path to the PNG file.

    Returns:
        np.ndarray: The loaded image in BGRA format, or None if loading fails.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"[WARNING] Could not load asset: {path}")
        return None
    # Ensure it has an alpha channel
    if img.shape[2] == 3:
        # Add a fully opaque alpha channel
        alpha = np.ones((img.shape[0], img.shape[1], 1), dtype=img.dtype) * 255
        img = np.concatenate([img, alpha], axis=2)
    return img
