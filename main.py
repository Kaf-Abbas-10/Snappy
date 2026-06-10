"""
Snappy — A Snapchat-like Face Filter App 📸

A real-time face filter application that overlays fun effects on your
webcam feed using OpenCV and MediaPipe Face Mesh.

Features:
    - 🐶 Dog Ears & Nose: Classic puppy filter with floppy ears
    - 😎 Sunglasses: Aviator shades that follow your face
    - 🔥 Flame Crown: Animated procedural flame crown
    - 🌊 Wobble Face: Sine-wave distortion effect
    - 📦 Pixelate Zoom: Dynamic face pixelation
    - ⚔️ Viking Helmet & Beard: Warrior outfit with beard
    - 👨‍🚀 Astronaut Helmet: Glass dome with reflection
    - 🎭 Masquerade Mask: Ornate venetian mask
    - 💚 Matrix Rain: Falling green characters
    - ⚡ Lightning Aura: Electric arcs effect
    - 🖤 Ink Splash: Animated ink blobs
    - 😊 Mood Meter: Expression-based emoji display
    - ❄️ Freeze Ray: Mouth-triggered frost effect
    - 🌸 Blush Reactor: Smile-reactive blush circles

Controls:
    1-9 — Filters 1-9
    Shift+1 to Shift+5 — Filters 10-14
    0 — No filter (off)
    s — Save screenshot
    q — Quit

Usage:
    python main.py

Requirements:
    pip install -r requirements.txt
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import os
from datetime import datetime

from filters import (
    DogFilter, SunglassesFilter, FlameCrownFilter,
    WobbleFilter, PixelateZoomFilter, VikingHelmetFilter,
    AstronautHelmetFilter, MasqueradeMaskFilter, MatrixRainFilter,
    LightningAuraFilter, InkSplashFilter, MoodMeterFilter,
    FreezeRayFilter, BlushReactorFilter
)


# ─── Configuration ─────────────────────────────────────────────────
WINDOW_NAME = "Snappy — Face Filters"
SCREENSHOT_DIR = "screenshots"
CAMERA_INDEX = 0       # Default webcam
TARGET_FPS = 30
MODEL_PATH = os.path.join(os.path.dirname(__file__), "assets", "face_landmarker_v2.task")


def draw_hud(frame, filter_name, fps):
    """
    Draw the heads-up display overlay on the frame.

    Shows:
    - App title and current filter name (top-left)
    - FPS counter (top-right)
    - Control hints (bottom)

    The HUD uses a semi-transparent dark background for readability.

    Args:
        frame (np.ndarray): The video frame to draw on.
        filter_name (str): Name of the currently active filter.
        fps (float): Current frames per second.
    """
    h, w = frame.shape[:2]

    # ── Top bar background ──
    top_bar = frame[0:70, :].copy()
    dark_overlay = np.zeros_like(top_bar)
    cv2.addWeighted(top_bar, 0.4, dark_overlay, 0.6, 0, top_bar)
    frame[0:70, :] = top_bar

    # ── Title ──
    cv2.putText(
        frame, "SNAPPY",
        (15, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8,
        (255, 255, 255), 2, cv2.LINE_AA
    )

    # ── Current filter name ──
    cv2.putText(
        frame, f"Filter: {filter_name}",
        (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        (150, 230, 255), 1, cv2.LINE_AA
    )

    # ── FPS counter ──
    fps_text = f"FPS: {int(fps)}"
    text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    cv2.putText(
        frame, fps_text,
        (w - text_size[0] - 15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        (100, 255, 100), 1, cv2.LINE_AA
    )

    # ── Bottom controls bar ──
    controls = "1-14: Filters  |  0: Off  |  S: Screenshot  |  Q: Quit"
    bottom_bar = frame[h - 40:h, :].copy()
    dark_bottom = np.zeros_like(bottom_bar)
    cv2.addWeighted(bottom_bar, 0.4, dark_bottom, 0.6, 0, bottom_bar)
    frame[h - 40:h, :] = bottom_bar

    text_size = cv2.getTextSize(controls, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
    text_x = (w - text_size[0]) // 2
    cv2.putText(
        frame, controls,
        (text_x, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
        (200, 200, 200), 1, cv2.LINE_AA
    )


def save_screenshot(frame):
    """
    Save the current frame as a PNG screenshot.

    Screenshots are saved to the 'screenshots/' directory with a
    timestamp-based filename.

    Args:
        frame (np.ndarray): The frame to save.

    Returns:
        str: Path to the saved screenshot.
    """
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"snappy_{timestamp}.png"
    filepath = os.path.join(SCREENSHOT_DIR, filename)
    cv2.imwrite(filepath, frame)
    print(f"📸 Screenshot saved: {filepath}")
    return filepath


def draw_screenshot_flash(frame):
    """
    Draw a brief white flash overlay to indicate a screenshot was taken.

    Args:
        frame (np.ndarray): The frame to flash.

    Returns:
        np.ndarray: The frame with white flash overlay.
    """
    flash = np.ones_like(frame, dtype=np.uint8) * 255
    return cv2.addWeighted(frame, 0.3, flash, 0.7, 0)


def main():
    """
    Main application loop.

    Initializes the webcam, MediaPipe FaceLandmarker, and all filters.
    Runs the main video processing loop with filter application,
    HUD rendering, and keyboard input handling.
    """
    print("=" * 50)
    print("  🎭 SNAPPY — Face Filter App")
    print("=" * 50)
    print()
    print("  Starting webcam...")
    print()

    # ── Initialize webcam ──
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam!")
        print("   Make sure your camera is connected and not in use.")
        return

    # Set camera properties for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  📷 Camera resolution: {actual_w}x{actual_h}")

    # ── Initialize MediaPipe FaceLandmarker (Tasks API) ──
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file not found at {MODEL_PATH}")
        print("   Download it from: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task")
        cap.release()
        return

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    face_landmarker = vision.FaceLandmarker.create_from_options(options)

    print("  ✅ MediaPipe FaceLandmarker initialized")

    # ── Initialize filters ──
    filters = {
        1: DogFilter(),
        2: SunglassesFilter(),
        3: FlameCrownFilter(),
        4: WobbleFilter(),
        5: PixelateZoomFilter(),
        6: VikingHelmetFilter(),
        7: AstronautHelmetFilter(),
        8: MasqueradeMaskFilter(),
        9: MatrixRainFilter(),
        10: LightningAuraFilter(),
        11: InkSplashFilter(),
        12: MoodMeterFilter(),
        13: FreezeRayFilter(),
        14: BlushReactorFilter(),
    }

    current_filter_key = 0  # 0 = no filter
    screenshot_flash_frames = 0  # Counter for flash effect

    print(f"  ✅ Loaded {len(filters)} filters:")
    for key, f in filters.items():
        print(f"     [{key}] {f.name}")
    print()
    print("  Press Q to quit")
    print("=" * 50)

    # ── FPS tracking ──
    fps = 0.0
    prev_time = time.time()
    frame_count = 0
    timestamp_ms = 0

    # ── Main loop ──
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Failed to read from webcam!")
            break

        # Flip horizontally for mirror effect (more natural interaction)
        frame = cv2.flip(frame, 1)

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image and detect landmarks
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms += 33  # ~30fps timestamps
        results = face_landmarker.detect_for_video(mp_image, timestamp_ms)

        # Apply the active filter if a face is detected
        if results.face_landmarks and current_filter_key in filters:
            for face_landmarks in results.face_landmarks:
                frame = filters[current_filter_key].apply(
                    frame, face_landmarks, frame.shape
                )

        # Calculate FPS
        frame_count += 1
        current_time = time.time()
        elapsed = current_time - prev_time
        if elapsed >= 0.5:  # Update FPS every 0.5 seconds
            fps = frame_count / elapsed
            frame_count = 0
            prev_time = current_time

        # Get current filter name for HUD
        if current_filter_key == 0:
            filter_name = "None"
        else:
            filter_name = filters[current_filter_key].name

        # Draw HUD
        draw_hud(frame, filter_name, fps)

        # Screenshot flash effect
        if screenshot_flash_frames > 0:
            frame = draw_screenshot_flash(frame)
            screenshot_flash_frames -= 1

        # Display the frame
        cv2.imshow(WINDOW_NAME, frame)

        # ── Keyboard input ──
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == ord('Q'):
            # Quit
            print("\n👋 Goodbye!")
            break

        elif key == ord('0'):
            # No filter
            current_filter_key = 0
            print("  🔄 Filter: None")

        elif key == ord('1'):
            current_filter_key = 1
            print(f"  🔄 Filter: {filters[1].name}")

        elif key == ord('2'):
            current_filter_key = 2
            print(f"  🔄 Filter: {filters[2].name}")

        elif key == ord('3'):
            current_filter_key = 3
            print(f"  🔄 Filter: {filters[3].name}")

        elif key == ord('4'):
            current_filter_key = 4
            print(f"  🔄 Filter: {filters[4].name}")

        elif key == ord('5'):
            current_filter_key = 5
            print(f"  🔄 Filter: {filters[5].name}")

        elif key == ord('6'):
            current_filter_key = 6
            print(f"  🔄 Filter: {filters[6].name}")

        elif key == ord('7'):
            current_filter_key = 7
            print(f"  🔄 Filter: {filters[7].name}")

        elif key == ord('8'):
            current_filter_key = 8
            print(f"  🔄 Filter: {filters[8].name}")

        elif key == ord('9'):
            current_filter_key = 9
            print(f"  🔄 Filter: {filters[9].name}")

        elif key == ord('!'):  # Shift+1
            current_filter_key = 10
            print(f"  🔄 Filter: {filters[10].name}")

        elif key == ord('@'):  # Shift+2
            current_filter_key = 11
            print(f"  🔄 Filter: {filters[11].name}")

        elif key == ord('#'):  # Shift+3
            current_filter_key = 12
            print(f"  🔄 Filter: {filters[12].name}")

        elif key == ord('$'):  # Shift+4
            current_filter_key = 13
            print(f"  🔄 Filter: {filters[13].name}")

        elif key == ord('%'):  # Shift+5
            current_filter_key = 14
            print(f"  🔄 Filter: {filters[14].name}")

        elif key == ord('s') or key == ord('S'):
            # Save screenshot (without HUD)
            save_screenshot(frame)
            screenshot_flash_frames = 5  # Flash for 5 frames

    # ── Cleanup ──
    cap.release()
    cv2.destroyAllWindows()
    face_landmarker.close()
    print("  Resources released. App closed.")


if __name__ == "__main__":
    main()

