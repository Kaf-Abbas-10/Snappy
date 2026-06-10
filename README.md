# 🎭 Snappy — Snapchat-like Face Filter App

A real-time face filter desktop application built with Python, OpenCV, and MediaPipe. Apply fun filters to your webcam feed — just like Snapchat, but running locally on your machine!

---

## ✨ Features

| Filter | Key | Description |
|---|---|---|
| 🐶 **Dog Ears & Nose** | `1` | Classic puppy filter with floppy cartoon ears and a cute nose |
| 😎 **Sunglasses** | `2` | Aviator-style shades that follow your face tilt |
| 🔥 **Flame Crown** | `3` | Animated procedural flame crown floating above your head |

All filters:
- Track your face in real-time using **468 facial landmarks**
- Scale proportionally to your face size
- Rotate to match your head tilt
- Work at full webcam FPS

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- A working webcam

### Installation

```bash
# Clone or navigate to the project directory
cd Snappy

# Install dependencies
pip install -r requirements.txt
```

### Run the App

```bash
python main.py
```

---

## 🎮 Controls

| Key | Action |
|---|---|
| `1` | Activate Dog Ears & Nose filter |
| `2` | Activate Sunglasses filter |
| `3` | Activate Flame Crown filter |
| `0` | Turn off all filters |
| `S` | Save a screenshot to `screenshots/` |
| `Q` | Quit the application |

---

## 🏗️ Architecture

```
Snappy/
├── main.py                 # Entry point — webcam loop, keyboard handler, HUD
├── utils.py                # Overlay helpers, landmark extraction, asset loading
├── filters/
│   ├── __init__.py         # Package exports
│   ├── base.py             # BaseFilter abstract class
│   ├── dog.py              # Dog ears & nose filter
│   ├── sunglasses.py       # Sunglasses filter
│   └── flame_crown.py      # Procedural flame crown filter
├── assets/
│   ├── dog_ears.png        # Cartoon dog ears (transparent PNG)
│   ├── dog_nose.png        # Cartoon dog nose (transparent PNG)
│   └── sunglasses.png      # Aviator sunglasses (transparent PNG)
├── screenshots/            # Saved screenshots (created automatically)
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

### Core Components

#### `main.py` — Application Entry Point
- Opens the webcam using `cv2.VideoCapture`
- Initializes MediaPipe Face Mesh with 468-point landmark detection
- Runs the main video processing loop at ~30 FPS
- Handles keyboard input for filter switching and screenshots
- Draws the HUD (title, FPS, controls) with semi-transparent backgrounds

#### `utils.py` — Utility Functions
- **`overlay_image()`**: Alpha-blends a PNG overlay onto a frame, handling boundary clipping
- **`overlay_image_rotated()`**: Same as above but with rotation support for face tilt
- **`get_landmark_point()`**: Converts normalized MediaPipe coordinates to pixel positions
- **`get_face_width()`**: Calculates face width using ear-to-ear landmarks (234 → 454)
- **`get_face_angle()`**: Calculates face tilt using eye corner landmarks (33, 263)
- **`load_asset()`**: Loads PNG files with alpha channel preservation

#### `filters/base.py` — Base Filter Class
Abstract base class that all filters inherit from. Defines the interface:
```python
class BaseFilter(ABC):
    @property
    def name(self) -> str: ...        # Display name for HUD
    def apply(self, frame, landmarks, frame_shape) -> np.ndarray: ...
```

#### `filters/dog.py` — Dog Filter
- Places dog ears above the forehead using landmark **10** (forehead top)
- Places dog nose on landmark **1** (nose tip)
- Scales to ~1.3x face width (ears) and ~0.3x face width (nose)
- Rotates with face tilt

#### `filters/sunglasses.py` — Sunglasses Filter
- Centers sunglasses between eye outer corners (landmarks **33** and **263**)
- Scales to ~1.15x face width
- Rotates to match face tilt angle

#### `filters/flame_crown.py` — Flame Crown Filter
- **Fully procedural** — no image assets needed
- Draws 12 flame tongues using `cv2.ellipse()` along an arc above the head
- Each flame has a randomized height that updates every 4 frames (flickering)
- Uses a warm color palette: deep red → orange → bright yellow
- Applies Gaussian blur for glow, then additive blending onto the frame

---

## 🔧 How It Works

### Face Detection Pipeline

```
Webcam Frame → Flip (mirror) → RGB Conversion → MediaPipe Face Mesh
                                                         ↓
                                              468 Facial Landmarks
                                                         ↓
                                              Active Filter.apply()
                                                         ↓
                                              Draw HUD → Display
```

### MediaPipe Face Mesh Landmarks

MediaPipe provides **468 facial landmarks** as normalized (0.0–1.0) coordinates. Key landmarks used by Snappy:

| Landmark | Location | Used By |
|---|---|---|
| 1 | Nose tip | Dog nose placement |
| 10 | Forehead top center | Dog ears, Flame crown |
| 33 | Left eye outer corner | Sunglasses, face angle |
| 109 | Left forehead | Flame crown positioning |
| 133 | Left eye inner corner | Eye width reference |
| 234 | Left ear | Face width calculation |
| 263 | Right eye outer corner | Sunglasses, face angle |
| 338 | Right forehead | Flame crown positioning |
| 362 | Right eye inner corner | Eye width reference |
| 454 | Right ear | Face width calculation |

### Alpha Blending

PNG overlays are applied using per-pixel alpha blending:

```
result = overlay × alpha + background × (1 - alpha)
```

This ensures transparent regions of the overlay show the webcam feed underneath, while opaque regions display the filter.

### Rotation Handling

Filters that track face tilt use `cv2.warpAffine()` to rotate the overlay:

1. Calculate the angle between left and right eye corners
2. Build a rotation matrix centered on the overlay
3. Expand the bounding box to prevent cropping
4. Apply the affine transformation
5. Overlay the rotated image using alpha blending

---

## 🎨 Adding Custom Filters

Creating a new filter is simple:

1. **Create a new file** in `filters/`, e.g., `filters/my_filter.py`
2. **Inherit from `BaseFilter`**:

```python
from .base import BaseFilter
from utils import get_landmark_point, get_face_width

class MyFilter(BaseFilter):
    @property
    def name(self) -> str:
        return "✨ My Filter"

    def apply(self, frame, landmarks, frame_shape):
        # Your filter logic here
        # Use get_landmark_point(landmarks, INDEX, frame_shape)
        # to get pixel coordinates for any of the 468 landmarks
        return frame
```

3. **Register it** in `filters/__init__.py`:
```python
from .my_filter import MyFilter
```

4. **Add it to `main.py`**:
```python
filters = {
    1: DogFilter(),
    2: SunglassesFilter(),
    3: FlameCrownFilter(),
    4: MyFilter(),  # Add your filter here
}
```

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| `opencv-python` | ≥ 4.8.0 | Video capture, image processing, display |
| `mediapipe` | ≥ 0.10.0 | Face mesh landmark detection (468 points) |
| `numpy` | ≥ 1.24.0 | Array manipulation, coordinate math |

---

## 🐛 Troubleshooting

| Issue | Solution |
|---|---|
| **Webcam not opening** | Make sure no other app is using the camera. Try changing `CAMERA_INDEX` in `main.py` |
| **Low FPS** | Reduce camera resolution in `main.py` (set to 640×480) |
| **Filter not aligned** | Ensure good lighting and face the camera directly |
| **Import errors** | Run `pip install -r requirements.txt` again |
| **MediaPipe warning on M1 Mac** | This is normal — MediaPipe still works fine on Apple Silicon |

---

## 📄 License

This project is for educational and personal use. Have fun! 🎉
