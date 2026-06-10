# 🎭 Snappy — Snapchat-like Face Filter App

A real-time face filter desktop application built with Python, OpenCV, and MediaPipe. Apply fun filters to your webcam feed — just like Snapchat, but running locally on your machine!

---

## ✨ Features

| Filter | Key | Category | Description |
|---|---|---|---|
| 🐶 **Dog Ears & Nose** | `1` | Wearables | Classic puppy filter with floppy cartoon ears and a cute nose |
| 😎 **Sunglasses** | `2` | Wearables | Aviator-style shades that follow your face tilt |
| 🔥 **Flame Crown** | `3` | Animated | Procedural flame crown flickering above your head |
| 🌊 **Wobble Face** | `4` | Distortion | Sine-wave jelly ripple effect across the entire face |
| 📦 **Pixelate Zoom** | `5` | Distortion | Dynamic pixelation that reacts to movement (censor blur) |
| ⚔️ **Viking Helmet & Beard** | `6` | Wearables | Ornate warrior helmet with flowing beard using PNG asset |
| 👨‍🚀 **Astronaut Helmet** | `7` | Wearables | Transparent glass dome with reflection sheen using PNG asset |
| 🎭 **Masquerade Mask** | `8` | Wearables | Ornate venetian half-mask covering eyes and nose using PNG asset |
| 💚 **Matrix Rain** | `9` | Procedural | Green cascading katakana characters with glitch effects |
| ⚡ **Lightning Aura** | Shift+1 | Procedural | Electric arcs sparking outward from face outline |
| 🖤 **Ink Splash** | Shift+2 | Procedural | Animated ink blobs spreading from forehead, resets on movement |
| 😊 **Mood Meter** | Shift+3 | Reactive | Real-time emoji + mood label based on facial expression |
| ❄️ **Freeze Ray** | Shift+4 | Reactive | Frost/crystallize effect spreading when mouth opens wide |
| 🌸 **Blush Reactor** | Shift+5 | Reactive | Soft blush circles on cheeks, intensifies with smile |

All filters:
- Track your face in real-time using **468 facial landmarks**
- Scale proportionally to your face size
- Rotate to match your head tilt
- Work at full webcam FPS

---

##  Quick Start

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
| `1`–`9` | Activate filters 1–9 |
| `Shift+1`–`Shift+5` | Activate filters 10–14 (`!` `@` `#` `$` `%`) |
| `0` | Turn off all filters |
| `S` | Save a screenshot to `screenshots/` |
| `Q` | Quit the application |

---

##  Architecture

```
Snappy/
├── main.py                      # Entry point — webcam loop, keyboard handler, HUD
├── utils.py                     # Overlay helpers, landmark extraction, asset loading
├── filters/
│   ├── __init__.py              # Package exports
│   ├── base.py                  # BaseFilter abstract class
│   ├── dog.py                   # Dog ears & nose filter
│   ├── sunglasses.py            # Sunglasses filter
│   ├── flame_crown.py           # Procedural flame crown filter
│   ├── wobble_face.py           # Sine-wave distortion filter
│   ├── pixelate_zoom.py         # Dynamic pixelation filter
│   ├── viking_helmet.py         # Viking helmet & beard (PNG-based)
│   ├── astronaut_helmet.py      # Astronaut helmet dome (PNG-based)
│   ├── masquerade_mask.py       # Venetian mask overlay (PNG-based)
│   ├── matrix_rain.py           # Falling characters with glitch effects
│   ├── lightning_aura.py        # Procedural electric arcs
│   ├── ink_splash.py            # Animated ink blob spread
│   ├── mood_meter.py            # Expression-based emoji display
│   ├── freeze_ray.py            # Mouth-triggered frost effect
│   └── blush_reactor.py         # Smile-reactive blush circles
├── assets/
│   ├── dog_ears.png             # Cartoon dog ears (transparent PNG)
│   ├── dog_nose.png             # Cartoon dog nose (transparent PNG)
│   ├── sunglasses.png           # Aviator sunglasses (transparent PNG)
│   ├── viking_helmet.png        # Viking warrior helmet & beard (transparent PNG)
│   ├── astronaut_helmet.png     # Glass dome helmet (transparent PNG)
│   ├── masquerade_mask.png      # Ornate venetian mask (transparent PNG)
│   └── face_landmarker_v2.task  # MediaPipe Face Mesh model (required)
├── screenshots/                 # Saved screenshots (created automatically)
├── requirements.txt             # Python dependencies
└── README.md                    # This file
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

#### `filters/wobble_face.py` — Wobble Face Filter
- Applies sine-wave horizontal distortion to the face region
- Creates a rippling jelly-like effect
- Updates continuously with animated wave motion
- Procedurally generated without assets

#### `filters/pixelate_zoom.py` — Pixelate Zoom Filter
- Downscales and upscales the face region to create pixelation
- Blocksize = 12 pixels for censor blur effect
- Background remains sharp and unpixelated
- Reacts to movement in real-time

#### `filters/viking_helmet.py` — Viking Helmet & Beard Filter
- Uses PNG asset (`viking_helmet.png`) for realistic appearance
- Scales to 1.35x face width
- Positioned above forehead with rotation awareness
- Includes helmet, horns, and flowing beard in one asset

#### `filters/astronaut_helmet.py` — Astronaut Helmet Filter
- Uses PNG asset (`astronaut_helmet.png`) — glass dome design
- Scales to 1.4x face width
- Covers entire head region
- Rotates with face tilt angle

#### `filters/masquerade_mask.py` — Masquerade Mask Filter
- Uses PNG asset (`masquerade_mask.png`) — ornate venetian design
- Scales to 1.2x face width
- Centered on eyes and nose bridge
- Rotates to follow face angle

#### `filters/matrix_rain.py` — Matrix Rain Filter
- **Enhanced procedural** — green cascading katakana characters
- Features:
  - Per-column speed and brightness variation
  - Exponential fade curve for smooth trails
  - Glow effect on leading character
  - Occasional scanline flash effects
  - Subtle screen glitch artifacts (5% chance per frame)
  - Dynamic trail lengths (6-15 characters per column)

#### `filters/lightning_aura.py` — Lightning Aura Filter
- Generates procedural electric arcs radiating from face outline
- Creates branching lightning paths
- Animates with random variations each frame
- Glow ring around face center

#### `filters/ink_splash.py` — Ink Splash Filter
- Animated ink blob spread from forehead downward
- Resets animation on significant face movement
- Features dripping trails and splatter effects
- 2-second animation cycle with smooth fading

#### `filters/mood_meter.py` — Mood Meter Filter
- Detects facial expression using:
  - Mouth openness (landmarks **13**, **14**)
  - Eyebrow height (landmarks **105**, **334**)
- Displays emoji + mood label above head:
  - 😄 **Happy** — wide mouth, raised brows
  - 😲 **Surprised** — very wide mouth, high eyebrows
  - 🤔 **Curious** — neutral mouth, raised brows
  - 😢 **Sad** — closed mouth, low eyebrows
  - 😐 **Neutral** — default state
- Real-time updates with colored background box

#### `filters/freeze_ray.py` — Freeze Ray Filter
- Detects mouth wide open (aspect ratio > 0.16)
- Activates with smooth ramping (0.1 per frame)
- Draws crystalline ice patterns:
  - Snowflake-shaped crystals radiating outward
  - Blue-tinted overlay glow
  - Spreading frost radius effect
- Deactivates when mouth closes

#### `filters/blush_reactor.py` — Blush Reactor Filter
- Calculates smile intensity using:
  - Mouth width (landmarks **61**, **291**)
  - Mouth height (landmarks **13**, **14**)
- Displays on cheeks (landmarks **226**, **446**):
  - Base radius: 15 pixels × smile intensity multiplier
  - Pink color with variable opacity
  - Subtle highlight effect when smiling (> 30% intensity)

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
| 6 | Nose bridge | Masquerade mask centering |
| 10 | Forehead top center | Dog ears, Flame crown, Viking helmet |
| 13 | Upper lip center | Mood meter, Freeze ray (mouth detection) |
| 14 | Lower lip center | Mood meter, Freeze ray, Blush reactor |
| 33 | Left eye outer corner | Sunglasses, face angle, Masquerade mask |
| 61 | Left mouth corner | Blush reactor (smile width) |
| 105 | Left eyebrow | Mood meter (raise height) |
| 109 | Left forehead | Flame crown positioning |
| 133 | Left eye inner corner | Eye width reference |
| 152 | Chin center | Face height reference |
| 226 | Left cheek | Blush reactor placement |
| 234 | Left ear | Face width calculation |
| 263 | Right eye outer corner | Sunglasses, face angle, Masquerade mask |
| 291 | Right mouth corner | Blush reactor (smile width) |
| 334 | Right eyebrow | Mood meter (raise height) |
| 338 | Right forehead | Flame crown positioning |
| 362 | Right eye inner corner | Eye width reference |
| 446 | Right cheek | Blush reactor placement |
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

##  Troubleshooting

| Issue | Solution |
|---|---|
| **Webcam not opening** | Make sure no other app is using the camera. Try changing `CAMERA_INDEX` in `main.py` |
| **Low FPS** | Reduce camera resolution in `main.py` (set to 640×480) |
| **Filter not aligned** | Ensure good lighting and face the camera directly |
| **Import errors** | Run `pip install -r requirements.txt` again |
| **MediaPipe warning on M1 Mac** | This is normal — MediaPipe still works fine on Apple Silicon |

---

## 📄 License

Copyright (c) 2026 Kaf Abbas. All rights reserved.

This project is proprietary software.

No permission is granted to use, copy, modify, merge, publish, distribute, sublicense, or sell copies of this software, in whole or in part, without prior written permission from the copyright holder.

Viewing the source code on GitHub does not grant any rights to use or redistribute the software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED.