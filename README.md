# VisionAI: Giving Sight Through Sound

**An AI-powered, real-time assistive navigation and safety system for the visually impaired, designed for Indian roads.**

---

## 1. Overview

Visually impaired people in India face hazards that most commercial assistive devices were never built for: potholes, open drains, construction work, stray animals, uneven footpaths and dense mixed traffic. White canes only sense what is within reach, and existing smart glasses are expensive, need constant internet, and are tuned for Western streets.

VisionAI is a low-cost, two-part wearable system:

| Part | Role |
|------|------|
| **Smart glasses** (camera) | Look ahead at eye level. YOLOv8 + OpenCV detect obstacles, stray animals, vehicles, people and construction zones, estimate how far away they are, and speak guidance ("Dog on your left, 2 metres. Move slightly right."). |
| **Smart stick** (ultrasonic + vibration + buzzer) | Covers what the glasses **cannot see**: the ground directly ahead. It measures ground-level distance to detect potholes, kerbs, steps and drops, and warns through vibration and buzzer. |

**Safety layer (on the stick):**

- **SOS button**: one press sends an alert with live location to saved emergency contacts.
- **Emergency contact**: a pre-configured trusted person who is notified.
- **Medical emergency alert**: sends a direct message to a close person or friend of the user, separate from the general SOS.
- Alerts and location sharing are sent through **Twilio** over an internet/GSM module.
- Includes a **women's safety** mode built on the same SOS and location-sharing path.

Feedback to the user is **multi-modal**: audio (earphones/speaker), vibration, and buzzer, plus voice commands.

---

## 2. System architecture

```
 Smart glasses (camera)          Smart stick (ESP32)
        |                                |
   live video                  ultrasonic ground-distance
        |                       vibration / buzzer / SOS button
        v                                |
 Processing unit  <-----------------------+
 (Laptop / Raspberry Pi / ESP32)
   - OpenCV pre-processing
   - YOLOv8 object detection
   - Distance estimation + decision logic
        |
        +--> Audio guidance (text-to-speech) + vibration alerts
        +--> Voice commands (speech recognition)
        +--> SOS / medical alert --> Twilio --> emergency contact (+ location)
```

See `docs/visionai.pdf` (slides 9, 10 and 13) for the full diagrams.

---

## 3. Project structure

```
VisionAI/
├── README.md
├── requirements-windows.txt
├── requirements-raspberry-pi.txt
├── docs/
│   └── visionai.pdf          Project presentation (14 slides)
├── src/
│   ├── test.py               Windows desktop prototype (Tkinter GUI)
│   ├── rpi_test.py           Raspberry Pi version (Picamera2)
│   ├── v.py                  Piper offline-voice test (Windows)
│   └── yolov8n.pt            YOLOv8-nano weights (COCO-pretrained)
└── training/                 Train the Indian-road hazard model (see training/README.md)
    ├── train_colab.ipynb     One-notebook pipeline for a free Colab GPU
    ├── class_map.yaml        Master classes + dataset name mapping
    ├── merge_datasets.py / voc_to_yolo.py / preview_labels.py / train.py
    └── requirements-training.txt
```

Run every script from inside `src/` so `yolov8n.pt` is found.

### What each script does

**`src/test.py`: Windows prototype.** Tkinter app with Start / Stop / Exit buttons and a live annotated camera view. Runs YOLOv8n at 640 px on the webcam, estimates distance from bounding-box width (pinhole model), classifies each object as LEFT / CENTER / RIGHT, and speaks guidance through Windows SAPI5 (`pyttsx3`). Only the newest message is kept in the speech queue so guidance never lags behind.

**`src/rpi_test.py`: Raspberry Pi version.** Same detection and guidance logic, tuned for the Pi: Picamera2 at 640x480, inference at 320 px, `espeak` voice, shorter spoken messages. For better FPS, export the model to NCNN (`yolov8n_ncnn_model`) as noted in the script.

**`src/v.py`: Piper voice test.** Checks the offline neural voice (`en_US-lessac-medium`) using sample guidance sentences. Needs `en_US-lessac-medium.onnx` and its `.onnx.json` file placed in `src/` (not included; download from the Piper voices repository).

---

## 4. Setup and run

### Windows (laptop prototype)

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements-windows.txt
cd src
python test.py
```

### Raspberry Pi

```bash
sudo apt update
sudo apt install -y python3-picamera2 espeak-ng python3-opencv
python3 -m venv --system-site-packages venv
source venv/bin/activate
pip install -r requirements-raspberry-pi.txt
cd src
python rpi_test.py        # press 'q' in the window to quit
```

### Tuning

| Setting | Where | Meaning |
|---------|-------|---------|
| `FOCAL_LENGTH_PX` | both scripts | Camera calibration for distance. **Recalibrate for your camera** (700 on laptop, 550 on Pi). |
| `CONFIDENCE_THRESHOLD` | both | Minimum detection confidence (0.40 / 0.45). |
| `VERY_CLOSE_METERS` | both | Distance (0.8 m) that triggers the "Stop" warning. |
| `OBSTACLE_CHECK_DISTANCE` | both | Range (2.5 m) used to decide whether a side is blocked. |
| `REDETECTION_DELAY` | both | Seconds before the same object is announced again. |
| `OBJECT_WIDTHS` | both | Real-world widths (m) used for distance estimation. |

---

## 5. Training the hazard model

The stock weights cannot see potholes, barricades or open drains. `training/` contains a
Colab notebook and scripts that merge public Indian road datasets (DriveIndia, RDD2022 India,
Roboflow sets) plus your own images, fine-tune YOLOv8n, and export `best.pt` / NCNN.
Start with `training/README.md`.

## 6. Implementation status

| Feature | Status |
|---------|--------|
| Real-time object detection (YOLOv8n + OpenCV) | Done (`test.py`, `rpi_test.py`) |
| Camera-based distance estimation | Done (approximate; needs calibration) |
| Left / centre / right guidance with avoidance advice | Done |
| Spoken audio feedback (SAPI5 / espeak) | Done |
| Raspberry Pi deployment | Done (`rpi_test.py`) |
| Offline neural voice (Piper) | Tested standalone (`v.py`), not yet integrated |
| **Pothole, construction-zone, open-drain detection** | **Training kit ready, model not yet trained.** The included `yolov8n.pt` is stock COCO and lacks these classes. Follow `training/README.md` to fine-tune on Indian-roads datasets. |
| **Stray animal detection** | **Partial.** COCO covers dog, cat, cow, horse, sheep and bird only. Fine-tune to cover other cases. |
| Smart stick: ground-level ultrasonic distance, vibration, buzzer | To do (firmware not in this folder) |
| SOS button, emergency contact, medical-emergency message | To do |
| Twilio alerts with location sharing | To do |
| Voice commands (speech recognition) | To do |

---

## 7. Plan of execution (from the presentation)

| Phase | Weeks | Work |
|-------|-------|------|
| 1. Data acquisition | 1-2 | Capture real-time video from the smart glasses |
| 2. Data processing | 3 | Pre-process frames with OpenCV |
| 3. Object detection | 4-5 | Detect objects and obstacles with YOLOv8 |
| 4. Decision and navigation | 6 | Estimate distance, generate navigation instructions |
| 5. Feedback system | 7 | Audio and vibration alerts; integrate smart-stick sensing |
| 6. Safety and deployment | 8 | Trigger SOS alerts via Twilio; deploy |

---

## 8. Requirements summary

**Hardware:** smart glasses with camera; processing unit (laptop / Raspberry Pi / ESP32); smart stick with ultrasonic sensor, vibration motor, buzzer, microphone and speaker; earphones; internet module / GSM for alerts and location sharing.

**Software:** Python, YOLOv8 (Ultralytics), OpenCV, speech-recognition library, text-to-speech engine, Twilio.

**Non-functional goals:** low latency, high detection accuracy, simple interface, low cost and portability, reliability outdoors.
