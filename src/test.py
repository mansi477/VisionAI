
import cv2
import time
import queue
import threading
import tkinter as tk
from tkinter import ttk
from ultralytics import YOLO
from PIL import Image, ImageTk
import pyttsx3


# =========================================================
# CONFIGURATION
# =========================================================

MODEL_PATH = "yolov8n.pt"
CAMERA_INDEX = 0

CONFIDENCE_THRESHOLD = 0.40
IMAGE_SIZE = 640
FOCAL_LENGTH_PX = 700

VERY_CLOSE_METERS = 0.8
OBSTACLE_CHECK_DISTANCE = 2.5
REDETECTION_DELAY = 1.5

# Approximate real-world object widths in metres.
OBJECT_WIDTHS = {
    "person": 0.45,
    "bicycle": 0.60,
    "car": 1.80,
    "motorcycle": 0.80,
    "airplane": 8.00,
    "bus": 2.50,
    "train": 3.00,
    "truck": 2.50,
    "boat": 2.00,
    "traffic light": 0.30,
    "fire hydrant": 0.25,
    "stop sign": 0.75,
    "parking meter": 0.25,
    "bench": 1.50,
    "bird": 0.20,
    "cat": 0.25,
    "dog": 0.45,
    "horse": 0.70,
    "sheep": 0.60,
    "cow": 0.90,
    "elephant": 2.00,
    "bear": 0.90,
    "zebra": 0.80,
    "giraffe": 1.00,
    "backpack": 0.35,
    "umbrella": 0.90,
    "handbag": 0.30,
    "suitcase": 0.45,
    "frisbee": 0.25,
    "skis": 0.15,
    "snowboard": 0.30,
    "sports ball": 0.22,
    "kite": 0.80,
    "baseball bat": 0.08,
    "baseball glove": 0.25,
    "skateboard": 0.25,
    "surfboard": 0.50,
    "tennis racket": 0.30,
    "bottle": 0.08,
    "wine glass": 0.08,
    "cup": 0.10,
    "fork": 0.03,
    "knife": 0.03,
    "spoon": 0.04,
    "bowl": 0.18,
    "banana": 0.10,
    "apple": 0.08,
    "sandwich": 0.15,
    "orange": 0.08,
    "broccoli": 0.15,
    "carrot": 0.05,
    "hot dog": 0.12,
    "pizza": 0.30,
    "donut": 0.10,
    "cake": 0.20,
    "chair": 0.55,
    "couch": 1.80,
    "potted plant": 0.40,
    "bed": 1.50,
    "dining table": 1.20,
    "toilet": 0.40,
    "tv": 0.90,
    "laptop": 0.33,
    "mouse": 0.07,
    "remote": 0.05,
    "keyboard": 0.45,
    "cell phone": 0.08,
    "microwave": 0.50,
    "oven": 0.60,
    "toaster": 0.30,
    "sink": 0.60,
    "refrigerator": 0.80,
    "book": 0.16,
    "clock": 0.25,
    "vase": 0.20,
    "scissors": 0.08,
    "teddy bear": 0.25,
    "hair drier": 0.10,
    "toothbrush": 0.02,
}

DEFAULT_OBJECT_WIDTH = 0.45


# =========================================================
# LOAD YOLO MODEL
# =========================================================

print("Loading YOLOv8 model...")

try:
    model = YOLO(MODEL_PATH)
    print("YOLOv8 model loaded successfully.")
except Exception as error:
    raise RuntimeError(
        f"Could not load YOLO model: {error}"
    )


# =========================================================
# SHARED STATE
# =========================================================

state_lock = threading.Lock()

latest_frame = None
latest_status = "Camera stopped"
latest_details = "Press Start Camera to begin."

camera_running = False
camera_thread = None
camera = None

# One pending message at a time. New pending messages replace
# old ones so outdated guidance does not accumulate.
speech_queue = queue.Queue(maxsize=1)

speech_enabled = True
app_closing = False


# =========================================================
# RELIABLE WINDOWS VOICE OUTPUT
# =========================================================

def speak(message):
    """Queue a message without blocking the camera thread."""

    if not speech_enabled or not message or app_closing:
        return

    # Remove any message that has not started speaking yet.
    try:
        while True:
            speech_queue.get_nowait()
            speech_queue.task_done()
    except queue.Empty:
        pass

    try:
        speech_queue.put_nowait(message)
    except queue.Full:
        pass


def speech_worker():
    """
    Keep the speech engine on one dedicated thread.
    Windows SAPI5 is used for speech synthesis.
    """

    engine = None
    pythoncom_module = None

    try:
        try:
            import pythoncom
            pythoncom_module = pythoncom
            pythoncom_module.CoInitialize()
        except ImportError:
            print(
                "pywin32 COM support is unavailable; "
                "trying the default speech engine."
            )

        try:
            engine = pyttsx3.init("sapi5")
        except Exception:
            engine = pyttsx3.init()

        engine.setProperty("rate", 165)
        engine.setProperty("volume", 1.0)

        # Select an English voice if one is installed.
        try:
            for voice in engine.getProperty("voices"):
                voice_name = getattr(voice, "name", "").lower()
                voice_id = getattr(voice, "id", "").lower()

                if "english" in voice_name or "english" in voice_id:
                    engine.setProperty("voice", voice.id)
                    break
        except Exception as error:
            print("Could not select voice:", error)

        print("Voice engine ready.")

        engine.say("Voice guidance started.")
        engine.runAndWait()

        while True:
            message = speech_queue.get()

            try:
                if message is None:
                    break

                if speech_enabled and message.strip():
                    print("SPEAKING:", message)
                    engine.say(message)
                    engine.runAndWait()

            except Exception as error:
                print("Speech playback error:", repr(error))

            finally:
                speech_queue.task_done()

    except Exception as error:
        print("VOICE ENGINE ERROR:", repr(error))
        print(
            "Check Windows audio output, install pyttsx3 "
            "and pywin32, and verify your system speech settings."
        )

    finally:
        if engine is not None:
            try:
                engine.stop()
            except Exception:
                pass

        if pythoncom_module is not None:
            try:
                pythoncom_module.CoUninitialize()
            except Exception:
                pass


# =========================================================
# DISTANCE ESTIMATION
# =========================================================

def estimate_distance(name, box_width):
    if box_width <= 1:
        return None

    real_width = OBJECT_WIDTHS.get(
        name,
        DEFAULT_OBJECT_WIDTH
    )

    distance = (
        real_width * FOCAL_LENGTH_PX
    ) / box_width

    if distance <= 0:
        return None

    return distance


def distance_text(distance):
    if distance is None:
        return "distance uncertain"

    return f"approximately {distance:.1f} metres away"


# =========================================================
# LEFT / CENTER / RIGHT
# =========================================================

def get_side(center_x, frame_width):
    position = center_x / max(frame_width, 1)

    if position < 0.35:
        return "LEFT"

    if position > 0.65:
        return "RIGHT"

    return "CENTER"


# =========================================================
# SIDE OCCUPANCY CHECK
# =========================================================

def side_is_occupied(side, detections):
    for obj in detections:
        if obj["side"] != side:
            continue

        distance = obj["distance"]

        # Be conservative if distance cannot be estimated.
        if distance is None:
            return True

        if distance <= OBSTACLE_CHECK_DISTANCE:
            return True

    return False


# =========================================================
# SPOKEN GUIDANCE
# =========================================================

def get_guidance(obj, detections):
    name = obj["name"]
    side = obj["side"]
    distance = obj["distance"]

    description = (
        f"{name} detected, {distance_text(distance)}"
    )

    # Warn about a very close object.
    if (
        distance is not None
        and distance <= VERY_CLOSE_METERS
    ):
        return (
            f"{description}, on your {side.lower()}. "
            "Object very close. Stop and check your surroundings."
        )

    if side == "LEFT":
        if not side_is_occupied("RIGHT", detections):
            return (
                f"{description}, on your left. "
                "Move slightly to the right if the path is clear."
            )

        return (
            f"{description}, on your left. "
            "The right side may also have an obstacle. "
            "Check your surroundings before moving."
        )

    if side == "RIGHT":
        if not side_is_occupied("LEFT", detections):
            return (
                f"{description}, on your right. "
                "Move slightly to the left if the path is clear."
            )

        return (
            f"{description}, on your right. "
            "The left side may also have an obstacle. "
            "Check your surroundings before moving."
        )

    left_blocked = side_is_occupied("LEFT", detections)
    right_blocked = side_is_occupied("RIGHT", detections)

    if not left_blocked and not right_blocked:
        return (
            f"{description}, ahead. "
            "The left and right sides appear clear. "
            "Move slightly to the left if safe."
        )

    if not left_blocked:
        return (
            f"{description}, ahead. "
            "Move slightly to the left if the path is clear."
        )

    if not right_blocked:
        return (
            f"{description}, ahead. "
            "Move slightly to the right if the path is clear."
        )

    return (
        f"{description}, ahead. "
        "Both sides may have obstacles. "
        "Check your surroundings before choosing a direction."
    )


# =========================================================
# CAMERA PROCESSING THREAD
# =========================================================

def camera_worker():
    global camera
    global camera_running
    global latest_frame
    global latest_status
    global latest_details

    local_camera = cv2.VideoCapture(CAMERA_INDEX)

    if not local_camera.isOpened():
        local_camera.release()

        with state_lock:
            camera_running = False
            latest_status = "Camera error"
            latest_details = (
                "Could not open the camera. "
                "Check CAMERA_INDEX and camera permissions."
            )

        return

    local_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    local_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    with state_lock:
        camera = local_camera

    # Tracks currently visible objects by name and side.
    local_last_seen = {}

    try:
        while True:
            with state_lock:
                if not camera_running or app_closing:
                    break

            success, frame = local_camera.read()

            if not success:
                with state_lock:
                    latest_status = "Camera read error"
                    latest_details = "Check the camera connection."

                time.sleep(0.1)
                continue

            frame_height, frame_width = frame.shape[:2]
            now = time.time()

            results = model.predict(
                source=frame,
                conf=CONFIDENCE_THRESHOLD,
                imgsz=IMAGE_SIZE,
                verbose=False
            )

            detections = []
            result = results[0]

            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0].tolist()
                    )

                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    name = str(model.names[class_id])

                    box_width = max(x2 - x1, 1)
                    center_x = (x1 + x2) / 2

                    side = get_side(center_x, frame_width)

                    distance = estimate_distance(
                        name,
                        box_width
                    )

                    detections.append({
                        "name": name,
                        "confidence": confidence,
                        "side": side,
                        "distance": distance,
                        "box": (x1, y1, x2, y2)
                    })

            # Generate messages after all detections are known.
            for obj in detections:
                obj["message"] = get_guidance(
                    obj,
                    detections
                )

            visible_keys = set()

            # Announce when an object first appears, then again
            # after it has disappeared for the redetection delay.
            for obj in detections:
                key = (obj["name"], obj["side"])
                visible_keys.add(key)

                previous_time = local_last_seen.get(key)

                if (
                    previous_time is None
                    or now - previous_time > REDETECTION_DELAY
                ):
                    speak(obj["message"])

                local_last_seen[key] = now

            # Remove keys for objects absent long enough.
            for key in list(local_last_seen.keys()):
                if key not in visible_keys:
                    if (
                        now - local_last_seen[key]
                        > REDETECTION_DELAY
                    ):
                        del local_last_seen[key]

            # Draw boxes and object labels only.
            # No vertical or horizontal guide lines.
            for obj in detections:
                x1, y1, x2, y2 = obj["box"]

                if obj["distance"] is None:
                    dist_label = "Distance uncertain"
                else:
                    dist_label = f"{obj['distance']:.1f} m"

                label = (
                    f"{obj['name']} "
                    f"{obj['confidence'] * 100:.0f}% "
                    f"| {dist_label} | {obj['side']}"
                )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 220, 0),
                    2
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 10, 22)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 220, 0),
                    2,
                    cv2.LINE_AA
                )

            if detections:
                known_distances = [
                    obj for obj in detections
                    if obj["distance"] is not None
                ]

                if known_distances:
                    nearest = min(
                        known_distances,
                        key=lambda item: item["distance"]
                    )
                else:
                    nearest = detections[0]

                status = f"{len(detections)} object(s) detected"

                details = (
                    f"Object: {nearest['name'].title()}\n"
                    f"Confidence: "
                    f"{nearest['confidence'] * 100:.1f}%\n"
                    f"Position: {nearest['side']}\n"
                    f"Distance: {distance_text(nearest['distance'])}\n"
                    f"Guidance: {nearest['message']}"
                )
            else:
                status = "Searching for objects..."
                details = "No objects detected."

            with state_lock:
                latest_frame = frame.copy()
                latest_status = status
                latest_details = details

    except Exception as error:
        print("Camera processing error:", repr(error))

        with state_lock:
            latest_status = "Detection error"
            latest_details = str(error)

    finally:
        local_camera.release()

        with state_lock:
            camera = None
            camera_running = False

            if latest_status not in (
                "Camera error",
                "Detection error"
            ):
                latest_status = "Camera stopped"


# =========================================================
# TKINTER UI
# =========================================================

root = tk.Tk()
root.title("Smart Object Detection and Voice Guidance")
root.geometry("1100x780")
root.minsize(850, 650)
root.configure(bg="#101827")

style = ttk.Style()
style.theme_use("clam")

style.configure(
    "TButton",
    font=("Segoe UI", 11, "bold"),
    padding=(18, 9)
)

title_label = tk.Label(
    root,
    text="SMART OBJECT DETECTION",
    font=("Segoe UI", 22, "bold"),
    fg="#60a5fa",
    bg="#101827"
)
title_label.pack(pady=(12, 4))

subtitle_label = tk.Label(
    root,
    text="Real-time detection • Distance estimate • Voice guidance",
    font=("Segoe UI", 10),
    fg="#cbd5e1",
    bg="#101827"
)
subtitle_label.pack(pady=(0, 10))


# =========================================================
# BUTTONS ABOVE CAMERA
# =========================================================

button_frame = tk.Frame(root, bg="#101827")
button_frame.pack(pady=(0, 10))


# =========================================================
# CAMERA DISPLAY
# =========================================================

video_label = tk.Label(
    root,
    bg="#020617",
    text="Camera is stopped",
    fg="white",
    font=("Segoe UI", 14)
)

video_label.pack(
    padx=15,
    pady=5,
    fill="both",
    expand=True
)


# =========================================================
# STATUS AND DETAILS
# =========================================================

status_label = tk.Label(
    root,
    text="Camera stopped",
    font=("Segoe UI", 12, "bold"),
    fg="#34d399",
    bg="#101827"
)
status_label.pack(pady=(6, 3))

details_label = tk.Label(
    root,
    text="Press Start Camera to begin.",
    font=("Segoe UI", 10),
    fg="#e2e8f0",
    bg="#101827",
    justify="left",
    anchor="w",
    wraplength=1000
)

details_label.pack(
    fill="x",
    padx=20,
    pady=(4, 8)
)


# =========================================================
# BUTTON FUNCTIONS
# =========================================================

def start_camera():
    global camera_running
    global camera_thread
    global latest_status
    global latest_details
    global latest_frame

    with state_lock:
        if camera_running or app_closing:
            return

        camera_running = True
        latest_status = "Starting camera..."
        latest_details = "Initializing camera and detection..."
        latest_frame = None

    video_label.config(
        image="",
        text="Starting camera...",
        fg="white"
    )
    video_label.image = None

    status_label.config(
        text="Starting camera...",
        fg="#fbbf24"
    )

    details_label.config(
        text="Initializing camera and object detection..."
    )

    camera_thread = threading.Thread(
        target=camera_worker,
        daemon=True
    )
    camera_thread.start()


def stop_camera():
    global camera_running

    with state_lock:
        if not camera_running:
            return

        camera_running = False
        latest_status = "Stopping camera..."
        latest_details = "Camera is stopping."

    status_label.config(
        text="Stopping camera...",
        fg="#fbbf24"
    )


def close_app():
    global camera_running
    global app_closing

    with state_lock:
        app_closing = True
        camera_running = False

    # Discard any waiting message before sending shutdown signal.
    try:
        while True:
            speech_queue.get_nowait()
            speech_queue.task_done()
    except queue.Empty:
        pass

    try:
        speech_queue.put_nowait(None)
    except queue.Full:
        pass

    root.destroy()


start_button = ttk.Button(
    button_frame,
    text="Start Camera",
    command=start_camera
)
start_button.grid(row=0, column=0, padx=8)

stop_button = ttk.Button(
    button_frame,
    text="Stop Camera",
    command=stop_camera
)
stop_button.grid(row=0, column=1, padx=8)

exit_button = ttk.Button(
    button_frame,
    text="Exit",
    command=close_app
)
exit_button.grid(row=0, column=2, padx=8)


# =========================================================
# GUI REFRESH
# =========================================================

def update_display():
    if app_closing:
        return

    with state_lock:
        frame_copy = (
            latest_frame.copy()
            if latest_frame is not None
            else None
        )
        status_copy = latest_status
        details_copy = latest_details

    status_label.config(text=status_copy)
    details_label.config(text=details_copy)

    if frame_copy is not None:
        rgb_frame = cv2.cvtColor(
            frame_copy,
            cv2.COLOR_BGR2RGB
        )

        display_image = Image.fromarray(rgb_frame)
        display_image.thumbnail((1050, 500))

        photo = ImageTk.PhotoImage(display_image)

        video_label.config(
            image=photo,
            text=""
        )
        video_label.image = photo

    root.after(100, update_display)


# =========================================================
# START VOICE THREAD AND APPLICATION
# =========================================================

threading.Thread(
    target=speech_worker,
    daemon=True
).start()

root.protocol("WM_DELETE_WINDOW", close_app)
root.after(100, update_display)
root.mainloop()
