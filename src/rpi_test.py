import time
import queue
import threading
import cv2
import pyttsx3
from picamera2 import Picamera2
from ultralytics import YOLO

# =========================================================
# CONFIGURATION
# =========================================================
# Use yolov8n.pt or yolov8n_ncnn_model for better Pi FPS
MODEL_PATH = "yolov8n.pt" 
CAMERA_RESOLUTION = (640, 480)

CONFIDENCE_THRESHOLD = 0.45
FOCAL_LENGTH_PX = 550  # Tuned for standard Pi Camera Module at 640x480

VERY_CLOSE_METERS = 0.8
OBSTACLE_CHECK_DISTANCE = 2.5
REDETECTION_DELAY = 2.0  # Seconds before repeating the same object callout

OBJECT_WIDTHS = {
    "person": 0.45, "bicycle": 0.60, "car": 1.80, "motorcycle": 0.80,
    "bus": 2.50, "truck": 2.50, "traffic light": 0.30, "bench": 1.50,
    "dog": 0.45, "cat": 0.25, "backpack": 0.35, "chair": 0.55,
    "couch": 1.80, "bed": 1.50, "dining table": 1.20, "tv": 0.90,
    "laptop": 0.33, "cell phone": 0.08, "bottle": 0.08, "cup": 0.10
}
DEFAULT_OBJECT_WIDTH = 0.45

# =========================================================
# AUDIO / SPEECH WORKER (ESPEAK-NG FOR PI)
# =========================================================
speech_queue = queue.Queue(maxsize=1)
app_running = True

def speak(message):
    """Drop stale messages and push newest warning."""
    if not message or not app_running:
        return
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
    """Dedicated voice thread using espeak on Linux."""
    try:
        engine = pyttsx3.init("espeak")
        engine.setProperty("rate", 160)
        engine.setProperty("volume", 1.0)
    except Exception as e:
        print(f"[VOICE] Error initializing espeak: {e}")
        return

    print("[VOICE] Audio thread active.")
    engine.say("System ready.")
    engine.runAndWait()

    while app_running:
        try:
            message = speech_queue.get(timeout=0.2)
            if message is None:
                break
            print(f"[SPEAKING]: {message}")
            engine.say(message)
            engine.runAndWait()
            speech_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[VOICE] Error during playback: {e}")

# =========================================================
# DISTANCE & DIRECTION LOGIC
# =========================================================
def estimate_distance(name, box_width):
    if box_width <= 2:
        return None
    real_width = OBJECT_WIDTHS.get(name, DEFAULT_OBJECT_WIDTH)
    distance = (real_width * FOCAL_LENGTH_PX) / box_width
    return round(distance, 1)

def get_side(center_x, frame_width):
    ratio = center_x / frame_width
    if ratio < 0.35:
        return "LEFT"
    if ratio > 0.65:
        return "RIGHT"
    return "CENTER"

def side_is_occupied(side, detections):
    for obj in detections:
        if obj["side"] == side:
            dist = obj["distance"]
            if dist is None or dist <= OBSTACLE_CHECK_DISTANCE:
                return True
    return False

def get_guidance(obj, detections):
    name = obj["name"]
    side = obj["side"]
    dist = obj["distance"]

    dist_str = f"at {dist} meters" if dist else "ahead"

    if dist is not None and dist <= VERY_CLOSE_METERS:
        return f"Warning: {name} very close on your {side.lower()}. Stop."

    left_blocked = side_is_occupied("LEFT", detections)
    right_blocked = side_is_occupied("RIGHT", detections)

    if side == "LEFT":
        if not right_blocked:
            return f"{name} on your left {dist_str}. Step right."
        return f"{name} on left. Right side blocked."

    elif side == "RIGHT":
        if not left_blocked:
            return f"{name} on your right {dist_str}. Step left."
        return f"{name} on right. Left side blocked."

    else:  # CENTER
        if not left_blocked:
            return f"{name} directly ahead {dist_str}. Move left."
        elif not right_blocked:
            return f"{name} directly ahead {dist_str}. Move right."
        return f"{name} directly ahead. Both sides blocked."

# =========================================================
# MAIN EXECUTION
# =========================================================
def main():
    global app_running

    # 1. Start voice thread
    voice_thread = threading.Thread(target=speech_worker, daemon=True)
    voice_thread.start()

    # 2. Load YOLO model
    print("[YOLO] Loading model...")
    model = YOLO(MODEL_PATH)
    print("[YOLO] Model loaded.")

    # 3. Setup Picamera2
    print("[CAMERA] Initializing Picamera2...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": CAMERA_RESOLUTION, "format": "XBGR8888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1.5)  # Sensor warmup

    print("[SYSTEM] Running. Press 'q' in the window to quit.")
    last_spoken_times = {}

    try:
        while True:
            # Grab BGR frame
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            frame_h, frame_w = frame.shape[:2]
            now = time.time()

            # YOLO inference
            results = model.predict(
                source=frame,
                conf=CONFIDENCE_THRESHOLD,
                imgsz=320,  # 320x320 keeps Pi inference much faster
                verbose=False
            )

            detections = []
            res = results[0]

            if res.boxes is not None:
                for box in res.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    name = str(model.names[int(box.cls[0])])
                    conf = float(box.conf[0])
                    box_w = max(x2 - x1, 1)
                    center_x = (x1 + x2) / 2

                    side = get_side(center_x, frame_w)
                    dist = estimate_distance(name, box_w)

                    detections.append({
                        "name": name,
                        "confidence": conf,
                        "side": side,
                        "distance": dist,
                        "box": (x1, y1, x2, y2)
                    })

            # Check guidance and prioritize speech
            visible_keys = set()
            for obj in detections:
                obj["message"] = get_guidance(obj, detections)
                key = (obj["name"], obj["side"])
                visible_keys.add(key)

                last_time = last_spoken_times.get(key)
                if last_time is None or (now - last_time > REDETECTION_DELAY):
                    speak(obj["message"])
                    last_spoken_times[key] = now

            # Cleanup expired speech locks
            for key in list(last_spoken_times.keys()):
                if key not in visible_keys and (now - last_spoken_times[key] > REDETECTION_DELAY):
                    del last_spoken_times[key]

            # Render bounding boxes on frame
            for obj in detections:
                x1, y1, x2, y2 = obj["box"]
                d_label = f"{obj['distance']}m" if obj['distance'] else "?"
                label = f"{obj['name']} {d_label} [{obj['side']}]"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, max(y1 - 8, 18)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("Raspberry Pi - Obstacle Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        app_running = False
        speech_queue.put_nowait(None)
        picam2.stop()
        cv2.destroyAllWindows()
        print("Camera released and system shut down cleanly.")

if __name__ == "__main__":
    main()