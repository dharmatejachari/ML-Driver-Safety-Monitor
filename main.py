import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import urllib.request
import os
from ultralytics import YOLO
import asyncio
import telegram
from datetime import datetime

# ============================================================
#  TELEGRAM SETUP
# ============================================================
from dotenv import load_dotenv
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID   = int(os.getenv("CHAT_ID"))
import threading


def send_alert(message, frame):
    """Send alert in a separate thread so video never freezes."""
    def _send():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def _do():
                from telegram.request import HTTPXRequest
                request = HTTPXRequest(connect_timeout=20, read_timeout=20)
                fresh_bot = telegram.Bot(token=BOT_TOKEN, request=request)
                async with fresh_bot:
                    await fresh_bot.send_message(chat_id=CHAT_ID, text=message)
                    img_path = "alert_snapshot.jpg"
                    cv2.imwrite(img_path, frame.copy())
                    with open(img_path, "rb") as photo:
                        await fresh_bot.send_photo(chat_id=CHAT_ID, photo=photo)

            loop.run_until_complete(_do())
            loop.close()
            print("Alert sent!")
        except Exception as e:
            print(f"Telegram error: {e}")

    threading.Thread(target=_send, daemon=True).start()

def trigger_alert(message, frame):
    send_alert(message, frame)
# ============================================================
#  MODEL DOWNLOADS
# ============================================================
model_path = "face_landmarker.task"
if not os.path.exists(model_path):
    print("Downloading face landmark model (~30MB)...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        model_path
    )
    print("Done!")

# ============================================================
#  MEDIAPIPE SETUP
# ============================================================
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    output_facial_transformation_matrixes=True
)
landmarker = vision.FaceLandmarker.create_from_options(options)

# ============================================================
#  YOLO SETUP
# ============================================================
yolo = YOLO("yolov8s.pt")
DANGER_OBJECTS  = {67: "phone"}
DANGER_KEYWORDS = ["phone", "cell", "mobile"]

# ============================================================
#  CONSTANTS
# ============================================================
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

EAR_THRESHOLD    = 0.25
CLOSED_SECONDS   = 2.0
YAW_LIMIT        = 25
PITCH_LIMIT      = 15
AWAY_SECONDS     = 2.0
OBJECT_SECONDS   = 2.0

# ============================================================
#  HELPER FUNCTIONS
# ============================================================
def eye_aspect_ratio(landmarks, eye_points, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_points]
    v1  = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    v2  = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    hz  = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (v1 + v2) / (2.0 * hz)

def rotation_matrix_to_angles(matrix):
    r     = np.array(matrix).reshape(4, 4)[:3, :3]
    pitch = np.degrees(np.arcsin(-r[1, 2]))
    yaw   = np.degrees(np.arctan2(r[0, 2], r[2, 2]))
    return yaw, pitch

def draw_status_row(frame, y, label, value, color):
    cv2.putText(frame, f"{label}: {value}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

# ============================================================
#  ALERT STATES
# ============================================================
eyes_closed_since  = None
looking_away_since = None
object_since       = None

# ============================================================
#  ALERT COOLDOWN (prevent spam)
# ============================================================
last_alert_time     = {}
ALERT_COOLDOWN_SECS = 30   # send max one alert per 30 seconds per type

def should_alert(alert_type):
    """Returns True if enough time has passed since last alert of this type."""
    now = time.time()
    if alert_type not in last_alert_time:
        last_alert_time[alert_type] = 0
    if now - last_alert_time[alert_type] > ALERT_COOLDOWN_SECS:
        last_alert_time[alert_type] = now
        return True
    return False

# ============================================================
#  MAIN LOOP
# ============================================================
cap = cv2.VideoCapture(0)
print("Driver Monitor started. Press ESC to stop.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # ---------- YOLO object detection ----------
    yolo_results  = yolo(frame, verbose=False, conf=0.30)
    danger_found  = False
    danger_label  = ""

    for result in yolo_results:
        for box in result.boxes:
            class_id   = int(box.cls[0])
            confidence = float(box.conf[0])
            label      = yolo.names[class_id].lower()
            is_danger  = (class_id in DANGER_OBJECTS or
                          any(kw in label for kw in DANGER_KEYWORDS))
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if is_danger:
                danger_found = True
                danger_label = label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{label} {confidence:.0%}",
                            (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # ---------- face landmarks ----------
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_img)

    # default status values
    drowsy_text  = "No face"
    drowsy_color = (100, 100, 100)
    gaze_text    = "No face"
    gaze_color   = (100, 100, 100)
    obj_text     = "No object"
    obj_color    = (0, 255, 0)

    if result.face_landmarks and result.facial_transformation_matrixes:
        landmarks = result.face_landmarks[0]

        # --- drowsiness ---
        left_ear  = eye_aspect_ratio(landmarks, LEFT_EYE,  w, h)
        right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
        avg_ear   = (left_ear + right_ear) / 2.0

        for idx in LEFT_EYE + RIGHT_EYE:
            lm = landmarks[idx]
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 2, (0, 255, 255), -1)

        if avg_ear < EAR_THRESHOLD:
            if eyes_closed_since is None:
                eyes_closed_since = time.time()
            dur = time.time() - eyes_closed_since
            if dur >= CLOSED_SECONDS:
                drowsy_text  = f"DROWSY ALERT! ({dur:.1f}s)"
                drowsy_color = (0, 0, 255)
                if should_alert("drowsy"):
                    now = datetime.now().strftime("%H:%M:%S")
                    trigger_alert(
                        f"🚨 DROWSY ALERT!\nDriver eyes closed for {dur:.1f} seconds\nTime: {now}",
                        frame
                    )
            else:
                drowsy_text  = f"Eyes closing ({dur:.1f}s)"
                drowsy_color = (0, 165, 255)
        else:
            eyes_closed_since = None
            drowsy_text       = f"Eyes open  EAR:{avg_ear:.2f}"
            drowsy_color      = (0, 255, 0)

        # --- head pose ---
        matrix     = result.facial_transformation_matrixes[0].data
        yaw, pitch = rotation_matrix_to_angles(matrix)

        nose  = landmarks[1]
        nx, ny = int(nose.x * w), int(nose.y * h)
        ax    = int(nx + np.sin(np.radians(yaw))   * 60)
        ay    = int(ny + np.sin(np.radians(pitch))  * 60)
        cv2.arrowedLine(frame, (nx, ny), (ax, ay), (255, 255, 0), 2, tipLength=0.3)

        if yaw > YAW_LIMIT:
            direction = "Looking LEFT"
            is_away   = True
        elif yaw < -YAW_LIMIT:
            direction = "Looking RIGHT"
            is_away   = True
        elif pitch > PITCH_LIMIT:
            direction = "Looking DOWN"
            is_away   = True
        else:
            direction = "Forward"
            is_away   = False

        if is_away:
            if looking_away_since is None:
                looking_away_since = time.time()
            dur = time.time() - looking_away_since
            if dur >= AWAY_SECONDS:
                gaze_text  = f"DISTRACTION! {direction} ({dur:.1f}s)"
                gaze_color = (0, 0, 255)
                if should_alert("gaze"):
                    now = datetime.now().strftime("%H:%M:%S")
                    trigger_alert(
                        f"👀 DISTRACTION ALERT!\nDriver {direction} for {dur:.1f} seconds\nTime: {now}",
                        frame
                    )
            else:
                gaze_text  = f"{direction} ({dur:.1f}s)"
                gaze_color = (0, 165, 255)
        else:
            looking_away_since = None
            gaze_text          = f"Gaze: {direction}"
            gaze_color         = (0, 255, 0)

    # --- object alert ---
    if danger_found:
        if object_since is None:
            object_since = time.time()
        dur      = time.time() - object_since
        if dur >= OBJECT_SECONDS:
                obj_text  = f"PHONE ALERT! ({dur:.1f}s)"
                obj_color = (0, 0, 255)
                if should_alert("phone"):
                    now = datetime.now().strftime("%H:%M:%S")
                    trigger_alert(
                        f"📱 PHONE ALERT!\nDriver using phone while driving!\nTime: {now}",
                        frame
                    )
        else:
            obj_text  = f"Phone detected ({dur:.1f}s)"
            obj_color = (0, 165, 255)
    else:
        object_since = None
        obj_text     = "No phone"
        obj_color    = (0, 255, 0)

    # ---------- draw status panel ----------
    cv2.rectangle(frame, (0, 0), (w, 110), (0, 0, 0), -1)
    draw_status_row(frame,  30, "Drowsiness", drowsy_text,  drowsy_color)
    draw_status_row(frame,  60, "Gaze      ", gaze_text,    gaze_color)
    draw_status_row(frame,  90, "Object    ", obj_text,     obj_color)

    cv2.imshow("Driver Safety Monitor", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("Stopped.")