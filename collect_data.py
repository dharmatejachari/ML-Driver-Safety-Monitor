import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import os
import urllib.request
import csv
from datetime import datetime
from ultralytics import YOLO

# ============================================================
# MODEL SETUP
# ============================================================

MODEL_PATH = "face_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading MediaPipe face model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        MODEL_PATH
    )

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    output_facial_transformation_matrixes=True
)

landmarker = vision.FaceLandmarker.create_from_options(options)

print("Loading YOLO...")
yolo = YOLO("yolov8s.pt")

# ============================================================
# SETTINGS
# ============================================================

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

EAR_THRESHOLD = 0.25

SAMPLE_EVERY = 5

CSV_PATH = "data/driver_behavior.csv"

os.makedirs("data", exist_ok=True)

# ============================================================
# FUNCTIONS
# ============================================================

def eye_aspect_ratio(landmarks, eye_points, w, h):

    pts = [
        (landmarks[i].x * w, landmarks[i].y * h)
        for i in eye_points
    ]

    v1 = np.linalg.norm(
        np.array(pts[1]) - np.array(pts[5])
    )

    v2 = np.linalg.norm(
        np.array(pts[2]) - np.array(pts[4])
    )

    hz = np.linalg.norm(
        np.array(pts[0]) - np.array(pts[3])
    )

    return (v1 + v2) / (2.0 * hz)


def rotation_matrix_to_angles(matrix):

    r = np.array(matrix).reshape(4, 4)[:3, :3]

    pitch = np.degrees(
        np.arcsin(-r[1, 2])
    )

    yaw = np.degrees(
        np.arctan2(r[0, 2], r[2, 2])
    )

    roll = np.degrees(
        np.arctan2(r[1, 0], r[1, 1])
    )

    return yaw, pitch, roll


# ============================================================
# CSV SETUP
# ============================================================

columns = [
    "timestamp",
    "EAR",
    "eye_closure_duration",
    "yaw",
    "pitch",
    "roll",
    "abs_yaw",
    "abs_pitch",
    "head_movement",
    "phone_detected",
    "phone_confidence",
    "driver_state"
]

file_exists = os.path.exists(CSV_PATH)

# IMPORTANT:
# Existing CSV has old columns, so create a new dataset.
if file_exists:
    old_file = CSV_PATH
    backup_file = "data/driver_behavior_old.csv"

    if not os.path.exists(backup_file):
        os.rename(old_file, backup_file)
        print("Old dataset backed up to:", backup_file)

csv_file = open(CSV_PATH, "a", newline="")

writer = csv.DictWriter(
    csv_file,
    fieldnames=columns
)

writer.writeheader()

# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Camera could not be opened.")
    csv_file.close()
    exit()

print()
print("==============================================")
print(" DRIVER BEHAVIOR DATA COLLECTION")
print("==============================================")
print("1 = ALERT")
print("2 = DROWSY")
print("3 = DISTRACTED")
print("4 = PHONE USAGE")
print("Q = QUIT")
print()
print("Collect approximately 100 samples per class.")
print("==============================================")

current_state = "Alert"

eyes_closed_since = None

previous_yaw = None
previous_pitch = None

frame_count = 0
sample_count = 0

# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    h, w = frame.shape[:2]

    # ========================================================
    # PHONE DETECTION
    # ========================================================

    phone_detected = 0
    phone_confidence = 0.0

    yolo_results = yolo(
        frame,
        verbose=False,
        conf=0.30
    )

    for result in yolo_results:

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            label = yolo.names[class_id].lower()

            if (
                class_id == 67
                or "phone" in label
                or "cell" in label
            ):

                phone_detected = 1

                phone_confidence = max(
                    phone_confidence,
                    confidence
                )

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    2
                )

    # ========================================================
    # FACE LANDMARKS
    # ========================================================

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_img = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect(mp_img)

    EAR = 0.0
    eye_closure_duration = 0.0

    yaw = 0.0
    pitch = 0.0
    roll = 0.0

    head_movement = 0.0

    if result.face_landmarks:

        landmarks = result.face_landmarks[0]

        # ====================================================
        # EAR
        # ====================================================

        left_ear = eye_aspect_ratio(
            landmarks,
            LEFT_EYE,
            w,
            h
        )

        right_ear = eye_aspect_ratio(
            landmarks,
            RIGHT_EYE,
            w,
            h
        )

        EAR = (left_ear + right_ear) / 2.0

        # ====================================================
        # EYE CLOSURE
        # ====================================================

        if EAR < EAR_THRESHOLD:

            if eyes_closed_since is None:
                eyes_closed_since = time.time()

            eye_closure_duration = (
                time.time() - eyes_closed_since
            )

        else:

            eyes_closed_since = None
            eye_closure_duration = 0.0

        # ====================================================
        # HEAD POSE
        # ====================================================

        if result.facial_transformation_matrixes:

            matrix = (
                result.facial_transformation_matrixes[0].data
            )

            yaw, pitch, roll = rotation_matrix_to_angles(
                matrix
            )

        # ====================================================
        # HEAD MOVEMENT
        # ====================================================

        if previous_yaw is not None:

            head_movement = np.sqrt(
                (yaw - previous_yaw) ** 2 +
                (pitch - previous_pitch) ** 2
            )

        previous_yaw = yaw
        previous_pitch = pitch

    # ========================================================
    # SAVE DATA
    # ========================================================

    if (
        frame_count % SAMPLE_EVERY == 0
        and result.face_landmarks
    ):

        writer.writerow({

            "timestamp":
                datetime.now().isoformat(),

            "EAR":
                round(EAR, 4),

            "eye_closure_duration":
                round(eye_closure_duration, 3),

            "yaw":
                round(yaw, 3),

            "pitch":
                round(pitch, 3),

            "roll":
                round(roll, 3),

            "abs_yaw":
                round(abs(yaw), 3),

            "abs_pitch":
                round(abs(pitch), 3),

            "head_movement":
                round(head_movement, 3),

            "phone_detected":
                phone_detected,

            "phone_confidence":
                round(phone_confidence, 3),

            "driver_state":
                current_state
        })

        csv_file.flush()

        sample_count += 1

    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.rectangle(
        frame,
        (0, 0),
        (w, 120),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        f"STATE: {current_state}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"EAR: {EAR:.2f}",
        (10, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Yaw: {yaw:.1f} Pitch: {pitch:.1f} Roll: {roll:.1f}",
        (10, 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Samples: {sample_count}",
        (10, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "Driver ML Dataset Collection",
        frame
    )

    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("1"):

        current_state = "Alert"
        print("State -> ALERT")

    elif key == ord("2"):

        current_state = "Drowsy"
        print("State -> DROWSY")

    elif key == ord("3"):

        current_state = "Distracted"
        print("State -> DISTRACTED")

    elif key == ord("4"):

        current_state = "Phone_Usage"
        print("State -> PHONE USAGE")

    elif key == ord("q") or key == ord("Q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
csv_file.close()
cv2.destroyAllWindows()

print()
print("==============================================")
print("DATA COLLECTION FINISHED")
print("Dataset saved to:", CSV_PATH)
print("Total samples:", sample_count)
print("==============================================")