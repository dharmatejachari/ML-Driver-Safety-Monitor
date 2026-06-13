import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import urllib.request
import os

# ---------- download model (reuses same file from Day 2) ----------
model_path = "face_landmarker.task"
if not os.path.exists(model_path):
    print("Downloading face landmark model (~30MB)...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        model_path
    )
    print("Download complete!")

# ---------- mediapipe setup ----------
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=True
)
landmarker = vision.FaceLandmarker.create_from_options(options)

# ---------- thresholds (degrees) ----------
YAW_LIMIT   = 25   # left/right turn limit
PITCH_LIMIT = 15  # up/down tilt limit
AWAY_SECONDS = 2.0 # alert after looking away this long

def rotation_matrix_to_angles(matrix):
    r = np.array(matrix).reshape(4, 4)[:3, :3]
    pitch = np.degrees(np.arcsin(-r[1, 2]))
    yaw   = np.degrees(np.arctan2( r[0, 2], r[2, 2]))
    roll  = np.degrees(np.arctan2(-r[1, 0], r[1, 1]))
    return yaw, pitch, roll

# ---------- main loop ----------
cap = cv2.VideoCapture(0)
print("Head pose detector started. Press ESC to stop.")

looking_away_since = None
alert_active       = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = landmarker.detect(mp_img)

    status_text  = "No face detected"
    status_color = (100, 100, 100)

    if results.face_landmarks and results.facial_transformation_matrixes:
        matrix = results.facial_transformation_matrixes[0].data
        yaw, pitch, roll = rotation_matrix_to_angles(matrix)

        # determine gaze direction
        if yaw > YAW_LIMIT:
            direction = "Looking LEFT"
            is_away   = True
        elif yaw < -YAW_LIMIT:
            direction = "Looking RIGHT"
            is_away   = True
        elif pitch > 15:
            direction = "Looking DOWN"
            is_away   = True
        else:
            direction = "Looking FORWARD"
            is_away   = False

        # draw direction arrow on frame
        nose_tip = results.face_landmarks[0][1]
        nose_x   = int(nose_tip.x * w)
        nose_y   = int(nose_tip.y * h)
        arrow_x  = int(nose_x + np.sin(np.radians(yaw))   * 80)
        arrow_y  = int(nose_y + np.sin(np.radians(pitch))  * 80)
        cv2.arrowedLine(frame, (nose_x, nose_y),
                        (arrow_x, arrow_y), (255, 255, 0), 3, tipLength=0.3)

        # show angle values
        cv2.putText(frame, f"Yaw:{yaw:+.0f} Pitch:{pitch:+.0f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

        # alert logic
        if is_away:
            if looking_away_since is None:
                looking_away_since = time.time()
            away_duration = time.time() - looking_away_since

            if away_duration >= AWAY_SECONDS:
                status_text  = f"DISTRACTION ALERT! {direction} ({away_duration:.1f}s)"
                status_color = (0, 0, 255)
            else:
                status_text  = f"{direction} ({away_duration:.1f}s)"
                status_color = (0, 165, 255)
        else:
            looking_away_since = None
            status_text        = f"{direction} - OK"
            status_color       = (0, 255, 0)

    # status bar
    cv2.rectangle(frame, (0, 0), (w, 55), (0, 0, 0), -1)
    cv2.putText(frame, status_text, (10, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, status_color, 2)

    cv2.imshow("Head Pose Detector", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("Stopped.")