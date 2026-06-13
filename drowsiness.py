import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import urllib.request
import os

# ---------- download model (only first time) ----------
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
    num_faces=1
)
landmarker = vision.FaceLandmarker.create_from_options(options)

# ---------- eye landmark indexes ----------
# These are the 6 points around each eye in mediapipe's 468-point face mesh
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

EAR_THRESHOLD  = 0.25   # below this = eye is closed
CLOSED_SECONDS = 2.0    # alert after this many seconds of closed eyes

# ---------- EAR formula ----------
def eye_aspect_ratio(landmarks, eye_points, w, h):
    pts = []
    for idx in eye_points:
        lm = landmarks[idx]
        pts.append((lm.x * w, lm.y * h))

    # vertical distances
    v1 = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    v2 = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    # horizontal distance
    hz = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))

    ear = (v1 + v2) / (2.0 * hz)
    return ear

# ---------- main loop ----------
cap = cv2.VideoCapture(0)
print("Drowsiness detector started. Press ESC to stop.")

eyes_closed_since = None   # timestamp when eyes first closed
alert_active       = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = landmarker.detect(mp_img)

    status_text  = "No face detected"
    status_color = (100, 100, 100)

    if results.face_landmarks:
        landmarks = results.face_landmarks[0]

        left_ear  = eye_aspect_ratio(landmarks, LEFT_EYE,  w, h)
        right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
        avg_ear   = (left_ear + right_ear) / 2.0

        # draw eye points
        for idx in LEFT_EYE + RIGHT_EYE:
            lm = landmarks[idx]
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 2, (0, 255, 255), -1)

        # show EAR value
        cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        if avg_ear < EAR_THRESHOLD:
            # eyes are closed
            if eyes_closed_since is None:
                eyes_closed_since = time.time()

            closed_duration = time.time() - eyes_closed_since

            if closed_duration >= CLOSED_SECONDS:
                alert_active  = True
                status_text   = f"DROWSY ALERT! ({closed_duration:.1f}s)"
                status_color  = (0, 0, 255)
            else:
                status_text   = f"Eyes closing... ({closed_duration:.1f}s)"
                status_color  = (0, 165, 255)
        else:
            # eyes are open — reset
            eyes_closed_since = None
            alert_active      = False
            status_text       = "Alert - Eyes Open"
            status_color      = (0, 255, 0)

    # draw status bar
    cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 0), -1)
    cv2.putText(frame, status_text, (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)

    cv2.imshow("Drowsiness Detector", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("Stopped.")