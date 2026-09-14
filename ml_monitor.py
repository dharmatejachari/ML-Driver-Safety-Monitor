import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import time
import joblib
from ultralytics import YOLO


# ============================================================
# LOAD TRAINED ML MODEL
# ============================================================

MODEL_PATH = "models/driver_behavior_model.pkl"

if not os.path.exists(MODEL_PATH):
    print("ERROR: Trained model not found.")
    print("Run: python train_model.py")
    exit()

model = joblib.load(MODEL_PATH)

print("ML model loaded successfully.")

# Check expected number of features
if hasattr(model, "n_features_in_"):
    print(
        "Model expects",
        model.n_features_in_,
        "features."
    )


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

MODEL_FILE = "face_landmarker.task"

if not os.path.exists(MODEL_FILE):
    print("ERROR: face_landmarker.task not found.")
    exit()

base_options = python.BaseOptions(
    model_asset_path=MODEL_FILE
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    output_facial_transformation_matrixes=True
)

landmarker = vision.FaceLandmarker.create_from_options(
    options
)


# ============================================================
# YOLO SETUP
# ============================================================

print("Loading YOLO...")

yolo = YOLO("yolov8s.pt")


# ============================================================
# CONSTANTS
# ============================================================

LEFT_EYE = [
    362, 385, 387,
    263, 373, 380
]

RIGHT_EYE = [
    33, 160, 158,
    133, 153, 144
]

EAR_THRESHOLD = 0.25


# ============================================================
# FUNCTIONS
# ============================================================

def eye_aspect_ratio(
    landmarks,
    eye_points,
    w,
    h
):

    pts = [
        (
            landmarks[i].x * w,
            landmarks[i].y * h
        )
        for i in eye_points
    ]

    v1 = np.linalg.norm(
        np.array(pts[1])
        -
        np.array(pts[5])
    )

    v2 = np.linalg.norm(
        np.array(pts[2])
        -
        np.array(pts[4])
    )

    horizontal = np.linalg.norm(
        np.array(pts[0])
        -
        np.array(pts[3])
    )

    if horizontal == 0:
        return 0.0

    return (
        v1 + v2
    ) / (
        2.0 * horizontal
    )


def rotation_matrix_to_angles(matrix):

    r = np.array(
        matrix
    ).reshape(4, 4)[:3, :3]

    pitch = np.degrees(
        np.arcsin(
            np.clip(
                -r[1, 2],
                -1.0,
                1.0
            )
        )
    )

    yaw = np.degrees(
        np.arctan2(
            r[0, 2],
            r[2, 2]
        )
    )

    roll = np.degrees(
        np.arctan2(
            r[1, 0],
            r[1, 1]
        )
    )

    return yaw, pitch, roll


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Camera could not be opened.")
    exit()


print()
print("==============================================")
print("       ML DRIVER SAFETY MONITOR")
print("==============================================")
print("Model: Random Forest")
print("Features: 10")
print("Press Q to quit.")
print("==============================================")


# ============================================================
# TRACKING VARIABLES
# ============================================================

eyes_closed_since = None

previous_yaw = None
previous_pitch = None

prediction_history = []

prediction_confidence = 0.0


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    h, w = frame.shape[:2]


    # ========================================================
    # PHONE DETECTION USING YOLO
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

            class_id = int(
                box.cls[0]
            )

            confidence = float(
                box.conf[0]
            )

            label = yolo.names[
                class_id
            ].lower()


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


                cv2.putText(
                    frame,
                    f"PHONE {confidence:.0%}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )


    # ========================================================
    # MEDIAPIPE FACE LANDMARKS
    # ========================================================

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    mp_img = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )


    result = landmarker.detect(
        mp_img
    )


    # ========================================================
    # DEFAULT VALUES
    # ========================================================

    driver_state = "No Face"

    EAR = 0.0

    eye_closure_duration = 0.0

    yaw = 0.0
    pitch = 0.0
    roll = 0.0

    abs_yaw = 0.0
    abs_pitch = 0.0

    head_movement = 0.0

    prediction_confidence = 0.0


    # ========================================================
    # FACE DETECTED
    # ========================================================

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


        EAR = (
            left_ear
            +
            right_ear
        ) / 2.0


        # ====================================================
        # EYE CLOSURE DURATION
        # ====================================================

        if EAR < EAR_THRESHOLD:

            if eyes_closed_since is None:

                eyes_closed_since = time.time()


            eye_closure_duration = (
                time.time()
                -
                eyes_closed_since
            )

        else:

            eyes_closed_since = None

            eye_closure_duration = 0.0


        # ====================================================
        # HEAD POSE
        # ====================================================

        if result.facial_transformation_matrixes:

            matrix = (
                result
                .facial_transformation_matrixes[0]
                .data
            )


            yaw, pitch, roll = (
                rotation_matrix_to_angles(
                    matrix
                )
            )


            # =================================================
            # ABSOLUTE HEAD ANGLES
            # =================================================

            abs_yaw = abs(yaw)

            abs_pitch = abs(pitch)


            # =================================================
            # HEAD MOVEMENT
            # =================================================

            if (
                previous_yaw is not None
                and
                previous_pitch is not None
            ):

                head_movement = np.sqrt(
                    (
                        yaw
                        -
                        previous_yaw
                    ) ** 2
                    +
                    (
                        pitch
                        -
                        previous_pitch
                    ) ** 2
                )


            previous_yaw = yaw

            previous_pitch = pitch


        # ====================================================
        # 10 ML FEATURES
        # ====================================================

        features = np.array([[
            EAR,
            eye_closure_duration,
            yaw,
            pitch,
            roll,
            abs_yaw,
            abs_pitch,
            head_movement,
            phone_detected,
            phone_confidence
        ]])


        # ====================================================
        # ML PREDICTION
        # ====================================================

        prediction = model.predict(
            features
        )[0]

        driver_state = str(
            prediction
        )


        # ====================================================
        # PREDICTION CONFIDENCE
        # ====================================================

        if hasattr(
            model,
            "predict_proba"
        ):

            probabilities = (
                model.predict_proba(
                    features
                )
            )

            prediction_confidence = float(
                np.max(
                    probabilities
                )
            )


        # ====================================================
        # PREDICTION SMOOTHING
        # ====================================================

        prediction_history.append(
            driver_state
        )


        if len(
            prediction_history
        ) > 5:

            prediction_history.pop(
                0
            )


        if prediction_history:

            driver_state = max(
                set(
                    prediction_history
                ),
                key=prediction_history.count
            )


    else:

        # Reset face tracking
        eyes_closed_since = None

        previous_yaw = None

        previous_pitch = None

        prediction_history.clear()


    # ========================================================
    # DISPLAY INFORMATION PANEL
    # ========================================================

    cv2.rectangle(
        frame,
        (0, 0),
        (w, 180),
        (0, 0, 0),
        -1
    )


    # ========================================================
    # DRIVER STATE
    # ========================================================

    cv2.putText(
        frame,
        f"ML DRIVER STATE: {driver_state}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (0, 255, 0),
        2
    )


    # ========================================================
    # ML CONFIDENCE
    # ========================================================

    cv2.putText(
        frame,
        f"ML Confidence: {prediction_confidence:.0%}",
        (10, 57),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        2
    )


    # ========================================================
    # EYE INFORMATION
    # ========================================================

    cv2.putText(
        frame,
        f"EAR: {EAR:.2f}  Closure: {eye_closure_duration:.1f}s",
        (10, 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        2
    )


    # ========================================================
    # HEAD POSE
    # ========================================================

    cv2.putText(
        frame,
        f"Yaw: {yaw:.1f}  Pitch: {pitch:.1f}  Roll: {roll:.1f}",
        (10, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        2
    )


    # ========================================================
    # HEAD MOVEMENT
    # ========================================================

    cv2.putText(
        frame,
        f"Head Movement: {head_movement:.2f}",
        (10, 137),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        2
    )


    # ========================================================
    # PHONE INFORMATION
    # ========================================================

    cv2.putText(
        frame,
        f"Phone: {phone_detected}  Confidence: {phone_confidence:.2f}",
        (10, 164),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        2
    )


    # ========================================================
    # SHOW WINDOW
    # ========================================================

    cv2.imshow(
        "ML Driver Safety Monitor",
        frame
    )


    # ========================================================
    # QUIT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    if (
        key == ord("q")
        or
        key == ord("Q")
    ):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print()
print("==============================================")
print("ML Driver Monitor stopped.")
print("==============================================")