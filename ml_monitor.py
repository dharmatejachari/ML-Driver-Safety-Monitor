import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import time
import joblib
import requests
import threading
from dotenv import load_dotenv
from ultralytics import YOLO


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")


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

if hasattr(model, "n_features_in_"):

    print(
        "Model expects",
        model.n_features_in_,
        "features."
    )


# ============================================================
# TELEGRAM SETUP
# ============================================================

if BOT_TOKEN and CHAT_ID:

    TELEGRAM_ENABLED = True

    print("Telegram alerts: ENABLED")

else:

    TELEGRAM_ENABLED = False

    print("Telegram alerts: DISABLED")
    print("Check BOT_TOKEN and CHAT_ID in your .env file.")


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

# Unsafe state must persist for this many seconds
UNSAFE_DURATION = 2.0

# Minimum time between Telegram alerts
ALERT_COOLDOWN = 30.0


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
# TELEGRAM ALERT FUNCTION
# ============================================================

def send_telegram_alert(
    frame,
    driver_state,
    confidence
):

    if not TELEGRAM_ENABLED:

        return

    try:

        # Encode current frame as JPEG
        success, encoded_image = cv2.imencode(
            ".jpg",
            frame
        )

        if not success:

            print("ERROR: Could not encode snapshot.")

            return

        image_bytes = encoded_image.tobytes()

        # Alert message
        message = (
            "🚨 DRIVER SAFETY ALERT\n\n"
            f"Driver State: {driver_state}\n"
            f"ML Confidence: {confidence:.0%}\n\n"
            "Unsafe behavior detected by "
            "the ML Driver Safety Monitor."
        )

        url = (
            f"https://api.telegram.org/bot"
            f"{BOT_TOKEN}/sendPhoto"
        )

        files = {
            "photo": (
                "driver_alert.jpg",
                image_bytes,
                "image/jpeg"
            )
        }

        data = {
            "chat_id": CHAT_ID,
            "caption": message
        }

        response = requests.post(
            url,
            data=data,
            files=files,
            timeout=10
        )

        if response.ok:

            print(
                f"Telegram alert sent: {driver_state}"
            )

        else:

            print(
                "Telegram error:",
                response.text
            )

    except Exception as e:

        print(
            "Telegram alert failed:",
            e
        )


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
print("Telegram: " +
      ("Enabled" if TELEGRAM_ENABLED else "Disabled"))
print()
print("Unsafe states:")
print("- Drowsy")
print("- Distracted")
print("- Phone_Usage")
print()
print("Press Q to quit.")
print("==============================================")


# ============================================================
# TRACKING VARIABLES
# ============================================================

eyes_closed_since = None

previous_yaw = None
previous_pitch = None

prediction_history = []

# Current unsafe state being monitored
unsafe_state = None

# When the current unsafe state started
unsafe_state_since = None

# Last Telegram alert time
last_alert_time = 0


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:

        break

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
                    (
                        x1,
                        max(20, y1 - 10)
                    ),
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
        # EYE CLOSURE
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


            # Absolute values

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
        # ML CONFIDENCE
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


        # ====================================================
        # SAFETY MONITORING
        # ====================================================

        unsafe_states = [
            "Drowsy",
            "Distracted",
            "Phone_Usage"
        ]


        if driver_state in unsafe_states:

            # New unsafe state
            if unsafe_state != driver_state:

                unsafe_state = driver_state

                unsafe_state_since = time.time()


            # Same unsafe state continues
            else:

                unsafe_duration = (
                    time.time()
                    -
                    unsafe_state_since
                )


                # Check persistence
                if unsafe_duration >= UNSAFE_DURATION:

                    current_time = time.time()


                    # Check alert cooldown
                    if (
                        current_time
                        -
                        last_alert_time
                        >= ALERT_COOLDOWN
                    ):

                        # Send Telegram in background
                        alert_frame = frame.copy()

                        threading.Thread(
                            target=send_telegram_alert,
                            args=(
                                alert_frame,
                                driver_state,
                                prediction_confidence
                            ),
                            daemon=True
                        ).start()


                        last_alert_time = current_time


        else:

            # Driver returned to safe state
            unsafe_state = None

            unsafe_state_since = None


    else:

        # ====================================================
        # NO FACE
        # ====================================================

        eyes_closed_since = None

        previous_yaw = None

        previous_pitch = None

        prediction_history.clear()

        unsafe_state = None

        unsafe_state_since = None


    # ========================================================
    # DISPLAY PANEL
    # ========================================================

    cv2.rectangle(
        frame,
        (0, 0),
        (w, 185),
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
    # CONFIDENCE
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
    # EAR
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
        (10, 111),
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
        (10, 138),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        2
    )


    # ========================================================
    # PHONE
    # ========================================================

    cv2.putText(
        frame,
        f"Phone: {phone_detected}  Confidence: {phone_confidence:.2f}",
        (10, 165),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        2
    )


    # ========================================================
    # MONITORING STATUS
    # ========================================================

    if unsafe_state is not None:

        monitoring_time = (
            time.time()
            -
            unsafe_state_since
        )

        status_text = (
            f"Monitoring: {unsafe_state} "
            f"({monitoring_time:.1f}s)"
        )

        cv2.putText(
            frame,
            status_text,
            (10, 182),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 165, 255),
            1
        )


    # ========================================================
    # DISPLAY
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