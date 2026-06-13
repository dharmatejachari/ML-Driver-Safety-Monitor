import cv2
import time
from ultralytics import YOLO

# ---------- load model (downloads ~6MB on first run) ----------
model = YOLO("yolov8s.pt")

# ---------- objects we care about ----------
# COCO dataset class names for phone and headphones
DANGER_OBJECTS = {
    67: "phone",
    75: "remote",        # sometimes detects earphones as remote
}

# any object name containing these words also triggers alert
DANGER_KEYWORDS = ["phone", "cell", "mobile", "earphone",
                   "headphone", "airpod", "earbud"]

DETECTION_SECONDS = 2.0   # alert after object visible this long

# ---------- main loop ----------
cap = cv2.VideoCapture(0)
print("Object detector started. Press ESC to stop.")

detected_since = None
alert_active   = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # run YOLO on current frame
    results = model(frame, verbose=False, conf=0.30)

    danger_found    = False
    danger_label    = ""

    for result in results:
        for box in result.boxes:
            class_id   = int(box.cls[0])
            confidence = float(box.conf[0])
            label      = model.names[class_id].lower()

            # check if it's a danger object
            is_danger = (class_id in DANGER_OBJECTS or
                         any(kw in label for kw in DANGER_KEYWORDS))

            # get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if is_danger:
                danger_found = True
                danger_label = label
                # draw red box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{label} {confidence:.0%}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # draw gray box for other objects
                cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 1)
                cv2.putText(frame, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

    # alert logic
    if danger_found:
        if detected_since is None:
            detected_since = time.time()
        duration = time.time() - detected_since

        if duration >= DETECTION_SECONDS:
            status_text  = f"ALERT! {danger_label} detected! ({duration:.1f}s)"
            status_color = (0, 0, 255)
        else:
            status_text  = f"Detecting {danger_label}... ({duration:.1f}s)"
            status_color = (0, 165, 255)
    else:
        detected_since = None
        status_text    = "No dangerous object"
        status_color   = (0, 255, 0)

    # status bar
    cv2.rectangle(frame, (0, 0), (w, 55), (0, 0, 0), -1)
    cv2.putText(frame, status_text, (10, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, status_color, 2)

    cv2.imshow("Object Detector", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("Stopped.")