# 🚗 AI-Based Driver Safety Monitoring System

A real-time driver monitoring system using computer vision and deep learning
to detect drowsiness, distraction, and phone usage — with instant alerts
sent to the vehicle owner via Telegram.

---

## 📌 Project Overview

Road accidents caused by driver negligence are a major problem in transport
businesses. This system continuously monitors the driver using a camera and
automatically alerts the vehicle owner when dangerous behavior is detected.

---

## ✨ Features

- 😴 **Drowsiness Detection** — Detects if driver's eyes are closed for more
  than 2 seconds using Eye Aspect Ratio (EAR) algorithm
- 👀 **Distraction Detection** — Detects if driver is looking left, right,
  or down instead of the road using 3D head pose estimation
- 📱 **Phone Detection** — Detects if driver is using a mobile phone while
  driving using YOLOv8 object detection
- 📲 **Telegram Alerts** — Instantly notifies vehicle owner with alert
  message and photo snapshot of the driver
- ⏱️ **Alert Cooldown** — Prevents alert spam with 30 second cooldown per
  alert type

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Core programming language |
| OpenCV | Video capture and frame processing |
| MediaPipe | Face landmark detection (468 points) |
| YOLOv8 | Real-time object detection |
| Telegram Bot API | Owner alert notifications |
| NumPy | Mathematical computations |

---

## 📁 Project Structure

Driver-Safety-Monitor/

├── main.py            # Main system — runs all detectors together

├── drowsiness.py      # Drowsiness detection module

├── head_pose.py       # Head pose / distraction detection module

├── object_detect.py   # Phone detection module

├── requirements.txt   # Python dependencies

├── .env               # API keys (not uploaded — create locally)

└── README.md          # Project documentation

---

## ⚙️ Installation & Setup

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/Driver-Safety-Monitor.git
cd Driver-Safety-Monitor
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Create your `.env` file**

BOT_TOKEN=your_telegram_bot_token
CHAT_ID=your_telegram_chat_id

**4. Run the system**
```bash
python main.py
```

> On first run, model files are downloaded automatically (~50MB total)

---

## 📲 How to set up Telegram alerts

1. Open Telegram and search `@BotFather`
2. Send `/newbot` and follow the steps
3. Copy the token and paste in `.env` as `BOT_TOKEN`
4. Message your bot once, then visit:
   `https://api.telegram.org/botYOUR_TOKEN/getUpdates`
5. Copy the `chat.id` value and paste in `.env` as `CHAT_ID`

---

## 🚨 Alert Examples

When an alert fires, the owner receives:

🚨 DROWSY ALERT!

Driver eyes closed for 3.2 seconds

Time: 14:32:08
👀 DISTRACTION ALERT!

Driver Looking LEFT for 2.5 seconds

Time: 14:35:21
📱 PHONE ALERT!

Driver using phone while driving!

Time: 14:40:05

Followed immediately by a photo snapshot of the driver.

---

## 💻 System Requirements

- Python 3.11 or higher
- Webcam (built-in or external)
- Internet connection (for Telegram alerts)
- Minimum 4GB RAM recommended
- No GPU required — runs on CPU only

---

## 🔮 Future Improvements

- [ ] Alcohol detection using MQ-3 sensor + Arduino
- [ ] Emergency SOS button for driver
- [ ] GPS tracking with real coordinates (NEO-6M module)
- [ ] Raspberry Pi deployment for permanent in-vehicle installation
- [ ] Headphone/earphone detection with custom trained model

---

## 👨‍💻 Developer

Built as part of a transport business safety initiative to reduce road
accidents caused by driver negligence.

---

## ⚠️ Disclaimer

This system is a prototype built for educational and demonstration purposes.
For production deployment in vehicles, additional hardware, testing, and
safety certifications are recommended.