# ML Driver Safety Monitor

## Machine Learning Based Driver Behavior Analysis and Safety Monitoring System

An AI-based driver safety monitoring system that combines computer vision, deep learning, feature engineering, and machine learning to analyze driver behavior in real time.

The system extracts meaningful driver-related features from camera input and uses a machine learning model to classify the driver's state as:

- Alert
- Drowsy
- Distracted
- Phone Usage

---

## Project Objective

Driver distraction, drowsiness, and mobile phone usage are major causes of road accidents.

The objective of this project is to develop an intelligent driver monitoring system that:

- Monitors the driver through a camera
- Extracts facial and head-pose features
- Detects mobile phone usage
- Processes extracted features
- Classifies driver behavior using machine learning
- Provides a foundation for real-time safety alerts

---

## System Architecture

Camera Input
        ↓
OpenCV
        ↓
MediaPipe + YOLOv8
        ↓
Feature Extraction
        ↓
Data Preprocessing
        ↓
Feature Engineering
        ↓
Machine Learning Models
        ↓
Driver Behavior Classification
        ↓
Safety Monitoring / Alerts

---

## Technologies Used

### Computer Vision

- OpenCV
- MediaPipe

### Deep Learning

- YOLOv8

### Machine Learning

- Scikit-learn
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)

### Data Science

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn

### Other Technologies

- Joblib
- Telegram Bot API
- Git & GitHub

---

## Features

### 1. Drowsiness Detection

Eye Aspect Ratio (EAR) is extracted from facial landmarks.

The system also calculates eye closure duration to identify prolonged eye closure.

### 2. Driver Distraction Detection

Head-pose information is extracted using facial transformation matrices.

The system uses:

- Yaw
- Pitch
- Roll
- Absolute yaw
- Absolute pitch
- Head movement

to represent driver head behavior.

### 3. Phone Usage Detection

YOLOv8 is used to detect mobile phones in the camera frame.

The following information is extracted:

- Phone detected
- Phone confidence

### 4. Machine Learning Classification

The extracted features are provided to supervised machine learning models to classify driver behavior.

Classes:

- Alert
- Drowsy
- Distracted
- Phone Usage

---

## Dataset

A custom driver behavior dataset was created using real-time feature extraction.

The final dataset contains:

- 101 observations
- 10 input features
- 4 driver behavior classes

### Features

1. EAR
2. Eye Closure Duration
3. Yaw
4. Pitch
5. Roll
6. Absolute Yaw
7. Absolute Pitch
8. Head Movement
9. Phone Detection
10. Phone Confidence

---

## Machine Learning Models

The following models were evaluated:

- Logistic Regression
- Random Forest
- SVM
- Tuned SVM

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 80.95% | 82.14% | 80.95% | 81.16% |
| Random Forest | 76.19% | 81.90% | 76.19% | 75.23% |
| SVM | 71.43% | 73.61% | 71.43% | 70.88% |
| Tuned SVM | 80.95% | 83.33% | 80.95% | 81.09% |

### Best Model

**Logistic Regression**

Test Accuracy: **80.95%**

F1 Score: **81.16%**

The Logistic Regression model was selected as the final model based on the highest test F1-score.

---

## Data Analysis

Exploratory Data Analysis was performed using:

- Pandas
- NumPy
- Matplotlib
- Seaborn

The analysis includes:

- Driver state distribution
- EAR analysis
- Yaw analysis
- Pitch analysis
- Feature correlation analysis
- Confusion matrix
- Model comparison

Generated analysis files are available in:

```text
analysis/
Project Structure
ML-Driver-Safety-Monitor/
│
├── analysis/
│   ├── eda.py
│   ├── create_dashboard_data.py
│   ├── model_comparison.csv
│   ├── tuned_svm_metrics.csv
│   ├── driver_safety_dashboard.csv
│   └── plots/
│
├── data/
│   └── driver_behavior.csv
│
├── models/
│   └── driver_behavior_model.pkl
│
├── collect_data.py
├── train_model.py
├── ml_monitor.py
├── main.py
├── drowsiness.py
├── head_pose.py
├── object_detect.py
├── requirements.txt
└── README.md
How to Run
Install dependencies
pip install -r requirements.txt
Collect driver behavior data
python collect_data.py
Train machine learning models
python train_model.py
Run the ML driver monitor
python ml_monitor.py
Machine Learning Workflow

The project follows the following Data Science workflow:

Data Collection
      ↓
Data Cleaning
      ↓
Feature Extraction
      ↓
Feature Engineering
      ↓
Data Preprocessing
      ↓
Train-Test Split
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Hyperparameter Tuning
      ↓
Best Model Selection
      ↓
Real-Time Prediction
Internship Relevance

This project demonstrates practical application of Machine Learning and Data Science concepts including:

NumPy
Pandas
Data preprocessing
Feature engineering
Supervised learning
Scikit-learn
Model evaluation
Model selection
Ensemble learning
Hyperparameter tuning
Exploratory Data Analysis
Data visualization

Computer vision and deep learning techniques are used for feature extraction, while machine learning is used for driver behavior classification.

Future Scope

Future improvements can include:

Larger and more diverse datasets
Improved real-time prediction stability
More advanced feature engineering
Deep learning based driver behavior classification
Cloud-based monitoring
Mobile application integration
Advanced safety analytics dashboards
Deployment using Flask or other web frameworks
Disclaimer

This project is an educational prototype developed for learning and demonstration purposes. It should not be considered a certified automotive safety system.