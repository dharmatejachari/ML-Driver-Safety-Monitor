# ML Driver Safety Monitor

## Machine Learning Based Driver Behavior Analysis and Safety Monitoring System

An AI-based driver safety monitoring system that combines computer vision, deep learning, feature engineering, and machine learning to analyze driver behavior in real time.

The system extracts meaningful driver-related features from camera input and uses machine learning models to classify the driver's state as:

- Alert
- Drowsy
- Distracted
- Phone Usage

---

## Project Objective

Driver drowsiness, distraction, and mobile phone usage are major safety concerns while driving.

The objective of this project is to develop an intelligent driver monitoring system that:

- Monitors the driver through a camera
- Extracts facial and head-pose features
- Detects mobile phone usage
- Performs data preprocessing and feature engineering
- Trains supervised machine learning models
- Evaluates and compares different ML algorithms
- Selects the best-performing model
- Performs real-time driver behavior prediction

---

## System Architecture

```text
                    CAMERA INPUT
                         |
                         v
               +-------------------+
               | OpenCV Processing |
               +-------------------+
                         |
                         v
              +--------------------+
              | MediaPipe + YOLOv8 |
              +--------------------+
                         |
                         v
                 FEATURE EXTRACTION
                         |
                         v
              +--------------------+
              | 10 Driver Features |
              +--------------------+
                         |
                         v
                DATA PREPROCESSING
                         |
                         v
                FEATURE ENGINEERING
                         |
                         v
              MACHINE LEARNING MODELS
                         |
          +--------------+--------------+
          |              |              |
          v              v              v
   Logistic Regression  Random Forest  SVM
          |              |              |
          +--------------+--------------+
                         |
                         v
                 MODEL EVALUATION
                         |
                         v
                  BEST ML MODEL
                  Random Forest
                         |
                         v
              REAL-TIME PREDICTION
                         |
          +--------------+--------------+
          |              |              |
          v              v              v
        Alert         Drowsy       Distracted
                         |
                         v
                   Phone Usage
Technologies Used
Programming Language
Python
Computer Vision
OpenCV
MediaPipe
Deep Learning
YOLOv8
Machine Learning
Scikit-learn
Logistic Regression
Random Forest
Support Vector Machine (SVM)
Data Science
NumPy
Pandas
Matplotlib
Seaborn
Model Management
Joblib
Communication
Telegram Bot API
Version Control
Git
GitHub
Key Features
1. Drowsiness Detection

The system extracts facial landmarks using MediaPipe and calculates the Eye Aspect Ratio (EAR).

Eye closure duration is also measured to capture prolonged eye closure behavior.

Important features include:

EAR
Eye closure duration
2. Driver Distraction Detection

The system uses facial transformation data to estimate the driver's head orientation.

The following head-pose features are extracted:

Yaw
Pitch
Roll
Absolute Yaw
Absolute Pitch
Head Movement

These features help represent changes in driver head position and movement.

3. Phone Usage Detection

YOLOv8 is used to detect mobile phones in the camera frame.

The system extracts:

Phone detection
Phone confidence

These values are provided as input features to the machine learning model.

4. Machine Learning Based Classification

The extracted features are used as inputs to supervised machine learning algorithms.

The system classifies driver behavior into four classes:

Alert
Drowsy
Distracted
Phone_Usage
Dataset

A custom driver behavior dataset was created using real-time camera-based feature extraction.

The final dataset contains:

303 observations
10 input features
4 driver behavior classes
Class Distribution
Driver State	Samples
Phone Usage	104
Alert	74
Drowsy	74
Distracted	51
Total	303
Input Features

The machine learning model uses the following 10 features:

Feature	Description
EAR	Eye Aspect Ratio
Eye Closure Duration	Duration of detected eye closure
Yaw	Horizontal head rotation
Pitch	Vertical head rotation
Roll	Head tilt rotation
Absolute Yaw	Absolute value of yaw
Absolute Pitch	Absolute value of pitch
Head Movement	Change in head orientation
Phone Detection	Whether a phone is detected
Phone Confidence	YOLO phone detection confidence
Data Science Workflow

The project follows a complete machine learning workflow:

Data Collection
       |
       v
Data Cleaning
       |
       v
Feature Extraction
       |
       v
Feature Engineering
       |
       v
Exploratory Data Analysis
       |
       v
Train-Test Split
       |
       v
Model Training
       |
       v
Model Evaluation
       |
       v
Hyperparameter Tuning
       |
       v
Best Model Selection
       |
       v
Real-Time Prediction
Machine Learning Models

The following supervised learning algorithms were trained and evaluated:

1. Logistic Regression

Used as a baseline classification model for driver behavior prediction.

2. Random Forest

An ensemble learning algorithm that combines multiple decision trees to improve classification performance.

3. Support Vector Machine

Used to identify decision boundaries between different driver behavior classes.

4. Tuned SVM

Hyperparameter tuning was performed using GridSearchCV to identify better SVM parameters.

Best parameters found:

C = 10
gamma = scale
kernel = rbf
Model Evaluation

The models were evaluated using:

Accuracy
Precision
Recall
F1-score
Confusion Matrix
Model Comparison
Model	Accuracy	Precision	Recall	F1 Score
Logistic Regression	88.52%	90.57%	88.52%	88.81%
Random Forest	90.16%	91.05%	90.16%	90.31%
SVM	88.52%	92.18%	88.52%	88.92%
Tuned SVM	86.89%	89.75%	86.89%	87.33%
Best Model
Random Forest

The Random Forest model achieved the highest F1-score among the evaluated models.

Performance
Accuracy  : 90.16%
Precision : 91.05%
Recall    : 90.16%
F1 Score  : 90.31%

Therefore, Random Forest was selected as the final machine learning model for real-time driver behavior classification.

The trained model is saved as:

models/driver_behavior_model.pkl
Exploratory Data Analysis

Exploratory Data Analysis was performed using Pandas, NumPy, Matplotlib, and Seaborn.

The project includes analysis of:

Driver behavior class distribution
Eye Aspect Ratio
Yaw
Pitch
Feature correlations
Model performance
Confusion matrix

Generated analysis files are available in:

analysis/

The project also contains generated visualization files in:

analysis/plots/
Real-Time Machine Learning Prediction

The trained Random Forest model is integrated with the real-time monitoring system.

The live workflow is:

Camera
   |
   v
MediaPipe + YOLOv8
   |
   v
Feature Extraction
   |
   v
10 Features
   |
   v
Random Forest Model
   |
   v
Driver Behavior

The predicted states are:

Alert
Drowsy
Distracted
Phone_Usage
Project Structure
ML-Driver-Safety-Monitor/
│
├── analysis/
│   ├── eda.py
│   ├── create_dashboard_data.py
│   ├── model_comparison.csv
│   ├── tuned_svm_metrics.csv
│   ├── driver_safety_dashboard.csv
│   │
│   └── plots/
│       ├── class_distribution.png
│       ├── confusion_matrix.png
│       ├── correlation_heatmap.png
│       ├── ear_by_state.png
│       ├── pitch_by_state.png
│       └── yaw_by_state.png
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
How to Run the Project
1. Install Dependencies

Open a terminal in the project directory and run:

pip install -r requirements.txt
2. Collect Training Data

Run:

python collect_data.py

During data collection:

1 = ALERT
2 = DROWSY
3 = DISTRACTED
4 = PHONE USAGE
Q = QUIT

The collected data is stored in:

data/driver_behavior.csv
3. Train Machine Learning Models

Run:

python train_model.py

This will:

Load the dataset
Prepare the features
Split the data into training and testing sets
Train multiple ML models
Evaluate model performance
Perform SVM hyperparameter tuning
Select the best model
Save the trained model

The final model is saved as:

models/driver_behavior_model.pkl
4. Run Real-Time ML Monitoring

Run:

python ml_monitor.py

The system will access the camera and perform real-time driver behavior prediction using the trained machine learning model.

Press:

Q

to stop the monitoring system.

Original Safety Monitoring System

The project also contains the original rule-based safety monitoring implementation:

main.py

It combines:

MediaPipe
YOLOv8
OpenCV
Drowsiness detection
Head-pose analysis
Phone detection
Telegram alerts

The original system can be started using:

python main.py

The machine-learning-based workflow is demonstrated using:

python ml_monitor.py
Internship Relevance

This project demonstrates practical application of Machine Learning and Data Science concepts including:

NumPy
Pandas
Data preprocessing
Feature engineering
Exploratory Data Analysis
Supervised learning
Scikit-learn
Model evaluation
Model comparison
Model selection
Ensemble learning
Hyperparameter tuning
Data visualization

Computer vision and deep learning techniques are used for extracting meaningful features from video data, while supervised machine learning is used for driver behavior classification.

Key Results

The final system achieved:

Dataset Size       : 303 samples
Number of Features : 10
Number of Classes  : 4

Best Model         : Random Forest

Accuracy           : 90.16%
Precision          : 91.05%
Recall             : 90.16%
F1 Score           : 90.31%
Future Scope

Future improvements can include:

Increasing the size and diversity of the dataset
Collecting data from multiple drivers
Improving real-time prediction stability
Adding more driver behavior classes
Advanced feature engineering
Deep learning based driver behavior classification
Cloud-based driver safety monitoring
Mobile application integration
Advanced analytics dashboards
Web-based deployment using Flask
Integration with vehicle safety systems
Disclaimer

This project is an educational prototype developed for learning and demonstration purposes.

It should not be considered a certified automotive safety system and should not be relied upon as the sole safety mechanism in a vehicle.