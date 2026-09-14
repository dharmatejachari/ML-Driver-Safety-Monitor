import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt


# ============================================================
# LOAD DATA
# ============================================================

DATA_PATH = "data/driver_behavior.csv"

df = pd.read_csv(DATA_PATH)

print("\nDataset shape:", df.shape)

print("\nClass distribution:")
print(df["driver_state"].value_counts())


# ============================================================
# FEATURES
# ============================================================

features = [
    "EAR",
    "eye_closure_duration",
    "yaw",
    "pitch",
    "roll",
    "abs_yaw",
    "abs_pitch",
    "head_movement",
    "phone_detected",
    "phone_confidence"
]

X = df[features].copy()
y = df["driver_state"].copy()

X = X.replace([np.inf, -np.inf], np.nan)

valid = X.notna().all(axis=1)

X = X[valid]
y = y[valid]


# ============================================================
# TRAIN / TEST
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# MODELS
# ============================================================

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(
            kernel="linear",
            probability=True
        ))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        class_weight="balanced"
    ),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(
            probability=True
        ))
    ])
}


# ============================================================
# TRAIN BASE MODELS
# ============================================================

results = []

best_model = None
best_f1 = 0
best_name = ""

for name, model in models.items():

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1
    })

    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_name = name


# ============================================================
# SVM HYPERPARAMETER TUNING
# ============================================================

print("\n" + "=" * 60)
print("SVM HYPERPARAMETER TUNING")
print("=" * 60)

svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(probability=True))
])

param_grid = {
    "svm__C": [0.1, 1, 10, 100],
    "svm__gamma": ["scale", "auto", 0.01, 0.1],
    "svm__kernel": ["rbf", "linear"]
}

grid = GridSearchCV(
    svm_pipeline,
    param_grid,
    cv=5,
    scoring="f1_weighted",
    n_jobs=-1
)

grid.fit(X_train, y_train)

tuned_model = grid.best_estimator_

tuned_predictions = tuned_model.predict(X_test)

tuned_accuracy = accuracy_score(
    y_test,
    tuned_predictions
)

tuned_precision = precision_score(
    y_test,
    tuned_predictions,
    average="weighted",
    zero_division=0
)

tuned_recall = recall_score(
    y_test,
    tuned_predictions,
    average="weighted",
    zero_division=0
)

tuned_f1 = f1_score(
    y_test,
    tuned_predictions,
    average="weighted",
    zero_division=0
)

print("\nBest parameters:")
print(grid.best_params_)

print("\nTuned SVM:")
print(f"Accuracy : {tuned_accuracy:.4f}")
print(f"Precision: {tuned_precision:.4f}")
print(f"Recall   : {tuned_recall:.4f}")
print(f"F1 Score : {tuned_f1:.4f}")


# ============================================================
# CHECK IF TUNED SVM IS BEST
# ============================================================

if tuned_f1 > best_f1:

    best_model = tuned_model
    best_f1 = tuned_f1
    best_name = "Tuned SVM"


# ============================================================
# MODEL COMPARISON
# ============================================================

results.append({
    "Model": "Tuned SVM",
    "Accuracy": tuned_accuracy,
    "Precision": tuned_precision,
    "Recall": tuned_recall,
    "F1_Score": tuned_f1
})

results_df = pd.DataFrame(results)

print("\n\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(results_df.to_string(index=False))


# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs("analysis/plots", exist_ok=True)
os.makedirs("models", exist_ok=True)

results_df.to_csv(
    "analysis/model_comparison.csv",
    index=False
)


# ============================================================
# CONFUSION MATRIX FOR BEST MODEL
# ============================================================

best_predictions = best_model.predict(X_test)

cm = confusion_matrix(
    y_test,
    best_predictions,
    labels=sorted(y.unique())
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=sorted(y.unique())
)

fig, ax = plt.subplots(figsize=(8, 6))

disp.plot(
    ax=ax,
    xticks_rotation=20
)

plt.title(
    f"Driver Behavior Classification - {best_name}"
)

plt.tight_layout()

plt.savefig(
    "analysis/plots/confusion_matrix.png"
)

plt.close()


# ============================================================
# SAVE BEST MODEL
# ============================================================

model_path = "models/driver_behavior_model.pkl"

joblib.dump(
    best_model,
    model_path
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("FINAL MODEL")
print("=" * 60)

print("Best Model :", best_name)
print("Best F1    :", round(best_f1, 4))
print("Saved Model:", model_path)
print("=" * 60)