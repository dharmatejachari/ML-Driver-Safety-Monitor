import pandas as pd
import os

# ============================================================
# LOAD DATA
# ============================================================

INPUT_FILE = "data/driver_behavior.csv"
OUTPUT_FILE = "analysis/driver_safety_dashboard.csv"

if not os.path.exists(INPUT_FILE):
    print("ERROR: Dataset not found.")
    exit()

df = pd.read_csv(INPUT_FILE)

print("Original dataset shape:", df.shape)

# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

df.columns = df.columns.str.strip()

# ============================================================
# CREATE DASHBOARD DATASET
# ============================================================

dashboard_df = df.copy()

# Add numeric event indicators
dashboard_df["Drowsy_Event"] = (
    dashboard_df["driver_state"] == "Drowsy"
).astype(int)

dashboard_df["Distracted_Event"] = (
    dashboard_df["driver_state"] == "Distracted"
).astype(int)

dashboard_df["Phone_Usage_Event"] = (
    dashboard_df["driver_state"] == "Phone_Usage"
).astype(int)

dashboard_df["Alert_Event"] = (
    dashboard_df["driver_state"] == "Alert"
).astype(int)

# ============================================================
# SAVE
# ============================================================

dashboard_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nDashboard dataset created successfully.")
print("Saved to:", OUTPUT_FILE)

print("\nDataset shape:", dashboard_df.shape)

print("\nDriver state distribution:")
print(dashboard_df["driver_state"].value_counts())

print("\nColumns:")
print(list(dashboard_df.columns))