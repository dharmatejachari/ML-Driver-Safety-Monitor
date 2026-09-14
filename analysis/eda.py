import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
df = pd.read_csv("data/driver_behavior.csv")

os.makedirs("analysis/plots", exist_ok=True)

# ============================================================
# 1. CLASS DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="driver_state")
plt.title("Driver Behavior Class Distribution")
plt.xlabel("Driver State")
plt.ylabel("Number of Samples")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("analysis/plots/class_distribution.png")
plt.close()

# ============================================================
# 2. EAR DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="driver_state", y="EAR")
plt.title("Eye Aspect Ratio by Driver State")
plt.xlabel("Driver State")
plt.ylabel("EAR")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("analysis/plots/ear_by_state.png")
plt.close()

# ============================================================
# 3. YAW DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="driver_state", y="yaw")
plt.title("Head Yaw by Driver State")
plt.xlabel("Driver State")
plt.ylabel("Yaw (degrees)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("analysis/plots/yaw_by_state.png")
plt.close()

# ============================================================
# 4. PITCH DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="driver_state", y="pitch")
plt.title("Head Pitch by Driver State")
plt.xlabel("Driver State")
plt.ylabel("Pitch (degrees)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("analysis/plots/pitch_by_state.png")
plt.close()

# ============================================================
# 5. CORRELATION HEATMAP
# ============================================================

numeric_columns = [
    "EAR",
    "eye_closure_duration",
    "yaw",
    "pitch",
    "phone_detected",
    "phone_confidence"
]

plt.figure(figsize=(9, 7))
sns.heatmap(
    df[numeric_columns].corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("analysis/plots/correlation_heatmap.png")
plt.close()

print("EDA completed successfully.")
print("Plots saved in: analysis/plots/")