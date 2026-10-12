from pathlib import Path
import json
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = ROOT / "data" / "processed" / "features.csv"
MODEL_FILE = ROOT / "models" / "book_classifier.joblib"
REPORT_FOLDER = ROOT / "report"
METRICS_FILE = REPORT_FOLDER / "metrics.json"
MATRIX_FILE = REPORT_FOLDER / "confusion_matrix.png"

REPORT_FOLDER.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(DATA_FILE)

X = df.drop(columns=["category", "target"])
y = df["target"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Load model and train
model = joblib.load(MODEL_FILE)
model.fit(X_train, y_train)

# Predictions and evaluation
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Holdout accuracy:", accuracy)
print(classification_report(y_test, predictions))

# Save confusion matrix plot
matrix = confusion_matrix(y_test, predictions)
plt.figure(figsize=(6, 5))
sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig(MATRIX_FILE)
plt.close()

# Cross validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
print("CV Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())

# Save metrics to json
results = {
    "holdout_accuracy": float(accuracy),
    "cv_mean": float(cv_scores.mean()),
    "cv_scores": list(cv_scores)
}

with open(METRICS_FILE, "w") as f:
    json.dump(results, f, indent=4)

print("Saved metrics and matrix successfully!")