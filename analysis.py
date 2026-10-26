from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, RocCurveDisplay, roc_auc_score

BASE = Path(__file__).parent
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)
data = pd.read_csv(BASE / "student_performance.csv")
features = ["study_hours_per_day", "attendance_percent", "previous_exam_score", "assignments_completed"]
X, y = data[features], data["passed"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)
pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]
print("Dataset rows:", len(data))
print(f"Accuracy: {accuracy_score(y_test, pred):.3f}")
print(f"ROC-AUC: {roc_auc_score(y_test, prob):.3f}")
print("\nClassification report:\n", classification_report(y_test, pred, zero_division=0))
results = X_test.copy()
results["actual_passed"] = y_test.values
results["predicted_passed"] = pred
results["predicted_pass_probability"] = prob.round(3)
results.to_csv(OUT / "test_predictions.csv", index=False)
importance = pd.DataFrame({"feature": features, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
importance.to_csv(OUT / "feature_importance.csv", index=False)
fig, ax = plt.subplots(figsize=(6,5))
ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["Not passed","Passed"], ax=ax, colorbar=False)
ax.set_title("Confusion Matrix — Student Pass Prediction")
fig.tight_layout(); fig.savefig(OUT / "confusion_matrix.png", dpi=150); plt.close(fig)
fig, ax = plt.subplots(figsize=(6,5))
RocCurveDisplay.from_predictions(y_test, prob, ax=ax)
ax.set_title("ROC Curve — Student Pass Prediction")
fig.tight_layout(); fig.savefig(OUT / "roc_curve.png", dpi=150); plt.close(fig)
fig, ax = plt.subplots(figsize=(7,5))
importance.sort_values("importance").plot(x="feature", y="importance", kind="barh", legend=False, ax=ax)
ax.set_title("Feature Importance"); ax.set_xlabel("Importance")
fig.tight_layout(); fig.savefig(OUT / "feature_importance.png", dpi=150); plt.close(fig)
print("\nFeature importance:\n", importance.to_string(index=False))
print("Outputs saved in:", OUT)
print("Educational demo using synthetic data only; do not use to make real decisions about students.")
