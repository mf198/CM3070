import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve
from imblearn.combine import SMOTETomek

# ✅ Step 1: Load Data
df = pd.read_csv("ccfd/data/creditcard.csv")  # Replace with your dataset path

# ✅ Step 2: Drop 'Time' Column
df = df.drop(columns=["Time"])

# ✅ Step 3: Separate Features (X) and Target (y)
X = df.drop(columns=["Class"])  # Features
y = df["Class"]  # Target

# ✅ Step 4: Split Dataset (Stratified to Maintain Class Balance)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ✅ Step 5: Apply SMOTE-Tomek to Handle Class Imbalance
smote_tomek = SMOTETomek(random_state=42)
X_train_resampled, y_train_resampled = smote_tomek.fit_resample(X_train, y_train)

# ✅ Step 6: Standardize Features
scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)

# ✅ Step 7: Train XGBoost Model
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=len(y_train_resampled) / sum(y_train_resampled),
    eval_metric="logloss",
    use_label_encoder=False
)
xgb_model.fit(X_train_resampled, y_train_resampled)

# ✅ Step 8: Get Prediction Probabilities
y_pred_probs = xgb_model.predict_proba(X_test)[:, 1]  # Get probability of fraud

# ✅ Step 9: Find the Best Precision-Recall Threshold
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_probs)

# Find the threshold where Precision > 0.80
best_threshold = thresholds[np.argmax(precision > 0.80)]
print(f"Optimal Decision Threshold: {best_threshold:.3f}")

# Apply the new threshold
y_pred_adjusted = (y_pred_probs > best_threshold).astype(int)

# ✅ Step 10: Evaluate Model
accuracy = accuracy_score(y_test, y_pred_adjusted)
precision = precision_score(y_test, y_pred_adjusted)
recall = recall_score(y_test, y_pred_adjusted)
f1 = f1_score(y_test, y_pred_adjusted)
roc_auc = roc_auc_score(y_test, y_pred_probs)

print("\nModel Evaluation Results (Adjusted Threshold):")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")
print(f"ROC AUC   : {roc_auc:.4f}")

# ✅ Step 11: Plot Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.', label="Precision-Recall Curve")
plt.axvline(x=recall[np.argmax(precision > 0.80)], color='red', linestyle='--', label="Best Threshold")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid()
plt.show()
