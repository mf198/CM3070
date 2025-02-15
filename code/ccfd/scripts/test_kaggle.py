
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# Load dataset
df = pd.read_csv('ccfd/data/creditcard.csv')

# Display class distribution
print(df['Class'].value_counts())
sns.countplot(x=df['Class'])
plt.title("Class Distribution")
plt.show()

# Extract features & target
X = df.drop(['Class'], axis=1)
y = df['Class']

# Standardize 'Amount' column
scaler = StandardScaler()
X['Amount'] = scaler.fit_transform(X[['Amount']])

# Train-test split before SMOTE (to prevent data leakage)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE to balance classes
smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Check new class distribution
print(pd.Series(y_train_resampled).value_counts())

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train_resampled, y_train_resampled)

# Train XGBoost
xgb_model = XGBClassifier(scale_pos_weight=5, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train_resampled, y_train_resampled)

# Predictions
rf_preds = rf_model.predict(X_test)
xgb_preds = xgb_model.predict(X_test)

# Print classification reports
print("Random Forest Model")
print(classification_report(y_test, rf_preds))

print("XGBoost Model")
print(classification_report(y_test, xgb_preds))

# Compute AUC-ROC scores
rf_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:,1])
xgb_auc = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:,1])

print(f"Random Forest AUC-ROC: {rf_auc:.4f}")
print(f"XGBoost AUC-ROC: {xgb_auc:.4f}")

# Plot Confusion Matrix for XGBoost
sns.heatmap(confusion_matrix(y_test, xgb_preds), annot=True, fmt='d', cmap='coolwarm')
plt.title("Confusion Matrix - XGBoost")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()