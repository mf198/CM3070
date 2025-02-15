import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier


# ✅ Step 1: Load Data
df = pd.read_csv("ccfd/data/creditcard.csv")  # Replace with your file path

# ✅ Step 2: Drop 'Time' Column (Not Useful for Training)
df = df.drop(columns=["Time"])

# Apply scaling to the 'Amount' feature
df['Amount'] = StandardScaler().fit_transform(df[['Amount']])

# ✅ Step 3: Separate Features (X) and Target (y)
X = df.drop(columns=["Class"])  # Features
y = df["Class"]  # Target (Fraud or Not Fraud)

# ✅ Step 4: Split the Dataset into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"X_train\n: {X_test.head(2)}")
print(f"y_train\n: {X_test.head(2)}")
print(f"X_test\n: {X_train.head(2)}")
print(f"y_test\n: {X_train.head(2)}")

# ✅ Step 5: Apply SMOTE to Handle Class Imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# ✅ Step 7: Train Logistic Regression Model
#model = LogisticRegression()
#model.fit(X_train_resampled, y_train_resampled)
model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
model.fit(X_train_resampled, y_train_resampled)


# ✅ Step 8: Make Predictions
y_pred = model.predict(X_test)

# ✅ Step 9: Evaluate Model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

# ✅ Step 10: Print Evaluation Results
print("\nModel Evaluation Results:")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")
print(f"ROC AUC   : {roc_auc:.4f}")
