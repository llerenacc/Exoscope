# PHASE 2: BASELINE XGBOOST MODEL

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    f1_score
)
import joblib

df_path = "data/kepler_koi_processed.csv"
if not os.path.exists(df_path):
    raise FileNotFoundError(f"❌ File not found: {df_path}") # make sure to run Phase 1 first

df = pd.read_csv(df_path)
print("✅ Data loaded successfully.")
print("Available columns:", df.columns.tolist())

label_map = {'confirmed': 1, 'false positive': 0, 'candidate': 2} # avoid inconsistencies; categories written in lowercase

if 'koi_disposition' not in df.columns:
    raise KeyError("❌ Column 'koi_disposition' was not found in the dataset.")

df['koi_disposition'] = df['koi_disposition'].astype(str).str.lower()
y = df['koi_disposition'].map(label_map)

# Filtering valid rows and X/y separation
mask_valid = y.notnull()
X = df.loc[mask_valid].drop('koi_disposition', axis=1)
y = y.loc[mask_valid]

cat_cols = [c for c in ['kepoi_name', 'kepler_name', 'koi_pdisposition', 'koi_tce_delivname'] if c in X.columns]
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str)) # ensure all data is string type

X = X.fillna(X.mean(numeric_only=True))

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42 # 70% train, 30% temp (validation, testing)
) # REMEMBER: phase used to evaluate initial predictive capacity
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

print("✅ Sizes:", X_train.shape, X_val.shape, X_test.shape)

y_train = y_train.map(lambda x: 1 if x == 1 else 0)
y_val   = y_val.map(lambda x: 1 if x == 1 else 0)
y_test  = y_test.map(lambda x: 1 if x == 1 else 0)
# generates a binary baseline model where 1 = confirmed, 0 = not confirmed (false positive + candidate)

print("Label distribution:")
print("Train:", np.bincount(y_train))
print("Val:", np.bincount(y_val))
print("Test:", np.bincount(y_test))

neg, pos = np.bincount(y_train)
scale_pos_weight = neg / pos if pos != 0 else 1.0
print("scale_pos_weight =", scale_pos_weight) # useful for unbalanced datasets

model_xgb = XGBClassifier( # Set up an XGBoost model w/
    objective="binary:logistic", # binary classification
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss", # logistic loss
    use_label_encoder=False,
    random_state=42
)

print("\n🔄 Training XGBoost model...")
model_xgb.fit(X_train, y_train)
print("✅ Training completed.")

y_pred = model_xgb.predict(X_test)
y_proba = model_xgb.predict_proba(X_test)[:, 1]

f1 = f1_score(y_test, y_pred)
pr_auc = average_precision_score(y_test, y_proba)

print(f"\nF1 = {f1:.4f}")
print(f"PR-AUC = {pr_auc:.4f}\n")

# Confusion Matrix for true positive/negative analysis
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted values")
plt.ylabel("Actual values")
plt.title("Confusion Matrix")
plt.show()

# Precision-Recall Curve
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
plt.plot(recall, precision, marker='.')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show()
# shows the trade-off between precision and recall values

# 10 most important features
importances = model_xgb.feature_importances_
indexes = np.argsort(importances)[-10:]
plt.barh(range(len(indexes)), importances[indexes], align='center')
plt.yticks(range(len(indexes)), [X.columns[i] for i in indexes])
plt.xlabel("Importance")
plt.title("Top 10 most important features")
plt.show()

os.makedirs("models", exist_ok=True)
joblib.dump(model_xgb, "kepler_koi_processed.csv")

print("\n💾 Model saved in 'models/xgb_baseline.pkl'")