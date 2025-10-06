# PHASE 2.5 — GENERATES AND SAVES PROCESSED FEATURES FROM KEPLER koi processed.csv

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/kepler_koi_processed.csv")
print("✅ Loaded dataset:", df.shape)

print("\nTypes of columns:")
print(df.dtypes.head(10))

target_col = None # Automatically detect target column
for col in df.columns:
    if "disposition" in col.lower() or "label" in col.lower() or "target" in col.lower():
        target_col = col
        break

if not target_col:
    raise ValueError("❌ The system could not find a specific column. Check CSV.")

print(f"\nTarget column automatically detected: {target_col}")

X = df.drop(columns=[target_col])
y = df[target_col]
# Ensure target is binary

X_numeric = X.select_dtypes(include=["number"])
non_numeric = X.columns.difference(X_numeric.columns)
if len(non_numeric) > 0:
    print(f"\nNon-numeric columns removed: {list(non_numeric)}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric) # prevents variables with large magnitudes from dominating others.

X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

os.makedirs("data", exist_ok=True)
np.save("data/processed_features.npy", X_scaled)
np.save("data/X_train.npy", X_train)
np.save("data/y_train.npy", y_train)
np.save("data/X_val.npy", X_val)
np.save("data/y_val.npy", y_val)
np.save("data/X_test.npy", X_test)
np.save("data/y_test.npy", y_test)
joblib.dump(scaler, "data/scaler.pkl")

print("\n✅ Processed and split numeric data stored in 'data/'")
print("Sizes:")
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
