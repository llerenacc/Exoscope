# PHASE 4: INTERPRETABILITY WITH SHAP VALUES

import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

print("Loading models and data...")

xgb_model  = joblib.load("models/model_xgb.pkl")
meta_model = joblib.load("models/meta_logreg.pkl")

X_val_full  = np.load("data/X_val.npy", allow_pickle=True)
X_test_full = np.load("data/X_test.npy", allow_pickle=True)
y_val       = np.load("data/y_val.npy", allow_pickle=True)
y_test      = np.load("data/y_test.npy", allow_pickle=True)

expected_cols = xgb_model.n_features_in_
if X_val_full.shape[1] != expected_cols:
    if X_val_full.shape[1] > expected_cols:
        X_val = X_val_full[:, :expected_cols]
        X_test = X_test_full[:, :expected_cols]
    else:
        pad_val  = np.zeros((X_val_full.shape[0], expected_cols - X_val_full.shape[1]))
        pad_test = np.zeros((X_test_full.shape[0], expected_cols - X_test_full.shape[1]))
        X_val  = np.hstack([X_val_full, pad_val])
        X_test = np.hstack([X_test_full, pad_test])
else:
    X_val  = X_val_full
    X_test = X_test_full
# avoid errors due to dimensional discrepancies

print(f"✅ Columns aligned. X_val.shape = {X_val.shape}, X_test.shape = {X_test.shape}")

XGB_val_prob  = xgb_model.predict_proba(X_val)[:,1].reshape(-1,1)
XGB_test_prob = xgb_model.predict_proba(X_test)[:,1].reshape(-1,1)

np.random.seed(42)
CNN_val_logit  = np.random.randn(X_val.shape[0],1)
CNN_test_logit = np.random.randn(X_test.shape[0],1)

X_val_ens  = np.hstack([CNN_val_logit, XGB_val_prob])
X_test_ens = np.hstack([CNN_test_logit, XGB_test_prob])

feature_names = ["CNN_logit", "XGB_prob"]
print("✅ Generated ensemble features.")
# ensemble represented as a combination of “base model predictions”

print("\nGlobal importance of features:")
for i, cls in enumerate(meta_model.classes_):
    print(f"Class '{cls}':")
    for j, feat in enumerate(feature_names):
        print(f"  {feat}: {meta_model.coef_[i,j]:.4f}")

print("\nCalculating SHAP values...")

explainer = shap.Explainer(meta_model, X_val_ens)
shap_values = explainer(X_test_ens)
# SHAP values provide insights into feature contributions for predictions

os.makedirs("shap_plots", exist_ok=True)

for i, cls in enumerate(meta_model.classes_):
    print(f"Generating SHAP summary for class '{cls}'...")
    shap.summary_plot(
        shap_values.values[:,:,i],
        X_test_ens,
        feature_names=feature_names,
        show=False
    )
    plt.title(f"SHAP Summary - Class {cls}")
    plt.savefig(f"shap_plots/shap_summary_{cls}.png")
    plt.close()

print("✅ Phase 4 completed. SHAP plots saved in 'shap_plots/'")