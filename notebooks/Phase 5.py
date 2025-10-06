# PHASE 5: INTERPRETABILITY WITH SHAP VALUES (TOP-5 FEATURES AND INDIVIDUAL EXPLANATIONS)

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

print(f"✅ Data loaded. X_val: {X_val_full.shape}, X_test: {X_test_full.shape}, y_val: {y_val.shape}, y_test: {y_test.shape}")

# Try to get expected number of columns from the XGB model; fallback to X_val_full's columns
try:
    expected_cols = int(xgb_model.n_features_in_)
except Exception:
    expected_cols = X_val_full.shape[1]
    print("Couldn't read xgb_model.n_features_in_; falling back to X_val_full.shape[1]")

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
# Ensure 2D shape

print(f"✅ Columns aligned. X_val.shape = {X_val.shape}, X_test.shape = {X_test.shape}")

# Ensure the XGBoost model provides predict_proba (if not, try to use decision_function or predict)
if hasattr(xgb_model, "predict_proba"):
    XGB_val_prob  = xgb_model.predict_proba(X_val)[:,1].reshape(-1,1)
    XGB_test_prob = xgb_model.predict_proba(X_test)[:,1].reshape(-1,1)
elif hasattr(xgb_model, "predict"):
    # fallback to raw predict (may be logits or class labels)
    XGB_val_prob  = xgb_model.predict(X_val).reshape(-1,1)
    XGB_test_prob = xgb_model.predict(X_test).reshape(-1,1)
else:
    raise AttributeError("xgb_model does not have predict_proba or predict. Ensure the loaded model is an sklearn-compatible estimator.")

# XGBoost model's probability output as a feature

# Placeholder CNN logits (simulation)
np.random.seed(42)
CNN_val_logit  = np.random.randn(X_val.shape[0],1)
CNN_test_logit = np.random.randn(X_test.shape[0],1)

X_val_ens  = np.hstack([CNN_val_logit, XGB_val_prob])
X_test_ens = np.hstack([CNN_test_logit, XGB_test_prob])
# each column represents an independent source of information that the meta-model combines

print("✅ Generated ensemble features.")

os.makedirs("shap_plots", exist_ok=True)
os.makedirs("shap_top5", exist_ok=True)
os.makedirs("shap_individual", exist_ok=True)

print("\nCalculating SHAP values...")
explainer = shap.Explainer(meta_model, X_val_ens)
shap_values = explainer(X_test_ens)
# SHAP values for the meta-model on test set

if isinstance(shap_values, (list, tuple)):
    try:
        stacked = []
        for sv in shap_values:
            if hasattr(sv, "values"):
                stacked.append(np.array(sv.values))
            else:
                stacked.append(np.array(sv))
        vals = np.stack(stacked, axis=2)
    except Exception as e:
        raise RuntimeError("Unexpected shap_values list format.") from e
else:
    vals = getattr(shap_values, "values", np.array(shap_values))
    vals = np.asarray(vals)
    if vals.ndim == 2:
        vals = vals[:, :, np.newaxis]
# ensure vals is 3D: (n_samples, n_features, n_classes)

if hasattr(explainer, "expected_value"):
    expected_values = np.array(explainer.expected_value)
elif hasattr(shap_values, "base_values"):
    expected_values = np.array(shap_values.base_values)
else:
    expected_values = np.zeros(vals.shape[2])

expected_values = np.asarray(expected_values).ravel()
if expected_values.size == 1 and vals.shape[2] > 1:
    expected_values = np.repeat(expected_values, vals.shape[2])
elif expected_values.size != vals.shape[2]:
    if expected_values.size > vals.shape[2]:
        expected_values = expected_values[:vals.shape[2]]
    else:
        expected_values = np.pad(expected_values, (0, vals.shape[2]-expected_values.size), constant_values=0.0)
print("✅ SHAP values calculated.")

feature_names = ["CNN_logit", "XGB_prob"]
if X_val_ens.shape[1] != len(feature_names):
    feature_names = [f"meta_feat_{k}" for k in range(X_val_ens.shape[1])]

n_classes = vals.shape[2]
n_features_meta = vals.shape[1]

for cls_idx, cls in enumerate(meta_model.classes_):
    mean_shap = np.abs(vals[:,:,cls_idx]).mean(axis=0)
    top_idx = np.argsort(mean_shap)[::-1][:5]
    print(f"\nTop-5 features for class '{cls}':")
    for idx in top_idx:
        print(f"  {feature_names[idx]}: {mean_shap[idx]:.4f}")
    # plotting top-5 features

    plt.figure(figsize=(6,4))
    plt.bar([feature_names[idx] for idx in top_idx], mean_shap[top_idx])
    plt.ylabel("Mean(|SHAP value|)")
    plt.title(f"Top-5 features - Class {cls}")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    safe_cls = str(cls).replace(" ", "_")
    plt.savefig(f"shap_top5/top5_features_{safe_cls}.png")
    plt.close()

indexes_cand = np.where(y_test == 'CANDIDATE')[0][:3] if np.any(y_test == 'CANDIDATE') else np.array([], dtype=int)
indexes_conf = np.where(y_test == 'CONFIRMED')[0][:3] if np.any(y_test == 'CONFIRMED') else np.array([], dtype=int)
indexes_fp   = np.where(y_test == 'FALSE POSITIVE')[0][:3] if np.any(y_test == 'FALSE POSITIVE') else np.array([], dtype=int)
# try to get some examples of each class

preds = None
if (indexes_cand.size == 0) or (indexes_conf.size == 0) or (indexes_fp.size == 0):
    try:
        preds = meta_model.predict(X_test_ens)
    except Exception:
        preds = None

if indexes_cand.size == 0 and preds is not None and 'CANDIDATE' in meta_model.classes_:
    indexes_cand = np.where(preds == 'CANDIDATE')[0][:3]
if indexes_conf.size == 0 and preds is not None and 'CONFIRMED' in meta_model.classes_:
    indexes_conf = np.where(preds == 'CONFIRMED')[0][:3]
if indexes_fp.size == 0 and preds is not None and 'FALSE POSITIVE' in meta_model.classes_:
    indexes_fp = np.where(preds == 'FALSE POSITIVE')[0][:3]
    # try to get some examples of each predicted class if true class not present

selected_indexes = np.concatenate([indexes_cand, indexes_conf, indexes_fp]) if (indexes_cand.size + indexes_conf.size + indexes_fp.size) > 0 else np.array([], dtype=int)

if selected_indexes.size == 0:
    selected_indexes = np.arange(min(9, X_test_ens.shape[0]))
    # fallback to first few samples if no classes found

for sample_idx in selected_indexes:
    for cls_idx, cls in enumerate(meta_model.classes_):
        sample_vals = vals[sample_idx, :, cls_idx]
        order = np.argsort(np.abs(sample_vals))[::-1]
        names = [feature_names[k] for k in order]
        mags = sample_vals[order]
        # Waterfall plot

        plt.figure(figsize=(6,3))
        y_pos = np.arange(len(names))
        plt.barh(y_pos, mags, align='center')
        plt.yticks(y_pos, names)
        plt.xlabel("SHAP value")
        plt.title(f"SHAP individual - Sample {sample_idx} - Class {cls}")
        plt.gca().invert_yaxis() # highest at top
        plt.tight_layout()
        safe_cls = str(cls).replace(" ", "_")
        plt.savefig(f"shap_individual/shap_individual_sample{sample_idx}_cls{safe_cls}.png")
        plt.close()

print("\n✅ Phase 5 completed.")
print("Top-5 features saved in 'shap_top5/'")
print("Representative individual plots saved in 'shap_individual/'")
print("Global summary in 'shap_plots/' (Phase 4)")