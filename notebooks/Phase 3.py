# PHASE 3: ENSEMBLE LEARNING WITH LOGISTIC REGRESSION META-MODEL

import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve
from sklearn.preprocessing import label_binarize

# Retrieve the preprocessed sets in Phase 2.5.
print("Loading processed datasets")
X_train = np.load("data/X_train.npy")
X_val   = np.load("data/X_val.npy")
X_test  = np.load("data/X_test.npy")

y_train = np.load("data/y_train.npy", allow_pickle=True)
y_val   = np.load("data/y_val.npy", allow_pickle=True)
y_test  = np.load("data/y_test.npy", allow_pickle=True)

print("✅ Data loaded successfully.")

try:
    cnn_logits_val   = np.load("models/cnn_logits_val.npy")
    xgb_probs_val    = np.load("models/xgb_probs_val.npy")
    cnn_logits_test  = np.load("models/cnn_logits_test.npy")
    xgb_probs_test   = np.load("models/xgb_probs_test.npy")
    print("CNN and XGBoost outputs loaded.")
    # if these files do not exist, simulate some data for demonstration purposes.
except FileNotFoundError:
    print("No previous outputs found. Generating simulations (for testing only)...")
    np.random.seed(42)
    cnn_logits_val   = np.random.rand(len(y_val), 1)
    xgb_probs_val    = np.random.rand(len(y_val), 1)
    cnn_logits_test  = np.random.rand(len(y_test), 1)
    xgb_probs_test   = np.random.rand(len(y_test), 1)
    # in a real scenario, replace the above with actual model outputs.
X_val_ens  = np.concatenate([cnn_logits_val, xgb_probs_val], axis=1)
X_test_ens = np.concatenate([cnn_logits_test, xgb_probs_test], axis=1)
# stack the outputs as features for the meta-model

print("Features of the created ensemble:")
print(f"Val:  {X_val_ens.shape}, Test: {X_test_ens.shape}")

meta_model = LogisticRegression(max_iter=1000, multi_class='ovr')
meta_model.fit(X_val_ens, y_val)
# train a logistic regression as meta-model on validation set outputs.

y_pred_meta       = meta_model.predict(X_test_ens)
y_pred_proba_meta = meta_model.predict_proba(X_test_ens)
# get predictions and probabilities on the test set.

classes = meta_model.classes_
y_test_bin = label_binarize(y_test, classes=classes)


print("\nMeta-model metrics per class:")

for i, c in enumerate(classes):
    y_true_i = y_test_bin[:, i]
    y_score_i = y_pred_proba_meta[:, i]

    pr_auc = average_precision_score(y_true_i, y_score_i)
    
    precision, recall, thresholds = precision_recall_curve(y_true_i, y_score_i)
    
    f1 = f1_score(y_true_i, (y_score_i >= 0.5).astype(int))  # umbral 0.5 por defecto
    
    recall_at_08 = recall[np.argmax(precision >= 0.8)] if np.any(precision >= 0.8) else 0
    
    print(f"\nClase '{c}':")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"Recall@0.8 Precision: {recall_at_08:.4f}")

os.makedirs("models", exist_ok=True)
joblib.dump(meta_model, "models/meta_logreg.pkl")
np.save("models/ensemble_val.npy", X_val_ens)
np.save("models/ensemble_test.npy", X_test_ens)

print("\n💾 Meta-model saved in 'models/meta_logreg.pkl'")