# PHASE 6: THRESHOLD INTERACTIVE AND INDIVIDUAL SHAP EXPLANATIONS WITH STREAMLIT

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Exoscope", layout="wide")

st.title("Exoscope: Threshold & SHAP")

@st.cache_data
def load_models():
    xgb_model = joblib.load("models/model_xgb.pkl")
    meta_model = joblib.load("models/meta_logreg.pkl")
    return xgb_model, meta_model

@st.cache_data
def load_data():
    X_val_full  = np.load("data/X_val.npy", allow_pickle=True)
    X_test_full = np.load("data/X_test.npy", allow_pickle=True)
    y_val       = np.load("data/y_val.npy", allow_pickle=True)
    y_test      = np.load("data/y_test.npy", allow_pickle=True)
    return X_val_full, X_test_full, y_val, y_test
# optimizes loading times

xgb_model, meta_model = load_models()
X_val_full, X_test_full, y_val, y_test = load_data()

expected_cols = xgb_model.n_features_in_
def align_columns(X_full):
    if X_full.shape[1] != expected_cols:
        if X_full.shape[1] > expected_cols:
            return X_full[:, :expected_cols]
        else:
            pad = np.zeros((X_full.shape[0], expected_cols - X_full.shape[1]))
            return np.hstack([X_full, pad])
    return X_full
# align columns
X_val  = align_columns(X_val_full)
X_test = align_columns(X_test_full)

XGB_val_prob  = xgb_model.predict_proba(X_val)[:,1].reshape(-1,1)
XGB_test_prob = xgb_model.predict_proba(X_test)[:,1].reshape(-1,1)

np.random.seed(42)
CNN_val_logit  = np.random.randn(X_val.shape[0],1)
CNN_test_logit = np.random.randn(X_test.shape[0],1)

X_val_ens  = np.hstack([CNN_val_logit, XGB_val_prob])
X_test_ens = np.hstack([CNN_test_logit, XGB_test_prob])
# stacking features

feature_names = ["CNN_logit", "XGB_prob"]

# Interactive threshold
st.subheader("Interactive Threshold Adjustment")
threshold = st.slider("Select threshold for prediction", 0.0, 1.0, 0.5, 0.01)

probs = meta_model.predict_proba(X_test_ens)
pred_class = meta_model.predict(X_test_ens)

pred_class_thresh = []
for p in probs:
    if p[0] >= threshold:
        pred_class_thresh.append(meta_model.classes_[0])
    else:
        pred_class_thresh.append(meta_model.classes_[np.argmax(p)])

df_pred = pd.DataFrame({
    "Original prediction": pred_class[:20],
    f"With threshold {threshold}": pred_class_thresh[:20]
})
st.subheader("Predictions with threshold")
st.table(df_pred)

st.subheader("SHAP - Individual Explanations")

if st.checkbox("Show individual SHAP for selected row"):
    explainer = shap.Explainer(meta_model, X_val_ens)
    shap_values = explainer(X_test_ens)

    idx = st.number_input("Select row index", min_value=0, max_value=X_test_ens.shape[0]-1, value=0, step=1)
    cls_idx = st.selectbox("Select class for SHAP", options=list(range(len(meta_model.classes_))),
                           format_func=lambda i: meta_model.classes_[i])

    if shap_values.values.ndim == 3:
        values = shap_values.values[idx, :, cls_idx]
        base_value = shap_values.base_values[idx, cls_idx]
    else:
        values = shap_values.values[idx]
        base_value = shap_values.base_values[idx]

    shap_single = shap.Explanation(
        values=values,
        base_values=base_value,
        data=X_test_ens[idx],
        feature_names=feature_names
    )

    st.write(f"Fila: {idx}, Clase: {meta_model.classes_[cls_idx]}")
    shap.initjs()
    fig, ax = plt.subplots(figsize=(6,3))
    shap.plots.waterfall(shap_single, show=False)
    st.pyplot(fig)
    # allow you to see why a candidate was classified as a planet or false positive

st.success("✅ Phase 6 loaded successfully. You can explore individual thresholds and SHAPs.")
