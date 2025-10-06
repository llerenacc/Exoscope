# PHASE 7: STREAMLIT APP FOR PREDICTIONS AND SHAP EXPLANATIONS

import os
import joblib
import pandas as pd
import streamlit as st
import shap
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

XGB_MODEL_PATH = os.path.join(MODELS_DIR, "model_xgb.pkl")
META_MODEL_PATH = os.path.join(MODELS_DIR, "meta_logreg_multiclass.pkl")

@st.cache_data
def load_models():
    xgb_model = joblib.load(XGB_MODEL_PATH)
    meta_model = joblib.load(META_MODEL_PATH)
    return xgb_model, meta_model

xgb_model, meta_model = load_models()

st.title("Exoscope: Exoplanet predictions")
st.write("Upload a CSV with the feature columns to get predictions.")

uploaded_file = st.file_uploader("Select a CSV", type=["csv"])
if uploaded_file:
    input_data = pd.read_csv(uploaded_file)

    st.write("Data preview:")
    st.dataframe(input_data.head())

    try:
        meta_input = input_data

        xgb_pred = xgb_model.predict_proba(meta_input)
        xgb_pred_class = xgb_model.predict(meta_input)

        st.write("XGB prediction (probabilidades):")
        st.dataframe(pd.DataFrame(xgb_pred, columns=["Not exoplanet", "Exoplanet"]))

        st.write("XGB prediction (class):")
        st.dataframe(pd.DataFrame(xgb_pred_class, columns=["Prediction"]))

        meta_pred = meta_model.predict(meta_input)
        st.write("Meta Model prediction:")
        st.dataframe(pd.DataFrame(meta_pred, columns=["Meta prediction"]))
        # assuming meta_model is a classifier with classes [0, 1, 2]

        try:
            explainer = shap.LinearExplainer(meta_model, meta_input, feature_perturbation="interventional")
            shap_values = explainer.shap_values(meta_input)

            st.write("SHAP Values:")
            st.dataframe(pd.DataFrame(shap_values, columns=meta_input.columns))

            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.write("Shap summary plot:")
            shap.summary_plot(shap_values, meta_input, show=False)
            st.pyplot(bbox_inches='tight')

        except Exception as e:
            st.warning(f"Could not generate SHAP: {e}")
            # continue without SHAP

    except Exception as e:
        st.error(f"Error in predictions: {e}")
        # handle error gracefully