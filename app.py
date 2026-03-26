import streamlit as st
import numpy as np
import joblib

# Load trained model and scaler
model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("💧 Water Potability Prediction")

st.write("Enter water quality parameters to predict if water is potable.")

# Input fields
ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0)
hardness = st.number_input("Hardness", min_value=0.0, value=200.0)
solids = st.number_input("Solids", min_value=0.0, value=10000.0)
chloramines = st.number_input("Chloramines", min_value=0.0, value=7.0)
sulfate = st.number_input("Sulfate", min_value=0.0, value=300.0)
conductivity = st.number_input("Conductivity", min_value=0.0, value=400.0)
organic_carbon = st.number_input("Organic Carbon", min_value=0.0, value=10.0)
trihalomethanes = st.number_input("Trihalomethanes", min_value=0.0, value=60.0)
turbidity = st.number_input("Turbidity", min_value=0.0, value=4.0)

# Collect inputs
features = np.array([[ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]])

# Scale features
features_scaled = scaler.transform(features)

# Predict
prediction = model.predict(features_scaled)[0]
st.write("Prediction:", "✅ Potable" if prediction==1 else "❌ Not Potable")
