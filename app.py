import streamlit as st
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Try to import TensorFlow/Keras safely
try:
    from tensorflow.keras.models import load_model
    nn = load_model("nn_model.keras")   # adjust filename if you saved as .h5
    nn_available = True
except ImportError:
    nn_available = False

# Load classical models
rf = joblib.load("rf_model.pkl")
dt = joblib.load("dt_model.pkl")
svm = joblib.load("svm_model.pkl")
logreg = joblib.load("logreg_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load test set for metrics
X_test = joblib.load("X_test.pkl")
y_test = joblib.load("y_test.pkl")

st.title("💧 Water Potability Prediction")

st.write("Enter water quality parameters to predict potability using different ML models.")

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

features = np.array([[ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]])
features_scaled = scaler.transform(features)

# Build model list dynamically
models = ["Random Forest", "Decision Tree", "SVM", "Logistic Regression"]
if nn_available:
    models.append("Neural Network")

model_choice = st.selectbox("Choose a model:", models)

if st.button("Predict"):
    if model_choice == "Random Forest":
        model = rf
        y_pred = model.predict(X_test)
        prediction = model.predict(features_scaled)[0]
    elif model_choice == "Decision Tree":
        model = dt
        y_pred = model.predict(X_test)
        prediction = model.predict(features_scaled)[0]
    elif model_choice == "SVM":
        model = svm
        y_pred = model.predict(X_test)
        prediction = model.predict(features_scaled)[0]
    elif model_choice == "Logistic Regression":
        model = logreg
        y_pred = model.predict(X_test)
        prediction = model.predict(features_scaled)[0]
    elif model_choice == "Neural Network" and nn_available:
        y_pred = (nn.predict(X_test) > 0.5).astype("int32")
        prediction = (nn.predict(features_scaled) > 0.5).astype("int32")[0][0]

    st.write("Prediction:", "✅ Potable" if prediction == 1 else "❌ Not Potable")

    st.subheader("📊 Model Metrics")
    st.write("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
    st.write("Precision:", round(precision_score(y_test, y_pred) * 100, 2), "%")
    st.write("Recall:", round(recall_score(y_test, y_pred) * 100, 2), "%")
    st.write("F1 Score:", round(f1_score(y_test, y_pred) * 100, 2), "%")
    st.write("ROC-AUC:", round(roc_auc_score(y_test, y_pred) * 100, 2), "%")
