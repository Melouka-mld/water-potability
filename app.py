import streamlit as st
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Try to import TensorFlow safely
try:
    from tensorflow.keras.models import load_model
    nn = load_model("nn_model.keras")   # or "nn_model.h5"
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

# ---------------- UI Layout ----------------
st.set_page_config(page_title="Water Potability Prediction", layout="wide")

st.title("💧 Water Potability Prediction")
st.markdown("""
This app predicts whether water is **potable** or **not potable** using multiple machine learning models.  
Choose a model, enter water quality parameters, and view predictions along with evaluation metrics.
""")

# Sidebar for inputs
st.sidebar.header("🔧 Input Parameters")
ph = st.sidebar.number_input("pH", min_value=0.0, max_value=14.0, value=7.0)
hardness = st.sidebar.number_input("Hardness", min_value=0.0, value=200.0)
solids = st.sidebar.number_input("Solids", min_value=0.0, value=10000.0)
chloramines = st.sidebar.number_input("Chloramines", min_value=0.0, value=7.0)
sulfate = st.sidebar.number_input("Sulfate", min_value=0.0, value=300.0)
conductivity = st.sidebar.number_input("Conductivity", min_value=0.0, value=400.0)
organic_carbon = st.sidebar.number_input("Organic Carbon", min_value=0.0, value=10.0)
trihalomethanes = st.sidebar.number_input("Trihalomethanes", min_value=0.0, value=60.0)
turbidity = st.sidebar.number_input("Turbidity", min_value=0.0, value=4.0)

features = np.array([[ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]])
features_scaled = scaler.transform(features)

# Model selection
models = ["Random Forest", "Decision Tree", "SVM", "Logistic Regression"]
if nn_available:
    models.append("Neural Network")

model_choice = st.sidebar.selectbox("📌 Choose a model", models)

# ---------------- Prediction Section ----------------
st.subheader("🔮 Prediction Result")

if st.sidebar.button("Run Prediction"):
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

    st.success("✅ Potable" if prediction == 1 else "❌ Not Potable")

    # ---------------- Metrics Section ----------------
    st.subheader("📊 Model Performance Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred)*100:.2f}%")
    col2.metric("Precision", f"{precision_score(y_test, y_pred)*100:.2f}%")
    col3.metric("Recall", f"{recall_score(y_test, y_pred)*100:.2f}%")

    col4, col5 = st.columns(2)
    col4.metric("F1 Score", f"{f1_score(y_test, y_pred)*100:.2f}%")
    col5.metric("ROC-AUC", f"{roc_auc_score(y_test, y_pred)*100:.2f}%")
