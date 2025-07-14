import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('heart_model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title("🫀 Heart Disease Prediction App")
st.write("This app uses a trained ML model to predict the risk of heart disease.")

# Input form
st.sidebar.header("Patient Input")

def get_input():
    age = st.sidebar.slider('Age', 29, 77, 55)
    sex = st.sidebar.selectbox('Sex (0 = Female, 1 = Male)', [0, 1])
    cp = st.sidebar.selectbox('Chest Pain Type (0-3)', [0, 1, 2, 3])
    trestbps = st.sidebar.slider('Resting BP', 90, 200, 120)
    chol = st.sidebar.slider('Cholesterol (mg/dl)', 100, 600, 200)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 (1 = Yes)', [0, 1])
    restecg = st.sidebar.selectbox('Resting ECG (0-2)', [0, 1, 2])
    thalach = st.sidebar.slider('Max Heart Rate', 70, 210, 150)
    exang = st.sidebar.selectbox('Exercise Induced Angina (1 = Yes)', [0, 1])
    oldpeak = st.sidebar.slider('ST Depression', 0.0, 6.5, 1.0, 0.1)
    slope = st.sidebar.selectbox('Slope of ST Segment (0-2)', [0, 1, 2])
    ca = st.sidebar.selectbox('Major Vessels (0-4)', [0, 1, 2, 3, 4])
    thal = st.sidebar.selectbox('Thal (1 = Normal, 2 = Fixed, 3 = Reversible)', [1, 2, 3])

    data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
        'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
        'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }
    return pd.DataFrame(data, index=[0])

input_df = get_input()
scaled_input = scaler.transform(input_df)
prediction = model.predict(scaled_input)
prediction_proba = model.predict_proba(scaled_input)

# Output
st.subheader("Prediction Result")
if prediction[0] == 1:
    st.error("⚠️ High Risk of Heart Disease")
else:
    st.success("✅ Low Risk of Heart Disease")

st.subheader("Prediction Probability")
st.write(f"🟥 Heart Disease: {prediction_proba[0][1]:.2f}")
st.write(f"🟩 No Heart Disease: {prediction_proba[0][0]:.2f}")
