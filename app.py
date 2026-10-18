
import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

st.title("Heart Disease Prediction System")
st.write("Machine Learning Fundamentals - Assignment 2")
st.write("Enter the following details to predict heart disease.")

st.info(
    "This application is for educational purposes only "
    "and is not intended for medical diagnosis."
)

# Load trained model
model_path = Path(__file__).resolve().parent / "svc_trained_model.pkl"

with open(model_path, "rb") as file:
    model = pickle.load(file)

# User inputs
chest_pain = st.selectbox(
    "Chest Pain Type",
    [
        "Asymptomatic",
        "Atypical Angina",
        "Non-Anginal Pain",
        "Typical Angina"
    ]
)

exercise_angina = st.selectbox(
    "Exercise-Induced Angina",
    ["No", "Yes"]
)

vessels = st.selectbox(
    "Number of Major Vessels",
    ["Zero", "One", "Two", "Three"]
)

thal = st.selectbox(
    "Thalassemia Test Result",
    ["Normal", "Fixed Defect", "Reversible Defect"]
)

# Numerical encoding based on training dataset
chest_pain_map = {
    "Asymptomatic": 0,
    "Atypical Angina": 1,
    "Non-Anginal Pain": 2,
    "Typical Angina": 3
}

exercise_angina_map = {
    "No": 0,
    "Yes": 1
}

vessels_map = {
    "One": 0,
    "Three": 1,
    "Two": 2,
    "Zero": 3
}

thal_map = {
    "Fixed Defect": 0,
    "Normal": 1,
    "Reversible Defect": 2
}

if st.button("Predict Heart Disease"):

    input_data = pd.DataFrame({
        "ChestPain": [chest_pain_map[chest_pain]],
        "ExerciseAngina": [exercise_angina_map[exercise_angina]],
        "Vessels": [vessels_map[vessels]],
        "Thal": [thal_map[thal]]
    })

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.warning("Prediction: Heart Disease Present")
    else:
        st.success("Prediction: No Heart Disease Predicted")

st.caption("Educational ML prediction application using UCI Heart Disease data.")
