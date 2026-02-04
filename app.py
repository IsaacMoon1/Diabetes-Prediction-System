import streamlit as st
import numpy as np
import joblib

# Load the model and scaler
classifier = joblib.load("diabetes_classifier.pkl")
scaler = joblib.load("scaler.pkl")

# Page configuration
st.set_page_config(page_title="Diabetes Prediction Pro", page_icon="🏥")

st.title(" Diabetes Prediction Dashboard")
st.markdown("Enter the patient's metrics below to calculate the risk.")

# Create two columns for inputs
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, step=1, value=0)
    glucose = st.number_input("Glucose Level", min_value=0.0, value=120.0)
    blood_pressure = st.number_input("Blood Pressure", min_value=0.0, value=70.0)
    skin_thickness = st.number_input("Skin Thickness", min_value=0.0, value=20.0)

with col2:
    insulin = st.number_input("Insulin Level", min_value=0.0, value=80.0)
    bmi = st.number_input("BMI", min_value=0.0, value=25.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.5)
    age = st.number_input("Age", min_value=0, step=1, value=30)

# Prediction Logic
if st.button("Run Diagnostic Analysis"):
    # 1. Prepare data
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                           insulin, bmi, dpf, age]])
    
    # 2. Scale
    std_data = scaler.transform(input_data)
    
    # 3. Predict
    prediction = classifier.predict(std_data)
    
    # 4. Show Result
    st.divider()
    if prediction[0] == 1:
        st.error("### Result: The person is likely Diabetic")
    else:
        st.success("### Result: The person is likely NOT Diabetic")