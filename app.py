import streamlit as st
import pickle
import numpy as np

# Load the model and scaler
with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Streamlit app title
st.title("Autism Detection Prediction")

# Input fields for the user to enter their details
age = st.number_input("Age", min_value=0, max_value=100, value=3)
gender = st.selectbox("Gender", ["Male", "Female"])
iq_score = st.number_input("IQ Score", min_value=0, max_value=200, value=70)
social_skills = st.slider("Social Skills Rating (1-10)", 1, 10, value=1)
communication_skills = st.slider("Communication Skills Rating (1-10)", 1, 10, value=1)
repetitive_behaviors = st.slider("Repetitive Behaviors Rating (1-10)", 1, 10, value=1)
sensory_sensitivity = st.slider("Sensory Sensitivity (1-10)", 1, 10, value=1)
eye_contact = st.slider("Eye Contact Frequency (1-10)", 1, 10, value=1)
sleep_issues = st.selectbox("Sleep Issues (0 = No, 1 = Yes)", [0, 1])
parental_involvement = st.selectbox("Parental Involvement (0 = No, 1 = Yes)", [0, 1])
family_history = st.selectbox("Family History of ASD (0 = No, 1 = Yes)", [0, 1])
motor_skills = st.slider("Motor Skills Rating (1-10)", 1, 10, value=1)

# Combine inputs into a feature array
def prepare_inputs():
    gender_val = 1 if gender == "Male" else 0  # Male: 1, Female: 0
    inputs = np.array([[age, gender_val, iq_score, social_skills, communication_skills,
                        repetitive_behaviors, sensory_sensitivity, eye_contact, sleep_issues,
                        parental_involvement, family_history, motor_skills]])
    # Scale the inputs
    inputs_scaled = scaler.transform(inputs)
    return inputs_scaled

# Submit button for prediction
if st.button("Predict Autism"):
    # Prepare the input data
    input_data = prepare_inputs()
    
    # Make prediction
    prediction = model.predict(input_data)

    # Display prediction result
    if prediction == 1:
        st.write("The model predicts: Autism Spectrum Disorder (ASD)")
    else:
        st.write("The model predicts: No Autism Spectrum Disorder (Non-ASD)")
