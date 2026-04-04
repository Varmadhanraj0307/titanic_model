import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open('model.pkl', 'rb'))

st.title("🚢 Titanic Survival Prediction App")

st.write("Enter passenger details:")

# User inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["Male", "Female"])
age = st.slider("Age", 1, 80)
fare = st.number_input("Fare", value=50.0)

# Convert inputs
sex = 1 if sex == "Male" else 0

# Prediction
if st.button("Predict"):
    input_data = np.array([[pclass, sex, age, fare]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Passenger Survived")
    else:
        st.error("❌ Passenger Did Not Survive")
