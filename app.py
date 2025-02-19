import streamlit as st
import joblib
import pandas as pd

# Load Model
model = joblib.load("models/model.pkl")

st.title("Medical Insurance Cost Predictor")

# Input Fields
age = st.number_input("Age", min_value=18, max_value=100, value=30)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
children = st.number_input("Number of Children", min_value=0, max_value=5, value=0)
sex = st.radio("Sex", ["Male", "Female"])
smoker = st.radio("Smoker", ["Yes", "No"])
region = st.selectbox("Region", ["Northeast", "Northwest", "Southeast", "Southwest"])

# Convert inputs into a DataFrame (required for ColumnTransformer)
input_data = pd.DataFrame(
    {
        "age": [age],
        "sex": [sex],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker],
        "region": [region],
    }
)

if st.button("Predict"):
    prediction = model.predict(input_data)
    st.success(f"Estimated Insurance Cost: ${prediction[0]:.2f}")
