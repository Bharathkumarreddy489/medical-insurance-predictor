import joblib
import pandas as pd

# Load model
model = joblib.load("models/model.pkl")

# Define function for prediction
def predict_insurance_cost(input_data):
    # Ensure categorical columns use the correct format
    sex_map = {0: "male", 1: "female"}
    smoker_map = {0: "no", 1: "yes"}
    region_map = {0: "southwest", 1: "southeast", 2: "northwest", 3: "northeast"}

    input_dict = {
        "age": input_data[0],
        "sex": sex_map[input_data[1]],  # Convert numerical input back to string
        "bmi": input_data[2],
        "children": input_data[3],
        "smoker": smoker_map[input_data[4]],  # Convert numerical input back to string
        "region": region_map[input_data[5]],  # Convert numerical input back to string
    }

    # Convert input to DataFrame
    input_df = pd.DataFrame([input_dict])

    # Predict
    prediction = model.predict(input_df)
    return prediction[0]

# Example test input (Ensure the values match the format used during training)
if __name__ == "__main__":
    sample_input = [30, 1, 25.4, 1, 0, 2]  # [age, sex (1=female), bmi, children, smoker (0=no), region (2=northwest)]
    print("Predicted Insurance Cost:", predict_insurance_cost(sample_input))
