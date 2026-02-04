from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import os

# 1. Initialize FastAPI app
app = FastAPI(title="Diabetes Prediction API")

# 2. Load the trained model and scaler
# Make sure these files are in the same folder as main.py
MODEL_PATH = "diabetes_classifier.pkl"
SCALER_PATH = "scaler.pkl"

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    classifier = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model and Scaler loaded successfully.")
else:
    print("CRITICAL ERROR: 'diabetes_classifier.pkl' or 'scaler.pkl' not found!")

# 3. DEFINE THE CLASS FIRST (Crucial for fixing your error)
class DiabetesInput(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

# 4. Root Endpoint
@app.get("/")
def home():
    return {
        "status": "API is running",
        "message": "Visit /docs for the automated testing UI"
    }

# 5. Prediction Endpoint
@app.post("/predict")
def predict(data: DiabetesInput):
    try:
        # Extract values in the exact order defined in the class
        # .model_dump() replaces the older .dict() method
        input_dict = data.model_dump()
        input_values = list(input_dict.values())

        # Convert to numpy array and reshape (1 row, N columns)
        input_array = np.array(input_values).reshape(1, -1)

        # Standardize the data using the same scaler from your training
        std_data = scaler.transform(input_array)

        # Make prediction
        prediction = classifier.predict(std_data)

        # Map to human-readable result
        # 1 usually means Diabetic, 0 means Not Diabetic in this dataset
        result = "The person is diabetic" if prediction[0] == 1 else "The person is not diabetic"

        return {
            "prediction": int(prediction[0]),
            "result": result
        }

    except Exception as e:
        # Provide a clear error message if something fails
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

# To run the server, use this command in your terminal:

# uvicorn main:app --reload