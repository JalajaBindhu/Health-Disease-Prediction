from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
import os

app = FastAPI(title="Health Disease Prediction API")

@app.get("/")
def home():
    return {
        "message": "Health Disease Prediction API is running",
        "endpoints": [
            "/predict/heart",
            "/predict/diabetes",
            "/docs"
        ]
    }
# -----------------------
# Input Schemas
# -----------------------

class HeartInput(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int


class DiabetesInput(BaseModel):
    pregnancies: int
    glucose: int
    bloodpressure: int
    skinthickness: int
    insulin: int
    bmi: float
    dpf: float
    age: int


# -----------------------
# Prediction APIs
# -----------------------

@app.post("/predict/heart")
def predict_heart(data: HeartInput):

    input_df = pd.DataFrame([data.dict()], columns=heart_columns)
    prediction = heart_model.predict(input_df)[0]

    return {
        "disease": "Heart Disease",
        "prediction": int(prediction),
        "result": "Detected" if prediction == 1 else "Not Detected"
    }


@app.post("/predict/diabetes")
def predict_diabetes(data: DiabetesInput):
    try:
        input_df = pd.DataFrame([data.dict()])

        # Rename columns to match model training
        input_df = input_df.rename(columns={
            "pregnancies": "Pregnancies",
            "glucose": "Glucose",
            "bloodpressure": "BloodPressure",
            "skinthickness": "SkinThickness",
            "insulin": "Insulin",
            "bmi": "BMI",
            "dpf": "DiabetesPedigreeFunction",
            "age": "Age"
        })

        # Reorder columns
        input_df = input_df[diabetes_columns]

        prediction = diabetes_model.predict(input_df)[0]

        return {
            "disease": "Diabetes",
            "prediction": int(prediction),
            "result": "Detected" if prediction == 1 else "Not Detected"
        }
    except Exception as e:
        return {
            "error": str(e),
            "received_columns": list(input_df.columns),
            "expected_columns": diabetes_columns
        }



# -----------------------
# Absolute Path Fix
# -----------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

heart_model_path = os.path.join(MODEL_DIR, "heart_model.pkl")
diabetes_model_path = os.path.join(MODEL_DIR, "diabetes_model.pkl")
heart_columns_path = os.path.join(MODEL_DIR, "heart_columns.pkl")
diabetes_columns_path = os.path.join(MODEL_DIR, "diabetes_columns.pkl")

print("Model directory:", MODEL_DIR)
print("Heart model exists:", os.path.exists(heart_model_path))

heart_model = pickle.load(open(heart_model_path, "rb"))
diabetes_model = pickle.load(open(diabetes_model_path, "rb"))
heart_columns = pickle.load(open(heart_columns_path, "rb"))
diabetes_columns = pickle.load(open(diabetes_columns_path, "rb"))
