from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
from keras.models import load_model

app = FastAPI()

class FullInput(BaseModel):
    radius_mean: float
    texture_mean: float
    perimeter_mean: float
    area_mean: float
    smoothness_mean: float
    compactness_mean: float
    concavity_mean: float
    concave_points_mean: float
    symmetry_mean: float
    fractal_dimension_mean: float
    radius_se: float
    texture_se: float
    perimeter_se: float
    area_se: float
    smoothness_se: float
    compactness_se: float
    concavity_se: float
    concave_points_se: float
    symmetry_se: float
    fractal_dimension_se: float
    radius_worst: float
    texture_worst: float
    perimeter_worst: float
    area_worst: float
    smoothness_worst: float
    compactness_worst: float
    concavity_worst: float
    concave_points_worst: float
    symmetry_worst: float
    fractal_dimension_worst: float

# Reduced input for logistic regression
class ReducedInput(BaseModel):
    radius_mean: float
    texture_mean: float
    compactness_worst: float
    concave_points_worst: float
    area_worst: float

rf_model = joblib.load("./models/random_forest_model.pkl")
logreg_model = joblib.load("./models/logistic_regression_model.pkl")
nn_model = load_model("./models/nn_model.h5")

scaler_full = joblib.load("./models/scaler_full.pkl")
scaler_logreg = joblib.load("./models/scaler_logreg.pkl")

@app.post("/predict/randomforest")
def predict_rf(input: FullInput):
    data = np.array([[value for value in input.dict().values()]])
    data_scaled = scaler_full.transform(data)
    prediction = rf_model.predict(data_scaled)[0]
    label = "Malignant" if prediction == 1 else "Benign"
    return {
        "model": "random_forest",
        "prediction": int(prediction),
        "label": label
    }

@app.post("/predict/neuralnet")
def predict_nn(input: FullInput):
    data = np.array([[value for value in input.dict().values()]])
    data_scaled = scaler_full.transform(data)
    prob = nn_model.predict(data_scaled)[0][0] # type: ignore
    prediction = int(prob >= 0.5)
    label = "Malignant" if prediction == 1 else "Benign"
    return {
        "model": "neural_network",
        "prediction": prediction,
        "label": label,
        "malignancy_probability": round(float(prob), 4)
    }

@app.post("/predict/logistic")
def predict_logreg(input: ReducedInput):
    data = np.array([[value for value in input.dict().values()]])
    data_scaled = scaler_logreg.transform(data)
    prediction = logreg_model.predict(data_scaled)[0]
    label = "Malignant" if prediction == 1 else "Benign"
    return {
        "model": "logistic_regression",
        "prediction": int(prediction),
        "label": label
    }