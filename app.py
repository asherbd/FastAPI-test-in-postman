from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import sys


num_cols = [
    "Age",
    "BMI",
    "Heart_Rate"
]


def add_features(data):

    d = pd.DataFrame(
        data,
        columns=num_cols
    ).astype(float)

    d["BMI_sq"] = d["BMI"] ** 2

    d["HR_x_Age"] = (
        d["Heart_Rate"] *
        d["Age"]
    )

    d["HR_per_BMI"] = (
        d["Heart_Rate"] /
        d["BMI"]
    )

    return d.values


# Make add_features available for the old pickle file
sys.modules["__main__"].add_features = add_features


# Load trained model
with open("best_stress_model.pkl", "rb") as file:
    model = pickle.load(file)


app = FastAPI(
    title="Stress Level Prediction API"
)


class StressInput(BaseModel):
    Gender: str
    Smoking: str
    Alcohol_Consumption: str
    Age: float
    BMI: float
    Heart_Rate: float


@app.get("/")
def home():
    return {
        "message": "Stress Level Prediction API is running"
    }


@app.post("/predict")
def predict_stress(data: StressInput):

    input_data = pd.DataFrame([{
        "Gender": data.Gender,
        "Smoking": data.Smoking,
        "Alcohol_Consumption": data.Alcohol_Consumption,
        "Age": data.Age,
        "BMI": data.BMI,
        "Heart_Rate": data.Heart_Rate
    }])

    prediction = model.predict(input_data)

    label = {
        0: "Low",
        1: "Medium",
        2: "High"
    }

    predicted_class = int(prediction[0])

    return {
        "prediction": predicted_class,
        "stress_level": label.get(
            predicted_class,
            "Unknown"
        )
    }