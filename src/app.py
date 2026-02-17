from fastapi import FastAPI, Query
import pandas as pd
from train.train_util import load_model
from data_models.models import PatientData
from data_models.enums import ModelName
import uvicorn

# Load model
loaded_logreg=load_model("logistic_regression_pipeline.pkl")
loaded_rf=load_model("random_forest_pipeline.pkl")

# -------------------------------------------------
# App initialization
# -------------------------------------------------
app = FastAPI(
    title="Heart Disease Prediction API",
    description="Predict heart disease using ML models",
    version="1.0.0"
)

# -------------------------------------------------
# Health check endpoint
# -------------------------------------------------
@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(
    data: PatientData,
    model_name: ModelName = Query(..., description="Choose model: rf or logreg")
):
    # Convert input to DataFrame
    df = pd.DataFrame([data.dict(by_alias=True)])

    # Select model
    if model_name == ModelName.rf:
        model = loaded_rf
    elif model_name == ModelName.logreg:
        model = loaded_logreg

    # Predict
    df
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    return {
        "model_used": model_name,
        "prediction": int(pred),
        "label": "Heart Disease" if pred == 1 else "No Heart Disease"
    }

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
