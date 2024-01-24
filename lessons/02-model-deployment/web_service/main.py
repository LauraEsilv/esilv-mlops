# main.py
from fastapi import FastAPI, Depends
from app_config import AppConfig

app = FastAPI()

# Dependency to get app configuration
def get_app_config():
    return AppConfig

@app.get("/")
async def read_root(config: AppConfig = Depends(get_app_config)):
    return {
        "title": config.TITLE,
        "description": config.DESCRIPTION,
        "app_version": config.APP_VERSION,
        "model_version": config.MODEL_VERSION,
    }

from fastapi import Depends
from lib.model_operations import run_inference
from lib.models import Input, Prediction
from lib.prediction_utils import load_preprocessor, load_pickle

# Define a POST operation for the path "/greet"
@app.post("/predict", response_model=Prediction, status_code=201)
def predict(payload: Input):
    dv = load_preprocessor(AppConfig.PATH_TO_PREPROCESOR)
    model = load_pickle(AppConfig.PATH_TO_PIPELINE)
    y = run_inference([payload], dv, model)
    return {"trip_duration_prediction": y[0]}