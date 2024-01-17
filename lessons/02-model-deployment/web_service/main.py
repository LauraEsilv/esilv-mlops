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
from lib.models import Input

# Define a POST operation for the path "/greet"
@app.post("/run_inference")
async def run_inference_endpoint(data: Input):
    result = run_inference(data)
    return {"message": f"StartLocation: {data.PULocationID}, "
                       f"EndLocation: {data.DOLocationID}, "
                       f"Passenger Count: {data.passenger_count}",
            "result": result}