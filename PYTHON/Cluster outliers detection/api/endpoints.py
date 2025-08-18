from fastapi import APIRouter, BackgroundTasks
from core.trainer import train_all_models  # Function to train all models
from core.registry import (
    get_model_status,         # Returns current status of all models
    get_model_report,         # Retrieves performance report for a specific model
    predict_with_model,       # Makes predictions using a specified model
    predict_with_ensemble     # Makes predictions using an ensemble of models
)
from api.models import PredictionRequest  # Pydantic model for incoming prediction data
import pandas as pd

# Create a router instance to define API endpoints
router = APIRouter()

@router.get("/status")
def status():
    """
    Returns the status of all registered models.
    """
    return get_model_status()

@router.post("/train")
async def train(background_tasks: BackgroundTasks):
    """
    Initiates training of all models in the background.

    Parameters:
    - background_tasks: FastAPI's BackgroundTasks object to run tasks asynchronously
    """
    background_tasks.add_task(train_all_models)
    return {"message": "Training started in background."}

@router.post("/score")
def score(request: PredictionRequest):
    """
    Generates predictions using the ensemble of models.

    Parameters:
    - request: JSON payload containing input data

    Returns:
    - Ensemble prediction results
    """
    df = pd.DataFrame(request.data)
    return predict_with_ensemble(df)

@router.get("/report/{model_name}")
def report(model_name: str):
    """
    Retrieves the performance report for a specific model.

    Parameters:
    - model_name: Name of the model to fetch the report for

    Returns:
    - Model performance metrics
    """
    return get_model_report(model_name)

@router.post("/predict/{model_name}")
def predict(model_name: str, request: PredictionRequest):
    """
    Generates predictions using a specified model.

    Parameters:
    - model_name: Name of the model to use
    - request: JSON payload containing input data

    Returns:
    - Prediction results from the specified model
    """
    df = pd.DataFrame(request.data)
    return predict_with_model(model_name, df)
