import joblib
import os
from models.AutoencoderModel import AutoencoderModel
from models.KMeansModel import KMeansModel
from models.DBSSCANModel import DBSCANModel
from models.cluster_ensemble_model import ClusterEnsembleModel
from models.PCAModel import PCAModel


def get_model_status():
    """
    Checks whether model reports exist and returns the training status.

    Returns:
    - A dictionary indicating whether models are trained and listing available models.
    """
    reports_path = "saved_models/reports.pkl"
    if not os.path.exists(reports_path):
        return {"trained": False, "message": "Models not trained yet."}

    reports = joblib.load(reports_path)
    return {"trained": True, "models": list(reports.keys())}


def get_model_report(model_name: str):
    """
    Retrieves the performance report for a specific model.

    Parameters:
    - model_name: Name of the model

    Returns:
    - Report dictionary or error message if model is not found
    """
    reports = joblib.load("saved_models/reports.pkl")
    return reports.get(model_name, {"error": "Model not found."})


def predict_with_model(model_name: str, df):
    """
    Generates predictions using a specified model after PCA transformation.

    Parameters:
    - model_name: Name of the model to use
    - df: Input data as a pandas DataFrame

    Returns:
    - Dictionary containing prediction results or error message
    """
    # Apply PCA transformation
    pca = PCAModel()
    pca.load("saved_models/pca_model")
    x = pca.score(df.to_numpy())

    # Map model names to their corresponding classes
    model_map = {
        "Autoencoder": AutoencoderModel,
        "KMeans": KMeansModel,
        "DBSCAN": DBSCANModel
    }

    if model_name not in model_map:
        return {"error": "Unsupported model."}

    # Load and apply the selected model
    model = model_map[model_name]()
    model.load(f"saved_models/{model_name.lower()}")
    preds = model.score(x)

    return {"predictions": preds.tolist()}


def predict_with_ensemble(df):
    """
    Generates predictions using the ensemble model after PCA transformation.

    Parameters:
    - df: Input data as a pandas DataFrame

    Returns:
    - Dictionary containing ensemble prediction results
    """
    # Apply PCA transformation
    pca = PCAModel()
    pca.load("saved_models/pca_model")
    x = pca.score(df.to_numpy())

    # Load and apply the ensemble model
    ensemble = ClusterEnsembleModel()
    ensemble.load("saved_models/ensemble")
    preds = ensemble.score(x)

    return {"predictions": preds.tolist()}
