from utils.data_loader import load_data  # Loads training data
from models.KMeansModel import KMeansModel
from models.DBSSCANModel import DBSCANModel
from models.AutoencoderModel import AutoencoderModel
from models.cluster_ensemble_model import ClusterEnsembleModel
from models.PCAModel import PCAModel
from utils.evaluator import evaluate_model  # Evaluates model performance
import joblib
import os

def train_all_models():
    """
    Trains all individual models and the ensemble model.

    Workflow:
    - Loads training data
    - Applies PCA transformation
    - Trains KMeans, DBSCAN, and Autoencoder models
    - Evaluates each model and stores reports
    - Computes weights based on F1 scores
    - Builds and saves the ensemble model
    - Saves performance reports
    """
    # Load training data
    x_train, y_train, _ = load_data()

    # Train PCA model and transform data
    pca_model = PCAModel(n_components=50)
    pca_model.train(x_train)
    x_train_pca = pca_model.score(x_train)
    pca_model.save("saved_models/pca_model")

    # Initialize models
    models = {
        "KMeans": KMeansModel(n_clusters=2),
        "DBSCAN": DBSCANModel(),
        "Autoencoder": AutoencoderModel(input_dim=50)
    }

    reports = {}

    # Train and evaluate each model
    for name, model in models.items():
        model.train(x_train_pca)
        model.save(f"saved_models/{name.lower()}")
        reports[name] = evaluate_model(model, x_train_pca, y_train)

    # Compute ensemble weights based on F1 scores
    weights = compute_weights(reports)

    # Load trained models for ensemble
    ae = AutoencoderModel(); ae.load("saved_models/autoencoder")
    km = KMeansModel(); km.load("saved_models/kmeans")
    db = DBSCANModel(); db.load("saved_models/dbscan")

    # Create and save ensemble model
    ensemble = ClusterEnsembleModel(ae, km, db, weights)
    ensemble.save("saved_models/ensemble")

    # Evaluate ensemble and add to reports
    reports["ClusterEnsemble"] = evaluate_model(ensemble, x_train_pca, y_train)

    # Save all reports to disk
    joblib.dump(reports, "saved_models/reports.pkl")  # Uncomment to persist reports

def compute_weights(reports):
    """
    Computes normalized weights for ensemble based on F1 scores.

    Parameters:
    - reports: Dictionary of model evaluation reports

    Returns:
    - Dictionary of weights for Autoencoder, KMeans, and DBSCAN
    """
    total = sum(r["f1"] for k, r in reports.items() if k != "ClusterEnsemble")
    return {
        "ae": reports["Autoencoder"]["f1"] / total,
        "kmeans": reports["KMeans"]["f1"] / total,
        "dbscan": reports["DBSCAN"]["f1"] / total
    }
