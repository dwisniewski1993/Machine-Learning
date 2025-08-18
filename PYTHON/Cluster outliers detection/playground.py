import pprint
import pandas as pd
from models.KMeansModel import KMeansModel
from models.DBSSCANModel import DBSCANModel
from models.AutoencoderModel import AutoencoderModel
from models.cluster_ensemble_model import ClusterEnsembleModel
from models.PCAModel import PCAModel
from utils.data_loader import load_data
from utils.evaluator import evaluate_model
from absl import logging as log
from utils.factory import PCA_plot, analyze_target_proportions

# Set logging verbosity
log.set_verbosity(log.INFO)

def compute_weights(reports):
    """
    Computes normalized weights for ensemble voting based on F1 scores.

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

def main():
    """
    Full training and evaluation pipeline(task script.py):
    - Loads data
    - Trains PCA and base models
    - Evaluates models and computes ensemble weights
    - Builds and saves ensemble model
    - Generates predictions on test data
    """
    # Load training and test data
    x_train, y_train, x_pred = load_data()

    analyze_target_proportions(y_train)

    # Generate PCA visualization
    PCA_plot(x_train, y_train)

    # Train PCA model and transform data
    pca_model = PCAModel(n_components=2)
    pca_model.train(x_train)
    x_train_pca = pca_model.score(x_train)
    x_pred_pca = pca_model.score(x_pred)
    pca_model.save("saved_models/pca_model")

    # Initialize base models
    model_kmeans = KMeansModel(n_clusters=2)
    model_dbscan = DBSCANModel()
    model_autoenc = AutoencoderModel(input_dim=2)

    models = {
        "KMeans": model_kmeans,
        "DBSCAN": model_dbscan,
        "Autoencoder": model_autoenc
    }

    # Train and save each model
    for name, model in models.items():
        log.info(f"Training model: {name}")
        model.train(x_train_pca)
        model.save(f"saved_models/{name.lower()}")

    # Evaluate models
    log.info("Generating reports...")
    reports = {}
    for name, model in models.items():
        reports[name] = evaluate_model(model, x_train_pca, y_train)

    # Display reports
    print("\nModel reports:")
    pprint.pprint(reports)

    # Compute ensemble weights
    weights = compute_weights(reports)

    # Load trained models for ensemble
    ae_loaded = AutoencoderModel(); ae_loaded.load("saved_models/autoencoder")
    km_loaded = KMeansModel(); km_loaded.load("saved_models/kmeans")
    db_loaded = DBSCANModel(); db_loaded.load("saved_models/dbscan")

    # Create and save ensemble model
    ensemble = ClusterEnsembleModel(
        autoencoder_model=ae_loaded,
        kmeans_model=km_loaded,
        dbscan_model=db_loaded,
        weights=weights
    )
    ensemble.save("saved_models/ensemble")

    # Load ensemble and evaluate
    ensemble_loaded = ClusterEnsembleModel()
    ensemble_loaded.load("saved_models/ensemble")
    reports["ClusterEnsemble"] = evaluate_model(ensemble_loaded, x_train_pca, y_train)

    # Display updated reports
    print("\nModel reports (with ensemble):")
    pprint.pprint(reports)

    # Predict on test data and save results
    preds = ensemble_loaded.score(x_pred_pca)
    df = pd.DataFrame(preds, columns=["prediction"])
    df.to_csv("test_labels.csv", index=False)
    log.info("Ensemble predictions saved → test_labels.csv")

if __name__ == "__main__":
    main()
