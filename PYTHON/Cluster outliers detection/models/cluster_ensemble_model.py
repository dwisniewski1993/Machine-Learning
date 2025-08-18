import numpy as np
import joblib
from models.AMI import AbstractModelInterface

class ClusterEnsembleModel(AbstractModelInterface):
    """
    Ensemble model that combines predictions from Autoencoder, KMeans, and DBSCAN
    using weighted voting. Only models that detect anomalies contribute to the vote.
    """

    def __init__(self, autoencoder_model=None, kmeans_model=None, dbscan_model=None, weights=None):
        self.autoencoder_model = autoencoder_model
        self.kmeans_model = kmeans_model
        self.dbscan_model = dbscan_model
        self.weights = weights if weights else {"ae": 1.0, "kmeans": 1.0, "dbscan": 1.0}

    def train(self, data: np.ndarray):
        """
        No training required — ensemble relies on pre-trained models.
        """
        pass

    def score(self, data: np.ndarray) -> np.ndarray:
        """
        Generates ensemble predictions using weighted voting.

        Parameters:
        - data: Input data as a NumPy array

        Returns:
        - Array of predictions: -1 for anomaly, 1 for normal
        """
        # Get predictions from each model
        ae_votes = self.autoencoder_model.score(data)
        kmeans_votes = self.kmeans_model.score(data)
        dbscan_votes = self.dbscan_model.score(data)

        # Stack votes into a matrix: shape (n_samples, 3)
        votes = np.stack([ae_votes, kmeans_votes, dbscan_votes], axis=1)

        # Check which models detect any anomalies
        model_has_signal = {
            "ae": np.any(ae_votes == -1),
            "kmeans": np.any(kmeans_votes == -1),
            "dbscan": np.any(dbscan_votes == -1)
        }

        # Filter weights: ignore models that don't detect anomalies
        filtered_weights = {
            name: self.weights[name] if model_has_signal[name] else 0.0
            for name in self.weights
        }

        # Compute weighted anomaly votes
        weighted_votes = (
            filtered_weights["ae"] * (votes[:, 0] == -1).astype(float) +
            filtered_weights["kmeans"] * (votes[:, 1] == -1).astype(float) +
            filtered_weights["dbscan"] * (votes[:, 2] == -1).astype(float)
        )

        # Normalize threshold based on active models
        total_weight = sum(filtered_weights.values())
        threshold = total_weight * 0.5  # majority vote

        # Final prediction: anomaly if weighted vote exceeds threshold
        final_preds = np.where(weighted_votes >= threshold, -1, 1)
        return final_preds

    def save(self, path: str):
        """
        Saves ensemble metadata and individual models to disk.

        Parameters:
        - path: Base path for saving files
        """
        joblib.dump({
            "weights": self.weights
        }, path + "_ensemble_meta.pkl")

        self.autoencoder_model.save(path + "_ae")
        self.kmeans_model.save(path + "_kmeans")
        self.dbscan_model.save(path + "_dbscan")

    def load(self, path: str):
        """
        Loads ensemble metadata and individual models from disk.

        Parameters:
        - path: Base path for loading files
        """
        meta = joblib.load(path + "_ensemble_meta.pkl")
        self.weights = meta["weights"]

        # Lazy imports to avoid circular dependencies
        from models.AutoencoderModel import AutoencoderModel
        from models.KMeansModel import KMeansModel
        from models.DBSSCANModel import DBSCANModel

        ae = AutoencoderModel(); ae.load(path + "_ae")
        km = KMeansModel(); km.load(path + "_kmeans")
        db = DBSCANModel(); db.load(path + "_dbscan")

        self.autoencoder_model = ae
        self.kmeans_model = km
        self.dbscan_model = db
