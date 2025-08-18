import numpy as np
import joblib
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from models.AMI import AbstractModelInterface

class DBSCANModel(AbstractModelInterface):
    """
    Anomaly detection model using DBSCAN clustering.
    Points labeled as noise (-1) are considered anomalies.
    """

    def __init__(self, min_samples=100):
        """
        Initializes the DBSCAN model.

        Parameters:
        - min_samples: Minimum number of points required to form a dense region
        """
        self.min_samples = min_samples
        self.model = None
        self.eps = None  # Distance threshold for DBSCAN

    def train(self, data: np.ndarray):
        """
        Estimates the optimal eps value and trains the DBSCAN model.

        Parameters:
        - data: Input data as a NumPy array
        """
        # Estimate eps using the k-distance method
        neighbors = NearestNeighbors(n_neighbors=self.min_samples)
        neighbors_fit = neighbors.fit(data)
        distances, _ = neighbors_fit.kneighbors(data)
        distances = np.sort(distances[:, -1])
        self.eps = np.percentile(distances, 70)

        # Train DBSCAN with estimated eps
        self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.model.fit(data)

    def score(self, data: np.ndarray) -> np.ndarray:
        """
        Predicts anomalies using DBSCAN.

        Parameters:
        - data: Input data as a NumPy array

        Returns:
        - Array of predictions: -1 for anomaly, 1 for normal
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")
        labels = self.model.fit_predict(data)
        return np.where(labels == -1, -1, 1)

    def save(self, path: str):
        """
        Saves the DBSCAN model and metadata to disk.

        Parameters:
        - path: Base path for saving the model
        """
        joblib.dump({
            "model": self.model,
            "eps": self.eps,
            "min_samples": self.min_samples
        }, path + "_dbscan.pkl")

    def load(self, path: str):
        """
        Loads the DBSCAN model and metadata from disk.

        Parameters:
        - path: Base path for loading the model
        """
        data = joblib.load(path + "_dbscan.pkl")
        self.model = data["model"]
        self.eps = data["eps"]
        self.min_samples = data["min_samples"]
