from sklearn.cluster import KMeans
import joblib
import numpy as np
from models.AMI import AbstractModelInterface

class KMeansModel(AbstractModelInterface):
    """
    Anomaly detection model using KMeans clustering.
    Points far from any cluster centroid are considered anomalies.
    """

    def __init__(self, n_clusters=2, percentile=75):
        """
        Initializes the KMeans model.

        Parameters:
        - n_clusters: Number of clusters to form
        - percentile: Threshold percentile for anomaly detection
        """
        self.model = KMeans(n_clusters=n_clusters, random_state=42)
        self.percentile = percentile
        self.centroids = None
        self.delta = None  # Distance threshold for anomaly detection

    def train(self, data: np.ndarray):
        """
        Trains the KMeans model and computes the anomaly threshold.

        Parameters:
        - data: Input data as a NumPy array
        """
        self.model.fit(data)
        self.centroids = self.model.cluster_centers_

        # Compute distance of each point to its nearest centroid
        distances = []
        for x in data:
            dists = np.linalg.norm(self.centroids - x, axis=1)
            min_dist = np.min(dists)
            distances.append(min_dist)

        # Set delta as the specified percentile of distances
        self.delta = np.percentile(distances, self.percentile)

    def score(self, data: np.ndarray) -> np.ndarray:
        """
        Predicts anomalies based on distance to nearest centroid.

        Parameters:
        - data: Input data as a NumPy array

        Returns:
        - Array of predictions: -1 for anomaly, 1 for normal
        """
        predictions = []
        for x in data:
            dists = np.linalg.norm(self.centroids - x, axis=1)
            min_dist = np.min(dists)
            if min_dist > self.delta:
                predictions.append(-1)  # anomaly
            else:
                predictions.append(1)   # normal
        return np.array(predictions)

    def save(self, path: str):
        """
        Saves the KMeans model and metadata to disk.

        Parameters:
        - path: Base path for saving the model
        """
        joblib.dump({
            "model": self.model,
            "percentile": self.percentile,
            "delta": self.delta,
            "centroids": self.centroids
        }, path + "_kmeans.pkl")

    def load(self, path: str):
        """
        Loads the KMeans model and metadata from disk.

        Parameters:
        - path: Base path for loading the model
        """
        data = joblib.load(path + "_kmeans.pkl")
        self.model = data["model"]
        self.percentile = data["percentile"]
        self.delta = data["delta"]
        self.centroids = data["centroids"]
