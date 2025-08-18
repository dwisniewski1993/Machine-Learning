from sklearn.decomposition import PCA
import joblib
import numpy as np
from models.AMI import AbstractModelInterface

class PCAModel(AbstractModelInterface):
    """
    Dimensionality reduction model using Principal Component Analysis (PCA).
    """

    def __init__(self, n_components=50):
        """
        Initializes the PCA model.

        Parameters:
        - n_components: Number of principal components to retain
        """
        self.model = PCA(n_components=n_components)

    def train(self, data: np.ndarray):
        """
        Fits the PCA model to the input data.

        Parameters:
        - data: Input data as a NumPy array
        """
        self.model.fit(data)

    def score(self, data: np.ndarray) -> np.ndarray:
        """
        Transforms the input data using the trained PCA model.

        Parameters:
        - data: Input data as a NumPy array

        Returns:
        - Transformed data in reduced dimensionality
        """
        return self.model.transform(data)

    def save(self, path: str):
        """
        Saves the PCA model to disk.

        Parameters:
        - path: File path to save the model
        """
        joblib.dump(self.model, path)

    def load(self, path: str):
        """
        Loads the PCA model from disk.

        Parameters:
        - path: File path to load the model from
        """
        self.model = joblib.load(path)
