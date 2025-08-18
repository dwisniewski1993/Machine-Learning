from abc import ABC, abstractmethod
import numpy as np

class AbstractModelInterface(ABC):
    """
    Abstract base class defining the interface for anomaly detection models.
    All models must implement the following methods.
    """

    @abstractmethod
    def train(self, data: np.ndarray):
        """
        Trains the model using the provided data.

        Parameters:
        - data: A NumPy array of input features
        """
        pass

    @abstractmethod
    def score(self, data: np.ndarray) -> np.ndarray:
        """
        Generates predictions for the input data.

        Returns:
        - A NumPy array of predictions: -1 for anomalies, 1 for normal instances
        """
        pass

    @abstractmethod
    def save(self, path: str):
        """
        Saves the model (and optionally PCA) to a .pkl file.

        Parameters:
        - path: File path to save the model
        """
        pass

    @abstractmethod
    def load(self, path: str):
        """
        Loads the model (and optionally PCA) from a .pkl file.

        Parameters:
        - path: File path to load the model from
        """
        pass
