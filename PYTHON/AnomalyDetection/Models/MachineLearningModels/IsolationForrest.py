import numpy as np
from sklearn.ensemble import IsolationForest

from Models.MachineLearningModels.AbstractMlModel import AbstractMlModel
from config import ISOLATION_MODEL, ISOLATION_CORES, ISOLATION_ESTIMATORS_NUMBER, VERBOSE


class IsolationForrestModel(AbstractMlModel):
    def __init__(self, healthy_data: np.ndarray, broken_data: np.ndarray, data_labels: np.ndarray,
                 dataset_name: str) -> None:
        """
        Initialize the IsolationForrestModel class.

        :param healthy_data: Healthy data for training
        :param broken_data: Broken data with anomalies to detect
        :param data_labels: Data labels
        :param dataset_name: Name of the dataset
        """
        super().__init__(healthy_data, broken_data, data_labels, dataset_name)

        self.model_name: str = ISOLATION_MODEL
        self.model: IsolationForest = IsolationForest(n_estimators=ISOLATION_ESTIMATORS_NUMBER, n_jobs=ISOLATION_CORES,
                                                      verbose=VERBOSE)
