import numpy as np
from sklearn.svm import OneClassSVM

from Models.MachineLearningModels.AbstractMlModel import AbstractMlModel
from config import VERBOSE, ONECLASSSVM_MODEL


class OneClassSVMModel(AbstractMlModel):
    def __init__(self, healthy_data: np.ndarray, broken_data: np.ndarray, data_labels: np.ndarray,
                 dataset_name: str) -> None:
        """
        Initialize the OneClassSVMModel class.

        :param healthy_data: Healthy data for training
        :param broken_data: Broken data with anomalies to detect
        :param data_labels: Data labels
        :param dataset_name: Name of the dataset
        """
        super().__init__(healthy_data, broken_data, data_labels, dataset_name)

        self.model_name: str = ONECLASSSVM_MODEL
        self.model: OneClassSVM = OneClassSVM(verbose=VERBOSE)
