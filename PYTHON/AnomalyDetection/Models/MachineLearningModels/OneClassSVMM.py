from numpy import ndarray, array
from sklearn.svm.classes import OneClassSVM
from Models.MachineLearningModels.AbstractMlModel import AbstractMlModel
from config import VERBOSE, ONECLASSSVM_MODEL


class OneClassSVMModel(AbstractMlModel):
    def __init__(self, healthy_data: ndarray, broken_data: ndarray, data_labels: array, dataset_name: str) -> None:
        super().__init__(healthy_data, broken_data, data_labels, dataset_name)

        self.model_name = ONECLASSSVM_MODEL
        self.model = OneClassSVM(verbose=VERBOSE)
