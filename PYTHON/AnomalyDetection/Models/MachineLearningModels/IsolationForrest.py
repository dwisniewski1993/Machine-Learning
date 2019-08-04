from numpy import ndarray, array
from sklearn.ensemble.iforest import IsolationForest
from Models.MachineLearningModels.AbstractMlModel import AbstractMlModel
from config import ISOLATION_MODEL, ISOLATION_CORES, ISOLATION_ESTIMATORS_NUMBER, VERBOSE


class IsolationForrestModel(AbstractMlModel):
    def __init__(self, healthy_data: ndarray, broken_data: ndarray, data_labels: array, dataset_name: str) -> None:
        super().__init__(healthy_data, broken_data, data_labels, dataset_name)

        self.model_name = ISOLATION_MODEL
        self.model = IsolationForest(n_estimators=ISOLATION_ESTIMATORS_NUMBER, n_jobs=ISOLATION_CORES, verbose=VERBOSE)
