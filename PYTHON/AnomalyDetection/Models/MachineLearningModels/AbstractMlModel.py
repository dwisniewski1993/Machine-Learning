import os.path

import absl.logging as log
import numpy as np
from tqdm import tqdm

from Models.AMI import AbstractModelInterface
from Models.Utils import Preprocessing
from config import DEFAULT_SCALER

log.set_verbosity(log.INFO)


class AbstractMlModel(AbstractModelInterface):
    def __init__(self, healthy_data: np.ndarray, broken_data: np.ndarray, data_labels: np.ndarray,
                 dataset_name: str) -> None:
        """
        Abstract machine learning model for anomaly detection.
        Init the dataset, dataset shapes and pre-processing.

        :param healthy_data: Healthy data array
        :param broken_data: Data with anomalies array
        :param data_labels: Data labels array
        :param dataset_name: Unique dataset name for models
        """
        self.data_name = dataset_name

        self.normal_data: np.ndarray = healthy_data
        self.anomaly_data: np.ndarray = broken_data
        self.Y: np.ndarray = data_labels

        scaler = Preprocessing(scaler=DEFAULT_SCALER)
        self.normal_data = scaler.scale_data(data=self.normal_data)
        self.anomaly_data = scaler.scale_data(data=self.anomaly_data)

        self.model_name: str = f""
        self.model = None

    def train(self, retrain: bool = False) -> None:
        """
        Training machine learning model.

        :param retrain: Optional param, default False, if true network will train model even if one already exists.
        """
        data = self.normal_data

        if retrain:
            log.info(f"Start training {self.model_name} model...")
            self.model.fit(data)
        else:
            if os.path.exists(f"{self.data_name}__{self.model_name}_model.npy"):
                log.info(f"Loading {self.model_name} model...")
                self.model = np.load(f"{self.data_name}__{self.model_name}_model.npy", allow_pickle=True)
            else:
                log.info(f"Start training {self.model_name} model...")
                self.model.fit(data)
                np.save(f"{self.data_name}__{self.model_name}_model.npy", self.model)

    def score(self, data: np.ndarray) -> None:
        """
        Calculate the anomaly score.

        :param data: Data for prediction
        """
        log.info('Calculating anomaly score...')
        if os.path.exists(f"{self.model_name}_detected_{self.data_name}.csv"):
            log.info('Predictions already exist')
        else:
            with open(f"{self.model_name}_detected_{self.data_name}.csv", 'w') as results:
                predictions = [self.model.predict(data[i].reshape(1, -1)) for i in tqdm(range(len(data)))]
                for i in tqdm(range(len(predictions))):
                    if predictions[i]:
                        results.write(f"1.0,{self.Y[i]},Normal\n")
                    else:
                        results.write(f"0.0,{self.Y[i]},Anomaly\n")
            results.close()
