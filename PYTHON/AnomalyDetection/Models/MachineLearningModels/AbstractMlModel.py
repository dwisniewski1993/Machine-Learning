import absl.logging as log
import os.path

import numpy as np
from tqdm import tqdm

from Models.Utils import Preprocessing
from Models.AMI import AbstractModelInterface

from config import DEFAULT_SCALER


log.set_verbosity(log.INFO)


class AbstractMlModel(AbstractModelInterface):
    def __init__(self, healthy_data: np.ndarray, broken_data: np.ndarray, data_labels: np.array, dataset_name: str) \
            -> None:
        """
        Abstract machine learning model for anomaly detection.
        Init the dataset, dataset shapes and pre-processing.

        :param healthy_data: healthy data csv location
        :param broken_data: data with anomalies csv location
        :param dataset_name: unique dataset name for models
        """
        self.data_name = dataset_name

        self.normal_data = healthy_data
        self.anomaly_data = broken_data
        self.Y = data_labels

        scaler = Preprocessing(scaler=DEFAULT_SCALER)
        self.normal_data = scaler.scale_data(data=self.normal_data)
        self.anomaly_data = scaler.scale_data(data=self.anomaly_data)

        self.model_name = None
        self.model = None

    def train(self, retrain=False) -> None:
        """
        Training machine learning model.
        :param retrain: Optional param, default False, if true network will train model even if one already exist.
        :return: None, saving the network model file if not exist or retrain is True
        """
        data = self.normal_data

        if retrain:
            log.info(f"Start training {self.model_name} model...")
            self.model.fit(data)
        else:
            if os.path.exists(self.data_name + '__{}_model.npy'.format(self.model_name)):
                log.info(f"Loading {self.model_name} model...")
                self.model = np.load(self.data_name + '__{}_model.npy'.format(self.model_name), allow_pickle=True)
            else:
                log.info(f"Start training {self.model_name} model...")
                self.model.fit(data)
                np.save(self.data_name + '__{}_model.npy'.format(self.model_name), self.model)

    def score(self, data: np.array) -> None:
        """
        :param data: Data for prediction
        :return: Value matrix (numpy ndarray)
        """
        log.info('Calculating anomaly score...')
        if os.path.exists('{}_detected_{}.csv'.format(self.model_name, self.data_name)):
            log.info('Predictions already exist')
        else:
            with open('{}_detected_{}.csv'.format(self.model_name, self.data_name), 'w') as results:
                predictions = [self.model.predict(data[i].reshape(1, -1)) for i in tqdm(range(len(data)))]
                for i in tqdm(range(len(predictions))):
                    if predictions[i]:
                        results.write(f"1.0,{self.Y[i]},Normal\n")
                    else:
                        results.write(f"0.0,{self.Y[i]},Anomaly\n")
            results.close()
