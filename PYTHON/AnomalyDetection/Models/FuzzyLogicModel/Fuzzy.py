import logging
from typing import List, Tuple

import numpy as np
from pyFTS.common import Transformations
from pyFTS.models import chen
from pyFTS.partitioners import Grid
from scipy.spatial import distance

from Models.AMI import AbstractModelInterface
from Models.Utils import Preprocessing
from Models.exeptions import DataNotEqual
from config import DEFAULT_SCALER, DEFAULT_N_FUZZY_PARTS, SINGLE_UNIT

logging.basicConfig(level=logging.INFO)


class FuzzyModel(AbstractModelInterface):
    def __init__(self, healthy_data: np.ndarray, broken_data: np.ndarray, data_labels: np.ndarray, dataset_name: str):
        """
        Initialize the FuzzyModel class.

        :param healthy_data: Healthy data for training
        :param broken_data: Broken data with anomalies to detect
        :param data_labels: Data labels
        :param dataset_name: Name of the dataset
        """
        self.data_name: str = dataset_name
        self.threshold: float = 0.0

        self.normal_data: np.ndarray = healthy_data
        self.anomaly_data: np.ndarray = broken_data
        self.Y: np.ndarray = data_labels

        scaler = Preprocessing(scaler=DEFAULT_SCALER)
        self.normal_data = scaler.scale_data(data=self.normal_data)
        self.anomaly_data = scaler.scale_data(data=self.anomaly_data)

        logging.info('Initializing Fuzzy Time Series model...')
        self.dataset = np.std(a=self.normal_data[:, 1:], axis=0)
        diff = Transformations.Differential(SINGLE_UNIT)
        fuzzy_sets = Grid.GridPartitioner(data=self.dataset, npart=DEFAULT_N_FUZZY_PARTS, transformation=diff)
        self.model = chen.ConventionalFTS(name='Conventional FTS', partitioner=fuzzy_sets)
        self.model.append_transformation(diff)

    def train(self) -> None:
        """
        Train the FuzzyModel.
        """
        self.model.fit(self.dataset)

    def score(self, data: np.ndarray) -> List:
        """
        Calculate the fuzzy score for the given data.

        :param data: Data for prediction
        :return: Fuzzy score (list)
        """
        logging.info('Calculating fuzzy score...')
        yhat: List = []
        for record in data:
            yhat.append(self.model.predict(record))
        logging.info('Calculate Fuzzy Score Complete')
        return yhat

    def get_normal_data(self) -> np.ndarray:
        """
        Get the healthy (normal) data.

        :return: Healthy data
        """
        return self.normal_data

    def get_anomaly_data(self) -> np.ndarray:
        """
        Get the broken (anomaly) data.

        :return: Broken data
        """
        return self.anomaly_data

    def calculate_threshold(self, health: List) -> None:
        """
        Calculate and set the threshold for anomaly detection.

        :param health: Health matrix value from predict
        """
        logging.info('Calculate Fuzzy threshold...')
        actual = self.normal_data
        dists: List[float] = []
        if len(health) == len(actual):
            for i in range(len(health)):
                dist = distance.euclidean(health[i], actual[i])
                dists.append(dist)
            self.threshold = max(dists)
            logging.info('Threshold calculated Fuzzy: {}'.format(self.threshold))
        else:
            raise DataNotEqual("Both sets should have the same number of samples")

    def anomaly_score(self, pred: List) -> None:
        """
        Calculate the anomaly score and save the results to a file.

        :param pred: Matrix values from predicted anomaly set
        """
        logging.info('Calculating distance to anomaly...')
        dists: List[Tuple] = []
        actual = self.anomaly_data
        y = self.Y
        if len(pred) == len(actual):
            for i in range(len(pred)):
                dist = distance.euclidean(pred[i], actual[i])
                if dist > self.threshold:
                    score = 'Anomaly'
                else:
                    score = 'Normal'
                dists.append((dist, y[i], score))
            with open(f'Fuzzy_detected_{self.data_name}.csv', 'w') as file:
                for each in dists:
                    file.write(f"{each[0]},{each[1]},{each[2]}\n")
            logging.info('Anomaly score calculated')
        else:
            raise DataNotEqual("Both sets should have the same number of samples")
