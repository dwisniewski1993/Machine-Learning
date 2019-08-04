import absl.logging as log

import numpy as np
from pyFTS.common import Transformations
from pyFTS.models import chen
from pyFTS.partitioners import Grid
from scipy.spatial import distance

from Models.Utils import Preprocessing
from Models.AMI import AbstractModelInterface
from Models.exeptions import DataNotEqual

from config import DEFAULT_SCALER, DEFAULT_N_FUZZY_PARTS, SINGLE_UNIT


log.set_verbosity(log.INFO)


class FuzzyModel(AbstractModelInterface):
    def __init__(self, healthy_data: np.ndarray, broken_data: np.ndarray, data_labels: np.array, dataset_name: str) \
            -> None:
        self.data_name = dataset_name
        self.threshold = None

        self.normal_data = healthy_data
        self.anomaly_data = broken_data
        self.Y = data_labels

        scaler = Preprocessing(scaler=DEFAULT_SCALER)
        self.normal_data = scaler.scale_data(data=self.normal_data)
        self.anomaly_data = scaler.scale_data(data=self.anomaly_data)

        log.info('Initializing Fuzzy Time Series model...')
        self.dataset = np.std(a=self.normal_data[:, 1:], axis=0)
        diff = Transformations.Differential(SINGLE_UNIT)
        fuzzy_sets = Grid.GridPartitioner(data=self.dataset, npart=DEFAULT_N_FUZZY_PARTS, transformation=diff)
        self.model = chen.ConventionalFTS(name='Conventional FTS', partitioner=fuzzy_sets)
        self.model.append_transformation(diff)

    def train(self) -> None:
        """
        :return: None
        """
        self.model.fit(self.dataset)

    def score(self, data) -> np.array:
        """
        :param data: Data for prediction
        :return: Value matrix (numpy ndarray)
        """
        log.info('Calculating fuzzy score...')
        data = np.mean(a=data[:, 1:], axis=1)
        yhat = self.model.predict(data)
        log.info('Calculate Fuzzy Score Complete')
        return yhat

    def get_normal_data(self) -> np.array:
        """
        Give healthy data, training one
        :return: Healthy data
        """
        return self.normal_data

    def get_anomaly_data(self) -> np.array:
        """
        Give broken data with anomalie to detect
        :return: Broken data
        """
        return self.anomaly_data

    def calculate_threshold(self, helth: np.array):
        """
        Calculate and set threshold for anomaly detection
        :param helth: Health matrix value from predict
        :return: None
        """
        log.info('Calculate Fuzzy threshold...')
        actual = self.normal_data
        dists = []
        if len(helth) == len(actual):
            for i in range(len(helth)):
                helth[i] = helth[i].tolist()
                actual[i] = actual[i].tolist()
                dist = distance.euclidean(helth[i], actual[i])
                dists.append(dist)
            self.threshold = max(dists)
            log.info('Threshold calculated Fuzzy: {}'.format(self.threshold))
        else:
            raise DataNotEqual("Both sets should have the same number of samples")

    def anomaly_score(self, pred: np.array):
        """
        Anomaly detector, basing on calculated distance and threshold deciding if data is normal or with anomaly.
        :param pred: Matrix values from predicted anomaly set
        :return: None, saving files with pointed anomaly and real ones per calculated distance.
        """
        log.info('Calculating distance to anomaly...')
        dists = []
        actual = self.anomaly_data
        y = self.Y
        if len(pred) == len(actual):
            for i in range(len(pred)):
                pred[i] = pred[i].tolist()
                actual[i] = actual[i].tolist()
                dist = distance.euclidean(pred[i], actual[i])
                if dist > self.threshold:
                    score = 'Anomaly'
                else:
                    score = 'Normal'
                dists.append((dist, y[i], score))
            with open(r'Fuzzy_detected_' + self.data_name + '.csv', 'w') as file:
                for each in dists:
                    file.write(str(each[0]) + ',' + each[1] + ',' + each[2] + '\n')
            log.info('Anomaly score calculated')
        else:
            raise DataNotEqual("Both sets should have the same number of samples")
