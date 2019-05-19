import logging
import os.path

import numpy as np
from pyFTS.common import Transformations
from pyFTS.models import chen
from pyFTS.partitioners import Grid
from scipy.spatial import distance

from Models.Utils import SWATDataHandler, Preprocessing
from Models.exeptions import DataNotEqual


class FuzzyModel:
    def __init__(self, healthy_data: str, broken_data: str, dataset_name: str) -> None:
        logging.basicConfig(level=logging.INFO)
        self.data_name = dataset_name
        self.logger = logging.getLogger(__name__)
        self.threshold = None
        handler = SWATDataHandler(file_normal=healthy_data, file_brocken=broken_data)

        self.normal_data = handler.get_dataset_normal()
        self.attk_data = handler.get_dataset_broken()
        self.Y = handler.get_broken_labels()

        scaler = Preprocessing()
        self.normal_data = scaler.scaleData(data=self.normal_data)
        self.attk_data = scaler.scaleData(data=self.attk_data)

        self.logger.info('Initializing Fuzzy Time Series model...')
        self.dataset = np.std(a=self.normal_data[:, 1:], axis=0)
        diff = Transformations.Differential(1)
        fuzzy_sets = Grid.GridPartitioner(data=self.dataset, npart=15, transformation=diff)
        self.model = chen.ConventionalFTS(name='Conventional FTS', partitioner=fuzzy_sets)
        self.model.append_transformation(diff)

    def train(self, retrain: bool = False) -> None:
        """

        :param data: Data for prediction
        :return: Value matrix (numpy ndarray)
        """
        if retrain:
            self.logger.info('Start training Fuzzy Time Series model...')
            self.model.fit(self.dataset)
        else:
            if os.path.exists(self.data_name + '__FuzzyTimeSeries_model.npy'):
                self.logger.info('Loading Fuzzy Time Series model...')
                self.model = np.load(self.data_name + '__FuzzyTimeSeries_model.npy').item()
            else:
                self.logger.info('Start training Fuzzy Time Series model...')
                self.model.fit(self.dataset)
                np.save(self.data_name + '__FuzzyTimeSeries_model.npy', self.model)

    def score(self, data) -> None:
        """

        :param data: Data for prediction
        :return: Value matrix (numpy ndarray)
        """
        self.logger.info('Calculating fuzzy score...')
        data = np.mean(a=data[:, 1:], axis=1)
        yhat = self.model.predict(data)
        self.logger.info('Calculate Fuzzy Score Complete')
        return yhat

    def get_normal_data(self) -> np.array:
        """
        Give healthy data, training one
        :return: Healthy data
        """
        return self.normal_data

    def get_attk_data(self) -> np.array:
        """
        Give broken data with anomalie to detect
        :return: Broken data
        """
        return self.attk_data

    def calculate_threshold(self, helth: np.array):
        """
        Calculate and set threshold for anomaly detection
        :param helth: Health matrix value from predict
        :return: None
        """
        self.logger.info('Calculate Fuzzy threshold...')
        actual = self.normal_data
        dists = []
        if len(helth) == len(actual):
            for i in range(len(helth)):
                helth[i] = helth[i].tolist()
                actual[i] = actual[i].tolist()
                dist = distance.euclidean(helth[i], actual[i])
                dists.append(dist)
            self.threshold = max(dists)
            self.logger.info('Threshold calculated Fuzzy: {}'.format(self.threshold))
        else:
            raise DataNotEqual("Both sets should have the same number of samples")

    def anomaly_score(self, pred: np.array):
        """
        Anomaly detector, basing on calculated distance and threshold deciding if data is normal or with anomaly.
        :param pred: Matrix values from predicted anomaly set
        :return: None, saving files with pointed anomaly and real ones per calculated distance.
        """
        self.logger.info('Calculating distance to anomaly...')
        dists = []
        actual = self.attk_data
        Y = self.Y
        if len(pred) == len(actual):
            for i in range(len(pred)):
                pred[i] = pred[i].tolist()
                actual[i] = actual[i].tolist()
                dist = distance.euclidean(pred[i], actual[i])
                if dist > self.threshold:
                    score = 'Anomaly'
                else:
                    score = 'Normal'
                dists.append((dist, Y[i], score))
            with open(r'Fuzzy_detected_' + self.data_name + '.csv', 'w') as file:
                for each in dists:
                    file.write(str(each[0]) + ',' + each[1] + ',' + each[2] + '\n')
            self.logger.info('Anomaly score calculated')
        else:
            raise DataNotEqual("Both sets should have the same number of samples")
