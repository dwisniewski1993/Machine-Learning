import logging
import os
from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class SWATDataHandler:
    def __init__(self, file_normal: str, file_brocken: str) -> None:
        """

        :param file_normal: Healthy dataset csv location
        :param file_brocken: Broken data(with) anomalies csv location
        """
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.dataset_normal = self.load_data_normal(file=file_normal)
        self.dataset_broken, self.broken_labels = self.load_data_broken(file=file_brocken)

    def load_data_normal(self, file: str) -> np.array:
        """

        :param file: Healthy file
        :return: dataframe values
        """
        self.logger.info('Start reading healthy dataset')
        my_list = []
        for chunk in pd.read_csv(file, sep=';', header=0, dtype=object, low_memory=False, chunksize=2000):
            my_list.append(chunk)
        df = pd.concat(my_list, axis=0)
        df.drop(df.columns[[0, -1]], axis=1, inplace=True)
        df = df.stack().str.replace(',', '.').unstack()
        df = df.astype(float).fillna(0.0)
        self.logger.info('Healthy dataset loaded succesfully')
        return df.values

    def load_data_broken(self, file: str) -> tuple:
        """

        :param file: Broken data
        :return: Tuple, dataframes values, broken data and labels
        """
        self.logger.info('Start reading broken dataset')
        my_list = []
        for chunk in pd.read_csv(file, sep=';', header=0, dtype=object, low_memory=False, chunksize=2000):
            my_list.append(chunk)
        df = pd.concat(my_list, axis=0)
        labels = df[df.columns[-1]]
        df.drop(df.columns[[0, -1]], axis=1, inplace=True)
        df = df.stack().str.replace(',', '.').unstack()
        df = df.astype(float).fillna(0.0)
        self.logger.info('Broken dataset loaded succesfully')
        return df.values, labels.values

    def get_dataset_normal(self) -> np.array:
        """

        :return: Healthy data
        """
        return self.dataset_normal

    def get_dataset_broken(self) -> np.array:
        """

        :return: Broken data
        """
        return self.dataset_broken

    def get_broken_labels(self) -> np.array:
        """

        :return: Labels from broken data
        """
        return self.broken_labels


class Preprocessing:
    def __init__(self):
        """
        Needed preprocesing
        """
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def scaleData(self, data: np.array) -> np.array:
        """

        :param data: any data value
        :return: Scaled data
        """
        self.logger.info('Normalizing Data')
        return self.scaler.fit_transform(data)


class Results:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info('Calculating results anomaly detection results...')
        current_path = os.getcwd()
        results_files = [file for file in listdir(current_path) if isfile(join(current_path, file))
                         and file.split('.')[-1] == 'csv']
        for file in results_files:
            self.calculate_accuracy(file=file)

    def calculate_accuracy(self, file):
        with open(file, 'r') as rezfile:
            tp = 0
            tn = 0
            fp = 0
            fn = 0
            for line in rezfile:
                if line.split(',')[1] == 'Normal' and line.split(',')[2].replace('\n', '') == 'Normal':
                    tn += 1
                elif line.split(',')[1] == 'Normal' and line.split(',')[2].replace('\n', '') == 'Anomaly':
                    fp += 1
                elif line.split(',')[1] == 'Attack' and line.split(',')[2].replace('\n', '') == 'Normal':
                    fn += 1
                elif line.split(',')[1] == 'Attack' and line.split(',')[2].replace('\n', '') == 'Anomaly':
                    tp += 1
            accuracy = (tp + tn) / (tp + tn + fn + fp)
            self.logger.info(f"Accuracy for model {file.split('_')[0]} is {accuracy}")
