from os import listdir, getcwd
from os.path import isfile, join

import absl.logging as log
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, MaxAbsScaler, RobustScaler

log.set_verbosity(log.INFO)


class DataHandler:
    def __init__(self, file_normal: str, file_broken: str) -> None:
        """
        :param file_normal: Healthy dataset csv location
        :param file_broken: Broken data(with) anomalies csv location
        """
        self.dataset_normal, _ = self.load_data(file=file_normal)
        self.dataset_broken, self.broken_labels = self.load_data(file=file_broken)

    @staticmethod
    def load_data(file: str) -> np.array:
        """
        Loading data from csv files
        :param file: file path to load
        :return: pandas data frame values
        """
        log.info('Start reading healthy dataset')
        my_list = []
        for chunk in pd.read_csv(file, sep=';', header=0, dtype=object, low_memory=False, chunksize=2000):
            my_list.append(chunk)
        df = pd.concat(my_list, axis=0)
        labels = df[df.columns[-1]]
        df.drop(df.columns[[0, -1]], axis=1, inplace=True)
        df = df.stack().str.replace(',', '.').unstack()
        df = df.astype(float).fillna(0.0)
        log.info('Healthy dataset loaded successfully')
        return df.values, labels

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
    def __init__(self, scaler: str) -> None:
        """
        Needed preprocesing
        """
        self.available_scalers = {'Min-Max': MinMaxScaler(feature_range=(0, 1)),
                                  'Standard': StandardScaler(),
                                  'Normalize': Normalizer(),
                                  'Max-Abs': MaxAbsScaler(),
                                  'Robust': RobustScaler()}
        self.scaler = self.available_scalers[scaler]

    def scale_data(self, data: np.array) -> np.array:
        """
        :param data: any data value
        :return: Scaled data
        """
        log.info('Scaling Data')
        return self.scaler.fit_transform(data)


class Results:
    def __init__(self) -> None:
        log.info('Calculating results anomaly detection results...')
        current_path = getcwd()
        results_files = [file for file in listdir(current_path) if isfile(join(current_path, file))
                         and file.split('.')[-1] == 'csv']
        for file in results_files:
            self.calculate_accuracy(file=file)

    @staticmethod
    def calculate_accuracy(file):
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
            log.info(f"Accuracy for model {file.split('_')[0]} is {accuracy}")
