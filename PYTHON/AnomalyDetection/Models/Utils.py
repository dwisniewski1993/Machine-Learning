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
        DataHandler class for loading datasets.

        :param file_normal: File path of the healthy dataset
        :param file_broken: File path of the broken dataset (with anomalies)
        """
        self.dataset_normal, _ = self.load_data(file=file_normal)
        self.dataset_broken, self.broken_labels = self.load_data(file=file_broken)

    @staticmethod
    def load_data(file: str) -> np.array:
        """
        Load data from a CSV file.

        :param file: File path to load
        :return: Numpy array of the data values
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
        return df.values, labels.values

    def get_dataset_normal(self) -> np.array:
        """
        Get the healthy dataset.

        :return: Healthy data
        """
        return self.dataset_normal

    def get_dataset_broken(self) -> np.array:
        """
        Get the broken dataset (with anomalies).

        :return: Broken data
        """
        return self.dataset_broken

    def get_broken_labels(self) -> np.array:
        """
        Get the labels from the broken dataset.

        :return: Labels from the broken data
        """
        return self.broken_labels


class Preprocessing:
    def __init__(self, scaler: str) -> None:
        """
        Preprocessing class for data scaling.

        :param scaler: Name of the scaler to use ('Min-Max', 'Standard', 'Normalize', 'Max-Abs', 'Robust')
        """
        self.available_scalers = {
            'Min-Max': MinMaxScaler(feature_range=(0, 1)),
            'Standard': StandardScaler(),
            'Normalize': Normalizer(),
            'Max-Abs': MaxAbsScaler(),
            'Robust': RobustScaler()
        }
        self.scaler = self.available_scalers[scaler]

    def scale_data(self, data: np.array) -> np.array:
        """
        Scale the data using the specified scaler.

        :param data: Data values
        :return: Scaled data
        """
        log.info('Scaling Data')
        return self.scaler.fit_transform(data)


class Results:
    def __init__(self) -> None:
        """
        Results class for calculating anomaly detection results.
        """
        log.info('Calculating anomaly detection results...')
        current_path = getcwd()
        results_files = [file for file in listdir(current_path) if isfile(join(current_path, file))
                         and file.split('.')[-1] == 'csv']
        for file in results_files:
            self.calculate_accuracy(file=file)

    @staticmethod
    def calculate_accuracy(file: str) -> None:
        """
        Calculate accuracy for a given result file.

        :param file: Result file to calculate accuracy for
        """
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
