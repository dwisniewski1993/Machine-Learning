from os import listdir, getcwd
from os.path import isfile, join

import absl.logging as log
import dask.dataframe as dd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, MaxAbsScaler, RobustScaler

log.set_verbosity(log.INFO)


class DataHandler:
    def __init__(self, file_normal: str, file_broken: str) -> None:
        """
        DataHandler class for loading datasets.

        :param file_normal: File path of the healthy dataset
        :param file_broken: File path of the broken dataset (with anomalies)
        """
        self.dataset_normal, self.normal_labels = self.load_data(file_normal)
        self.dataset_broken, self.broken_labels = self.load_data(file_broken)

    @staticmethod
    def load_data(file: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Load data from a CSV file.

        :param file: File path to load
        :return: Tuple containing the data values and labels as numpy arrays
        """
        log.info(f'Start reading dataset from: {file}')
        df = dd.read_csv(file, sep=';', header=0, dtype=object, assume_missing=True)
        labels = df[df.columns[-1]].astype(str)
        data = df.drop(columns=[df.columns[0], df.columns[-1]])

        # Replace comma with period in float values
        data = data.replace({',': '.'}, regex=True).astype(float).fillna(0.0)

        data, labels = data.compute(), labels.compute()
        log.info(f'Dataset loaded successfully from: {file}')
        return data.values, labels.values

    def get_normal_dataset(self) -> np.ndarray:
        """
        Get the healthy dataset.

        :return: Healthy data
        """
        return self.dataset_normal

    def get_normal_labels(self) -> np.ndarray:
        """
        Get the labels from the healthy dataset.

        :return: Labels from the healthy data
        """
        return self.normal_labels

    def get_broken_dataset(self) -> np.ndarray:
        """
        Get the broken dataset (with anomalies).

        :return: Broken data
        """
        return self.dataset_broken

    def get_broken_labels(self) -> np.ndarray:
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
        :raises ValueError: If the specified scaler is not available
        """
        self.available_scalers = {
            'Min-Max': MinMaxScaler(feature_range=(0, 1)),
            'Standard': StandardScaler(),
            'Normalize': Normalizer(),
            'Max-Abs': MaxAbsScaler(),
            'Robust': RobustScaler()
        }

        if scaler not in self.available_scalers:
            raise ValueError(f"Scaler '{scaler}' is not recognized. Available scalers are:"
                             f" {', '.join(self.available_scalers.keys())}")

        self.scaler = self.available_scalers[scaler]

    def scale_data(self, data: np.ndarray) -> np.ndarray:
        """
        Scale the data using the specified scaler.

        :param data: Data values to scale
        :return: Scaled data
        """
        log.info('Scaling data using %s scaler', self.scaler.__class__.__name__)
        scaled_data = self.scaler.fit_transform(data)
        log.info('Data scaling completed successfully')
        return scaled_data


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
        with open(file, 'r') as results_file:
            tp = tn = fp = fn = 0

            for line in results_file:
                parts = line.strip().split(',')
                if len(parts) < 3:
                    log.warning(f"Skipping malformed line: {line.strip()}")
                    continue

                actual, predicted = parts[1], parts[2]

                if actual == 'Normal' and predicted == 'Normal':
                    tn += 1
                elif actual == 'Normal' and predicted == 'Anomaly':
                    fp += 1
                elif actual == 'Attack' and predicted == 'Normal':
                    fn += 1
                elif actual == 'Attack' and predicted == 'Anomaly':
                    tp += 1

        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0
        log.info(f"Accuracy for model {file.split('_')[0]} is {accuracy:.2f}")
