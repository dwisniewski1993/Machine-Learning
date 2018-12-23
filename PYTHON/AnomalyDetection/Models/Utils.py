import logging

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class SWATDataHandler:
    def __init__(self, file_normal, file_brocked):
        """

        :param file_normal: Healthy dataset csv location
        :param file_brocked: Broken data(with) anomalies csv location
        """
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.dataset_normal = self.load_data_normal(file=file_normal)
        self.dataset_broken, self.broken_labels = self.load_data_broken(file=file_brocked)

    def load_data_normal(self, file):
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

    def load_data_broken(self, file):
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

    def get_dataset_normal(self):
        """

        :return: Healthy data
        """
        return self.dataset_normal

    def get_dataset_broken(self):
        """

        :return: Broken data
        """
        return self.dataset_broken

    def get_broken_labels(self):
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

    def scaleData(self, data):
        """

        :param data: any data value
        :return: Scaled data
        """
        self.logger.info('Normalizing Data')
        return self.scaler.fit_transform(data)
