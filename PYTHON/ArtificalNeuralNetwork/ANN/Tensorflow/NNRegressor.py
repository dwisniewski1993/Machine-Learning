import logging as log

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class TfNeuralNetRegression:
    """
    Artificial Neural Network Regression
    """

    def __init__(self, train_file: str) -> None:
        """
        Loading and preparing data
        :param train_file: path to train file
        """
        log.getLogger().setLevel(log.INFO)
        log.info('Neural Network Regression With Tensorflow')

        self.trainFile = train_file
        train_data_frame = pd.read_csv(self.trainFile, sep='\t', header=None)

        # Mapping string and bool values to numeric
        self.mapping_string = self.map_columns(train_data_frame, 4)
        self.mapping_bool = self.map_columns(train_data_frame, 1)
        train_data_frame = train_data_frame.applymap(
            lambda x: self.mapping_string.get(x) if x in self.mapping_string else x)
        train_data_frame = train_data_frame.applymap(
            lambda x: self.mapping_bool.get(x) if x in self.mapping_bool else x)
        train_array = train_data_frame.values

        # Shuffle Data
        np.random.shuffle(train_array)

        # Extract values to numpy.Arrays
        self.X = train_array[:, 1:]
        self.Y = train_array[:, 0]

        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Dense(8, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Dense(1, activation=tf.nn.relu))
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

        # Split to train-test sets
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.2,
                                                                                random_state=0)

    def __str__(self) -> None:
        """
        Printing data
        :return: None
        """
        print("Features: {}, Labels: {}".format(self.X, self.Y))

    @staticmethod
    def map_columns(df: pd.DataFrame, col_number: int) -> dict:
        """
        Mapping non numeric values to numeric
        :param df: pandas dataframe that contain dataset
        :param col_number: number collumn to map
        :return: dictionary with mapped values
        """
        return dict([(y, x + 1) for x, y in enumerate(sorted(set(df[col_number].unique())))])

    def normalize(self) -> None:
        """
        Standardize the data
        :return: None
        """
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.fit_transform(self.X_test)

    def train_model(self) -> None:
        """
        Train model
        :return: None
        """
        self.model.fit(self.X, self.Y, batch_size=16, epochs=100, workers=4, use_multiprocessing=True, verbose=0)

    def output(self) -> tuple:
        """
        Print Validation loss and accuracy
        :return: tuple, validation loss and accuracy
        """
        val_loss, val_acc = self.model.evaluate(self.X_test, self.Y_test, verbose=0)

        return val_loss, val_acc
