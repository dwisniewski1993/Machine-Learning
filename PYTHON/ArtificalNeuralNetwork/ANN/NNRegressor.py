import logging as log

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class NNR:
    """
    Artificial Neural Network Regression
    """

    def __init__(self, trainfile: str) -> None:
        """
        Loading and preparing data
        :param trainfile: path to train file
        """
        log.getLogger().setLevel(log.INFO)
        log.info('Neural Network Regressor')

        self.trainFile = trainfile
        trainDataFrame = pd.read_csv(self.trainFile, sep='\t', header=None)

        # Mapping string and bool values to numeric
        self.mapping_string = self.map_columns(trainDataFrame, 4)
        self.mapping_bool = self.map_columns(trainDataFrame, 1)
        trainDataFrame = trainDataFrame.applymap(
            lambda x: self.mapping_string.get(x) if x in self.mapping_string else x)
        trainDataFrame = trainDataFrame.applymap(
            lambda x: self.mapping_bool.get(x) if x in self.mapping_bool else x)
        trainArray = trainDataFrame.values

        # Shuffle Data
        np.random.shuffle(trainArray)

        # Extract values to numpy.Arrays
        self.X = trainArray[:, 1:]
        self.Y = trainArray[:, 0]

        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Dense(5, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Dense(1, activation=tf.nn.softmax))
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

        # Split to train-test sets
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.3,
                                                                                random_state=0)

    def __str__(self) -> None:
        """
        Printing data
        :return: None
        """
        print("Features: {}, Labels: {}".format(self.X, self.Y))

    @staticmethod
    def map_columns(df: pd.DataFrame, colnumber: int) -> dict:
        """
        Mapping non numeric values to numeric
        :param df: pandas dataframe that contain dataset
        :param colnumber: number collumn to map
        :return: dictionary with mapped values
        """
        return dict([(y, x + 1) for x, y in enumerate(sorted(set(df[colnumber].unique())))])

    def normalize(self) -> None:
        """
        Standardlize the data
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
        self.model.fit(self.X, self.Y, batch_size=16, epochs=100)

    def output(self) -> None:
        """
        Print Validation loss and accuracy
        :return: None
        """
        val_loss, val_acc = self.model.evaluate(self.X_test, self.Y_test)

        print(f"Validation loss: {val_loss}, Validation accuracy: {val_acc}")
