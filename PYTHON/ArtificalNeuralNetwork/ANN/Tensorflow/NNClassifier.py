import logging as log

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


class TfNeuralNetClassifier:
    """
    Artificial Neural Network Classification using Tensorflow
    """

    def __init__(self, train_file: str) -> None:
        """
        Loading and preparing data
        :param train_file: path to train file
        """
        log.getLogger().setLevel(log.INFO)
        log.info('Neural Network Classifier With Tensorflow')

        # Load set
        self.trainFile = train_file
        train_data_frame = pd.read_csv(self.trainFile)
        train_array = train_data_frame.values

        # Shuffle Data
        np.random.shuffle(train_array)

        # Extract value to numpy.Array
        self.X = train_array[:, 0:4].astype(float)
        self.Y = train_array[:, 4]

        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Dense(8, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Dense(3, activation=tf.nn.softmax))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Map string labels to numeric
        self.Y = np.array(self.map_labels(self.Y)).astype(float)

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.2,
                                                                                random_state=0)

    def __str__(self) -> None:
        """
        Printing data
        :return: None
        """
        print("Features: {}, Labels: {}".format(self.X, self.Y))

    @staticmethod
    def map_labels(labels: np.array) -> list:
        """
        Mapping iris data labels to categorical values
        :param labels: numpy.Arrays contains labels
        :return: list of mapped values
        """
        mapped = [
            np.array([1, 0, 0]) if x == 'Iris-setosa' else np.array([0, 1, 0]) if x == 'Iris-versicolor' else np.array(
                [0, 0, 1]) for x in labels]
        return mapped

    def train_model(self) -> None:
        """
        Training model
        :return: None
        """
        self.model.fit(self.X_train, self.Y_train, batch_size=16, epochs=100, workers=4, use_multiprocessing=True,
                       verbose=0)

    def output(self) -> tuple:
        """
        Print Validation loss and accuracy
        :return: tuple, validation loss and accuracy
        """
        val_loss, val_acc = self.model.evaluate(self.X_test, self.Y_test, verbose=0)

        return val_loss, val_acc
