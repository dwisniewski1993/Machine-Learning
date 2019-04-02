import logging as log

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


class NNC:
    """
    Artificial Neural Network Classification
    """

    def __init__(self, trainfile: str) -> None:
        """
        Loading and preparing data
        :param trainfile: path to train file
        """
        log.getLogger().setLevel(log.INFO)
        log.info('Neural Network Classifier')

        # Load set
        self.trainFile = trainfile
        trainDataFrame = pd.read_csv(self.trainFile)
        trainArray = trainDataFrame.values

        # Shuffle Data
        np.random.shuffle(trainArray)

        # Extract value to numpy.Array
        self.X = trainArray[:, 0:4]
        self.Y = trainArray[:, 4]

        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Dense(8, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Dense(3, activation=tf.nn.softmax))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Map string labels to numeric
        self.Y = np.array(self.map_labels(self.Y))

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.3,
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
        Maping iris data labels to categorical values
        :param labels: numpy.Arrays contains labels
        :return: list of mapped values
        """
        maped = [
            np.array([1, 0, 0]) if x == 'Iris-setosa' else np.array([0, 1, 0]) if x == 'Iris-versicolor' else np.array(
                [0, 0, 1]) for x in labels]
        return maped

    def train_model(self) -> None:
        """
        Training model
        :return: None
        """
        self.model.fit(self.X_train, self.Y_train, batch_size=16, epochs=100)

    def output(self) -> None:
        """
        Print Validation loss and accuracy
        :return: None
        """
        val_loss, val_acc = self.model.evaluate(self.X_test, self.Y_test)

        print(f"Validation loss: {val_loss}, Validation accuracy: {val_acc}")
