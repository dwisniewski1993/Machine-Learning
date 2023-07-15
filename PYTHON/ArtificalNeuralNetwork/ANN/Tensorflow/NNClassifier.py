import logging as log

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


class TfNeuralNetClassifier:
    """
    Class implementing an artificial neural network for classification using TensorFlow
    """

    def __init__(self, train_file: str) -> None:
        """
        Load and prepare the data
        :param train_file: path to the training file
        """
        log.getLogger().setLevel(log.INFO)
        log.info('Neural Network Classifier With TensorFlow')

        self.trainFile = train_file
        train_data_frame = pd.read_csv(self.trainFile)
        train_array = train_data_frame.values

        # Shuffle the data
        np.random.shuffle(train_array)

        # Extract values into numpy arrays
        self.X = train_array[:, 0:4].astype(float)
        self.Y = train_array[:, 4]

        # Create the model architecture
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Dense(8, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Dense(3, activation=tf.nn.softmax))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Map string labels to numeric
        self.Y = np.array(self.map_labels(self.Y)).astype(float)

        # Split into train and test sets
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.2,
                                                                                random_state=0)

    def __str__(self) -> None:
        """
        Print the data
        :return: None
        """
        print("Features: {}, Labels: {}".format(self.X, self.Y))

    @staticmethod
    def map_labels(labels: np.array) -> list:
        """
        Map iris data labels to categorical values
        :param labels: numpy array containing the labels
        :return: list of mapped values
        """
        mapped = [
            np.array([1, 0, 0]) if x == 'Iris-setosa' else np.array([0, 1, 0]) if x == 'Iris-versicolor' else np.array(
                [0, 0, 1]) for x in labels]
        return mapped

    def train_model(self) -> None:
        """
        Train the model
        :return: None
        """
        self.model.fit(self.X_train, self.Y_train, batch_size=16, epochs=2000, workers=4, use_multiprocessing=True,
                       verbose=0)

    def output(self) -> tuple:
        """
        Print validation loss and accuracy
        :return: tuple containing validation loss and accuracy
        """
        val_loss, val_acc = self.model.evaluate(self.X_test, self.Y_test, verbose=0)
        return val_loss, val_acc

    def score(self) -> float:
        """
        Predict and calculate the F1 score
        :return: F1 score value
        """
        predict_out = self.model.predict(self.X_test, verbose=0)
        y_pred = f1_score(np.argmax(self.Y_test, axis=1), np.argmax(predict_out, axis=1), average='weighted')
        return y_pred
