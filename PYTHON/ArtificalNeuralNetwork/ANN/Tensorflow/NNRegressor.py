import logging as log

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class TfNeuralNetRegression:
    """
    Artificial Neural Network Regression using TensorFlow
    """

    def __init__(self, train_file: str) -> None:
        """
        Load and prepare the data
        :param train_file: path to the training file
        """
        log.getLogger().setLevel(log.INFO)
        log.info('Neural Network Regression With TensorFlow')

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

        # Shuffle the data
        np.random.shuffle(train_array)

        # Extract values into numpy arrays
        self.X = train_array[:, 1:]
        self.Y = train_array[:, 0]

        # Create the model architecture
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Dense(8, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Dense(1, activation=tf.nn.relu))
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

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
    def map_columns(df: pd.DataFrame, col_number: int) -> dict:
        """
        Map non-numeric values to numeric
        :param df: pandas DataFrame containing the dataset
        :param col_number: column number to map
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
        self.X_test = scaler.transform(self.X_test)

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

    def score(self):
        """
        Predict and calculate the R2 score
        :return: R2 score value
        """
        predict_out = self.model.predict(self.X_test, verbose=0)
        y_pred = r2_score(self.Y_test, predict_out)
        return y_pred
