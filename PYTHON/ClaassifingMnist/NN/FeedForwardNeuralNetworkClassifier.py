import logging as log
import os.path

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

from NN.DataUtils import Utils


class FFNeuralNetwork:
    """
    Feed Forward Neural Network Classification
    """

    def __init__(self) -> None:
        """
        Feed Forward Neural Network Constructor
        After loading and preparing data, build the neural network model
        """
        log.getLogger().setLevel(log.INFO)
        log.info('Feed Forward Neural Network Classifier')
        self.dataset = Utils()

        # Loading data
        self.X_train = self.dataset.get_x_train()
        self.X_test = self.dataset.get_x_test()
        self.Y_train = self.dataset.get_y_train()
        self.Y_test = self.dataset.get_y_test()
        self.nClasses = len(np.unique(self.Y_train))

        # Normalizing data
        self.normalize_data()

        # Build Neural Network Model
        log.info('Building network model')
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Flatten(input_shape=(self.X_train.shape[1:])))
        self.model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Dense(self.nClasses, activation=tf.nn.softmax))
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Print model summary
        self.model.summary()

        # Check if a trained model exists and load it instead of training a new one
        if os.path.exists('mnistnet_FFN'):
            self.load_model()
        else:
            self.train_network(epochs=5)
            self.save_model()

        self.val_loss, self.val_acc = self.model.evaluate(self.X_test, self.Y_test, verbose=0)

        self.f1_score = f1_score(y_true=self.Y_test,
                                 y_pred=np.argmax(self.model.predict(self.X_test, verbose=0), axis=1),
                                 average='weighted')

    def train_network(self, epochs: int) -> None:
        """
        Fit the network with data
        :param epochs: number of training epochs
        :return: None
        """
        log.info('Fitting neural network model')
        self.model.fit(self.X_train, self.Y_train, epochs=epochs)

    def save_model(self) -> None:
        """
        Save the trained model
        :return: None
        """
        log.info('Saving trained model')
        self.model.save('mnistnet_FFN')

    def load_model(self) -> None:
        """
        Load the trained model
        :return: None
        """
        log.info('Loading trained model')
        self.model = tf.keras.models.load_model('mnistnet_FFN')

    def get_val_loss(self) -> float:
        """
        Get the validation loss
        :return: Validation loss value
        """
        return self.val_loss

    def get_val_acc(self) -> float:
        """
        Get the validation accuracy
        :return: Validation accuracy value
        """
        return self.val_acc

    def get_f1_score(self) -> float:
        """
        Get the weighted F1 score
        :return: Weighted F1 score value
        """
        return self.f1_score

    def normalize_data(self) -> None:
        """
        Normalize the data in the dataset
        :return: None
        """
        log.info('Normalizing data')
        self.X_train = tf.keras.utils.normalize(self.X_train, axis=1)
        self.X_test = tf.keras.utils.normalize(self.X_test, axis=1)

    def get_prediction(self, data: np.array, id: int) -> np.array:
        """
        Predict the given image
        :param data: array of images
        :param id: index of the image to predict
        :return: Model predicted image array
        """
        log.info('Predicting data')
        output = self.model.predict([data], verbose=0)
        output = np.argmax(output[id])
        return output
