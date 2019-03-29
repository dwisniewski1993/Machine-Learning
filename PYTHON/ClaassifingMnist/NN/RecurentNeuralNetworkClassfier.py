import logging as log
import os.path

import numpy as np
import tensorflow as tf

from NN.DataUtils import Utils


class RecurentNeuralNetwork:
    """
    Recurent Neural Network Classification
    """

    def __init__(self) -> None:
        """
        Convolutional Neural Network Constructor
        After loading and preparing data, build neural network model
        """
        log.getLogger().setLevel(log.INFO)
        log.info('Reccurent Neural Network Classifier')
        self.dataset = Utils()

        # Loading data
        self.X_train = self.dataset.get_x_train()
        self.X_test = self.dataset.get_x_test()
        self.Y_train = self.dataset.get_y_train()
        self.Y_test = self.dataset.get_y_test()
        self.nClasses = len(np.unique(self.Y_train))

        # Normalizing data
        self.normalize_data()

        # Make Neural Network Model
        log.info('Building network model')
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.LSTM(128, input_shape=(self.X_train.shape[1:]), activation=tf.nn.relu,
                                            return_sequences=True))
        self.model.add(tf.keras.layers.Dropout(0.2))
        self.model.add(tf.keras.layers.LSTM(128, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Dropout(0.1))
        self.model.add(tf.keras.layers.Dense(32, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Dropout(0.2))
        self.model.add(tf.keras.layers.Dense(self.nClasses, activation=tf.nn.softmax))
        opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        # Checkout the model
        self.model.summary()

        # When trained model exist should be loaded insteed train new one
        if os.path.exists('mnistnet_RNN'):
            self.load_model()
        else:
            self.train_network(epochs=7)
            self.save_model()

        self.val_loss, self.val_acc = self.model.evaluate(self.X_test, self.Y_test)

    def normalize_data(self) -> None:
        """
        Normalizing data in dataset
        :return: None
        """
        log.info('Normalizing data...')
        self.X_train = tf.keras.utils.normalize(self.X_train, axis=1)
        self.X_test = tf.keras.utils.normalize(self.X_test, axis=1)

    def train_network(self, epochs) -> None:
        """
        Fiting the network with data
        :param epochs: number of training epochs
        :return: None
        """
        self.model.fit(self.X_train, self.Y_train, epochs=epochs, validation_data=(self.X_test, self.Y_test))

    def save_model(self) -> None:
        """
        Saving the trained model
        :return:
        """
        log.info('Training model...')
        self.model.save('mnistnet_RNN')

    def load_model(self) -> None:
        """
        Loading trained model
        :return: None
        """
        log.info('Loading trained model')
        self.model = tf.keras.models.load_model('mnistnet_RNN')

    def get_val_loss(self) -> float:
        """
        Get validation loss
        :return: Validation loss value
        """
        return self.val_loss

    def get_val_acc(self) -> float:
        """
        Get validation accuracy
        :return: Validation accuracy value
        """
        return self.val_acc

    def get_prediction(self, data: np.array, id: int) -> np.array:
        """
        Predict the given image
        :param data: array of images
        :param id: with one to see
        :return: Model predicted image array
        """
        log.info('Predicting data')
        output = self.model.predict([data])
        output = np.argmax(output[id])
        self.dataset.show_image(data[id])
        return output
