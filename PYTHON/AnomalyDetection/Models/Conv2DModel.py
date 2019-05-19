import logging
import os.path
import time

import numpy as np
import tensorflow as tf
from scipy.spatial import distance
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.python.keras.models import Sequential

from Models.Utils import SWATDataHandler, Preprocessing
from Models.exeptions import DataNotEqual, InvalidShapes


class Conv2DModel:
    def __init__(self, healthy_data: str, broken_data: str, dataset_name: str, timesteps: int) -> None:
        """
        AutoEncoder Conv1D Artifical Neural Network for anomaly detection.
        Init the dataset, dataset shapes and pre-processing.

        :param healthy_data: healthy data csv location
        :param broken_data: data with anomalies csv location
        :param dataset_name: unique dataset name for models
        :param timesteps: step in time per example
        """
        logging.basicConfig(level=logging.INFO)
        self.data_name = dataset_name
        self.logger = logging.getLogger(__name__)
        handler = SWATDataHandler(file_normal=healthy_data, file_brocken=broken_data)

        self.normal_data = handler.get_dataset_normal()
        self.attk_data = handler.get_dataset_broken()
        self.Y = handler.get_broken_labels()

        scaler = Preprocessing()
        self.normal_data = scaler.scaleData(data=self.normal_data)
        self.attk_data = scaler.scaleData(data=self.attk_data)

        normal_samples = len(self.normal_data)
        normal_dim = len(self.normal_data[0])
        self.timesteps = timesteps
        try:
            self.normal_data.shape = (int(normal_samples / self.timesteps), self.timesteps, normal_dim, 1)
        except InvalidShapes:
            raise InvalidShapes("Something wrong with dataset shapes -_-")

        attk_samples = len(self.attk_data)
        attk_dim = len(self.attk_data[0])
        try:
            self.attk_data.shape = (int(attk_samples / self.timesteps), self.timesteps, attk_dim, 1)
        except InvalidShapes:
            raise InvalidShapes("Something wrong with dataset shapes -_-")

        self.logger.info('Initializing Conv2D model...')

        self.dim = self.normal_data.shape[2]
        self.samples = self.normal_data.shape[0]
        self.model = self.define_model()
        self.threshold = 0
        NAME = "Conv2D-{}-{}".format(dataset_name, int(time.time()))
        self.tb = TensorBoard(log_dir='logs/{}'.format(NAME))

    def define_model(self) -> Sequential:
        """
        Defining the specify Conv2D architecteure: activation, number of layers, optimizer and error measure.
        :return: Keras Sequential Model
        """
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=[self.timesteps, self.dim, 1]))
        model.add(MaxPooling2D((2, 3)))

        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(UpSampling2D((2, 3)))
        model.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.summary()
        self.logger.info('Defining Conv2D network architecture...')
        return model

    def train(self, retrain: bool = False) -> None:
        """

        :param retrain: Optional param, default False, if true network will train model even if one already exist.
        :return: None, saving the network model file if not exist or retrain is True
        """
        data = self.normal_data
        print(data.shape)
        self.model.summary()
        if os.path.exists(self.data_name + '__Conv2D_model'):
            self.logger.info('Trained Conv2D model detected')
            if retrain:
                self.logger.info('Start training Conv2D Network.....')
                self.model.fit(data, data, epochs=100, batch_size=72, validation_data=(data, data), callbacks=[self.tb],
                               verbose=0)
                self.save_model()
                self.logger.info('Training Conv2D done!')
            else:
                self.load_model()
        else:
            self.logger.info('Start training Conv2D Network.....')
            self.model.fit(data, data, epochs=100, batch_size=72, validation_data=(data, data), callbacks=[self.tb],
                           verbose=0)
            self.save_model()
            self.logger.info('Training Conv2D done!')
        self.stats(data=data)

    def score(self, data: np.array) -> np.array:
        """

        :param data: Data for prediction
        :return: Value matrix (numpy ndarray)
        """
        self.logger.info('Calculate Conv2D Score...')
        yhat = self.model.predict(data)
        yhat.shape = (data.shape[0] * data.shape[1], data.shape[2])
        data.shape = (data.shape[0] * data.shape[1], data.shape[2])
        self.logger.info('Calculate Conv2D Score complete')
        return yhat

    def save_model(self):
        """
        Save trained model to file
        :return: None
        """
        self.logger.info('Saving Conv2D model...')
        self.model.save(self.data_name + '__Conv2D_model')

    def load_model(self):
        """
        Load model from file
        :return: None
        """
        self.logger.info('Load Conv2D model...')
        self.model = tf.keras.models.load_model(self.data_name + '__Conv2D_model')

    def get_normal_data(self) -> np.array:
        """
        Give healthy data, training one
        :return: Healthy data
        """
        return self.normal_data

    def get_attk_data(self) -> np.array:
        """
        Give broken data with anomalie to detect
        :return: Broken data
        """
        return self.attk_data

    def stats(self, data: np.array) -> None:
        """

        :param data: Data to validate the trained models
        :return: None
        """
        val_loss = self.model.evaluate(data, data, batch_size=72, verbose=0)
        self.logger.info('Validation Conv2D Loss: ' + str(val_loss))

    def calculate_threshold(self, helth: np.array):
        """
        Calculate and set threshold for anomaly detection
        :param helth: Health matrix value from predict
        :return: None
        """
        self.logger.info('Calculate Conv2D threshold...')
        actual = self.normal_data
        dists = []
        if len(helth) == len(actual):
            for i in range(len(helth)):
                helth[i] = helth[i].tolist()
                actual[i] = actual[i].tolist()
                dist = distance.euclidean(helth[i], actual[i])
                dists.append(dist)
            self.threshold = max(dists)
            self.logger.info('Threshold calculated Conv2D: {}'.format(self.threshold))
        else:
            raise DataNotEqual("Both sets should have the same number of samples")

    def anomaly_score(self, pred: np.array):
        """
        Anomaly detector, basing on calculated distance and threshold deciding if data is normal or with anomaly.
        :param pred: Matrix values from predicted anomaly set
        :return: None, saving files with pointed anomaly and real ones per calculated distance.
        """
        self.logger.info('Calculating distance to anomaly...')
        dists = []
        actual = self.attk_data
        Y = self.Y
        if len(pred) == len(actual):
            for i in range(len(pred)):
                pred[i] = pred[i].tolist()
                actual[i] = actual[i].tolist()
                dist = distance.euclidean(pred[i], actual[i])
                if dist > self.threshold:
                    score = 'Anomaly'
                else:
                    score = 'Normal'
                dists.append((dist, Y[i], score))
            with open(r'Conv2D_detected_' + self.data_name + '.csv', 'w') as file:
                for each in dists:
                    file.write(str(each[0]) + ',' + each[1] + ',' + each[2] + '\n')
            self.logger.info('Anomaly score calculated')
        else:
            raise DataNotEqual("Both sets should have the same number of samples")
