import logging as log

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import *


class NaiveBayesClassifier:
    """
    Naive Bayes Classification
    """
    def __init__(self, train_file):
        """
        Naive Bayes Constructor
        Loading and preparing data
        :param train_file: iris data csv path
        """
        log.getLogger().setLevel(log.INFO)
        log.info('Naive Bayes Classifier')

        # Load set
        self.trainFile = train_file
        train_data_frame = pd.read_csv(self.trainFile)
        train_array = train_data_frame.values

        # Shuffle Data
        np.random.shuffle(train_array)

        # Extract values to numpy.Arrays
        self.X = train_array[:, 0:4]
        self.Y = train_array[:, 4]

        self.model = None

        # Map string labels to numeric
        self.Y = self.map_labels(self.Y)

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.3,
                                                                                random_state=0)

    def __str__(self):
        """
        Printing data
        :return: None
        """
        print("Features: {}, Labels: {}".format(self.X, self.Y))

    @staticmethod
    def map_labels(labels):
        """
        Maping iris data labels to numeric
        :param labels: numpy.Arrays contains labels
        :return: list of mapped values
        """
        return [0.0 if x == 'Iris-setosa' else 1.0 if x == 'Iris-versicolor' else 2.0 for x in labels]

    def normalize(self):
        """
        Normalizing data in dataset
        :return: None
        """
        scaler = Normalizer()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.fit_transform(self.X_test)

    def train_model(self):
        """
        Fiting model with grid search hyper-parameters
        :return: None
        """
        self.model = GaussianNB()
        self.model.fit(self.X_train, self.Y_train)

    def score(self):
        """
        Calculating and logging f1 score
        :return: None
        """
        log.info(f"F1 Score: {f1_score(self.Y_test, self.model.predict(self.X_test), average='weighted'):.2f}")
