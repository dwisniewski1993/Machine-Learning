import logging as log

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import *


class SGDC:
    """
    Stochastic Gradient Descent Classification
    """

    def __init__(self, trainfile):
        """
        SGD Classification Constructor
        Loading and preparing data
        :param trainfile: iris data csv path
        """
        log.getLogger().setLevel(log.INFO)
        log.info('SGD Classification')

        # Load set
        self.trainFile = trainfile
        trainDataFrame = pd.read_csv(self.trainFile)
        trainArray = trainDataFrame.values

        # Shuffle Data
        np.random.shuffle(trainArray)

        # Extract values to numpy.Arrays
        self.X = trainArray[:, 0:4]
        self.Y = trainArray[:, 4]

        self.grided_params = []
        self.sgdc = None

        # Map string labels to numeric
        self.Y = self.map_labels(self.Y)

        # Split to train-test sets
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
        maped = [0.0 if x == 'Iris-setosa' else 1.0 if x == 'Iris-versicolor' else 2.0 for x in labels]
        return maped

    def rescale(self):
        """
        Rescaling data in dataset to [0,1]
        :return: None
        """
        scaler = MinMaxScaler(feature_range=(0, 1))
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.fit_transform(self.X_test)

    def normalize(self):
        """
        Normalizing data in dataset
        :return: None
        """
        scaler = Normalizer()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.fit_transform(self.X_test)

    def standalizer(self):
        """
        Standardlizing data in dataset
        :return: None
        """
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.fit_transform(self.X_test)

    def output(self):
        """
        Calculating and logging accuracy score
        :return: None
        """
        log.info(f"Accuracy: {self.sgdc.score(self.X_test, self.Y_test):.2f}")

    def grid_search(self):
        """
        Sklearn hyper-parameters grid search
        :return: None
        """
        hyperparam_grid = {
            'loss': ('hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'),
            'penalty': ('l1', 'l2', 'elasticnet'),
            'learning_rate': ('constant', 'optimal', 'invscaling', 'adaptive'),
            'eta0': [1, 3, 5, 7]
        }
        classifier = GridSearchCV(SGDClassifier(), hyperparam_grid, cv=5, iid=False)
        classifier.fit(self.X_train, self.Y_train)
        self.grided_params = [classifier.best_estimator_.loss, classifier.best_estimator_.penalty,
                              classifier.best_estimator_.learning_rate, classifier.best_estimator_.eta0]

    def train_model(self):
        """
        Fiting model with grid search hyper-parameters
        :return: None
        """
        self.sgdc = SGDClassifier(loss=self.grided_params[0], penalty=self.grided_params[1],
                                  learning_rate=self.grided_params[2], n_jobs=-1, eta0=self.grided_params[3]) \
            .fit(self.X_train, self.Y_train)
