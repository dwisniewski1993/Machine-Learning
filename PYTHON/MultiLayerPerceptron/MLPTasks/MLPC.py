import logging as log

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import multilayer_perceptron
from sklearn.preprocessing import *


class MultiLayerPerceptronClassifier:
    """
    Multilayer Perceptron Classifier
    """

    def __init__(self, trainfile):
        """
        MLP Classification Constructor
        Loading and preparing data
        :param trainfile: iris data csv path
        """
        log.getLogger().setLevel(log.INFO)
        log.info('MLP Classifier')

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
        self.mlpc = None

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
        log.info(f"Accuracy: {self.mlpc.score(self.X_test, self.Y_test):.2f}")

    def grid_search(self):
        """
        Sklearn hyper-parameters grid search
        :return: None
        """
        hyperparam_grid = {
            'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
            'learning_rate_init': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
        }
        classifier = GridSearchCV(multilayer_perceptron.MLPClassifier(max_iter=10000), hyperparam_grid, cv=5, iid=False)
        classifier.fit(self.X_train, self.Y_train)
        self.grided_params = [classifier.best_estimator_.alpha, classifier.best_estimator_.learning_rate_init]

    def train_model(self):
        """
        Fiting model with grid search hyper-parameters
        :return: None
        """
        self.mlpc = multilayer_perceptron.MLPClassifier(activation='relu', solver='adam', alpha=self.grided_params[0],
                                                        learning_rate_init=self.grided_params[1], max_iter=10000).fit(
            self.X_train, self.Y_train)
