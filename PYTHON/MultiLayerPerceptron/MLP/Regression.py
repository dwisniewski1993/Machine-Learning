import logging as log

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import neural_network
from config import parameters


class MultilayerPerceptronRegression:
    """
    Multilayer Perceptron Classifier
    """

    def __init__(self, train_file):
        """
        MLP Classification Constructor
        Loading and preparing data
        :param train_file: iris data csv path
        """
        log.getLogger().setLevel(log.INFO)
        log.info('MLP Regressor')

        # Load set
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

        # Shuffle Data
        np.random.shuffle(train_array)

        # Extract values to numpy.Arrays
        self.X = train_array[:, 1:]
        self.Y = train_array[:, 0]

        self.grid_params = []
        self.model = None

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
    def map_columns(df, col_number: int):
        """
        Mapping non numeric values to numeric
        :param df: pandas dataframe that contain dataset
        :param col_number: number collumn to map
        :return: dictionary with mapped values
        """
        return dict([(y, x + 1) for x, y in enumerate(sorted(set(df[col_number].unique())))])

    def score(self):
        """
        Predicting and log values
        :return: None
        """
        y_pred = self.model.predict(self.X_test)
        log.info(f"R2 Score: {r2_score(self.Y_test, y_pred)}")

    def grid_search(self):
        """
        Sklearn hyper-parameters grid search
        :return: None
        """
        classifier = GridSearchCV(neural_network.MLPRegressor(), parameters, cv=5,
                                  scoring=make_scorer(r2_score, greater_is_better=True))
        classifier.fit(self.X_train, self.Y_train)
        self.grid_params = classifier.best_params_

    def train_model(self):
        """
        Fiting model with grid search hyper-parameters
        :return: None
        """
        self.model = neural_network.MLPRegressor(**dict(self.grid_params)).fit(self.X_train, self.Y_train)
