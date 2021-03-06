import logging as log

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import *


class KNNRegression:
    def __init__(self, trainfile):
        log.getLogger().setLevel(log.INFO)
        log.info('KNN Regressor')

        # Load set
        self.trainFile = trainfile
        trainDataFrame = pd.read_csv(self.trainFile, sep='\t', header=None)

        # Mapping string and bool values to numeric
        self.mapping_string = self.map_columns(trainDataFrame, 4)
        self.mapping_bool = self.map_columns(trainDataFrame, 1)
        trainDataFrame = trainDataFrame.applymap(
            lambda x: self.mapping_string.get(x) if x in self.mapping_string else x)
        trainDataFrame = trainDataFrame.applymap(
            lambda x: self.mapping_bool.get(x) if x in self.mapping_bool else x)
        trainArray = trainDataFrame.values

        # Shuffle Data
        np.random.shuffle(trainArray)

        # Extract values to numpy.Arrays
        self.X = trainArray[:, 1:]
        self.Y = trainArray[:, 0]

        self.grided_params = []
        self.knnr = None

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
    def map_columns(df, colnumber: int):
        """
        Mapping non numeric values to numeric
        :param df: pandas dataframe that contain dataset
        :param colnumber: number collumn to map
        :return: dictionary with mapped values
        """
        return dict([(y, x + 1) for x, y in enumerate(sorted(set(df[colnumber].unique())))])

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
        Predicting and log values
        :return: None
        """
        y_pred = self.knnr.predict(self.X_test)
        log.info(f"MSE: {mean_squared_error(self.Y_test, y_pred)}")
        for x, y in zip(y_pred, self.Y_test):
            log.info(f"Predicted: {x}| Actual: {y}")

    def train_model(self):
        """
        Fiting model with grid search hyper-parameters
        :return: None
        """
        self.knnr = KNeighborsRegressor(n_neighbors=self.grided_params[0], weights=self.grided_params[1],
                                        algorithm=self.grided_params[2]).fit(self.X_train, self.Y_train)

    def grid_search(self):
        """
        Sklearn hyper-parameters grid search
        :return: None
        """
        hyperparam_grid = {
            'n_neighbors': [5, 10, 15, 20, 25, 30],
            'weights': ('uniform', 'distance'),
            'algorithm': ('ball_tree', 'kd_tree', 'brute')

        }
        classifier = GridSearchCV(KNeighborsRegressor(), hyperparam_grid, cv=5, iid=False)
        classifier.fit(self.X_train, self.Y_train)
        self.grided_params = [classifier.best_estimator_.n_neighbors, classifier.best_estimator_.weights,
                              classifier.best_estimator_.algorithm]
