import logging as log
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


class LinReg:
    """
    Linear Regression
    """
    def __init__(self, trainfile):
        """
        Linear Regression Constructor
        Loading and preparing data
        :param trainfile: Poznań flats data tsv path
        """
        log.getLogger().setLevel(log.INFO)
        log.info('Linear Regression')

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

        self.X = trainArray[:, 1:]
        self.Y = trainArray[:, 0]

        self.linreg = None

        # Split to train-test sets
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.3,
                                                                                random_state=0)

    def __str__(self):
        """
        Printing data
        :return: None
        """
        print("Features values: {}, Labels values {}".format(self.X, self.Y))

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

    def train_model(self):
        """
        Fiting model with grid search hyper-parameters
        :return: None
        """
        self.linreg = linear_model.LinearRegression()
        self.linreg.fit(self.X, self.Y)

    def output(self):
        """
        Predicting and log values
        :return: None
        """
        y_pred = self.linreg.predict(self.X_test)
        log.info(f"MSE: {mean_squared_error(self.Y_test, y_pred)}")
        for x, y in zip(y_pred, self.Y_test):
            log.info(f"Predicted: {x}| Actual: {y}")
