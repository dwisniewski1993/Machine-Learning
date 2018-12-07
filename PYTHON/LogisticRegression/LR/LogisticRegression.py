import logging as log

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import *


class LogReg:
    """
    Logistic Regression Classification
    """
    def __init__(self, trainfile):
        """
        Logistic Regression Constructor
        :param trainfile: path to file
        """
        log.getLogger().setLevel(log.INFO)
        log.info('Logistic Regression Classifier')
        self.trainFile = trainfile

        trainDataFrame = pd.read_csv(self.trainFile, header=0)
        trainDataFrame = trainDataFrame.dropna()
        trainDataFrame['education'] = np.where(trainDataFrame['education'] == 'basic.9y', 'Basic',
                                               trainDataFrame['education'])
        trainDataFrame['education'] = np.where(trainDataFrame['education'] == 'basic.6y', 'Basic',
                                               trainDataFrame['education'])
        trainDataFrame['education'] = np.where(trainDataFrame['education'] == 'basic.4y', 'Basic',
                                               trainDataFrame['education'])
        cat_vars = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week',
                    'poutcome']
        for var in cat_vars:
            cat_list = pd.get_dummies(trainDataFrame[var], prefix=var)
            data = trainDataFrame.join(cat_list)
            trainDataFrame = data

        cat_vars = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week',
                    'poutcome']
        data_vars = trainDataFrame.columns.values.tolist()
        to_keep = [i for i in data_vars if i not in cat_vars]
        finalData = trainDataFrame[to_keep]
        finalData.columns.values
        cols = ["previous", "euribor3m", "job_blue-collar", "job_retired", "job_services", "job_student", "default_no",
                "month_aug", "month_dec", "month_jul", "month_nov", "month_oct", "month_sep", "day_of_week_fri",
                "day_of_week_wed", "poutcome_failure", "poutcome_nonexistent", "poutcome_success"]
        self.features = finalData[cols]
        self.labels = finalData['y']
        self.grided_params = []
        self.lr = None

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.features, self.labels,
                                                                                test_size=0.3, random_state=0)

    def __str__(self):
        """
        Printing data
        :return: None
        """
        log.info(f"Train File: {self.trainFile}")

    def modelFiting(self):
        """
        Fiting model with grid search hyper-parameters
        :return: None
        """
        self.lr = LogisticRegression(C=self.grided_params[0], solver=self.grided_params[1], max_iter=10000)
        self.lr.fit(self.X_train, self.Y_train)

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
        log.info(f"Accuracy: {self.lr.score(self.X_test, self.Y_test):.2f}")

    def grid_search(self):
        """
        Fiting model with grid search hyper-parameters
        :return: None
        """
        hyperparam_grid = {
            'C': [1, 3, 5, 7, 9],
            'solver': ('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'),
        }
        classifier = GridSearchCV(LogisticRegression(), hyperparam_grid, cv=5, iid=False)
        classifier.fit(self.X_train, self.Y_train)
        self.grided_params = [classifier.best_estimator_.C,
                              classifier.best_estimator_.solver]
