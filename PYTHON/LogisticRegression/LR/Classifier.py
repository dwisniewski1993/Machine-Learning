import logging as log

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import *


class LogReg:
    """
    Logistic Regression Classification
    """
    def __init__(self, train_file):
        """
        Logistic Regression Constructor
        :param train_file: path to file
        """
        log.getLogger().setLevel(log.INFO)
        log.info('Logistic Regression Classifier')
        self.trainFile = train_file

        train_data_frame = pd.read_csv(self.trainFile, header=0)
        train_data_frame = train_data_frame.dropna()
        train_data_frame['education'] = np.where(train_data_frame['education'] == 'basic.9y', 'Basic',
                                                 train_data_frame['education'])
        train_data_frame['education'] = np.where(train_data_frame['education'] == 'basic.6y', 'Basic',
                                                 train_data_frame['education'])
        train_data_frame['education'] = np.where(train_data_frame['education'] == 'basic.4y', 'Basic',
                                                 train_data_frame['education'])
        cat_vars = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week',
                    'poutcome']
        for var in cat_vars:
            cat_list = pd.get_dummies(train_data_frame[var], prefix=var)
            data = train_data_frame.join(cat_list)
            train_data_frame = data

        cat_vars = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week',
                    'poutcome']
        data_vars = train_data_frame.columns.values.tolist()
        to_keep = [i for i in data_vars if i not in cat_vars]
        final_data = train_data_frame[to_keep]
        cols = ["previous", "euribor3m", "job_blue-collar", "job_retired", "job_services", "job_student", "default_no",
                "month_aug", "month_dec", "month_jul", "month_nov", "month_oct", "month_sep", "day_of_week_fri",
                "day_of_week_wed", "poutcome_failure", "poutcome_nonexistent", "poutcome_success"]
        self.features = final_data[cols]
        self.labels = final_data['y']
        self.grid_params = []
        self.model = None

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.features, self.labels,
                                                                                test_size=0.3, random_state=0)

    def __str__(self):
        """
        Printing data
        :return: None
        """
        log.info(f"Train File: {self.trainFile}")

    def train(self):
        """
        Fiting model with grid search hyper-parameters
        :return: None
        """
        self.model = LogisticRegression(**dict(self.grid_params))
        self.model.fit(self.X_train, self.Y_train)

    def normalize(self):
        """
        Normalizing data in dataset
        :return: None
        """
        scaler = Normalizer()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.fit_transform(self.X_test)

    def score(self):
        """
        Calculating and logging f1 score
        :return: None
        """
        log.info(f"F1 Score: {f1_score(self.Y_test, self.model.predict(self.X_test), average='weighted'):.2f}")

    def grid_search(self):
        """
        Fiting model with grid search hyper-parameters
        :return: None
        """
        parameters = {
            'C': [1, 3, 5, 7, 9],
            'solver': ('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'),
            'max_iter': [10000]
        }
        classifier = GridSearchCV(LogisticRegression(), parameters, cv=5)
        classifier.fit(self.X_train, self.Y_train)
        self.grid_params = classifier.best_params_
