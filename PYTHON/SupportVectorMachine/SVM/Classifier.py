import logging
from typing import List
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import Normalizer
from config import parameters

logging.basicConfig(level=logging.INFO)


class SupportVectorMachineClassification:
    """
    Support Vector Machine Classification
    """
    def __init__(self, train_file: str):
        """
        SVM Classifier Constructor
        Loading and preparing data
        :param train_file: iris data csv path
        """
        logging.getLogger().setLevel(logging.INFO)
        logging.info('SVM Classifier')

        self.trainFile = train_file
        train_data_frame = pd.read_csv(self.trainFile)
        train_array = train_data_frame.values

        np.random.shuffle(train_array)

        self.X = train_array[:, 0:4]
        self.Y = train_array[:, 4]

        self.grid_params = {}
        self.model = None

        self.Y = self.map_labels(self.Y)

        self.X_train = np.ndarray
        self.X_test = np.ndarray
        self.Y_train = np.ndarray
        self.Y_test = np.ndarray
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.X, self.Y, test_size=0.3, random_state=0
        )

    def __repr__(self):
        """
        Return a string representation of the object
        """
        return f"Features: {self.X}, Labels: {self.Y}"

    @staticmethod
    def map_labels(labels: np.ndarray) -> List[float]:
        """
        Mapping iris data labels to numeric
        """
        return [0.0 if x == 'Iris-setosa' else 1.0 if x == 'Iris-versicolor' else 2.0 for x in labels]

    def normalize(self):
        """
        Normalizing data in dataset
        """
        scaler = Normalizer()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def score(self):
        """
        Calculating and logging accuracy score
        """
        logging.info(
            f"F1 Score: {f1_score(self.Y_test, self.model.predict(self.X_test), average='weighted'):.2f}"
        )

    def grid_search(self):
        """
        Sklearn's hyperparameters grid search
        """
        classifier = GridSearchCV(SVC(), parameters, cv=5)
        classifier.fit(self.X_train, self.Y_train)
        self.grid_params = classifier.best_params_

    def train_model(self):
        """
        Fitting model with grid search hyperparameters
        """
        self.model = SVC(**self.grid_params).fit(self.X_train, self.Y_train)