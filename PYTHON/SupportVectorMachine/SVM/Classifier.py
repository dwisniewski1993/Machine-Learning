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
    def __init__(self, train_file: str) -> None:
        """
        SVM Regression Constructor
        Loading and preparing data
        :param train_file: iris data csv path
        """
        logging.getLogger().setLevel(logging.INFO)
        logging.info('SVM Classifier')

        self.trainFile: str = train_file
        train_data_frame: pd.DataFrame = pd.read_csv(self.trainFile)
        train_array: np.ndarray = train_data_frame.values

        np.random.shuffle(train_array)

        self.X: np.ndarray = train_array[:, 0:4]
        self.Y: np.ndarray = train_array[:, 4]

        self.grid_params: dict = {}
        self.model: SVC = None

        self.Y = self.map_labels(self.Y)

        self.X_train: np.ndarray
        self.X_test: np.ndarray
        self.Y_train: np.ndarray
        self.Y_test: np.ndarray
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.3,
                                                                                random_state=0)

    def __str__(self) -> None:
        """
        Printing data
        :return: None
        """
        print("Features: {}, Labels: {}".format(self.X, self.Y))

    @staticmethod
    def map_labels(labels: np.ndarray) -> List[float]:
        """
        Maping iris data labels to numeric
        :param labels: numpy.Arrays contains labels
        :return: list of mapped values
        """
        return [0.0 if x == 'Iris-setosa' else 1.0 if x == 'Iris-versicolor' else 2.0 for x in labels]

    def normalize(self) -> None:
        """
        Normalizing data in dataset
        :return: None
        """
        scaler = Normalizer()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def score(self) -> None:
        """
        Calculating and logging accuracy score
        :return: None
        """
        logging.info(f"F1 Score: {f1_score(self.Y_test, self.model.predict(self.X_test), average='weighted'):.2f}")

    def grid_search(self) -> None:
        """
        Sklearn hyper-parameters grid search
        :return: None
        """
        classifier = GridSearchCV(SVC(), parameters, cv=5)
        classifier.fit(self.X_train, self.Y_train)
        self.grid_params = classifier.best_params_

    def train_model(self) -> None:
        """
        Fitting model with grid search hyper-parameters
        :return: None
        """
        self.model = SVC(**self.grid_params).fit(self.X_train, self.Y_train)