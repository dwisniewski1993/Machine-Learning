import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from config import parameters, scaler


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

        self.trainFile = train_file
        train_data_frame = pd.read_csv(self.trainFile).dropna()
        train_array = train_data_frame.values

        np.random.shuffle(train_array)

        cols = len(train_array[0]) - 1

        self.X = train_array[:, 0:cols]
        self.Y = train_array[:, cols]

        self.grid_params = {}
        self.model = None

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

    def rescale_data(self, scaler_type: str):
        """
        Normalizing data in dataset
        """
        preprocessor = scaler[scaler_type]
        self.X_train = preprocessor.fit_transform(self.X_train)
        self.X_test = preprocessor.transform(self.X_test)

    def score(self) -> float:
        """
        Calculating F1 score
        """
        return f1_score(self.Y_test, self.model.predict(self.X_test), average='weighted')

    def grid_search(self):
        """
        Sklearn's hyperparameters grid search
        """
        classifier = GridSearchCV(SVC(), parameters, cv=10)
        classifier.fit(self.X_train, self.Y_train)
        self.grid_params = classifier.best_params_

    def train_model(self):
        """
        Fitting model with grid search hyperparameters
        """
        self.model = SVC(**self.grid_params).fit(self.X_train, self.Y_train)
