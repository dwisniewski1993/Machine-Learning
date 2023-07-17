import logging as log

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
from sklearn.tree import DecisionTreeClassifier


class DTClassifier:
    """
    Decision Tree Classification
    """

    def __init__(self, train_file: str) -> None:
        """
        Decision Tree Classification Constructor
        Loading and preparing data
        :param train_file: Path to the iris data CSV file
        """
        log.getLogger().setLevel(log.INFO)
        log.info('Decision Tree Classifier')

        # Load dataset
        self.train_file = train_file
        train_data_frame = pd.read_csv(self.train_file)
        train_array = train_data_frame.values

        # Shuffle Data
        np.random.shuffle(train_array)

        # Extract values to numpy arrays
        self.X = train_array[:, 0:4]
        self.Y = train_array[:, 4]

        self.grided_params = []
        self.dtc = None

        # Map string labels to numeric
        self.Y = self.map_labels(self.Y)

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.3,
                                                                                random_state=0)

    def __str__(self) -> None:
        """
        Printing data
        :return: None
        """
        print("Features: {}, Labels: {}".format(self.X, self.Y))

    @staticmethod
    def map_labels(labels: np.array) -> list:
        """
        Mapping iris data labels to numeric
        :param labels: numpy array containing labels
        :return: list of mapped values
        """
        mapped = [0.0 if x == 'Iris-setosa' else 1.0 if x == 'Iris-versicolor' else 2.0 for x in labels]
        return mapped

    def rescale(self) -> None:
        """
        Rescaling data in the dataset to [0, 1]
        :return: None
        """
        scaler = MinMaxScaler(feature_range=(0, 1))
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def normalize(self) -> None:
        """
        Normalizing data in the dataset
        :return: None
        """
        scaler = Normalizer()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def standardize(self) -> None:
        """
        Standardizing data in the dataset
        :return: None
        """
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def train_model(self) -> None:
        """
        Fitting the model with grid search hyperparameters
        :return: None
        """
        self.dtc = DecisionTreeClassifier(max_depth=self.grided_params[0])
        self.dtc.fit(self.X_train, self.Y_train)

    def output(self) -> None:
        """
        Calculating and logging F1 score
        :return: None
        """
        log.info(f"F1 Score: {f1_score(self.Y_test, self.dtc.predict(self.X_test), average='weighted'):.2f}")

    def grid_search(self) -> None:
        """
        Perform grid search for hyperparameters
        :return: None
        """
        hyperparam_grid = {'max_depth': np.arange(2, 15)}
        classifier = GridSearchCV(DecisionTreeClassifier(), hyperparam_grid, cv=5)
        classifier.fit(self.X_train, self.Y_train)
        self.grided_params = [classifier.best_estimator_.max_depth]
