import logging
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import r2_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from config import parameters

logging.basicConfig(level=logging.INFO)


class SupportVectorMachineRegression:
    def __init__(self, train_file: str) -> None:
        self.trainFile: str = train_file
        train_data_frame: pd.DataFrame = pd.read_csv(self.trainFile, sep='\t', header=None)

        self.mapping_string: Dict = self.map_columns(train_data_frame, 4)
        self.mapping_bool: Dict = self.map_columns(train_data_frame, 1)
        train_data_frame = train_data_frame.applymap(
            lambda x: self.mapping_string.get(x) if x in self.mapping_string else x)
        train_data_frame = train_data_frame.applymap(
            lambda x: self.mapping_bool.get(x) if x in self.mapping_bool else x)
        train_array: np.ndarray = train_data_frame.values

        np.random.shuffle(train_array)

        self.X: np.ndarray = train_array[:, 1:]
        self.Y: np.ndarray = train_array[:, 0]

        self.grid_params: Dict = {}
        self.model: SVR = None

        self.X_train: np.ndarray
        self.X_test: np.ndarray
        self.Y_train: np.ndarray
        self.Y_test: np.ndarray
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.3,
                                                                                random_state=0)

    def __str__(self) -> None:
        print("Features: {}, Labels: {}".format(self.X, self.Y))

    @staticmethod
    def map_columns(df: pd.DataFrame, col_number: int) -> Dict:
        return {y: x + 1 for x, y in enumerate(sorted(set(df[col_number].unique())))}

    def standardize(self) -> None:
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def score(self) -> None:
        y_pred = self.model.predict(self.X_test)
        logging.info(f"R2 Score: {r2_score(self.Y_test, y_pred)}")

    def grid_search(self) -> None:
        classifier = GridSearchCV(SVR(), parameters, cv=5, scoring=make_scorer(r2_score, greater_is_better=True))
        classifier.fit(self.X_train, self.Y_train)
        self.grid_params = classifier.best_params_

    def train_model(self) -> None:
        self.model = SVR(**self.grid_params).fit(self.X_train, self.Y_train)
