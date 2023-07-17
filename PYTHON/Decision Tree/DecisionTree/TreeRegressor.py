import logging as log

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
from sklearn.tree import DecisionTreeRegressor


class DTRegressor:
    """
    Decision Tree Regressor
    """

    def __init__(self, train_file: str) -> None:
        """
        Decision Tree Regression Constructor
        Loading and preparing data
        :param train_file: Path to the Poznań flats data TSV file
        """
        log.getLogger().setLevel(log.INFO)
        log.info('Decision Tree Regressor')

        # Load dataset
        self.train_file = train_file
        train_data_frame = pd.read_csv(self.train_file, sep='\t', header=None)

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

        # Extract values to numpy arrays
        self.X = train_array[:, 1:]
        self.Y = train_array[:, 0]

        self.grided_params = []
        self.dtr = None

        # Split into train-test sets
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.3,
                                                                                random_state=0)

    def __str__(self) -> None:
        """
        Printing data
        :return: None
        """
        print("Features: {}, Labels: {}".format(self.X, self.Y))

    @staticmethod
    def map_columns(df: pd.DataFrame, col_number: int) -> dict:
        """
        Mapping non-numeric values to numeric
        :param df: Pandas DataFrame that contains the dataset
        :param col_number: Column number to map
        :return: Dictionary with mapped values
        """
        return dict([(y, x + 1) for x, y in enumerate(sorted(set(df[col_number].unique())))])

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

    def output(self) -> None:
        """
        Predicting and logging values
        :return: None
        """
        y_pred = self.dtr.predict(self.X_test)
        log.info(f"R2 Score: {r2_score(self.Y_test, y_pred):.2f}")

    def train_model(self) -> None:
        """
        Fitting the model with grid search hyperparameters
        :return: None
        """
        self.dtr = DecisionTreeRegressor(max_depth=self.grided_params[0])
        self.dtr.fit(self.X_train, self.Y_train)

    def grid_search(self) -> None:
        """
        Perform grid search for hyperparameters
        :return: None
        """
        hyperparam_grid = {'max_depth': np.arange(2, 15)}
        classifier = GridSearchCV(DecisionTreeRegressor(), hyperparam_grid, cv=5, scoring='explained_variance')
        classifier.fit(self.X_train, self.Y_train)
        self.grided_params = [classifier.best_estimator_.max_depth]
