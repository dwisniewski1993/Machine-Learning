import numpy as np
import pandas as pd
import warnings
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import r2_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from config import parameters, scaler
from sklearn.exceptions import ConvergenceWarning

# Ignore convergence warnings
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)


class StochasticGradientDescentRegression:
    """
    Stochastic Gradient Descent Regression
    """

    def __init__(self, train_file):
        """
        Constructor for Stochastic Gradient Descent Regression class
        Loading and preparing data
        :param train_file: csv file path
        """

        self.trainFile = train_file
        train_data_frame = pd.read_csv(self.trainFile)

        # Finding columns containing string data
        string_columns = train_data_frame.select_dtypes(include='object').columns.tolist()

        for col in string_columns:
            self.mapping_string = self.map_columns(train_data_frame, col)
            train_data_frame = train_data_frame.map(
                lambda x: self.mapping_string.get(x) if x in self.mapping_string else x)

        train_array = train_data_frame.values

        np.random.shuffle(train_array)

        cols = len(train_array[0]) - 1

        self.X = train_array[:, 0:cols]  # Features
        self.Y = train_array[:, cols]  # Labels

        self.grid_params = {}  # Hyperparameters grid
        self.model = None  # Initialized model

        self.X_train = np.ndarray  # Training features
        self.X_test = np.ndarray  # Testing features
        self.Y_train = np.ndarray  # Training labels
        self.Y_test = np.ndarray  # Testing labels
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.X, self.Y, test_size=0.3, random_state=0
        )  # Split data into training and testing sets

    def __str__(self):
        """
        Printing data
        :return: None
        """
        print("Features: {}, Labels: {}".format(self.X, self.Y))

    @staticmethod
    def map_columns(df, col_number: str):
        """
        Mapping non-numeric values to numeric
        :param df: pandas dataframe that contains dataset
        :param col_number: column number to map
        :return: dictionary with mapped values
        """
        return dict([(y, x + 1) for x, y in enumerate(sorted(set(df[col_number].unique())))])

    def rescale_data(self, scaler_type: str):
        """
        Normalizing data in dataset
        """
        preprocessor = scaler[scaler_type]  # Select preprocessor based on the given scaler type
        self.X_train = preprocessor.fit_transform(self.X_train)  # Fit and transform training features
        self.X_test = preprocessor.transform(self.X_test)  # Transform testing features

    def score(self) -> float:
        """
        Predicting and logging values
        """
        y_pred = self.model.predict(self.X_test)
        return r2_score(self.Y_test, y_pred)

    def grid_search(self):
        """
        Sklearn hyper-parameters grid search
        :return: None
        """
        classifier = GridSearchCV(SGDRegressor(), parameters, cv=10,
                                  scoring=make_scorer(r2_score, greater_is_better=True))
        classifier.fit(self.X_train, self.Y_train)
        self.grid_params = classifier.best_params_

    def train_model(self):
        """
        Fitting model with grid search hyper-parameters
        :return: None
        """
        self.model = SGDRegressor(**dict(self.grid_params)).fit(self.X_train, self.Y_train)
