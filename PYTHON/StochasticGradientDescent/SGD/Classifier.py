import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from config import parameters, scaler  # Assuming `config` module contains predefined parameters and scalers


class StochasticGradientDescentClassification:
    """
    Stochastic Gradient Descent Classification
    """

    def __init__(self, train_file):
        """
        Constructor for SGD Classification
        Loads and prepares data
        """

        self.trainFile = train_file
        train_data_frame = pd.read_csv(self.trainFile).dropna()  # Load data and drop missing values
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
        Prints features and labels
        :return: None
        """
        print("Features: {}, Labels: {}".format(self.X, self.Y))

    def rescale_data(self, scaler_type: str):
        """
        Normalizes data in the dataset
        """
        preprocessor = scaler[scaler_type]  # Select preprocessor based on the given scaler type
        self.X_train = preprocessor.fit_transform(self.X_train)  # Fit and transform training features
        self.X_test = preprocessor.transform(self.X_test)  # Transform testing features

    def score(self) -> float:
        """
        Calculates F1 score
        :return: Score
        """
        return f1_score(self.Y_test, self.model.predict(self.X_test), average='weighted')

    def grid_search(self):
        """
        Performs hyperparameters grid search
        :return: None
        """
        classifier = GridSearchCV(SGDClassifier(), parameters, cv=10)  # Grid search with cross-validation
        classifier.fit(self.X_train, self.Y_train)  # Fit the classifier to the training data
        self.grid_params = classifier.best_params_  # Store the best parameters

    def train_model(self):
        """
        Fits the model with grid search hyperparameters
        :return: None
        """
        self.model = SGDClassifier(**dict(self.grid_params)).fit(self.X_train, self.Y_train)  # Fit the model
