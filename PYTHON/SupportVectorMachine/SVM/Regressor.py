from typing import Dict
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import r2_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from config import parameters, scaler


class SupportVectorMachineRegression:
    """
    Support Vector Machine Regression
    """

    def __init__(self, train_file: str):
        """
        SVM Regression Constructor
        Loading and preparing data
        :param train_file: Path to the training data file
        """
        # Initialize the SVR Regression
        self.trainFile = train_file

        # Read the CSV file
        train_data_frame = pd.read_csv(self.trainFile)

        # Find columns containing string data
        string_columns = train_data_frame.select_dtypes(include='object').columns.tolist()

        # Map string values to numeric
        for col in string_columns:
            self.mapping_string = self.map_columns(train_data_frame, col)
            train_data_frame = train_data_frame.map(
                lambda x: self.mapping_string.get(x) if x in self.mapping_string else x)

        # Convert DataFrame to NumPy array
        train_array = train_data_frame.values

        # Shuffle the rows of the array
        np.random.shuffle(train_array)

        # Determine the number of columns
        cols = len(train_array[0]) - 1

        # Split features (X) and labels (Y)
        self.X = train_array[:, 0:cols]
        self.Y = train_array[:, cols]

        # Initialize variables for grid search results and model
        self.grid_params = {}
        self.model = None

        # Initialize variables for training and testing data
        self.X_train = np.ndarray
        self.X_test = np.ndarray
        self.Y_train = np.ndarray
        self.Y_test = np.ndarray

        # Split the data into training and testing sets
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.X, self.Y, test_size=0.3, random_state=0
        )

    def __repr__(self):
        """
        Printing data
        """
        print("Features: {}, Labels: {}".format(self.X, self.Y))

    @staticmethod
    def map_columns(df: pd.DataFrame, col_number: str) -> Dict:
        """
        Mapping non-numeric values to numeric
        """
        return {y: x + 1 for x, y in enumerate(sorted(set(df[col_number].unique())))}

    def rescale_data(self, scaler_type: str):
        """
        Normalizing data in dataset
        :param scaler_type: Type of scaler to use
        """
        # Retrieve the scaler based on the specified type
        preprocessor = scaler[scaler_type]

        # Scale the training and testing data
        self.X_train = preprocessor.fit_transform(self.X_train)
        self.X_test = preprocessor.transform(self.X_test)

    def score(self) -> float:
        """
        Calculating R2 score
        """
        # Predict labels for the test data and calculate the R2 score
        y_pred = self.model.predict(self.X_test)
        return r2_score(self.Y_test, y_pred)

    def grid_search(self):
        """
        Perform grid search for hyperparameters
        """
        # Perform grid search to find the best hyperparameters
        classifier = GridSearchCV(SVR(), parameters, cv=5, scoring=make_scorer(r2_score, greater_is_better=True))
        classifier.fit(self.X_train, self.Y_train)

        # Store the best hyperparameters
        self.grid_params = classifier.best_params_

    def train_model(self):
        """
        Train the model using the best hyperparameters found
        """
        # Train the SVR model using the best hyperparameters
        self.model = SVR(**self.grid_params).fit(self.X_train, self.Y_train)
