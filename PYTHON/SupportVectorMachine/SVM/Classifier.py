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
        :param train_file: Path to the training data file
        """

        # Initialize the SVM Classifier
        self.trainFile = train_file

        # Read the CSV file and drop any rows with missing values
        train_data_frame = pd.read_csv(self.trainFile).dropna()

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
        Return a string representation of the object
        """
        return f"Features: {self.X}, Labels: {self.Y}"

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
        Calculate F1 score
        :return: F1 score
        """
        # Predict labels for the test data and calculate the F1 score
        return f1_score(self.Y_test, self.model.predict(self.X_test), average='weighted')

    def grid_search(self):
        """
        Perform grid search for hyperparameters
        """
        # Perform grid search to find the best hyperparameters
        classifier = GridSearchCV(SVC(), parameters, cv=10)
        classifier.fit(self.X_train, self.Y_train)

        # Store the best hyperparameters
        self.grid_params = classifier.best_params_

    def train_model(self):
        """
        Train the model using the best hyperparameters found
        """
        # Train the SVM model using the best hyperparameters
        self.model = SVC(**self.grid_params).fit(self.X_train, self.Y_train)
