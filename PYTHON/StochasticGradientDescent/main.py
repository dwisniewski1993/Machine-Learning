import os
import logging
from config import scaler_type  # Import scaler types from config file
from SGD.Classifier import StochasticGradientDescentClassification  # Import SGD Classification class
from SGD.Regressor import StochasticGradientDescentRegression  # Import SGD Regression class

# Set up logging configuration
logging.basicConfig(level=logging.INFO)


def main():
    """
    Main function. Run Stochastic Gradient Descent classification and regression tasks.
    :return: None
    """

    # SGD Classification
    path_class = "../../_datasets_classification"  # Path to classification datasets
    files = os.listdir(path_class)  # List files in the classification dataset directory
    for file in files:
        logging.info(f"----------{file}----------")
        logging.info('SGD Classifier')
        train_file = f"{path_class}/{file}"  # File path for classification training data
        for each in scaler_type:  # Iterate through each scaler type
            sgd_classifier = StochasticGradientDescentClassification(train_file=train_file)  # Create SGD instance
            sgd_classifier.rescale_data(scaler_type=each)  # Rescale data using specified scaler type
            sgd_classifier.grid_search()  # Perform grid search for hyperparameters
            sgd_classifier.train_model()  # Train the model
            score = sgd_classifier.score()  # Calculate F1 score
            logging.info(f"F1 score for scaler {each}: {score:.2f}")  # Log F1 score

    # SGD Regression
    path_reg = "../../_datasets_regression"  # Path to regression datasets
    files = os.listdir(path_reg)  # List files in the regression dataset directory
    for file in files:
        logging.info(f"----------{file}----------")
        logging.info('SGD Regressor')
        train_file = f"{path_reg}/{file}"  # File path for regression training data
        for each in scaler_type:  # Iterate through each scaler type
            sgd_regressor = StochasticGradientDescentRegression(train_file=train_file)  # Create SGD regressor instance
            sgd_regressor.rescale_data(scaler_type=each)  # Rescale data using specified scaler type
            sgd_regressor.grid_search()  # Perform grid search for hyperparameters
            sgd_regressor.train_model()  # Train the model
            score = sgd_regressor.score()  # Calculate R2 score
            logging.info(f"R2 score for scaler {each}: {score}")  # Log R2 score


if __name__ == '__main__':
    main()
