import os
import logging
from config import scaler_type  # Import scaler types from config file
from SVM.Classifier import SupportVectorMachineClassification  # Import SVM Classification class
from SVM.Regressor import SupportVectorMachineRegression  # Import SVM Regression class


# Set up logging configuration
logging.basicConfig(level=logging.INFO)


def main():
    """
    Main function. Run SVM classification and regression tasks.
    :return: None
    """
    # SVM Classification
    path_class = "../../_datasets_classification"  # Path to classification datasets
    files = os.listdir(path_class)  # List files in the classification dataset directory
    for file in files:
        logging.info(f"----------{file}----------")
        logging.info('SVM Classifier')
        train_file = f"{path_class}/{file}"  # File path for classification training data
        for each in scaler_type:  # Iterate through each scaler type
            svm_classifier = SupportVectorMachineClassification(train_file=train_file)  # Create SVM classifier instance
            svm_classifier.rescale_data(scaler_type=each)  # Rescale data using specified scaler type
            svm_classifier.grid_search()  # Perform grid search for hyperparameters
            svm_classifier.train_model()  # Train the model
            score = svm_classifier.score()  # Calculate F1 score
            logging.info(f"F1 score for scaler {each}: {score:.2f}")  # Log F1 score

    # SVM Regression
    path_reg = "../../_datasets_regression"  # Path to regression datasets
    files = os.listdir(path_reg)  # List files in the regression dataset directory
    for file in files:
        logging.info(f"----------{file}----------")
        logging.info('SVM Regressor')
        train_file = f"{path_reg}/{file}"  # File path for regression training data
        for each in scaler_type:  # Iterate through each scaler type
            svm_regressor = SupportVectorMachineRegression(train_file=train_file)  # Create SVM regressor instance
            svm_regressor.rescale_data(scaler_type=each)  # Rescale data using specified scaler type
            svm_regressor.grid_search()  # Perform grid search for hyperparameters
            svm_regressor.train_model()  # Train the model
            score = svm_regressor.score()  # Calculate R2 score
            logging.info(f"R2 score for scaler {each}: {score}")  # Log R2 score


if __name__ == '__main__':
    main()
