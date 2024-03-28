import os
import logging
from config import scaler_type
from SVM.Classifier import SupportVectorMachineClassification
from SVM.Regressor import SupportVectorMachineRegression


logging.basicConfig(level=logging.INFO)


def main():
    """
    Main function. Run SVM classification and regression tasks.
    :return: None
    """
    # SVR Classification
    path_class = "../../_datasets_classification"
    files = os.listdir(path_class)
    for file in files:
        logging.info(f"----------{file}----------")
        logging.info('SVM Classifier')
        train_file = f"{path_class}/{file}"
        for each in scaler_type:
            svm_classifier = SupportVectorMachineClassification(train_file=train_file)
            svm_classifier.rescale_data(scaler_type=each)
            svm_classifier.grid_search()
            svm_classifier.train_model()
            score = svm_classifier.score()
            logging.info(f"F1 score for scaler {each}: {score:.2f}")

    # SVR Regression
    path_reg = "../../_datasets_regression"
    files = os.listdir(path_reg)
    for file in files:
        logging.info(f"----------{file}----------")
        logging.info('SVM Regressor')
        train_file = f"{path_reg}/{file}"
        for each in scaler_type:
            svr_regression = SupportVectorMachineRegression(train_file=train_file)
            svr_regression.rescale_data(scaler_type=each)
            svr_regression.grid_search()
            svr_regression.train_model()
            score = svr_regression.score()
            logging.info(f"R2 score for scaler {each}: {score}")


if __name__ == '__main__':
    main()
