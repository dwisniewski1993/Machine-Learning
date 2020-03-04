from LR.Classifier import *


def main():
    """
    Main function. Run Logistic Regression classification tasks.
    :return:
    """
    train_file_location = r'train.csv'
    logistic_regression = LogReg(train_file=train_file_location)
    logistic_regression.normalize()
    logistic_regression.grid_search()
    logistic_regression.train()
    logistic_regression.score()
