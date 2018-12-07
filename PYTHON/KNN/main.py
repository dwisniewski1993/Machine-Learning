from KNN.KNNC import *
from KNN.KNNR import *


def main():
    """
    Main function. Run KNN classification and regression tasks.
    train.csv: iris dataset
    train.tsv: Poznań flats prices
    :return: None
    """
    # KNN Classification
    train_file_location = r'train.csv'
    knnclassfier = KNNClassifier(trainfile=train_file_location)
    knnclassfier.normalize()
    knnclassfier.grid_search()
    knnclassfier.train_model()
    knnclassfier.output()

    # KNN Regression
    train_file_location_regression = r'train.tsv'
    knnregressor = KNNRegression(trainfile=train_file_location_regression)
    knnregressor.normalize()
    knnregressor.grid_search()
    knnregressor.train_model()
    knnregressor.output()
