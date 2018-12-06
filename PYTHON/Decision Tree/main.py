from DecisionTree.TreeClassifier import DTClassifier
from DecisionTree.TreeRegressor import DTRegressior


def main():
    """
    Main function. Run Decision Tree classification and regression tasks.
    train.csv: iris dataset
    train.tsv: Poznań flats prices
    :return: None
    """
    # Decision Tree Classification
    train_file_location_classification = r'train.csv'
    dtc = DTClassifier(trainfile=train_file_location_classification)
    dtc.normalize()
    dtc.grid_search()
    dtc.train_model()
    dtc.output()

    # Decision Tree Regression
    train_file_location_regression = r'train.tsv'
    dtr = DTRegressior(trainfile=train_file_location_regression)
    dtr.normalize()
    dtr.grid_search()
    dtr.train_model()
    dtr.output()
