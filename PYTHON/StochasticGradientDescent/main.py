from SGD.SGDClassifier import SGDC
from SGD.SGDRegressor import SGDR


def main():
    """
    Main function. Run Stochastic Gradient Descent classification and regression tusks.
    train.csv: iris dataset
    train.tsv: Poznań flats prices
    :return: None
    """
    # SVR Classification
    train_file_location_classification = r'train.csv'
    rrclassifier = SGDC(trainfile=train_file_location_classification)
    rrclassifier.standalizer()
    rrclassifier.grid_search()
    rrclassifier.train_model()

    # SVR Regressor
    train_file_location_regression = r'train.tsv'
    rrregressor = SGDR(trainfile=train_file_location_regression)
    rrregressor.standalizer()
    rrregressor.grid_search()
    rrregressor.train_model()

    # Results
    rrregressor.output()
    rrclassifier.output()
