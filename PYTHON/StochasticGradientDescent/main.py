from SGD.Classifier import StochasticGradientDescentClassification
from SGD.Regressor import StochasticGradientDescentRegression


def main():
    """
    Main function. Run Stochastic Gradient Descent classification and regression tusks.
    train.csv: iris dataset
    train.tsv: Poznań flats prices
    :return: None
    """
    # SVR Classification
    train_file_location_classification = r'train.csv'
    sgd_classifier = StochasticGradientDescentClassification(train_file=train_file_location_classification)
    sgd_classifier.normalize()
    sgd_classifier.grid_search()
    sgd_classifier.train_model()
    sgd_classifier.score()

    # SVR Regression
    train_file_location_regression = r'train.tsv'
    sgd_regressor = StochasticGradientDescentRegression(train_file=train_file_location_regression)
    sgd_regressor.standardize()
    sgd_regressor.grid_search()
    sgd_regressor.train_model()
    sgd_regressor.score()
