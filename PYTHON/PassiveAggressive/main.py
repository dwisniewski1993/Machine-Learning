from PA.Classifier import PassiveAggressiveClassification
from PA.Regressor import PassiveAggressiveRegression


def main():
    """
    Main function. Run Passive Aggressive classification and regression tusks.
    train.csv: iris dataset
    train.tsv: Poznań flats prices
    :return: None
    """
    # Passive Aggressive Classification
    train_file_location_classification = r'train.csv'
    pa_classifier = PassiveAggressiveClassification(train_file=train_file_location_classification)
    pa_classifier.standardize()
    pa_classifier.grid_search()
    pa_classifier.train_model()
    pa_classifier.score()

    # Passive Aggressive Regression
    train_file_location_regression = r'train.tsv'
    pa_regressor = PassiveAggressiveRegression(train_file=train_file_location_regression)
    pa_regressor.standardize()
    pa_regressor.grid_search()
    pa_regressor.train_model()
    pa_regressor.score()
