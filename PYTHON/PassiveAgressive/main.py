from PA.PAC import PAClassifier
from PA.PAR import PARegressor


def main():
    """
    Main function. Run Passive Aggressive classification and regression tusks.
    train.csv: iris dataset
    train.tsv: Poznań flats prices
    :return: None
    """
    # SVR Classification
    train_file_location_classification = r'train.csv'
    rrclassifier = PAClassifier(trainfile=train_file_location_classification)
    rrclassifier.standalizer()
    rrclassifier.grid_search()
    rrclassifier.train_model()
    rrclassifier.output()

    # SVR Regressor
    train_file_location_regression = r'train.tsv'
    rrregressor = PARegressor(trainfile=train_file_location_regression)
    rrregressor.standalizer()
    rrregressor.grid_search()
    rrregressor.train_model()

    # Results
    rrclassifier.output()
    rrregressor.output()
