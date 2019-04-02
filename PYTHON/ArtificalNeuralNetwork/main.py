from ANN.NNClassifier import NNC
from ANN.NNRegressor import NNR


def main():
    """
    Main function. Run Artificial Neural Network classification and regression tasks.
    train.csv: iris dataset
    train.tsv: Poznań flats prices
    :return: None
    """
    # Artificial Neural Network Classification
    train_file_location_classification = r'train.csv'
    nnc = NNC(trainfile=train_file_location_classification)
    nnc.train_model()

    # Artificial Neural Network Regression
    train_file_location_regression = r'train.tsv'
    nnr = NNR(trainfile=train_file_location_regression)
    nnr.train_model()

    # Output
    nnc.output()
    nnr.output()
