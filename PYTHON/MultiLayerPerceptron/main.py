from MLP.Classification import MultiLayerPerceptronClassification
from MLP.Regression import MultilayerPerceptronRegression


def main():
    """
    Main function. Run MLP classification and regression tusks.
    train.csv: iris dataset
    train.tsv: Poznań flats prices
    :return: None
    """
    # MLP Classification
    train_file_location_classification = r'train.csv'
    mlp_classification = MultiLayerPerceptronClassification(train_file=train_file_location_classification)
    mlp_classification.grid_search()
    mlp_classification.train_model()
    mlp_classification.score()

    # MLP Regression
    train_file_location_regression = r'train.tsv'
    mlp_regression = MultilayerPerceptronRegression(train_file=train_file_location_regression)
    mlp_regression.grid_search()
    mlp_regression.train_model()
    mlp_regression.score()
