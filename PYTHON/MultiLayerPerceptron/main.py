from MLPTasks.MLPC import MultiLayerPerceptronClassifier
from MLPTasks.MLPR import MultolayerPerceptroRegressor


def main():
    """
    Main function. Run MLP classification and regression tusks.
    train.csv: iris dataset
    train.tsv: Poznań flats prices
    :return: None
    """
    # MLP Classification
    train_file_location_classification = r'train.csv'
    mlpc_executor = MultiLayerPerceptronClassifier(trainfile=train_file_location_classification)
    mlpc_executor.normalize()
    mlpc_executor.grid_search()
    mlpc_executor.train_model()
    mlpc_executor.output()

    # MLP Regression
    train_file_location_regression = r'train.tsv'
    mlpr_regression_executor = MultolayerPerceptroRegressor(trainfile=train_file_location_regression)
    mlpr_regression_executor.normalize()
    mlpr_regression_executor.grid_search()
    mlpr_regression_executor.train_model()
    mlpr_regression_executor.output()
