from SVM.SVMClassifier import SVMC
from SVM.SVMRegressor import SVMR


def main():
    """
    Main function. Run SVM classification and regression tusks.
    train.csv: iris dataset
    train.tsv: Poznań flats prices
    :return: None
    """
    # SVR Classification
    train_file_location_classification = r'train.csv'
    svmclassifier = SVMC(trainfile=train_file_location_classification)
    svmclassifier.normalize()
    svmclassifier.grid_search()
    svmclassifier.train_model()
    svmclassifier.output()

    # SVR Regressor
    train_file_location_regression = r'train.tsv'
    svrregressor = SVMR(trainfile=train_file_location_regression)
    svrregressor.normalize()
    svrregressor.grid_search()
    svrregressor.train_model()
    svrregressor.output()
