from SVM.Classifier import SupportVectorMachineClassification
from SVM.Regressor import SupportVectorMachineRegression


def main():
    """
    Main function. Run SVM classification and regression tusks.
    train.csv: iris dataset
    train.tsv: Poznań flats prices
    :return: None
    """
    # SVR Classification
    train_file_location_classification = r'train.csv'
    svm_classifier = SupportVectorMachineClassification(train_file=train_file_location_classification)
    svm_classifier.normalize()
    svm_classifier.grid_search()
    svm_classifier.train_model()
    svm_classifier.score()

    # SVR Regression
    train_file_location_regression = r'train.tsv'
    svr_regression = SupportVectorMachineRegression(train_file=train_file_location_regression)
    svr_regression.standardize()
    svr_regression.grid_search()
    svr_regression.train_model()
    svr_regression.score()
