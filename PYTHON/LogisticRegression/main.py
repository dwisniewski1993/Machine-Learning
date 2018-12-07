from LR.LogisticRegression import *


def main():
    """
    Main function. Run Logistic Regression classification tasks.
    :return:
    """
    train_file_location = r'train.csv'
    logreg = LogReg(trainfile=train_file_location)
    logreg.normalize()
    logreg.grid_search()
    logreg.modelFiting()
    logreg.output()
