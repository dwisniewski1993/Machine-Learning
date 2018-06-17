from LR.LogisticRegression import *


def main():
    print("Logistic Regression")

    trainFilelocation = r'train.csv'

    logreg = LogisticsRegression(trainfile=trainFilelocation)
    logreg.modelFiting()
    logreg.output()


if __name__ == "__main__":
    main()
