from SVMClassifier.SVMClassifier import SVMC


def main():
    print("SVM Classifier")

    trainFileLocation = r'train.csv'

    svmclassifier = SVMC(trainfile=trainFileLocation)
    svmclassifier.normalize()
    svmclassifier.grid_search()
    svmclassifier.train_model()
    svmclassifier.output()
