import logging as log

from SVMClassifier.SVMClassifier import SVMC


def main():
    log.getLogger().setLevel(log.INFO)
    log.info('SVM Classifier')

    trainFileLocation = r'train.csv'

    svmclassifier = SVMC(trainfile=trainFileLocation)
    svmclassifier.normalize()
    svmclassifier.grid_search()
    svmclassifier.train_model()
    svmclassifier.output()
