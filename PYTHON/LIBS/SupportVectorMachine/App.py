from SVMClassifier.SVMClassifier import SVMC

'''
DATA:ALG:ACCURACY
Normalize:SVC:0.64
Rescale:SVC:0.89
Standardlize:SVC:0.93
None:SVC:0.91
Normalize:LINEAR:0.84
Rescale:LINEAR:0.80
Standardlize:LINEAR:0.82
None:LINEAR:0.91
Normalize:RBF:0.64
Rescale:RBF:0.91
Standardlize:RBF:0.91
None:RBF:0.91
Normalize:POLYMOLMIAL:0.29
Rescale:POLYMOLMIAL:0.53
Standardlize:POLYMOLMIAL:0.91
None:POLYMOLMIAL:0.89
'''

def main():
    print("SVM Classifier")

    trainFileLocation = r'train.csv'

    svmclassifier = SVMC(trainfile=trainFileLocation)
    svmclassifier.standalizer()
    svmclassifier.train_svc()
    svmclassifier.output()


if __name__ == '__main__':
    main()
