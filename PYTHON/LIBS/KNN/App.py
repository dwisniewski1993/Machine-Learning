from KNN.kNN import *


def main():
    print("KNN")

    trainFileLocation = r'winequality-red.csv'

    knnobj = KNNClassifier(trainfile=trainFileLocation)
    knnobj.standardLizer()
    knnobj.train()
    knnobj.output()

if __name__ == "__main__":
    main()
