from NB.NB import NaiveBayesClassifier


def main():
    print('Naive-Bayes')

    trainFileLocation = r'train.csv'

    nb = NaiveBayesClassifier(trainfile=trainFileLocation)
    nb.normalize()
    nb.train()
    nb.output()


if __name__ == "__main__":
    main()
