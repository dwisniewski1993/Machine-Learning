from DC.DC import DecisionTree


def main():
    print("Decision Tree")

    trainFileLocation = r'train.csv'

    dc = DecisionTree(trainfile=trainFileLocation)
    dc.standalizer()
    dc.train()
    dc.output()

if __name__ == '__main__':
    main()
