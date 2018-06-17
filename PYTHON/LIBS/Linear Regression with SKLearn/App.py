from LR.LinearRegression import *


def main():
    print("Linear Regression")
    trainFileLocation = r'train.tsv'
    inFileLocation = r'in.tsv'

    lr = LinearRegression(trainfile=trainFileLocation, infile=inFileLocation)
    lr.train()
    scores = lr.output()

    for each in scores:
        print(each)


if __name__ == "__main__":
    main()
