from LR.LinearRegression import *


file = open(r'train.tsv')
file2 = open(r'in.tsv')


def main():
    print("Linear Regression")
    trainFileLocation = r'train.tsv'
    inFileLocation = r'in.tsv'

    lr = LinearRegression(trainfile=trainFileLocation, infile=inFileLocation)
    lr.__str__()
    lr.train()

    scores = lr.output()

    for each in scores:
        print(each)



if __name__ == "__main__":
    main()
