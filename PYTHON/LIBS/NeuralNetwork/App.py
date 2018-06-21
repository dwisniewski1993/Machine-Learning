from RegNN.RegNN import NNLR


def main():
    print("Naural Network for regression")

    trainFileLocation = r'train.tsv'
    inFileLocation = r'in.tsv'
    neuralnet = NNLR(trainfile=trainFileLocation, infile=inFileLocation)
    scores = neuralnet.output()
    print(scores)

if __name__ == "__main__":
    main()
