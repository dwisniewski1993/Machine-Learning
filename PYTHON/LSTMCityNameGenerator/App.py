from LSTMNN.LSTMNN import LSTMNN


def main():
    print("City Name Generator")

    trainFileLocation = r'Cities.txt'
    nnlstm = LSTMNN(path=trainFileLocation)
    nnlstm.train(1, 1.5, 10)
    score = nnlstm.output(1.5, 10)
    print(score)

if __name__ == '__main__':
    main()
