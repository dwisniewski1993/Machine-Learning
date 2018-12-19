from LSTMNN.LSTMNN import LSTMNN


def main():
    """
    Main function. Run City Name Generator
    :return: None
    """
    trainFileLocation = r'Cities.txt'
    nnlstm = LSTMNN(path=trainFileLocation)
    nnlstm.train(num_epochs=1, temperature=1.5, num_characters=10)
    score = nnlstm.output(temperature=1.5, num_characters=10)
    print(f"Final Score: {score}")
