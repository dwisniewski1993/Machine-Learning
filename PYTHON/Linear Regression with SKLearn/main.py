from LR.LinReg import *


def main():
    """
    Main function. Run Linear Regression regression tasks.
    train.tsv: Poznań flats prices
    :return: None
    """
    trainFileLocation = r'train.tsv'
    lr = LinReg(trainfile=trainFileLocation)
    lr.train_model()
    lr.output()
