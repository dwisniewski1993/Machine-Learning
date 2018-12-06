from random import seed
from LR.LR import LR


def main():
    print('Self Made Linear Regression')

    seed(123)

    filename = 'train.tsv'
    filename2 = 'in.tsv'

    lr = LR(filename, filename2)
    scores = lr.linear_regression_sgd()

    for each in scores:
        print(each)


if __name__ == "__main__":
    main()
