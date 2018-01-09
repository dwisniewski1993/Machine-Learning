import pandas as pd
import numpy as np

#String to float
def str_column_to_float(dataset, column):
    for row in dataset:
        col = row[column]
        row[column] = float(ord(row[column]))

def main():
    print("Neural Network for Classification")

    # In files
    train_file = 'train/train.tsv'
    dev0_file = 'dev-0/in.tsv'
    testA_file = 'test-A/in.tsv'

    # Out files
    out_dev0 = open(r'dev-0/out.tsv', 'w', encoding='utf-8')
    out_testA = open(r'test-A/out.tsv', 'w', encoding='utf-8')

    # Load files
    dataframe = pd.read_csv(train_file, sep='\t', header=None)
    array = dataframe.values
    dataframe2 = pd.read_csv(dev0_file, sep='\t', header=None)
    array2 = dataframe2.values
    dataframe3 = pd.read_csv(testA_file, sep='\t', header=None)
    array3 = dataframe3.values

    # Separate X and Y
    X = array[:, 1:23]
    Y = array[:, 0]

    dev0 = array2[:, 0:22]
    testA = array3[:, 0:22]

    # String to float
    for i in range(len(X[0])):
        str_column_to_float(X, i)
    for i in range(len(array2[0])):
        str_column_to_float(array2, i)
    for i in range(len(array3[0])):
        str_column_to_float(array3, i)

    label = np.array(Y).reshape(len(Y), 1)

if __name__ == "__main__":
    main()