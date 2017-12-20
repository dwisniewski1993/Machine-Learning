import pandas as pd
import numpy as np
import tflearn

#String to float
def str_column_to_float(dataset, column):
    for row in dataset:
        col = row[column]
        try:
            row[column] = float(row[column])
        except:
            if row[column] == True:
                row[column] = float(1)
            elif row[column] == False:
                row[column] = float(0)

def main():
    print("Naural Network for regression")

    # In files
    train_file = 'train/train.tsv'
    dev0_file = 'dev-0/in.tsv'
    testA_file = 'test-A/in.tsv'

    # Out files
    out_dev0 = open(r'dev-0/out.tsv', 'w', encoding='utf-8')
    out_testA = open(r'test-A/out.tsv', 'w', encoding='utf-8')

    #Load files
    dataframe = pd.read_csv(train_file, sep='\t', header=None)
    array = dataframe.values
    dataframe2 = pd.read_csv(dev0_file, sep='\t', header=None)
    array2 = dataframe2.values
    dataframe3 = pd.read_csv(testA_file, sep='\t', header=None)
    array3 = dataframe3.values

    # Separate X and Y
    X = array[:, 1:4]
    Y = array[:, 0]

    dev0 = array2[:, 0:3]
    testA = array3[:, 0:3]

    # String to float
    for i in range(len(X[0])):
        str_column_to_float(X, i)
    for i in range(len(array2[0])):
        str_column_to_float(array2, i)
    for i in range(len(array3[0])):
        str_column_to_float(array3, i)

    label = np.array(Y).reshape(1674, 1)

    net = tflearn.input_data(shape=[None, 3])
    net = tflearn.fully_connected(net, 64)
    net = tflearn.fully_connected(net, 32)
    net = tflearn.fully_connected(net, 16)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 3)
    linear = tflearn.fully_connected(net, 1, activation='linear')
    regression = tflearn.regression(linear, optimizer='sgd', metric='R2', loss='mean_square', learning_rate=0.01)

    model = tflearn.DNN(regression)

    model.fit(X, label, n_epoch=10, batch_size=16, show_metric=True)

    scores_dev0 = model.predict(dev0)
    scores_testA = model.predict(testA)

    for each in scores_dev0:
        out_dev0.write(str(each[0])+'\n')
    for each in scores_testA:
        out_testA.write(str(each[0])+'\n')

    #Close files
    out_testA.close()
    out_dev0.close()

if __name__ == "__main__":
    main()
