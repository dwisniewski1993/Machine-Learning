import pandas as pd
import numpy as np
import keras

# String to float
def str_column_to_float(dataset, column):
    for row in dataset:
        col = row[column]
        row[column] = int(ord(row[column]))

#Labels to int
def label_to_int(dataset):
    label = list()
    for each in dataset:
        each = 1 if each == 'e' else 0
        label.append(each)
    return label


def main():
    print("Neural Network for Classification")

    #Num of labels
    NUM_CLASSES = 2

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

    #Make arrays
    dev0 = array2[:, 0:22]
    testA = array3[:, 0:22]

    # String to float
    for i in range(len(X[0])):
        str_column_to_float(X, i)
    for i in range(len(array2[0])):
        str_column_to_float(array2, i)
    for i in range(len(array3[0])):
        str_column_to_float(array3, i)
    Y = label_to_int(Y)
    Y = np.array(Y)
    print("Y.shape=", Y.shape)
    print(Y)

    #one hot encoding
    Y = keras.utils.to_categorical(Y, NUM_CLASSES)

    print("Przykładów uczących: ", X.shape[0])
    print("Przykładów test-A: ", dev0.shape[0])
    print("Przykładów dev-0: ", testA.shape[0])

    print("X.shape=", X.shape)
    print("X.shape[1]=", X.shape[1])
    print("Y.shape=", Y.shape)

    #Making model
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(512, activation='relu', input_shape=(X.shape[1],)))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(NUM_CLASSES, activation='softmax'))
    model.summary()

    #Compile
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.RMSprop(), metrics=['accuracy'])

    #Fit model
    model.fit(X, Y, batch_size=128, epochs=5, verbose=1)

    #Predict scores
    scores_dev = model.predict(dev0)
    scores_test = model.predict(testA)

    #Saving dev scores
    for each in scores_dev:
        print(each)
        out_dev0.write(str(each) + '\n')

    #Saving test scores
    for each in scores_test:
        print(each)
        out_testA.write(str(each) + '\n')


if __name__ == "__main__":
    main()
