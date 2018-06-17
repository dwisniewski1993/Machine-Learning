from sklearn import linear_model
import numpy as np

file = open(r'train.tsv')
file2 = open(r'in.tsv')

def one_arg():
    print('ONE ARG LINEAR REGRESION')
    out = open(r'out.txt', 'w')
    flats_X = []
    flats_Y = []
    flats_X_test = []

    for line in file:
        line = line.split()

        # X
        flats_X.append(float(line[5]))

        # Y
        flats_Y.append(float(line[0]))

    for line in file2:
        line = line.split()
        flats_X_test.append(float(line[4]))

    # Reshape
    X = np.array(flats_X).reshape(len(flats_X), 1)
    Y = np.array(flats_Y)
    X_test = np.array(flats_X_test).reshape(len(flats_X_test), 1)

    # Objet regresji
    regr = linear_model.LinearRegression()

    # trenowanie modelu
    regr.fit(X, Y)

    # przewidywanie
    flats_y_pred = regr.predict(X_test)

    # Save to file
    for each in flats_y_pred:
        out.write(str(each) + '\n')
        print(each)