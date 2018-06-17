from sklearn import linear_model
import numpy as np

file = open(r'train.tsv')
file2 = open(r'in.tsv')


def many_arg():
    print('MANY ARGS LINEAR REGRESION')
    out_many = open(r'out_many.txt', 'w')
    flats_X = []
    flats_Y = []
    flats_X_test = []
    x2 = []
    x3 = []
    x5 = []
    x1_test = []
    x2_test = []
    x4_test = []

    for line in file:
        line = line.split()

        # X
        x2.append(float(line[2]))
        x3.append(float(line[3]))
        x5.append(float(line[5]))

        # Y
        flats_Y.append(float(line[0]))

    flats_X.append(x2)
    flats_X.append(x3)
    flats_X.append(x5)

    for line in file2:
        line = line.split()

        # X
        x1_test.append(float(line[1]))
        x2_test.append(float(line[2]))
        x4_test.append(float(line[4]))

    flats_X_test.append(x1_test)
    flats_X_test.append(x2_test)
    flats_X_test.append(x4_test)

    # Reshape
    X = np.array(flats_X).reshape((1674, 3))
    Y = np.array(flats_Y)
    X_test = np.array(flats_X_test).reshape(150, 3)

    # Objekt regresji
    regr = linear_model.LinearRegression()

    # Trenowanie modelu
    regr.fit(X, Y)

    # Prediction
    flats_predict = regr.predict(X_test)

    #Save to file
    for each in flats_predict:
        out_many.write(str(each ) +'\n')
        print(each)
