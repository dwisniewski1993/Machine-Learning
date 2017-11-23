from DataPreprocessing import FixingShityInSet, FixingShityTrainSet
from Functions import data_load, str_column_to_float
from sklearn import linear_model
import numpy as np
'''
Koniugacje
M:łem,łbym, ny
F:łam,łabym, na
'''

'''
NEW Dataset LOOK:
M/W | counted men con | counted women con
'''

def main():
    print('Ready-made logic Regression')
    #FixingShityInSet()
    #FixingShityTrainSet()

    out = open(r'out.tsv', 'w')

    filename = 'train_data_file.tsv'
    filename2 = 'in_data_file.tsv'

    dataset = data_load(filename)
    dataset2 = data_load(filename2)

    for i in range(len(dataset[0])):
        str_column_to_float(dataset, i)
    for i in range(len(dataset2[0])):
        str_column_to_float(dataset2, i)

    set_X = list()
    set_Y = list()
    arg1 = list()
    arg2 = list()

    set_X_test = list()
    arg1_test = list()
    arg2_test = list()

    for line in dataset:
        arg1.append(line[0])
        arg2.append(line[1])

        set_Y.append(line[2])
    set_X.append(arg1)
    set_X.append(arg2)

    for line in dataset2:
        arg1_test.append(line[0])
        arg2_test.append(line[1])
    set_X_test.append(arg1_test)
    set_X_test.append(arg2_test)

    X = np.array(set_X).reshape((3601424, 2))
    Y = np.array(set_Y)

    X_test = np.array(set_X_test).reshape((134675, 2))

    logreg = linear_model.LogisticRegression(C=1e5)

    logreg.fit(X, Y)

    predicts = logreg.predict(X_test)

    for each in predicts:
        if each == 10:
            each = 'M'
        elif each == 20:
            each = 'F'
        out.write(each + '\n')
        print(each)



    out.close()



if __name__ == "__main__":
    main()