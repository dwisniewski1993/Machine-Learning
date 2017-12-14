from functions import str_column_to_float
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def main():
    print("Logistic Regression on Mushrooms")

    #In files
    train_file = '../train/train.tsv'
    dev0_file = '../dev-0/in.tsv'
    testA_file = '../test-A/in.tsv'

    #Out files
    out_dev0 = open(r'../dev-0/out.tsv', 'w')
    out_testA = open(r'../test-A/out.tsv', 'w')

    #Load files
    dataframe = pd.read_csv(train_file, sep='\t')
    array = dataframe.values
    dataframe2 = pd.read_csv(dev0_file, sep='\t')
    array2 = dataframe2.values
    dataframe3 = pd.read_csv(testA_file, sep='\t')
    array3 = dataframe3.values

    # Separate X and Y
    X = array[:, 0:22]
    Y = array[:, 22]

    #String to float
    for i in range(len(X[0])):
        str_column_to_float(X, i)
    for i in range(len(array2[0])):
        str_column_to_float(array2, i)
    for i in range(len(array3[0])):
        str_column_to_float(array3, i)

    #Rescale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaledX = scaler.fit_transform(X)
    rescaledDev = scaler.fit_transform(array2)
    rescaledTestA = scaler.fit_transform(array3)

    #Making regression
    logreg = linear_model.LogisticRegression(C=1e5)
    logreg.fit(rescaledX, Y)

    #Predict Scores
    scores = logreg.predict(rescaledDev)
    scores2 = logreg.predict(rescaledTestA)

    #Print and write scores to files
    for each in scores:
        print(each)
        out_dev0.write(each + '\n')
    out_dev0.write('e \n')
    for each in scores2:
        print(each)
        out_testA.write(each + '\n')
    out_testA.write('e \n')

    #Close files
    out_testA.close()
    out_dev0.close()

if __name__ == "__main__":
    main()