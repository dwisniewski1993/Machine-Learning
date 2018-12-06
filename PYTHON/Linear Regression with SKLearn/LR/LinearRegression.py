import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler


class LinearRegression:
    def __init__(self, trainfile, infile):
        self.trainFile = trainfile
        self.inFile = infile

        trainDataFrame = pd.read_csv(self.trainFile, sep="\t")
        trainArray = trainDataFrame.values

        inDataFrame = pd.read_csv(self.inFile, sep="\t")
        self.inArray = inDataFrame.values

        self.X = trainArray[:, 1:6]
        self.Y = trainArray[:, 0]

        for i in range(len(self.X[0])):
            self.toFloat(self.X, i)
        for j in range(len(self.inArray[0])):
            self.toFloat(self.inArray, j)

    def __str__(self):
        print("Features values: {}, Labels values {}".format(self.X, self.Y))

    def rescale(self):
        scaler = MinMaxScaler(feature_range=(0, 1))
        self.X = scaler.fit_transform(self.X)
        self.inArray = scaler.fit_transform(self.inArray)

    def train(self):
        self.linreg = linear_model.LinearRegression()
        self.linreg.fit(self.X, self.Y)

    def output(self):
        output = self.linreg.predict(self.inArray)

        return output

    def toFloat(self, dataset, column):
        for row in dataset:
            try:
                row[column] = float(row[column])
            except:
                if row[column] == "True":
                    row[column] = float(1)
                elif row[column] == "False":
                    row[column] = float(0)
                else:
                    row[column] = 0.5
