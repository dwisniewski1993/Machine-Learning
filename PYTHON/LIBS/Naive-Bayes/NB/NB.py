import pandas as pd
from sklearn.preprocessing import *
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB


class NaiveBayesClassifier:
    def __init__(self, trainfile):
        self.trainFile = trainfile
        trainDataFrame = pd.read_csv(self.trainFile)
        trainArray = trainDataFrame.values

        self.X = trainArray[:, 0:36]
        self.Y = trainArray[:, 36]

        for i in range(len(self.X[0])):
            self.toFloat(self.X, i)
        self.Y = self.labels_to_binary(self.Y)

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.3,
                                                                                random_state=0)

    def __str__(self):
        print("Features: {}, Labels: {}".format(self.X, self.Y))

    def rescale(self):
        scaler = MinMaxScaler(feature_range=(0, 1))
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.fit_transform(self.X_test)

    def normalize(self):
        scaler = Normalizer()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.fit_transform(self.X_test)

    def standalizer(self):
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.fit_transform(self.X_test)

    def train(self):
        self.nbc = GaussianNB()
        self.nbc.fit(self.X_train, self.Y_train)

    def output(self):
        print("Accuracy: {:.2f}".format(self.nbc.score(self.X_test, self.Y_test)))

    def labels_to_binary(self, dataset):
        label = []
        for each in dataset:
            each = 1 if each == 'won' else 0
            label.append(each)
        return label

    def toFloat(self, dataset, column):
        for row in dataset:
            try:
                row[column] = float(row[column])
            except:
                s = row[column]
                q = ''.join(str(ord(c)) for c in s)
                row[column] = float(q)
