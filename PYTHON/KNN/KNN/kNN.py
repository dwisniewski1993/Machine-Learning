import pandas as pd
from sklearn.preprocessing import *
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier


class KNNClassifier:
    def __init__(self, trainfile):
        self.trainFile = trainfile
        trainDataFrame = pd.read_csv(self.trainFile, header=0, sep=';')
        trainArray = trainDataFrame.values

        self.X = trainArray[:, 0:11]
        self.Y = trainArray[:, 11]

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

    def standardLizer(self):
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.fit_transform(self.X_test)

    def train(self):
        self.knnc = KNeighborsClassifier(n_neighbors=10)
        self.knnc.fit(self.X_train, self.Y_train)

    def output(self):
        print("Accuracy: {:.2f}".format(self.knnc.score(self.X_test, self.Y_test)))
