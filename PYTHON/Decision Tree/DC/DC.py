import pandas as pd
from sklearn.preprocessing import *
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier


class DecisionTree:
    def __init__(self, trainfile):
        self.trainFile = trainfile
        trainDataFrame = pd.read_csv(self.trainFile)
        trainArray = trainDataFrame.values

        self.X = trainArray[:, 0:4]
        self.Y = trainArray[:, 4]

        self.Y = self.map_Labels(self.Y)

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.3,
                                                                                random_state=0)

    def __str__(self):
        print("Features: {}, Labels: {}".format(self.X, self.Y))

    def map_Labels(self, labels):
        maped = []
        for each in labels:
            if each == 'Iris-setosa':
                maped.append(0.0)
            elif each == 'Iris-versicolor':
                maped.append(1.0)
            else: maped.append(2.0)
        return maped

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
        self.dtc = DecisionTreeClassifier()
        self.dtc.fit(self.X_train, self.Y_train)

    def output(self):
        print("Accuracy: {:.2f}".format(self.dtc.score(self.X_test, self.Y_test)))