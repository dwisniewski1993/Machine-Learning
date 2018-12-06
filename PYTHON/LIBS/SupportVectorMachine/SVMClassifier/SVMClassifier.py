import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import *


class SVMC:
    def __init__(self, trainfile):
        self.trainFile = trainfile
        trainDataFrame = pd.read_csv(self.trainFile)
        trainArray = trainDataFrame.values
        np.random.shuffle(trainArray)

        self.X = trainArray[:, 0:4]
        self.Y = trainArray[:, 4]

        self.grided_params = []
        self.svmc = None

        self.Y = self.map_labels(self.Y)

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.3,
                                                                                random_state=0)

    def __str__(self):
        print("Features: {}, Labels: {}".format(self.X, self.Y))

    @staticmethod
    def map_labels(labels):
        maped = [0.0 if x == 'Iris-setosa' else 1.0 if x == 'Iris-versicolor' else 2.0 for x in labels]
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

    def output(self):
        print("Accuracy: {:.2f}".format(self.svmc.score(self.X_test, self.Y_test)))

    def grid_search(self):
        hyperparam_grid = {
            'kernel': ('linear', 'rbf', 'poly'),
            'gamma': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
            'C': [1, 3, 5, 7, 9]
        }
        classifier = GridSearchCV(svm.SVC(), hyperparam_grid)
        classifier.fit(self.X_train, self.Y_train)
        self.grided_params = [classifier.best_estimator_.kernel, classifier.best_estimator_.gamma,
                              classifier.best_estimator_.C]

    def train_model(self):
        self.svmc = svm.SVC(kernel=self.grided_params[0], gamma=self.grided_params[1], C=self.grided_params[2]). \
            fit(self.X_train, self.Y_train)
