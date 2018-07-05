import pandas as pd
import keras
import numpy as np


class NNC:
    def __init__(self, trainlocation, testalocation, dev0location):
        self.trainLocation = trainlocation
        self.testALocation = testalocation
        self.dev0Location = dev0location

        self.num_classes = 2

        trainDataFrame = pd.read_csv(self.trainLocation, sep="\t", header=None)
        trainArray = trainDataFrame.values

        testDataFrame = pd.read_csv(self.testALocation, sep="\t", header=None)
        testArray = testDataFrame.values

        devDataFrame = pd.read_csv(self.dev0Location, sep="\t", header=None)
        devArray = devDataFrame.values

        self.X = trainArray[:, 1:23]
        self.Y = trainArray[:, 0]
        print(self.X.shape)

        self.testA = testArray[:, 0:22]
        print(self.testA.shape)

        self.dev0 = devArray[:, 0:22]
        print(self.dev0.shape)

        for i in range(len(self.X[0])):
            self.toFloat(self.X, i)
        for j in range(len(self.testA[0])):
            self.toFloat(self.testA, j)
        for k in range(len(self.dev0[0])):
            self.toFloat(self.dev0, k)

        self.Y = self.label_to_one_hot(self.Y)

        self.model = self.build_model()

    def toFloat(self, dataset, column):
        for row in dataset:
            try:
                row[column] = int(ord(row[column]))
            except:
                pass

    def label_to_one_hot(self, dataset):
        label = []
        for each in dataset:
            each = [1, 0] if each == 'e' else [0, 1]
            label.append(each)
        return label

    def build_model(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(128, activation='relu', input_shape=(self.X.shape[1],)))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(256, activation='relu'))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(512, activation='relu'))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(1024, activation='relu'))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(512, activation='relu'))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(256, activation='relu'))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(128, activation='relu'))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(self.num_classes, activation='softmax'))
        model.summary()

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    def train_model(self):
        self.model.fit(self.X, self.Y, batch_size=128, epochs=10, verbose=1)

    def testA_output(self):
        output = self.model.predict(self.testA)
        return output

    def dev0_output(self):
        output = self.model.predict(self.dev0)
        return output
