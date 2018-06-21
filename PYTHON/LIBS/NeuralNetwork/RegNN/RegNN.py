import pandas as pd
import numpy as np
import tflearn
import os.path


class NNLR:
    def __init__(self, trainfile, infile):
        self.trainFile = trainfile
        self.inFile = infile
        self.model_name = 'poznan_flats_price_predict_model.model.index'

        trainDataFrame = pd.read_csv(self.trainFile, sep="\t")
        trainArray = trainDataFrame.values

        inDataFrame = pd.read_csv(self.inFile, sep="\t")
        self.inArray = inDataFrame.values

        self.num_features = len(trainArray[0])-1

        self.X = trainArray[:, 1:6]
        self.Y = trainArray[:, 0]

        for i in range(len(self.X[0])):
            self.toFloat(self.X, i)
        for j in range(len(self.inArray[0])):
            self.toFloat(self.inArray, j)

        self.Y = np.array(self.Y).reshape(len(self.Y), 1)

        #Define Network Model
        net = tflearn.input_data(shape=[None, self.num_features])
        net = tflearn.fully_connected(net, 32)
        net = tflearn.fully_connected(net, 64)
        net = tflearn.fully_connected(net, 128)
        net = tflearn.fully_connected(net, 256)
        net = tflearn.fully_connected(net, 128)
        net = tflearn.fully_connected(net, 64)
        net = tflearn.fully_connected(net, 32)
        net = tflearn.fully_connected(net, 4)
        linear = tflearn.fully_connected(net, 1, activation='linear')
        self.regression = tflearn.regression(linear, optimizer='sgd', metric='R2', loss='mean_square',
                                             learning_rate=0.001)
        self.model = tflearn.DNN(self.regression)

        if os.path.exists(self.model_name):
            self.load_model()
        else:
            self.train_model()
            self.save_model()

    def train_model(self):
        self.model.fit(self.X, self.Y, n_epoch=100, batch_size=32, show_metric=True)

    def save_model(self):
        self.model.save('poznan_flats_price_predict_model.model')

    def load_model(self):
        self.model.load('poznan_flats_price_predict_model.model')

    def output(self):
        output = self.model.predict(self.inArray)
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
