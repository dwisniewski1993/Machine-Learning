import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import tflearn


class MRNN:
    def __init__(self, featuresfile, labelsfile):
        self.featuresFile = featuresfile
        self.labelsFile = labelsfile

        featuresDataFrame = pd.read_csv(self.featuresFile, sep=",", header=None)
        featuresArray = featuresDataFrame.values

        labelsDataFrame = pd.read_csv(self.labelsFile, sep=",", header=None)
        labelsArray = labelsDataFrame.values

        self.INPUT_SHAPE = len(featuresArray[0])
        self.OUTPUT_NEURONS = len(labelsArray[0])
        self.LR = 0.001
        self.X = np.array(featuresArray)
        self.Y = np.array(labelsArray)

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.3,
                                                                                random_state=0)
        network = tflearn.input_data(shape=[None, self.INPUT_SHAPE])
        network = tflearn.fully_connected(network, 128, activation='sigmoid')
        network = tflearn.fully_connected(network, 256, activation='sigmoid')
        network = tflearn.fully_connected(network, 512, activation='sigmoid')
        network = tflearn.fully_connected(network, 1024, activation='sigmoid')
        network = tflearn.fully_connected(network, 2048, activation='sigmoid')
        network = tflearn.fully_connected(network, 1024, activation='sigmoid')
        network = tflearn.fully_connected(network, 512, activation='sigmoid')
        network = tflearn.fully_connected(network, 256, activation='sigmoid')
        network = tflearn.fully_connected(network, 128, activation='sigmoid')
        network = tflearn.fully_connected(network, 64, activation='sigmoid')
        network = tflearn.fully_connected(network, self.OUTPUT_NEURONS, activation='sigmoid')
        network = tflearn.regression(network, optimizer='adam', learning_rate=self.LR,
                                     loss='categorical_crossentropy', name='targets')
        self.model = tflearn.DNN(network)

    def train_network(self):
        self.model.fit(self.X_train, self.Y_train, validation_set=(self.X_test, self.Y_test),
                       n_epoch=10, show_metric=True, run_id='SILLY NET')


