import os.path

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential

from NN.Utils import Utils


class NeuralNetwork:
    def __init__(self):
        self.dataset = Utils()
        self.X_train = self.dataset.get_x_train()
        self.X_test = self.dataset.get_x_test()
        self.Y_train = self.dataset.get_y_train()
        self.Y_test = self.dataset.get_y_test()

        self.normalize_data()

        self.model = Sequential()

        self.model.add(LSTM(128, input_shape=(self.X_train.shape[1:]), activation='relu', return_sequences=True))
        self.model.add(Dropout(0.2))

        self.model.add(LSTM(128, activation='relu'))
        self.model.add(Dropout(0.1))

        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dropout(0.2))

        self.model.add(Dense(10, activation='softmax'))

        opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        if os.path.exists('MnistnetModel'):
            self.load_model()
        else:
            self.train_network(epochs=5)
            self.save_model()

        self.val_loss, self.val_acc = self.model.evaluate(self.X_test, self.Y_test)

    def normalize_data(self):
        self.X_train = tf.keras.utils.normalize(self.X_train, axis=1)
        self.X_test = tf.keras.utils.normalize(self.X_test, axis=1)

    def train_network(self, epochs):
        self.model.fit(self.X_train, self.Y_train, epochs=epochs, validation_data=(self.X_test, self.Y_test))

    def save_model(self):
        self.model.save('MnistnetModel')

    def load_model(self):
        self.model = tf.keras.models.load_model('MnistnetModel')

    def get_val_loss(self):
        return self.val_loss

    def get_val_acc(self):
        return self.val_acc

    def get_prediction(self, data, id):
        output = self.model.predict([data])
        output = np.argmax(output[id])
        self.dataset.show_image(data[id])
        return output
